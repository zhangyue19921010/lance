// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use std::cell::Cell;
use std::cell::OnceCell;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arrow_array::Array;
use itertools::Itertools;
use lance_core::Result;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;
use crate::scalar::inverted::builder::{BLOCK_SIZE, ScoredDoc};
use crate::scalar::inverted::encoding::MAX_POSTING_BLOCK_SIZE;
use crate::scalar::inverted::query::{FtsSearchParams, Operator};
use crate::scalar::inverted::scorer::{Scorer, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{DocInfo, PostingList};

#[cfg(test)]
use super::AndWindowStats;
use super::{
    BULK_AND_MODE, BulkAndMode, CompetitiveFloorMode, DocCandidate, FLAT_SEARCH_PERCENT_THRESHOLD,
    HeadPosting, NormKCache, PostingIterator, ScoreContribution, SharedNormAddends,
    TERMINATED_DOC_ID, TailPosting, TopKCollector, USE_MAXSCORE_SEARCH, Wand, WandDocuments,
    atomic_store_max_f32, conservative_score_sum, next_up_f64, norm_addends,
    score_contributions_in_query_order, score_sum_cannot_compete, score_sum_upper_bound_factor,
};

// we were using row id as doc id in the past, which is u64,
// but now we are using the index as doc id, which is u32.
// so here WAND is a generic struct that can be used for both u32 and u64 doc ids.
impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    pub(crate) fn new(
        operator: Operator,
        postings: impl Iterator<Item = PostingIterator>,
        documents: &'a D,
        scorer: S,
    ) -> Self {
        let mut head = BinaryHeap::new();
        let mut lead = Vec::new();
        for posting in postings {
            if posting.current_doc_id().is_none() {
                continue;
            }
            let posting = Box::new(posting);
            if operator == Operator::And {
                lead.push(posting);
            } else {
                head.push(HeadPosting::new(posting));
            }
        }
        if operator == Operator::And {
            lead.sort_unstable_by_key(|posting| posting.cost());
        }

        let score_first_and_enabled = operator == Operator::And
            && matches!(lead.len(), 4 | 5)
            && lead.iter().enumerate().all(|(index, posting)| {
                lead[..index]
                    .iter()
                    .all(|prior| prior.token_id != posting.token_id)
            })
            && lead.iter().all(|posting| {
                !posting.has_grouped_terms()
                    && !posting.use_scorer_upper_bound
                    && posting.query_weight.is_finite()
                    && posting.query_weight >= 0.0
                    && matches!(
                        &posting.list,
                        PostingList::Compressed(list)
                            if list.block_size == MAX_POSTING_BLOCK_SIZE
                                && list.impacts.is_some()
                    )
            })
            && scorer
                .doc_norm(1)
                .is_some_and(|norm| norm.is_finite() && norm >= 0.0);

        Self {
            // Standalone leaf top-k keeps its established exclusive floor.
            // Composable cursors switch to an inclusive floor because their
            // downstream collector applies the final document-key tie-breaker.
            threshold: 0.0,
            floor_mode: CompetitiveFloorMode::Exclusive,
            operator,
            num_terms: if operator == Operator::And {
                lead.len()
            } else {
                head.len()
            },
            head,
            lead,
            tail: BinaryHeap::new(),
            tail_max_score: 0.0,
            up_to: None,
            and_max_score: f32::INFINITY,
            and_last_doc: None,
            and_candidate_score: None,
            score_first_and_enabled,
            #[cfg(test)]
            disable_score_first_and: false,
            score_first_and_suffix_bounds: SmallVec::new(),
            score_first_and_bounds_up_to: None,
            score_first_and_dense_range: None,
            score_first_and_norm_k: OnceCell::new(),
            #[cfg(test)]
            and_window_stats: AndWindowStats::default(),
            bulk_and_mode_override: None,
            #[cfg(test)]
            bulk_and_searches: 0,
            #[cfg(test)]
            maxscore_single_essential_windows: 0,
            #[cfg(test)]
            maxscore_general_windows: 0,
            #[cfg(test)]
            maxscore_essential_blocks_skipped: 0,
            #[cfg(test)]
            maxscore_top2_gap_windows: 0,
            #[cfg(test)]
            maxscore_top2_gap_empty_span: 0,
            #[cfg(test)]
            phrase_position_checks: Cell::new(0),
            documents,
            scorer,
            shared_threshold: None,
            shared_norm_addends: None,
        }
    }

    /// Test hook: force one conjunction mode so parity tests can compare bulk
    /// and classic search within one process.
    #[cfg(test)]
    pub(super) fn with_bulk_and_mode(mut self, mode: BulkAndMode) -> Self {
        self.bulk_and_mode_override = Some(mode);
        self
    }

    /// Test hook for old/new score-first parity in one process.
    #[cfg(test)]
    pub(super) fn with_score_first_and(mut self, enabled: bool) -> Self {
        self.disable_score_first_and = !enabled;
        self
    }

    pub(super) fn with_floor_mode(mut self, mode: CompetitiveFloorMode) -> Self {
        self.floor_mode = mode;
        self
    }

    /// Share one cross-partition top-k floor across a query's partitions.
    pub(crate) fn with_shared_threshold(mut self, shared: Arc<AtomicU32>) -> Self {
        self.shared_threshold = Some(shared);
        self
    }

    /// Share one norm addend table across a query's partitions. The caller
    /// must pass the same table only to searches whose scorers agree on
    /// [`Scorer::doc_norm`].
    pub(crate) fn with_shared_norm_addends(mut self, shared: SharedNormAddends) -> Self {
        self.shared_norm_addends = Some(shared);
        self
    }

    /// Per-search norm→BM25-denominator cache (Lucene's norm cache): the doc
    /// byte-norm slab plus the 256 possible denominator addends. Available
    /// when the DocSet scores quantized (256-document-block partitions) and the scorer
    /// factors `doc_weight` as `(K1+1)*freq/(freq + addend)`. Scoring through
    /// the cache is bit-identical to `scorer.doc_weight`, because both
    /// evaluate the same expressions on the same quantized lengths.
    pub(super) fn norm_k_cache(&self) -> Option<NormKCache<'a>> {
        let norms = self.documents.scoring_norms()?;
        let addends = match &self.shared_norm_addends {
            Some(shared) => shared
                .0
                .get_or_init(|| norm_addends(&self.scorer))
                .clone()?,
            None => norm_addends(&self.scorer)?,
        };
        Some((norms, addends))
    }

    /// Set the pruning threshold from this partition's k-th best, raised to the
    /// shared cross-partition floor when one is attached.
    pub(super) fn update_threshold(&mut self, local_kth: f32, wand_factor: f32) {
        let mut t = local_kth * wand_factor;
        if let Some(shared) = self.shared_threshold.as_ref() {
            atomic_store_max_f32(shared, local_kth);
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > t {
                t = g;
            }
        }
        if self.score_first_and_enabled && self.threshold <= 0.0 && t > 0.0 {
            // A zero-floor window did not prepare block bounds. Force the next
            // conjunction candidate to establish a score-first window.
            self.up_to = None;
            self.invalidate_score_first_and_window();
        }
        self.threshold = t;
    }

    /// Raise the local threshold to the shared cross-partition floor, picking up
    /// updates published by sibling partitions.
    pub(super) fn raise_to_shared_floor(&mut self, wand_factor: f32) {
        if let Some(shared) = self.shared_threshold.as_ref() {
            let g = f32::from_bits(shared.load(Ordering::Relaxed)) * wand_factor;
            if g > self.threshold {
                if self.score_first_and_enabled && self.threshold <= 0.0 && g > 0.0 {
                    self.up_to = None;
                    self.invalidate_score_first_and_window();
                }
                self.threshold = g;
            }
        }
    }

    // search the top-k documents that contain the query
    // returns the row_id, frequency and doc length
    pub(crate) fn search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        if params.phrase_slop.is_some() {
            self.invalidate_score_first_and_window();
            self.score_first_and_enabled = false;
        }

        if self.operator == Operator::Or
            && let Some((num_docs_selected, documents)) = self.documents.flat_documents()
            && num_docs_selected.saturating_mul(100)
                <= (*FLAT_SEARCH_PERCENT_THRESHOLD as usize).saturating_mul(self.documents.len())
        {
            return self.flat_search(params, documents, metrics);
        }

        // Top-k disjunctions over compressed lists can opt into the bulk
        // MAXSCORE path (Lucene MaxScoreBulkScorer style): it streams whole
        // blocks of the essential clauses into a window accumulator instead of
        // advancing doc-at-a-time through a heap.
        if *USE_MAXSCORE_SEARCH
            && self.operator == Operator::Or
            && params.phrase_slop.is_none()
            && !self.head.is_empty()
            && self.head.iter().all(|posting| {
                posting.posting.is_compressed() && !posting.posting.has_grouped_terms()
            })
        {
            return self.maxscore_search(params, metrics);
        }

        // Top-k conjunctions (AND and phrase) over compressed lists use the
        // bulk path: the same block-max window pruning, but candidates come
        // from a slice-level merge over decompressed blocks instead of per-doc
        // `next()` leapfrogging through boxed iterators.
        if self.operator == Operator::And
            && !self.lead.is_empty()
            && self
                .lead
                .iter()
                .all(|posting| posting.is_compressed() && !posting.has_grouped_terms())
            && {
                let mode = self
                    .bulk_and_mode_override
                    .unwrap_or_else(|| *BULK_AND_MODE);
                mode.enabled_for(self.lead.len())
                    || (mode == BulkAndMode::Auto
                        && self.lead.len() >= 4
                        && self.lead.iter().all(|posting| {
                            matches!(&posting.list, PostingList::Compressed(list)
                                if list.block_size == MAX_POSTING_BLOCK_SIZE && list.impacts.is_some())
                        }))
            }
        {
            #[cfg(test)]
            {
                self.bulk_and_searches += 1;
            }
            return self.and_bulk_search(params, metrics);
        }

        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut num_comparisons = 0;
        loop {
            self.raise_to_shared_floor(params.wand_factor);
            let Some((doc, and_score)) = self.next()? else {
                break;
            };
            num_comparisons += 1;

            let posting_doc_id = doc.doc_id();
            let Some(document_key) = self.documents.document_key(&doc) else {
                if self.operator == Operator::Or {
                    self.push_back_leads(doc.doc_id() + 1);
                }
                continue;
            };

            let doc_length = self.documents.doc_length(&doc);

            let score = if self.operator == Operator::Or {
                self.advance_all_tail(doc.doc_id(), None, None);
                if let Some(slop) = params.phrase_slop {
                    // Only front-load the BM25 total when a floor exists: with
                    // no floor the skip cannot fire, and scoring before
                    // position confirmation would make every position-failing
                    // candidate pay for a score it never uses.
                    let early_score =
                        (self.threshold > 0.0).then(|| self.score_in_query_order(doc_length));
                    if early_score
                        .is_some_and(|score| self.exclusive_score_cannot_beat_floor(score))
                    {
                        self.push_back_leads(doc.doc_id() + 1);
                        continue;
                    }
                    if !self.check_positions(slop as i32)? {
                        self.push_back_leads(doc.doc_id() + 1);
                        continue;
                    }
                    early_score.unwrap_or_else(|| self.score_in_query_order(doc_length))
                } else {
                    self.score_in_query_order(doc_length)
                }
            } else {
                self.advance_all_tail(doc.doc_id(), None, None);
                if let Some(slop) = params.phrase_slop {
                    // Only front-load the BM25 total when a floor exists: with
                    // no floor the skip cannot fire, and scoring before
                    // position confirmation would make every position-failing
                    // candidate pay for a score it never uses.
                    let early_score = (self.threshold > 0.0).then(|| {
                        if self.and_candidate_score.is_some() {
                            and_score
                        } else {
                            self.score_in_query_order(doc_length)
                        }
                    });
                    if early_score
                        .is_some_and(|score| self.exclusive_score_cannot_beat_floor(score))
                    {
                        continue;
                    }
                    if !self.check_positions(slop as i32)? {
                        continue;
                    }
                    early_score.unwrap_or_else(|| {
                        if self.and_candidate_score.is_some() {
                            and_score
                        } else {
                            self.score_in_query_order(doc_length)
                        }
                    })
                } else if self.and_candidate_score.is_some() {
                    and_score
                } else {
                    self.score_in_query_order(doc_length)
                }
            };

            if candidates.insert(
                ScoredDoc::new(document_key, score),
                doc_length,
                posting_doc_id,
                self.iter_term_freqs(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, params.wand_factor);
            }
            if self.operator == Operator::Or {
                self.push_back_leads(doc.doc_id() + 1);
            }
        }
        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }

    pub(super) fn flat_search(
        &mut self,
        params: &FtsSearchParams,
        documents: Box<dyn Iterator<Item = (u64, u64)> + '_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(vec![]);
        }

        // Posting iterators are forward-only, so selected DocIds are sorted
        // before driving the sparse executor.
        let documents = documents.sorted_unstable().collect::<Vec<_>>();
        let is_compressed = self
            .head
            .peek()
            .map(|posting| matches!(posting.posting.list, PostingList::Compressed(_)))
            .or_else(|| {
                self.lead
                    .first()
                    .map(|posting| matches!(posting.list, PostingList::Compressed(_)))
            })
            .unwrap_or(false);

        let mut num_comparisons = 0;
        let mut candidates = TopKCollector::new(limit, 0);
        for (doc_id, document_key) in documents {
            num_comparisons += 1;
            self.move_head_before_target_to_tail(doc_id);
            self.move_head_doc_to_lead(doc_id);

            if self.lead.is_empty() && self.tail.is_empty() {
                continue;
            }

            if !self.can_target_beat_threshold(doc_id) {
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);

            if self.operator == Operator::And && self.lead.len() < self.num_terms {
                self.advance_lead_to_head(doc_id + 1);
                continue;
            }

            // score the doc
            let doc_length = self
                .documents
                .flat_doc_length(doc_id, document_key, is_compressed);
            if self.operator == Operator::Or && !self.refine_or_candidate(doc_id, doc_length) {
                // `flat_search` evaluates an explicit allow-list of doc ids. Unlike the
                // regular WAND path, skipping to the next block boundary is unsafe here
                // because later doc ids from the same block may still be present in the
                // allow-list and need to be evaluated individually.
                self.advance_tail_and_lead_to_head(doc_id + 1);
                continue;
            }

            self.collect_tail_matches(doc_id);
            // Only front-load the BM25 total when a floor exists: with no floor
            // the skip cannot fire, and scoring before position confirmation
            // would make every position-failing candidate pay for a score it
            // never uses.
            let early_score = (self.threshold > 0.0).then(|| self.score_in_query_order(doc_length));
            if let Some(slop) = params.phrase_slop {
                if early_score.is_some_and(|score| self.exclusive_score_cannot_beat_floor(score)) {
                    self.advance_lead_to_head(doc_id + 1);
                    continue;
                }
                if !self.check_positions(slop as i32)? {
                    self.advance_lead_to_head(doc_id + 1);
                    continue;
                }
            }
            let score = early_score.unwrap_or_else(|| self.score_in_query_order(doc_length));

            if candidates.insert(
                ScoredDoc::new(document_key, score),
                doc_length,
                doc_id,
                self.iter_term_freqs(),
            )? && let Some(kth) = candidates.kth_score_if_full()
            {
                self.update_threshold(kth, params.wand_factor);
            }

            self.advance_lead_to_head(doc_id + 1);
        }
        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }

    /// Exclusive top-k floor: a finite complete score at or below `threshold`
    /// cannot enter the heap, so phrase confirmation is wasted work.
    ///
    /// The score is the same query-order f32 the heap compares, so this uses
    /// `accepts_score` rather than the inflated partial-sum bound. Fail-closed:
    /// `threshold == 0` (heap not full) never skips.
    #[inline]
    pub(super) fn exclusive_score_cannot_beat_floor(&self, score: f32) -> bool {
        self.threshold > 0.0
            && score.is_finite()
            && !self.floor_mode.accepts_score(score, self.threshold)
    }

    /// Calculate the current document's score in query order.
    ///
    /// WAND deliberately reorders postings by doc id, cost, and score bound.
    /// Floating-point addition is not associative, so that execution order
    /// must not leak into the public score or top-k tie boundary.
    pub(super) fn score_in_query_order(&self, doc_length: u32) -> f32 {
        if self.lead.windows(2).all(|window| {
            (window[0].position, window[0].token_id) <= (window[1].position, window[1].token_id)
        }) {
            return self.lead.iter().fold(0.0_f32, |score, posting| {
                score
                    + posting.doc().map_or(0.0, |doc| {
                        posting.score(&self.scorer, doc.frequency(), doc_length)
                    })
            });
        }
        let contributions = self
            .lead
            .iter()
            .filter_map(|posting| {
                posting.doc().map(|doc| {
                    (
                        (posting.position, posting.token_id),
                        posting.score(&self.scorer, doc.frequency(), doc_length),
                    )
                })
            })
            .collect::<SmallVec<[ScoreContribution; 8]>>();
        score_contributions_in_query_order(contributions)
    }

    // iterate over all the preceding terms and collect the term index and frequency
    fn iter_term_freqs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.lead.iter().filter_map(|posting| {
            posting
                .doc()
                .map(|doc| (posting.term_index(), doc.frequency()))
        })
    }

    fn and_candidate_cannot_beat_threshold(&self, doc_length: u32) -> bool {
        if self.operator != Operator::And
            || self.threshold <= 0.0
            || self.num_terms < 2
            || self.lead.len() != self.num_terms
        {
            return false;
        }

        let Some((first, remaining)) = self.lead.split_first() else {
            return false;
        };
        let Some(doc) = first.doc() else {
            return false;
        };

        let remaining_upper_bound = remaining
            .iter()
            .map(|posting| f64::from(posting.block_max_score(&self.scorer)))
            .sum::<f64>();
        score_sum_cannot_compete(
            first.score(&self.scorer, doc.frequency(), doc_length),
            remaining_upper_bound,
            self.threshold,
            score_sum_upper_bound_factor(self.num_terms),
            self.floor_mode,
        )
    }

    pub(super) fn invalidate_score_first_and_window(&mut self) {
        if self.score_first_and_enabled {
            self.score_first_and_suffix_bounds.clear();
            self.score_first_and_bounds_up_to = None;
            self.score_first_and_dense_range = None;
        }
        self.and_candidate_score = None;
    }

    pub(super) fn prepare_score_first_and_window(&mut self) {
        if !self.score_first_and_enabled {
            return;
        }
        self.score_first_and_suffix_bounds.clear();
        self.score_first_and_bounds_up_to = None;
        self.score_first_and_dense_range = None;
        if self.threshold <= 0.0 {
            return;
        }
        #[cfg(test)]
        if self.disable_score_first_and {
            return;
        }
        let Some(up_to) = self.up_to else {
            return;
        };

        if self.and_max_score < self.threshold {
            return;
        }

        let mut dense_from = 0;
        let mut dense_to = up_to;
        for posting in &self.lead {
            let Some(current_doc) = posting.current_doc_id() else {
                return;
            };
            let PostingList::Compressed(list) = &posting.list else {
                return;
            };
            if list.block_size == 0 {
                return;
            }
            let expected_blocks = (list.length as usize).div_ceil(list.block_size);
            if list.blocks.len() != expected_blocks || posting.block_idx >= list.blocks.len() {
                return;
            }
            let Some(block_start) = posting.block_idx.checked_mul(list.block_size) else {
                return;
            };
            let Some(remaining) = (list.length as usize).checked_sub(block_start) else {
                return;
            };
            let block_count = remaining.min(list.block_size);
            if block_count == 0 {
                return;
            }
            let block_first = list.block_least_doc_id(posting.block_idx);
            let Some(block_last) = list
                .impacts
                .as_ref()
                .and_then(|impacts| impacts.level0_doc_up_to(posting.block_idx))
            else {
                return;
            };
            if block_last < block_first
                || u64::from(block_last) - u64::from(block_first) + 1 != block_count as u64
            {
                return;
            }
            dense_from = dense_from.max(current_doc).max(u64::from(block_first));
            dense_to = dense_to.min(u64::from(block_last));
        }
        if dense_from > dense_to {
            return;
        }

        self.score_first_and_suffix_bounds
            .resize(self.lead.len() + 1, 0.0);
        for index in (0..self.lead.len()).rev() {
            let bound = self.lead[index].block_max_score(&self.scorer);
            if !bound.is_finite() || bound < 0.0 {
                self.score_first_and_suffix_bounds.clear();
                return;
            }
            self.score_first_and_suffix_bounds[index] =
                next_up_f64(f64::from(bound) + self.score_first_and_suffix_bounds[index + 1]);
        }
        self.score_first_and_bounds_up_to = Some(up_to);
        self.score_first_and_dense_range = Some((dense_from, dense_to));
    }

    #[inline]
    fn score_first_and_cannot_compete(&self, partial_score: f32, next_clause: usize) -> bool {
        self.score_first_and_bounds_up_to == self.up_to
            && self
                .score_first_and_suffix_bounds
                .get(next_clause)
                .is_some_and(|remaining_upper_bound| {
                    score_sum_cannot_compete(
                        partial_score,
                        *remaining_upper_bound,
                        self.threshold,
                        score_sum_upper_bound_factor(self.num_terms),
                        self.floor_mode,
                    )
                })
    }

    fn and_next_target_after(&self, doc: u64) -> Option<u64> {
        if self
            .lead
            .first()
            .is_some_and(|posting| posting.is_compressed())
            && doc >= u64::from(u32::MAX)
        {
            None
        } else {
            doc.checked_add(1)
        }
    }

    // find the next doc candidate
    // Find the next term-level candidate doc. The returned score is the exact
    // contribution from the current `lead` set; additional score can still come
    // from `tail` iterators that are advanced to the same doc later.
    pub(super) fn next(&mut self) -> Result<Option<(DocInfo, f32)>> {
        if self.operator == Operator::And {
            let candidate = self.next_and_candidate();
            #[cfg(test)]
            {
                if candidate.is_some() {
                    self.and_window_stats.candidates_returned += 1;
                }
            }
            return Ok(candidate.map(|doc| (doc, self.and_candidate_score.unwrap_or(0.0))));
        }

        loop {
            let Some(target) = self.head_doc() else {
                if self.advance_tail_to_next_or_window() {
                    continue;
                }
                return Ok(None);
            };

            if self.up_to.is_none_or(|up_to| target > up_to) {
                self.update_max_scores(target);
            }
            self.move_head_doc_to_lead(target);
            if self.lead.is_empty() {
                continue;
            }

            // Block-Max WAND pruning: skip the whole window when its score upper
            // bound cannot reach the top-k threshold.
            if self.threshold > 0.0
                && self
                    .floor_mode
                    .rejects_upper_bound(f64::from(self.or_block_window_max()), self.threshold)
            {
                // On the final block `up_to` is the `u64::MAX` sentinel; step once
                // there to avoid seeking past the valid doc id range.
                let mut skip_to = match self.up_to {
                    Some(up_to) if up_to < u32::MAX as u64 => up_to + 1,
                    _ => target + 1,
                };
                // The narrow window is dead; if the whole level1 group is dead
                // too, hop over it in one advance.
                let group_skip = self.or_group_skip_to();
                if let Some(group_skip_to) = group_skip {
                    skip_to = skip_to.max(group_skip_to);
                }
                self.push_back_leads(skip_to);
                continue;
            }

            let Some(first_doc) = self.lead.first().and_then(|posting| posting.doc()) else {
                self.push_back_leads(target + 1);
                continue;
            };
            let doc_length = self.documents.doc_length(&first_doc);
            let mut lead_score = 0.0;
            if let Some(first_posting) = self.lead.first() {
                lead_score += first_posting.score(&self.scorer, first_doc.frequency(), doc_length);
            }
            for posting in self.lead.iter().skip(1) {
                if let Some(lead_doc) = posting.doc() {
                    lead_score += posting.score(&self.scorer, lead_doc.frequency(), doc_length);
                }
            }

            while !self.floor_mode.accepts_score(lead_score, self.threshold) {
                if score_sum_cannot_compete(
                    lead_score,
                    self.tail_upper_bound_sum(),
                    self.threshold,
                    score_sum_upper_bound_factor(self.num_terms),
                    self.floor_mode,
                ) {
                    self.push_back_leads(first_doc.doc_id() + 1);
                    break;
                }
                if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                    // Dynamic heap-order accumulation may be below the query-
                    // order score. Let the final canonical scorer decide.
                    break;
                }
            }

            if !self.lead.is_empty() {
                return Ok(Some((first_doc, lead_score)));
            }
        }
    }
    fn next_and_candidate(&mut self) -> Option<DocInfo> {
        self.and_candidate_score = None;
        if self.lead.len() < self.num_terms {
            return None;
        }
        // Two- and three-clause conjunctions use the bulk SIMD kernels by
        // default. Restrict this classic-path optimization to four and five
        // clauses, keeping one clause as the benchmark's drift control.
        let is_vectorized_search_enabled = matches!(self.lead.len(), 4 | 5);
        if let Some(last_doc) = self.and_last_doc
            && self
                .lead
                .first()
                .and_then(|posting| posting.current_doc_id())
                == Some(last_doc)
        {
            let next_target = self.and_advance_target(self.and_next_target_after(last_doc)?);
            if next_target == TERMINATED_DOC_ID {
                return None;
            }
            self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
        }

        'advance_head: loop {
            let doc = self
                .lead
                .first()
                .and_then(|posting| posting.current_doc_id())?;
            if self.up_to.is_none_or(|up_to| doc > up_to) {
                let next_target = self.and_advance_target(doc);
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                if next_target != doc {
                    self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                    continue;
                }
            }

            let mut is_score_first = self.score_first_and_bounds_up_to == self.up_to
                && self.score_first_and_suffix_bounds.len() == self.lead.len() + 1
                && self
                    .score_first_and_dense_range
                    .is_some_and(|(from, to)| doc >= from && doc <= to);
            let mut partial_score = 0.0_f32;
            let mut score_contributions = SmallVec::<[ScoreContribution; 8]>::new();
            let mut score_first_doc_norm = 0.0;
            if is_score_first {
                let lead_doc = self.lead.first().and_then(|posting| posting.doc())?;
                let cached_doc_norm = self
                    .score_first_and_norm_k
                    .get_or_init(|| {
                        self.norm_k_cache().filter(|(_, cache)| {
                            cache
                                .iter()
                                .all(|addend| addend.is_finite() && *addend >= 0.0)
                        })
                    })
                    .as_ref()
                    .and_then(|(norms, cache)| {
                        norms.get(doc as usize).map(|&code| cache[code as usize])
                    });
                let doc_norm = cached_doc_norm
                    .or_else(|| self.scorer.doc_norm(self.documents.doc_length(&lead_doc)));
                if let Some(doc_norm) = doc_norm
                    && doc_norm.is_finite()
                    && doc_norm >= 0.0
                {
                    score_first_doc_norm = doc_norm;
                    let contribution = self.lead[0].query_weight
                        * bm25_doc_weight_with_norm(lead_doc.frequency(), doc_norm);
                    partial_score = contribution;
                    score_contributions
                        .push(((self.lead[0].position, self.lead[0].token_id), contribution));
                    if self.score_first_and_cannot_compete(partial_score, 1) {
                        #[cfg(test)]
                        {
                            self.and_window_stats.score_first_rejections += 1;
                        }
                        let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                        if next_target == TERMINATED_DOC_ID {
                            return None;
                        }
                        self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                        continue;
                    }
                } else {
                    is_score_first = false;
                }
            }

            for index in 1..self.lead.len() {
                let posting = &mut self.lead[index];
                if posting.current_doc_id()? < doc {
                    posting.next_doc_id(doc, is_vectorized_search_enabled);
                }
                let next = posting.current_doc_id()?;
                if next > doc {
                    let next_target = self.and_advance_target(next);
                    if next_target == TERMINATED_DOC_ID {
                        return None;
                    }
                    self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                    continue 'advance_head;
                }

                if is_score_first {
                    let posting_doc = posting.doc()?;
                    let contribution = posting.query_weight
                        * bm25_doc_weight_with_norm(posting_doc.frequency(), score_first_doc_norm);
                    partial_score += contribution;
                    score_contributions.push(((posting.position, posting.token_id), contribution));
                    if self.score_first_and_cannot_compete(partial_score, index + 1) {
                        #[cfg(test)]
                        {
                            self.and_window_stats.score_first_rejections += 1;
                        }
                        let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                        if next_target == TERMINATED_DOC_ID {
                            return None;
                        }
                        self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                        continue 'advance_head;
                    }
                }
            }

            let lead_doc = self.lead.first().and_then(|posting| posting.doc())?;
            let doc_length = self.documents.doc_length(&lead_doc);
            if !is_score_first && self.and_candidate_cannot_beat_threshold(doc_length) {
                let next_target = self.and_advance_target(self.and_next_target_after(doc)?);
                if next_target == TERMINATED_DOC_ID {
                    return None;
                }
                self.lead[0].next_doc_id(next_target, is_vectorized_search_enabled);
                continue;
            }
            if is_score_first {
                self.and_candidate_score =
                    Some(score_contributions_in_query_order(score_contributions));
            }
            self.and_last_doc = Some(doc);
            return Some(lead_doc);
        }
    }

    #[allow(clippy::vec_box)]
    pub(super) fn head_doc(&self) -> Option<u64> {
        self.head.peek().map(HeadPosting::doc_id)
    }

    fn push_head(&mut self, posting: Box<PostingIterator>) {
        if posting.doc().is_some() {
            self.head.push(HeadPosting::new(posting));
        }
    }

    pub(super) fn move_head_doc_to_lead(&mut self, target: u64) {
        while self.head_doc() == Some(target) {
            if let Some(posting) = self.head.pop() {
                self.lead.push(posting.posting);
            }
        }
    }

    // Move all head iterators that are already known to be behind `target`
    // into `tail`, possibly overflowing low-value entries back into `head`.
    pub(super) fn move_head_before_target_to_tail(&mut self, target: u64) {
        if self.threshold <= 0.0 {
            while matches!(self.head_doc(), Some(doc_id) if doc_id < target) {
                if let Some(mut posting) = self.head.pop().map(|posting| posting.posting) {
                    posting.next(target);
                    self.push_head(posting);
                }
            }
            return;
        }

        while matches!(self.head_doc(), Some(doc_id) if doc_id < target) {
            if let Some(posting) = self.head.pop() {
                let upper_bound = posting.posting.global_upper_bound(&self.scorer);
                if let Some(mut evicted) =
                    self.insert_tail_with_overflow(posting.posting, upper_bound)
                {
                    evicted.next(target);
                    self.push_head(evicted);
                }
            }
        }
    }

    /// Upper bound on the score of any document in the window `[target, up_to]`
    /// for a disjunction. Sums the block-max of every overlapping iterator:
    /// `lead`, `head` (later docs still in the window, which
    /// `can_target_beat_threshold` omits), and the individual tail bounds.
    fn or_block_window_max(&self) -> f32 {
        conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.window_max_score(self.up_to, &self.scorer))
                .chain(
                    self.head
                        .iter()
                        .map(|posting| posting.posting.window_max_score(self.up_to, &self.scorer)),
                )
                .chain(self.tail.iter().map(|posting| posting.upper_bound)),
        )
    }

    fn can_target_beat_threshold(&mut self, target: u64) -> bool {
        if self.up_to.is_none_or(|up_to| target > up_to) {
            self.update_max_scores(target);
        }

        let mut possible_matches = self.lead.len();
        for posting in &self.tail {
            if matches!(posting.posting.block_first_doc(), Some(block_doc) if block_doc <= target) {
                possible_matches += 1;
            }
        }
        let sum = conservative_score_sum(
            self.lead
                .iter()
                .map(|posting| posting.window_max_score(self.up_to, &self.scorer))
                .chain(
                    self.tail
                        .iter()
                        .filter(|posting| {
                            matches!(
                                posting.posting.block_first_doc(),
                                Some(block_doc) if block_doc <= target
                            )
                        })
                        .map(|posting| posting.posting.window_max_score(self.up_to, &self.scorer)),
                ),
        );

        match self.operator {
            Operator::And => {
                possible_matches >= self.num_terms
                    && !self
                        .floor_mode
                        .rejects_upper_bound(f64::from(sum), self.threshold)
            }
            Operator::Or => !self
                .floor_mode
                .rejects_upper_bound(f64::from(sum), self.threshold),
        }
    }

    pub(super) fn update_max_scores(&mut self, target: u64) {
        // Refresh the block-max window for the current target. The resulting
        // `up_to` is the furthest doc id for which this block-max view remains
        // valid. Like Lucene's WANDScorer, the boundary comes from the cheap
        // clauses only, and the refresh avoids allocating: heaps are recycled
        // through their backing vectors (shallow_next never changes the doc a
        // head entry is ordered by, so heapify restores the same shape).
        let lead_cost = self
            .lead
            .iter()
            .map(|posting| posting.cost())
            .min()
            .unwrap_or(usize::MAX);
        let mut narrow_up_to = TERMINATED_DOC_ID;
        for posting in &mut self.lead {
            posting.shallow_next(target);
            narrow_up_to = narrow_up_to.min(posting.block_end_doc());
        }

        let mut head_postings = std::mem::take(&mut self.head).into_vec();
        for posting in &mut head_postings {
            // Unlike Lucene, every head clause participates in the boundary:
            // the refresh is allocation-free and answers from the anchored
            // level caches, so frequent refreshes are cheap, while keeping the
            // window inside every clause's current block keeps all the bounds
            // at tight level0 values.
            let doc_id = posting.doc_id();
            posting.posting.shallow_next(doc_id);
            narrow_up_to = narrow_up_to.min(posting.posting.block_end_doc());
        }

        let mut tail_postings = std::mem::take(&mut self.tail).into_vec();
        for tail_posting in &mut tail_postings {
            tail_posting.posting.shallow_next(target);
        }

        if narrow_up_to == TERMINATED_DOC_ID
            && let Some(top) = tail_postings
                .iter()
                .min_by_key(|posting| posting.posting.cost())
            && top.posting.cost() <= lead_cost
        {
            narrow_up_to = narrow_up_to.min(top.posting.block_end_doc().max(target));
        }

        self.up_to = Some(narrow_up_to);
        self.head = BinaryHeap::from(head_postings);

        self.tail_max_score = 0.0;
        for tail_posting in tail_postings {
            let posting = tail_posting.posting;
            let upper_bound = match posting.block_first_doc() {
                Some(block_doc) if block_doc <= target => {
                    posting.window_max_score(self.up_to, &self.scorer)
                }
                _ => 0.0,
            };
            if let Some(mut evicted) = self.insert_tail_with_overflow(posting, upper_bound) {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    /// After the narrow window proved skippable, try widening the skip to the
    /// level1 group boundary, in the spirit of Lucene's `getSkipUpTo`. All
    /// bounds come from the anchored level caches, so a failed attempt costs a
    /// few loads and float adds — unlike an eager wide-window probe.
    ///
    /// The group boundary is the minimum current-group end over every live
    /// iterator, so each iterator's level1 score is a valid bound over
    /// `[target, group_up_to]`.
    fn or_group_skip_to(&self) -> Option<u64> {
        let mut group_up_to = TERMINATED_DOC_ID;
        for posting in &self.lead {
            let (doc_up_to, _) = posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        for posting in self.head.iter() {
            let (doc_up_to, _) = posting.posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        for tail_posting in self.tail.iter() {
            let (doc_up_to, _) = tail_posting.posting.impact_group_bound(&self.scorer)?;
            group_up_to = group_up_to.min(doc_up_to);
        }
        if self.up_to.is_some_and(|up_to| group_up_to <= up_to) {
            // No gain over the narrow window skip.
            return None;
        }

        // Second pass over the memoized bounds: sum only the iterators that
        // can produce a doc inside the skipped range.
        let mut bounds = SmallVec::<[f32; 8]>::new();
        for posting in &self.lead {
            bounds.push(posting.impact_group_bound(&self.scorer)?.1);
        }
        for posting in self
            .head
            .iter()
            .filter(|posting| posting.doc_id() <= group_up_to)
        {
            bounds.push(posting.posting.impact_group_bound(&self.scorer)?.1);
        }
        for tail_posting in self.tail.iter().filter(|tail_posting| {
            matches!(
                tail_posting.posting.block_first_doc(),
                Some(block_doc) if block_doc <= group_up_to
            )
        }) {
            bounds.push(tail_posting.posting.impact_group_bound(&self.scorer)?.1);
        }
        let bounds_sum = conservative_score_sum(bounds.into_iter());
        self.floor_mode
            .rejects_upper_bound(f64::from(bounds_sum), self.threshold)
            .then_some(group_up_to.saturating_add(1))
    }

    fn refine_or_candidate(&mut self, target: u64, doc_length: u32) -> bool {
        if self.threshold <= 0.0 {
            return true;
        }

        let mut lead_score = self
            .lead
            .iter()
            .filter_map(|posting| {
                posting
                    .doc()
                    .map(|doc| posting.score(&self.scorer, doc.frequency(), doc_length))
            })
            .sum::<f32>();

        while !self.floor_mode.accepts_score(lead_score, self.threshold) {
            if score_sum_cannot_compete(
                lead_score,
                self.tail_upper_bound_sum(),
                self.threshold,
                score_sum_upper_bound_factor(self.num_terms),
                self.floor_mode,
            ) {
                return false;
            }
            if !self.advance_tail_top(target, doc_length, &mut lead_score) {
                // No remaining posting can be advanced, but f32 execution-
                // order rounding may still leave the canonical score at or
                // above the inclusive floor.
                return true;
            }
        }

        true
    }

    fn collect_tail_matches(&mut self, target: u64) {
        let mut remaining = Vec::with_capacity(self.tail.len());
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc().map(|doc| doc.doc_id()) {
                Some(doc_id) if doc_id == target => self.lead.push(posting),
                Some(_) => remaining.push(posting),
                None => {}
            }
        }

        for posting in remaining {
            self.push_head(posting);
        }
    }

    fn advance_tail_and_lead_to_head(&mut self, least_id: u64) {
        let mut postings = Vec::with_capacity(self.tail.len() + self.lead.len());
        while let Some(tail) = self.tail.pop() {
            postings.push(tail.posting);
        }
        self.tail_max_score = 0.0;
        postings.append(&mut self.lead);
        for mut posting in postings {
            posting.next(least_id);
            self.push_head(posting);
        }
    }

    fn advance_lead_to_head(&mut self, least_id: u64) {
        let lead = std::mem::take(&mut self.lead);
        for mut posting in lead {
            posting.next(least_id);
            self.push_head(posting);
        }
        // In the flat-search path this is only called after `collect_tail_matches`,
        // which drains the current tail into either `lead` or `head`. At this
        // point `tail` is expected to be empty, so clearing it is a no-op that
        // just resets the cached `tail_max_score`.
        debug_assert!(self.tail.is_empty());
        self.clear_tail();
    }

    fn clear_tail(&mut self) {
        self.tail.clear();
        self.tail_max_score = 0.0;
    }

    fn insert_tail(&mut self, posting: Box<PostingIterator>, upper_bound: f32) {
        self.tail_max_score = next_up_f64(self.tail_max_score + f64::from(upper_bound));
        self.tail
            .push(TailPosting::new(upper_bound, posting.cost(), posting));
    }

    fn insert_tail_with_overflow(
        &mut self,
        posting: Box<PostingIterator>,
        upper_bound: f32,
    ) -> Option<Box<PostingIterator>> {
        // Keep only the lagging iterators that are most useful for deciding the
        // current candidate. If a stronger tail entry arrives, evict the weakest
        // one back to the caller so it can be advanced into `head`.
        if self.threshold <= 0.0 || upper_bound <= 0.0 {
            return Some(posting);
        }

        let parked_upper_bound = self.tail_upper_bound_sum() + f64::from(upper_bound);
        if score_sum_cannot_compete(
            0.0,
            parked_upper_bound,
            self.threshold,
            score_sum_upper_bound_factor(self.num_terms),
            self.floor_mode,
        ) {
            self.insert_tail(posting, upper_bound);
            return None;
        }

        if self.tail.is_empty() {
            return Some(posting);
        }

        let candidate = TailPosting::new(upper_bound, posting.cost(), posting);
        if let Some(top) = self.tail.peek()
            && top > &candidate
        {
            let evicted = self.tail.pop().expect("peeked tail posting should exist");
            self.remove_tail_upper_bound(evicted.upper_bound);
            self.tail_max_score = next_up_f64(self.tail_max_score + f64::from(upper_bound));
            self.tail.push(candidate);
            return Some(evicted.posting);
        }

        Some(candidate.posting)
    }

    fn lead_to_tail_upper_bound(&self, posting: &PostingIterator, target: u64) -> f32 {
        if self.operator == Operator::Or
            && posting.is_compressed()
            && self.up_to.is_some_and(|up_to| target <= up_to)
        {
            posting.window_max_score(self.up_to, &self.scorer)
        } else {
            posting.global_upper_bound(&self.scorer)
        }
    }

    pub(super) fn advance_tail_to_next_or_window(&mut self) -> bool {
        if self.operator != Operator::Or || self.tail.is_empty() {
            return false;
        }

        let Some(up_to) = self.up_to else {
            return false;
        };
        if up_to >= u32::MAX as u64 {
            return false;
        }
        if !self
            .tail
            .iter()
            .any(|tail| tail.posting.has_next_compressed_block())
        {
            return false;
        }

        // A low-scoring tail can be the only iterator left in the current
        // window. Move to the next window so a later high-scoring block is still
        // reachable instead of ending the disjunction early.
        self.update_max_scores(up_to + 1);
        true
    }

    pub(super) fn push_back_leads(&mut self, target: u64) {
        // After finishing a candidate doc, convert the aligned iterators back
        // into lagging iterators. Entries that do not stay in `tail` are
        // advanced to `target` and returned to `head`.
        // pop() drains in place, keeping self.lead's capacity for reuse.
        if self.threshold <= 0.0 {
            while let Some(mut posting) = self.lead.pop() {
                posting.next(target);
                self.push_head(posting);
            }
            return;
        }

        while let Some(posting) = self.lead.pop() {
            let upper_bound = self.lead_to_tail_upper_bound(&posting, target);
            if let Some(mut evicted) = self.insert_tail_with_overflow(posting, upper_bound) {
                evicted.next(target);
                self.push_head(evicted);
            }
        }
    }

    fn advance_tail_top(&mut self, target: u64, doc_length: u32, lead_score: &mut f32) -> bool {
        // Advance the most promising lagging iterator to the current target.
        // If it lands on the target, fold its exact contribution into
        // `lead_score`; otherwise put it back into `head`.
        let Some(TailPosting {
            upper_bound,
            cost: _,
            mut posting,
        }) = self.tail.pop()
        else {
            return false;
        };
        self.remove_tail_upper_bound(upper_bound);
        posting.next(target);
        match posting.doc() {
            Some(doc) if doc.doc_id() == target => {
                *lead_score += posting.score(&self.scorer, doc.frequency(), doc_length);
                self.lead.push(posting);
            }
            Some(_) => self.push_head(posting),
            None => {}
        }
        true
    }

    #[inline]
    fn tail_upper_bound_sum(&self) -> f64 {
        self.tail_max_score
    }

    #[inline]
    fn remove_tail_upper_bound(&mut self, upper_bound: f32) {
        if self.tail.is_empty() {
            self.tail_max_score = 0.0;
            return;
        }
        self.tail_max_score = next_up_f64((self.tail_max_score - f64::from(upper_bound)).max(0.0));
    }

    pub(super) fn advance_all_tail(
        &mut self,
        target: u64,
        doc_length: Option<u32>,
        mut score: Option<&mut f32>,
    ) {
        // Materialize all remaining lagging iterators for `target`. This is
        // only done once we have already decided to fully score / validate the
        // candidate.
        let tail = std::mem::take(&mut self.tail);
        self.tail_max_score = 0.0;
        for tail_posting in tail.into_vec() {
            let mut posting = tail_posting.posting;
            posting.next(target);
            match posting.doc() {
                Some(doc) if doc.doc_id() == target => {
                    if let (Some(doc_length), Some(score)) = (doc_length, score.as_deref_mut()) {
                        *score += posting.score(&self.scorer, doc.frequency(), doc_length);
                    }
                    self.lead.push(posting)
                }
                Some(_) => self.push_head(posting),
                None => {}
            }
        }
    }

    pub(super) fn current_doc_postings(&self) -> Vec<&PostingIterator> {
        if !self.lead.is_empty() {
            return self.lead.iter().map(|posting| posting.as_ref()).collect();
        }

        let Some(target) = self.head_doc() else {
            return Vec::new();
        };
        self.head
            .iter()
            .filter(|posting| posting.doc_id() == target)
            .map(|posting| posting.posting.as_ref())
            .collect()
    }
}
