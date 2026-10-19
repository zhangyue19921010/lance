// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

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
}

pub(super) type ScoreContribution = ((u32, u32), f32);

#[inline]
pub(super) fn score_contributions_in_query_order(
    mut contributions: SmallVec<[ScoreContribution; 8]>,
) -> f32 {
    if !contributions
        .windows(2)
        .all(|window| window[0].0 <= window[1].0)
    {
        contributions.sort_unstable_by_key(|contribution| contribution.0);
    }
    contributions
        .into_iter()
        .fold(0.0_f32, |score, (_, contribution)| score + contribution)
}
