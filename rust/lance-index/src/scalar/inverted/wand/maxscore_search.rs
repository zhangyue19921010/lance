// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Result;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;
use crate::scalar::inverted::builder::{BLOCK_SIZE, ScoredDoc};
use crate::scalar::inverted::encoding::MAX_POSTING_BLOCK_SIZE;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::scorer::{Scorer, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{DocInfo, PostingList, RawDocInfo};

use super::{
    CompetitiveFloorMode, DocCandidate, MAXSCORE_INNER_WINDOW, MaxScoreClause, TERMINATED_DOC_ID,
    TopKCollector, Wand, WandDocuments, WindowAccumulator, maxscore, score_sum_cannot_compete,
    score_sum_upper_bound_factor,
};

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    /// Bulk MAXSCORE top-k disjunction, mirroring Lucene's MaxScoreBulkScorer.
    ///
    /// Per outer window (bounded by the essential clauses' block boundaries):
    /// clauses are partitioned into a non-essential prefix — sorted by window
    /// max score, as many as fit under the threshold — and the essential rest.
    /// Essential clauses stream their postings into a window accumulator in
    /// bulk; only accumulated candidates that could still beat the threshold
    /// probe the non-essential clauses. No per-doc heap maintenance happens
    /// anywhere on this path.
    pub(super) fn maxscore_search(
        &mut self,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate<D::Candidate>>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        let mut clauses = std::mem::take(&mut self.head)
            .into_vec()
            .into_iter()
            .map(|head| MaxScoreClause {
                posting: head.posting,
                query_rank: 0,
                bound: 0.0,
                prefix_bound: 0.0,
            })
            .collect::<Vec<_>>();
        // Before a competitive floor exists every clause is essential. Keep
        // that exhaustive accumulator in canonical query order so its score
        // can be emitted directly instead of recomputing every contribution.
        clauses.sort_unstable_by_key(|clause| (clause.posting.position, clause.posting.token_id));
        for (query_rank, clause) in clauses.iter_mut().enumerate() {
            clause.query_rank = query_rank;
        }
        let num_query_terms = clauses.len();
        let total_sum_upper_bound_factor = score_sum_upper_bound_factor(num_query_terms);

        let mut acc = WindowAccumulator::new(clauses.len());
        let mut essential_chunk = Vec::with_capacity(MAX_POSTING_BLOCK_SIZE);
        let mut soa_scratch = maxscore::MaxScoreSoaScratch::new();
        let mut candidates = TopKCollector::new(limit, std::cmp::min(limit, BLOCK_SIZE * 10));
        let norm_k = self.norm_k_cache();
        let norm_k_ref = norm_k
            .as_ref()
            .map(|(norms, cache)| (*norms, cache.as_ref()));
        let mut num_comparisons = 0usize;
        // Adaptive minimum window size (Lucene): grow windows when they yield
        // too few candidates to amortize the per-window bound computations.
        let mut min_window_size = 1u64;
        let mut num_windows = 0u64;
        let mut prev_first_essential = 0usize;

        let mut window_min = clauses
            .iter()
            .filter_map(|clause| clause.posting.doc().map(|doc| doc.doc_id()))
            .min()
            .unwrap_or(TERMINATED_DOC_ID);

        while window_min < TERMINATED_DOC_ID {
            clauses.retain(|clause| clause.posting.doc().is_some());
            if clauses.is_empty() {
                break;
            }
            self.raise_to_shared_floor(params.wand_factor);

            // Window boundary from the previous window's essential clauses
            // only: dense non-essential clauses must not fragment the window.
            let first_window_lead = prev_first_essential.min(clauses.len() - 1);
            let mut window_max = TERMINATED_DOC_ID;
            for clause in &mut clauses {
                let doc = clause
                    .posting
                    .doc()
                    .map(|doc| doc.doc_id())
                    .expect("exhausted clauses were retained out");
                clause.posting.shallow_next(doc.max(window_min));
            }
            for clause in &clauses[first_window_lead..] {
                window_max = window_max.min(clause.posting.block_end_doc());
            }
            if clauses.len() > 1 {
                // Target at least 32 candidates per clause per window on
                // average before shrinking windows back to block granularity.
                if (num_comparisons as u64) < num_windows * 32 * clauses.len() as u64 {
                    min_window_size = (min_window_size * 2).min(MAXSCORE_INNER_WINDOW as u64);
                } else {
                    min_window_size = 1;
                }
                window_max = window_max.max(window_min.saturating_add(min_window_size - 1));
            }

            for clause in &mut clauses {
                let doc = clause
                    .posting
                    .doc()
                    .map(|doc| doc.doc_id())
                    .expect("exhausted clauses were retained out");
                clause.bound = if doc > window_max {
                    0.0
                } else {
                    clause
                        .posting
                        .block_max_score_up_to_with_stats(window_max, &self.scorer)
                        .score
                };
            }
            let mut clauses_in_query_order = true;
            let mut first_essential = 0;
            let mut prefix = 0.0_f64;
            if self.threshold > 0.0 {
                clauses.sort_unstable_by(|a, b| a.bound.total_cmp(&b.bound));
                clauses_in_query_order = clauses
                    .windows(2)
                    .all(|window| window[0].query_rank <= window[1].query_rank);
                for (i, clause) in clauses.iter_mut().enumerate() {
                    let next_prefix = prefix + f64::from(clause.bound);
                    let widened = next_prefix * score_sum_upper_bound_factor(i + 1);
                    if widened > f64::from(self.threshold) {
                        break;
                    }
                    prefix = next_prefix;
                    clause.prefix_bound = prefix;
                    first_essential = i + 1;
                }
            }
            prev_first_essential = first_essential;
            num_windows += 1;

            if first_essential == clauses.len() {
                // No clause combination inside this window can beat the
                // threshold: skip it wholesale.
                window_min = match window_max {
                    TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                    max => max + 1,
                };
                // Single live clause: instead of re-running the window
                // machinery once per block, scan the baked per-block bounds
                // for the next block that can beat the threshold. This is the
                // slab form of Lucene's getSkipUpTo and turns dead stretches
                // of a dominant term into a tight load-mul-compare loop.
                if clauses.len() == 1 && window_min != TERMINATED_DOC_ID && self.threshold > 0.0 {
                    let posting = &clauses[0].posting;
                    if let PostingList::Compressed(ref list) = posting.list
                        && let Some(impacts) = list.impacts.as_ref()
                    {
                        let compressed = unsafe { &mut *posting.compressed_state_ptr() };
                        let bounds = impacts.level0_doc_weight_bounds_cached(
                            &self.scorer,
                            &mut compressed.block_max_window.impact_score_cache,
                        );
                        let query_weight = posting.query_weight;
                        // Position by binary search on the first-doc slab: the
                        // deep cursor lags arbitrarily far behind during long
                        // skip runs, and walking from it re-scans the same
                        // blocks on every dead window.
                        let first_docs = list.block_first_docs();
                        let mut block_idx = first_docs
                            .partition_point(|&first| u64::from(first) <= window_min)
                            .saturating_sub(1);
                        while block_idx < bounds.len()
                            && query_weight * bounds[block_idx] <= self.threshold
                        {
                            block_idx += 1;
                        }
                        window_min = if block_idx < bounds.len() {
                            window_min.max(u64::from(list.block_least_doc_id(block_idx)))
                        } else {
                            TERMINATED_DOC_ID
                        };
                    }
                }
                continue;
            }

            let total_non_essential_bound = if first_essential > 0 {
                clauses[first_essential - 1].prefix_bound
            } else {
                0.0
            };

            // Single essential clause (the common case once the threshold is
            // competitive): skip dead blocks, take one posting block at a
            // time, and complete ≤2 optionals in SoA buffers. More optionals
            // keep the per-doc LiveHit completion below.
            if first_essential + 1 == clauses.len() {
                #[cfg(test)]
                {
                    self.maxscore_single_essential_windows += 1;
                }
                if first_essential <= 2 {
                    let essential_query_rank = clauses[first_essential].query_rank;
                    let essential_term = clauses[first_essential].posting.term_index();
                    let essential_weight = clauses[first_essential].posting.query_weight;
                    let essential_window_bound = clauses[first_essential].bound;
                    if clauses[first_essential]
                        .posting
                        .doc()
                        .is_some_and(|doc| doc.doc_id() < window_min)
                    {
                        clauses[first_essential].posting.next(window_min);
                    }
                    self.score_single_essential_upto(
                        &mut clauses,
                        first_essential,
                        first_essential,
                        window_max,
                        essential_query_rank,
                        essential_term,
                        essential_weight,
                        essential_window_bound,
                        num_query_terms,
                        total_non_essential_bound,
                        total_sum_upper_bound_factor,
                        norm_k_ref,
                        &mut essential_chunk,
                        &mut soa_scratch,
                        &mut candidates,
                        &mut num_comparisons,
                        params.wand_factor,
                    )?;
                    window_min = match window_max {
                        TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                        max => max + 1,
                    };
                    continue;
                }
                let (non_essential, essential) = clauses.split_at_mut(first_essential);
                let essential_query_rank = essential[0].query_rank;
                let posting = &mut essential[0].posting;
                if posting.doc().is_some_and(|doc| doc.doc_id() < window_min) {
                    posting.next(window_min);
                }
                let essential_term = posting.term_index();
                let essential_weight = posting.query_weight;
                let sole_query_clause = non_essential.is_empty();

                macro_rules! consider_candidate {
                    ($doc:expr, $freq:expr) => {{
                        let doc = $doc;
                        let freq = $freq;
                        num_comparisons += 1;
                        // One byte-norm load + cached addend when available;
                        // the exact doc length is only needed at insert time.
                        let norm_addend =
                            norm_k_ref.map(|(norms, cache)| cache[norms[doc as usize] as usize]);
                        let score = match norm_addend {
                            Some(addend) => {
                                essential_weight * bm25_doc_weight_with_norm(freq, addend)
                            }
                            None => {
                                essential_weight
                                    * self.scorer.doc_weight(
                                        freq,
                                        self.documents.scoring_num_tokens(doc as u32),
                                    )
                            }
                        };
                        if !(self.threshold > 0.0
                            && score_sum_cannot_compete(
                                score,
                                total_non_essential_bound,
                                self.threshold,
                                total_sum_upper_bound_factor,
                                CompetitiveFloorMode::Exclusive,
                            ))
                        {
                            if let Some(document_key) =
                                self.documents.document_key_for_doc_id(doc as u32)
                            {
                                let mut total = score;
                                let mut scores_by_query_rank = SmallVec::<[f32; 8]>::new();
                                if !sole_query_clause {
                                    scores_by_query_rank.resize(num_query_terms, 0.0);
                                    scores_by_query_rank[essential_query_rank] = score;
                                }
                                let mut rejected = false;
                                for i in (0..non_essential.len()).rev() {
                                    if self.threshold > 0.0
                                        && score_sum_cannot_compete(
                                            total,
                                            non_essential[i].prefix_bound,
                                            self.threshold,
                                            total_sum_upper_bound_factor,
                                            CompetitiveFloorMode::Exclusive,
                                        )
                                    {
                                        rejected = true;
                                        break;
                                    }
                                    let query_rank = non_essential[i].query_rank;
                                    let probe = &mut non_essential[i].posting;
                                    if probe.doc().is_some_and(|d| d.doc_id() < doc) {
                                        probe.next(doc);
                                    }
                                    if let Some(d) = probe.doc()
                                        && d.doc_id() == doc
                                    {
                                        let contribution = match norm_addend {
                                            Some(addend) => {
                                                probe.query_weight
                                                    * bm25_doc_weight_with_norm(
                                                        d.frequency(),
                                                        addend,
                                                    )
                                            }
                                            None => probe.score(
                                                &self.scorer,
                                                d.frequency(),
                                                self.documents.scoring_num_tokens(doc as u32),
                                            ),
                                        };
                                        total += contribution;
                                        scores_by_query_rank[query_rank] = contribution;
                                    }
                                }

                                // Apply the caller's floor mode only after the
                                // canonical query-order score is available.
                                if !rejected {
                                    let canonical_score = if sole_query_clause {
                                        score
                                    } else {
                                        scores_by_query_rank
                                            .into_iter()
                                            .fold(0.0_f32, |sum, contribution| sum + contribution)
                                    };
                                    if canonical_score > self.threshold {
                                        let doc_length =
                                            self.documents.scoring_num_tokens(doc as u32);
                                        if candidates.insert(
                                            ScoredDoc::new(document_key, canonical_score),
                                            doc_length,
                                            doc,
                                            std::iter::once((essential_term, freq)).chain(
                                                non_essential.iter().filter_map(|clause| {
                                                    clause.posting.doc().and_then(|d| {
                                                        (d.doc_id() == doc).then(|| {
                                                            (
                                                                clause.posting.term_index(),
                                                                d.frequency(),
                                                            )
                                                        })
                                                    })
                                                }),
                                            ),
                                        )? && let Some(kth) = candidates.kth_score_if_full()
                                        {
                                            self.update_threshold(kth, params.wand_factor);
                                        }
                                    }
                                }
                            }
                        }
                    }};
                }

                match posting.list {
                    PostingList::Compressed(ref list) => {
                        let shift = list.block_shift();
                        let mask = list.block_mask();
                        'stream: while let Some(cur) = posting.current_doc {
                            if cur.doc_id() > window_max {
                                break;
                            }
                            let block_idx = posting.index >> shift;
                            let block_offset = posting.index & mask;
                            let compressed = unsafe {
                                &mut *posting.ensure_compressed_block_ptr(list, block_idx)
                            };
                            for offset in block_offset..compressed.doc_ids.len() {
                                let doc_id = compressed.doc_ids[offset];
                                if u64::from(doc_id) > window_max {
                                    posting.index = (block_idx << shift) + offset;
                                    posting.block_idx = block_idx;
                                    posting.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                        doc_id,
                                        frequency: compressed.freqs[offset],
                                    }));
                                    break 'stream;
                                }
                                consider_candidate!(u64::from(doc_id), compressed.freqs[offset]);
                            }
                            let next_start = (block_idx + 1) << shift;
                            if next_start >= list.length as usize {
                                posting.index = list.length as usize;
                                posting.block_idx = posting.index >> shift;
                                posting.current_doc = None;
                                break;
                            }
                            posting.index = next_start;
                            posting.block_idx = block_idx + 1;
                            let compressed = unsafe {
                                &mut *posting.ensure_compressed_block_ptr(list, block_idx + 1)
                            };
                            posting.current_doc = Some(DocInfo::Raw(RawDocInfo {
                                doc_id: compressed.doc_ids[0],
                                frequency: compressed.freqs[0],
                            }));
                        }
                    }
                    PostingList::Plain(_) => {
                        while let Some(cur) = posting.doc() {
                            let doc = cur.doc_id();
                            if doc > window_max {
                                break;
                            }
                            consider_candidate!(doc, cur.frequency());
                            posting.next(doc + 1);
                        }
                    }
                }

                window_min = match window_max {
                    TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                    max => max + 1,
                };
                continue;
            }

            // Stream the essential clauses through inner windows.
            #[cfg(test)]
            {
                self.maxscore_general_windows += 1;
            }
            let mut inner_min = window_min;
            loop {
                // Lucene `scoreInnerWindow`: when the second essential is at
                // least INNER_WINDOW/2 ahead of the first, the prefix matches
                // a single clause. Complete it with the SoA path instead of
                // scattering two sparse lists into a 4096-slot accumulator.
                //
                // One pass serves both purposes: `top_doc` is also the closest
                // essential cursor, which is what raises `inner_min` to the
                // window floor.
                let mut top_idx = None;
                let mut top_doc = TERMINATED_DOC_ID;
                let mut top2_doc = TERMINATED_DOC_ID;
                for (idx, clause) in clauses.iter().enumerate().skip(first_essential) {
                    let Some(doc) = clause.posting.doc() else {
                        continue;
                    };
                    let id = doc.doc_id();
                    if id < top_doc {
                        top2_doc = top_doc;
                        top_doc = id;
                        top_idx = Some(idx);
                    } else if id < top2_doc {
                        top2_doc = id;
                    }
                }
                inner_min = inner_min.max(top_doc);
                if inner_min == TERMINATED_DOC_ID || inner_min > window_max {
                    break;
                }
                if let Some(ess_idx) = top_idx
                    && top2_doc.saturating_sub(top_doc) >= (MAXSCORE_INNER_WINDOW / 2) as u64
                    && first_essential <= 2
                    // Reachable, not merely defensive. `shallow_next` only moves
                    // `block_idx` and leaves `current_doc` where it was, so a
                    // window that ends early -- wholesale skip, `inner_min >
                    // window_max`, or the single-essential stream -- can advance
                    // `window_min` past cursors that were never consumed. The
                    // next window then starts with `inner_min` above the smallest
                    // essential doc. Since `upto` is `top2_doc - 1`, the span is
                    // empty when `top2_doc == inner_min` (the call would return
                    // without advancing, leaving `inner_min` stuck) and ends
                    // below `inner_min` when `top2_doc < inner_min` (it would
                    // move `inner_min` backwards onto a span this window has
                    // already passed). Both fall through to the general path.
                    && {
                        let non_empty_span = top2_doc > inner_min;
                        #[cfg(test)]
                        if !non_empty_span {
                            self.maxscore_top2_gap_empty_span += 1;
                        }
                        non_empty_span
                    }
                {
                    let upto = window_max.min(top2_doc.saturating_sub(1));
                    if clauses[ess_idx]
                        .posting
                        .doc()
                        .is_some_and(|doc| doc.doc_id() < inner_min)
                    {
                        clauses[ess_idx].posting.next(inner_min);
                    }
                    let essential_query_rank = clauses[ess_idx].query_rank;
                    let essential_term = clauses[ess_idx].posting.term_index();
                    let essential_weight = clauses[ess_idx].posting.query_weight;
                    let essential_window_bound = clauses[ess_idx].bound;
                    #[cfg(test)]
                    {
                        self.maxscore_single_essential_windows += 1;
                        self.maxscore_top2_gap_windows += 1;
                    }
                    self.score_single_essential_upto(
                        &mut clauses,
                        first_essential,
                        ess_idx,
                        upto,
                        essential_query_rank,
                        essential_term,
                        essential_weight,
                        essential_window_bound,
                        num_query_terms,
                        total_non_essential_bound,
                        total_sum_upper_bound_factor,
                        norm_k_ref,
                        &mut essential_chunk,
                        &mut soa_scratch,
                        &mut candidates,
                        &mut num_comparisons,
                        params.wand_factor,
                    )?;
                    inner_min = match upto {
                        TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                        max => max + 1,
                    };
                    continue;
                }
                let inner_max =
                    window_max.min(inner_min.saturating_add(MAXSCORE_INNER_WINDOW as u64 - 1));

                for (clause_idx, clause) in clauses.iter_mut().enumerate().skip(first_essential) {
                    clause.posting.collect_window_scores(
                        inner_min,
                        inner_max,
                        clause_idx,
                        self.documents,
                        &self.scorer,
                        norm_k_ref,
                        &mut acc,
                    );
                }

                // Drain candidates in doc order, completing them with the
                // non-essential clauses ordered by descending bound.
                for word_idx in 0..acc.words.len() {
                    let mut word = acc.words[word_idx];
                    if word == 0 {
                        continue;
                    }
                    acc.words[word_idx] = 0;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        word &= word - 1;
                        let slot = (word_idx << 6) | bit;
                        let doc = inner_min + slot as u64;
                        let mut score = acc.scores[slot];
                        num_comparisons += 1;

                        if self.threshold > 0.0
                            && score_sum_cannot_compete(
                                score,
                                total_non_essential_bound,
                                self.threshold,
                                total_sum_upper_bound_factor,
                                CompetitiveFloorMode::Exclusive,
                            )
                        {
                            acc.clear_slot(slot);
                            continue;
                        }

                        let Some(document_key) = self.documents.document_key_for_doc_id(doc as u32)
                        else {
                            acc.clear_slot(slot);
                            continue;
                        };

                        // Doc length is only needed at heap-insert time; the
                        // non-essential completion scores go through the norm
                        // cache when available.
                        let norm_addend =
                            norm_k_ref.map(|(norms, cache)| cache[norms[doc as usize] as usize]);
                        let mut doc_length_cell: Option<u32> = None;
                        let needs_canonical_rescore =
                            first_essential != 0 || !clauses_in_query_order;
                        let mut scores_by_query_rank = SmallVec::<[f32; 8]>::new();
                        if needs_canonical_rescore {
                            scores_by_query_rank.resize(num_query_terms, 0.0);
                        }
                        let mut rejected = false;
                        for i in (0..first_essential).rev() {
                            if self.threshold > 0.0
                                && score_sum_cannot_compete(
                                    score,
                                    clauses[i].prefix_bound,
                                    self.threshold,
                                    total_sum_upper_bound_factor,
                                    CompetitiveFloorMode::Exclusive,
                                )
                            {
                                rejected = true;
                                break;
                            }
                            let query_rank = clauses[i].query_rank;
                            let posting = &mut clauses[i].posting;
                            if posting.doc().is_some_and(|d| d.doc_id() < doc) {
                                posting.next(doc);
                            }
                            if let Some(d) = posting.doc()
                                && d.doc_id() == doc
                            {
                                let contribution = match norm_addend {
                                    Some(addend) => {
                                        posting.query_weight
                                            * bm25_doc_weight_with_norm(d.frequency(), addend)
                                    }
                                    None => {
                                        let doc_length =
                                            *doc_length_cell.get_or_insert_with(|| {
                                                self.documents.scoring_num_tokens(doc as u32)
                                            });
                                        posting.score(&self.scorer, d.frequency(), doc_length)
                                    }
                                };
                                score += contribution;
                                if needs_canonical_rescore {
                                    scores_by_query_rank[query_rank] = contribution;
                                }
                            }
                        }

                        if !rejected {
                            let doc_length = doc_length_cell
                                .unwrap_or_else(|| self.documents.scoring_num_tokens(doc as u32));
                            let score = if !needs_canonical_rescore {
                                // `collect_window_scores` visited every
                                // essential clause in query order, so the
                                // accumulator already contains the canonical
                                // f32 fold. Avoid rescoring exhaustive hits.
                                score
                            } else {
                                for (i, clause) in clauses.iter().enumerate().skip(first_essential)
                                {
                                    let freq = acc.clause_freq(i, slot);
                                    if freq == 0 {
                                        continue;
                                    }
                                    let contribution = match norm_addend {
                                        Some(addend) => {
                                            clause.posting.query_weight
                                                * bm25_doc_weight_with_norm(freq, addend)
                                        }
                                        None => {
                                            clause.posting.score(&self.scorer, freq, doc_length)
                                        }
                                    };
                                    scores_by_query_rank[clause.query_rank] = contribution;
                                }
                                scores_by_query_rank
                                    .iter()
                                    .fold(0.0_f32, |sum, contribution| sum + contribution)
                            };
                            if score <= self.threshold {
                                acc.clear_slot(slot);
                                continue;
                            }
                            if candidates.insert(
                                ScoredDoc::new(document_key, score),
                                doc_length,
                                doc,
                                clauses.iter().enumerate().filter_map(|(i, clause)| {
                                    if i >= first_essential {
                                        let freq = acc.clause_freq(i, slot);
                                        (freq > 0).then(|| (clause.posting.term_index(), freq))
                                    } else {
                                        clause.posting.doc().and_then(|d| {
                                            (d.doc_id() == doc).then(|| {
                                                (clause.posting.term_index(), d.frequency())
                                            })
                                        })
                                    }
                                }),
                            )? && let Some(kth) = candidates.kth_score_if_full()
                            {
                                self.update_threshold(kth, params.wand_factor);
                            }
                        }
                        acc.clear_slot(slot);
                    }
                }
                if inner_max >= window_max {
                    break;
                }
                inner_min = inner_max + 1;
            }

            window_min = match window_max {
                TERMINATED_DOC_ID => TERMINATED_DOC_ID,
                max => max + 1,
            };
        }

        metrics.record_comparisons(num_comparisons);

        candidates.into_candidates(|key| self.documents.candidate_from_key(key))
    }
}
