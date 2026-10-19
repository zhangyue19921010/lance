// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::{
    cell::{OnceCell, RefCell, UnsafeCell},
    collections::{BinaryHeap, VecDeque},
};
use std::{cmp::Reverse, fmt::Debug};

use arrow::array::AsArray;
use arrow::datatypes::Int32Type;
use arrow_array::Array;
use itertools::Itertools;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use smallvec::SmallVec;

use crate::metrics::MetricsCollector;

mod and_bulk;
mod cursor;
mod doc_source;
mod intersection;
mod maxscore;
mod phrase;
mod posting;
mod search;

pub(super) use cursor::WandCursor;
pub use doc_source::DocCandidate;
#[allow(unused_imports)]
pub(super) use doc_source::{AllModernDocuments, ModernVisibility};
pub(super) use doc_source::{
    FlatDocuments, LegacyWandDocuments, ModernWandDocuments, WandDocuments,
};
pub(super) use posting::{GroupedTermScorer, validate_modern_posting_doc_ids};

#[cfg(test)]
use phrase::{POSITION_CURSOR_CALLS, counted_anchor_steps, take_position_cursor_calls};
use phrase::{PositionCursor, PositionValues, RecycledPositionValues};
#[cfg(test)]
use posting::find_next_geq_in_block;
use posting::{CompressedState, find_next_geq, find_next_geq_scalar};
#[cfg(target_arch = "x86_64")]
use posting::{HAS_AVX2, find_next_geq_avx2};

use super::{
    CompressedPositionStorage,
    documents::{DocId, DocLengths, DocVisibility},
    impact::{IMPACT_LEVEL1_BLOCKS, ImpactScoreCache, ImpactSkipData},
    index::{PositionStreamCodec, dequantize_doc_length},
    query::Operator,
    scorer::{BM25_DOC_WEIGHT_UPPER_BOUND, MemBM25Scorer, bm25_doc_weight_with_norm, idf},
};
use super::{
    CompressedPostingList, DocSet, PostingList, RawDocInfo,
    builder::ScoredDoc,
    encoding::{
        MAX_POSTING_BLOCK_SIZE, decode_position_stream_block, decompress_positions,
        decompress_posting_block_doc_ids, decompress_posting_block_frequencies,
        decompress_posting_remainder_doc_ids, decompress_posting_remainder_frequencies,
        seek_packed_doc_positions,
    },
    query::FtsSearchParams,
    scorer::Scorer,
};
use super::{DocInfo, builder::BLOCK_SIZE};

const TERMINATED_DOC_ID: u64 = u64::MAX;

/// Top-k heap entry: (scored doc, doc length, posting doc id, frequency slot).
/// The (term, freq) pairs live outside the heap so heap churn moves a compact
/// tuple instead of a `Vec`.
type TopKHeap = BinaryHeap<Reverse<(ScoredDoc, u32, u64, u32)>>;

type NormKCache<'a> = (&'a [u8], Arc<[f32; 256]>);

/// Norm-code BM25 denominator addends (Lucene's norm cache) computed once for
/// every partition a query searches with the same scorer. Each partition
/// otherwise rebuilds the same 256 entries for its first scored window.
///
/// Every [`Wand`] given one instance must use a scorer with identical
/// [`Scorer::doc_norm`] results; the first search to need the table fills it.
#[derive(Clone, Default)]
pub struct SharedNormAddends(Arc<OnceLock<Option<Arc<[f32; 256]>>>>);

fn norm_addends<S: Scorer + ?Sized>(scorer: &S) -> Option<Arc<[f32; 256]>> {
    let mut addends = [0f32; 256];
    for (code, slot) in addends.iter_mut().enumerate() {
        *slot = scorer.doc_norm(dequantize_doc_length(code as u8))?;
    }
    Some(Arc::new(addends))
}

/// Reusable (term, freq) storage for live heap candidates. Replacing the k-th
/// result reuses its slot and retained `Vec` capacity, so memory stays bounded
/// by the live top-k rather than every historical heap admission.
struct FrequencySlots {
    slots: Vec<Vec<(u32, u32)>>,
}

impl FrequencySlots {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, pairs: impl Iterator<Item = (u32, u32)>) -> Result<u32> {
        let slot = u32::try_from(self.slots.len()).map_err(|_| {
            Error::internal(format!(
                "FTS top-k frequency slot count {} exceeds u32::MAX",
                self.slots.len()
            ))
        })?;
        self.slots.push(pairs.collect());
        Ok(slot)
    }

    fn replace(&mut self, slot: u32, pairs: impl Iterator<Item = (u32, u32)>) -> Result<()> {
        let num_slots = self.slots.len();
        let slot = self.slots.get_mut(slot as usize).ok_or_else(|| {
            Error::internal(format!(
                "FTS top-k frequency slot {slot} is out of bounds for {num_slots} slots"
            ))
        })?;
        slot.clear();
        slot.extend(pairs);
        Ok(())
    }

    fn take(&mut self, slot: u32) -> Result<Vec<(u32, u32)>> {
        let num_slots = self.slots.len();
        self.slots
            .get_mut(slot as usize)
            .map(std::mem::take)
            .ok_or_else(|| {
                Error::internal(format!(
                    "FTS top-k frequency slot {slot} is out of bounds for {num_slots} slots"
                ))
            })
    }
}

/// Owns the top-k heap and the frequency slots referenced by its entries.
struct TopKCollector {
    limit: usize,
    heap: TopKHeap,
    frequency_slots: FrequencySlots,
}

impl TopKCollector {
    fn new(limit: usize, initial_capacity: usize) -> Self {
        let initial_capacity = initial_capacity.min(limit);
        Self {
            limit,
            heap: BinaryHeap::with_capacity(initial_capacity),
            frequency_slots: FrequencySlots::with_capacity(initial_capacity),
        }
    }

    /// Insert a competitive result. When the heap is full, its evicted entry's
    /// frequency slot is cleared and reused before the replacement is pushed.
    fn insert(
        &mut self,
        doc: ScoredDoc,
        doc_length: u32,
        posting_doc_id: u64,
        pairs: impl Iterator<Item = (u32, u32)>,
    ) -> Result<bool> {
        if self.limit == 0 {
            return Ok(false);
        }
        if self.heap.len() > self.limit {
            return Err(Error::internal(format!(
                "FTS top-k heap length {} exceeds limit {}",
                self.heap.len(),
                self.limit
            )));
        }

        let frequency_slot = if self.heap.len() == self.limit {
            let Some(kth_score) = self.heap.peek().map(|entry| entry.0.0.score.0) else {
                return Err(Error::internal(
                    "FTS top-k heap is empty while its nonzero limit is reached",
                ));
            };
            // Preserve the existing collector semantics for non-finite custom
            // scorer output: only a strictly greater raw f32 replaces k-th.
            if doc.score.0.partial_cmp(&kth_score) != Some(std::cmp::Ordering::Greater) {
                return Ok(false);
            }
            let Some(Reverse((_, _, _, frequency_slot))) = self.heap.pop() else {
                return Err(Error::internal(
                    "FTS top-k heap entry disappeared during replacement",
                ));
            };
            self.frequency_slots.replace(frequency_slot, pairs)?;
            frequency_slot
        } else {
            self.frequency_slots.push(pairs)?
        };

        self.heap
            .push(Reverse((doc, doc_length, posting_doc_id, frequency_slot)));
        Ok(true)
    }

    fn rejects_score(&self, score: f32) -> bool {
        if self.heap.len() < self.limit {
            return false;
        }
        let Some(kth_score) = self.heap.peek().map(|entry| entry.0.0.score.0) else {
            return false;
        };
        score.partial_cmp(&kth_score) != Some(std::cmp::Ordering::Greater)
    }

    fn kth_score_if_full(&self) -> Option<f32> {
        if self.heap.len() == self.limit {
            self.heap.peek().map(|entry| entry.0.0.score.0)
        } else {
            None
        }
    }

    fn into_candidates<C>(
        self,
        mut to_candidate: impl FnMut(u64) -> C,
    ) -> Result<Vec<DocCandidate<C>>> {
        let Self {
            heap,
            mut frequency_slots,
            ..
        } = self;
        heap.into_iter()
            .map(
                |Reverse((doc, doc_length, posting_doc_id, frequency_slot))| {
                    Ok(DocCandidate {
                        document: to_candidate(doc.row_id),
                        posting_doc_id,
                        freqs: frequency_slots.take(frequency_slot)?,
                        doc_length,
                    })
                },
            )
            .collect()
    }

    #[cfg(test)]
    fn num_frequency_slots(&self) -> usize {
        self.frequency_slots.slots.len()
    }
}
const LINEAR_BLOCK_SKIP_LIMIT: usize = 8;
pub static FLAT_SEARCH_PERCENT_THRESHOLD: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FLAT_SEARCH_PERCENT_THRESHOLD")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<u64>()
        .unwrap_or(10)
});
// Bulk MAXSCORE path for top-k disjunctions (Lucene MaxScoreBulkScorer
// style). Default on: with right-sized partitions it wins by a wide margin
// (Lucene-parity latency) and its results are score-identical to the classic
// WAND loop. LANCE_FTS_MAXSCORE=0 opts back into the classic loop.
static USE_MAXSCORE_SEARCH: LazyLock<bool> =
    LazyLock::new(|| std::env::var("LANCE_FTS_MAXSCORE").as_deref() != Ok("0"));

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CompetitiveFloorMode {
    #[default]
    Exclusive,
    Inclusive,
}

impl CompetitiveFloorMode {
    #[inline]
    fn rejects_upper_bound(self, upper_bound: f64, floor: f32) -> bool {
        match self {
            Self::Exclusive => upper_bound <= f64::from(floor),
            Self::Inclusive => upper_bound < f64::from(floor),
        }
    }

    #[inline]
    fn accepts_score(self, score: f32, floor: f32) -> bool {
        match self {
            Self::Exclusive => score > floor,
            Self::Inclusive => score >= floor,
        }
    }
}
// Bulk conjunction path for top-k AND / phrase queries: block-max window
// skipping plus a slice-level merge over decompressed blocks, replacing the
// per-doc `next()` leapfrog. Results are identical to the classic AND loop.
// LANCE_FTS_BULK_AND accepts auto (default), on/1, or off/0. Auto enables the
// bulk path for two and three clauses and for wider current-format conjunctions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BulkAndMode {
    #[default]
    Auto,
    On,
    Off,
}

impl BulkAndMode {
    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            Some(Self::Auto)
        } else if value.eq_ignore_ascii_case("on") || value == "1" {
            Some(Self::On)
        } else if value.eq_ignore_ascii_case("off") || value == "0" {
            Some(Self::Off)
        } else {
            None
        }
    }

    const fn enabled_for(self, num_clauses: usize) -> bool {
        match self {
            Self::Auto => matches!(num_clauses, 2 | 3),
            Self::On => true,
            Self::Off => false,
        }
    }
}

fn bulk_and_mode_from_env() -> BulkAndMode {
    match std::env::var("LANCE_FTS_BULK_AND") {
        Ok(value) => BulkAndMode::parse(&value).unwrap_or_else(|| {
            log::warn!(
                "Invalid LANCE_FTS_BULK_AND value {value:?}; expected auto, on/1, or off/0; \
                 falling back to auto"
            );
            BulkAndMode::Auto
        }),
        Err(std::env::VarError::NotPresent) => BulkAndMode::Auto,
        Err(std::env::VarError::NotUnicode(value)) => {
            log::warn!(
                "Invalid non-Unicode LANCE_FTS_BULK_AND value {value:?}; expected auto, on/1, \
                 or off/0; falling back to auto"
            );
            BulkAndMode::Auto
        }
    }
}

static BULK_AND_MODE: LazyLock<BulkAndMode> = LazyLock::new(bulk_and_mode_from_env);

pub struct PostingIterator {
    token: String,
    token_id: u32,
    position: u32,
    query_weight: f32,
    list: PostingList,
    // the index of current doc, this can be changed only by `next()`
    index: usize,
    // the index of current block, this can be changed by `next() and shallow_next()`
    block_idx: usize,
    current_doc: Option<DocInfo>,
    approximate_upper_bound: f32,
    // Stored block maxima may use partition-local statistics. Queries scored
    // with corpus-wide statistics must use a scorer-provided bound instead.
    use_scorer_upper_bound: bool,
    // The union posting drives iteration and pruning; these terms produce the
    // exact score for each candidate document.
    grouped_terms: Option<Arc<[GroupedTermScorer]>>,
    // Position cursors temporarily own this buffer and return it on drop. This
    // keeps repeated cursor creation allocation-free without lending a slice
    // out of the interior-mutable compressed state.
    position_scratch: RefCell<Option<Vec<u32>>>,

    // for compressed posting list
    compressed: Option<UnsafeCell<CompressedState>>,
}

/// Inner window span (in doc ids) of the bulk MAXSCORE path. Same as Lucene's
/// `MaxScoreBulkScorer.INNER_WINDOW_SIZE`.
const MAXSCORE_INNER_WINDOW: usize = 1 << 12;

/// Lucene's `MathUtil.sumUpperBound` factor, adapted to Lance's `f32` score
/// accumulation. Prefix bounds are summed in `f64`, then widened enough to
/// cover any recursive `f32` summation order of the same non-negative values.
#[inline]
pub(super) fn score_sum_upper_bound_factor(num_values: usize) -> f64 {
    if num_values <= 2 {
        1.0
    } else {
        let relative_error_bound = (num_values - 1) as f64 * f64::from(f32::EPSILON);
        1.0 + 2.0 * relative_error_bound
    }
}

#[inline]
fn score_sum_cannot_compete(
    partial_score: f32,
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
    floor_mode: CompetitiveFloorMode,
) -> bool {
    let upper_bound = (f64::from(partial_score) + remaining_upper_bound) * upper_bound_factor;
    floor_mode.rejects_upper_bound(f64::from(outward_f32_upper_bound(upper_bound)), floor)
}

type ScoreContribution = ((u32, u32), f32);

/// Largest `f32` partial score that `score_sum_cannot_compete` still rejects
/// under the exclusive floor, given the other clauses' upper bound. The
/// rejection predicate is monotone in the partial score, so every score at or
/// below the returned limit is rejected and every score above it competes.
/// Returns `None` when no finite score is rejected.
fn exclusive_partial_score_limit(
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
) -> Option<f32> {
    let rejects = |score: f32| {
        score_sum_cannot_compete(
            score,
            remaining_upper_bound,
            floor,
            upper_bound_factor,
            CompetitiveFloorMode::Exclusive,
        )
    };
    if !remaining_upper_bound.is_finite() || !floor.is_finite() {
        return None;
    }
    // Start from the real-valued boundary and walk at most a few ULPs to the
    // exact f32 boundary of the (monotone) predicate.
    let mut limit = ((f64::from(floor) / upper_bound_factor) - remaining_upper_bound) as f32;
    if !limit.is_finite() {
        return None;
    }
    if rejects(limit) {
        for _ in 0..8 {
            let up = next_up_f32(limit);
            if up == limit || !rejects(up) {
                return Some(limit);
            }
            limit = up;
        }
    } else {
        for _ in 0..8 {
            let down = next_down_f32(limit);
            if down == limit {
                return None;
            }
            if rejects(down) {
                return Some(down);
            }
            limit = down;
        }
    }
    // The walk did not converge (pathological inputs); fall back to an exact
    // binary search over the ordered f32 bit patterns.
    let mut lo = f32::MIN;
    let mut hi = f32::MAX;
    if !rejects(lo) {
        return None;
    }
    if rejects(hi) {
        return Some(hi);
    }
    for _ in 0..64 {
        let mid = f32::from_bits(((lo.to_bits() as i64 + hi.to_bits() as i64) / 2) as u32);
        if mid == lo || mid == hi {
            break;
        }
        if rejects(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

fn next_down_f32(value: f32) -> f32 {
    -next_up_f32(-value)
}

#[inline]
fn score_contributions_in_query_order(mut contributions: SmallVec<[ScoreContribution; 8]>) -> f32 {
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

/// One MAXSCORE clause: posting plus the current window bound / prefix used
/// to pick essential vs optional and to complete scores in query order.
struct MaxScoreClause {
    posting: Box<PostingIterator>,
    query_rank: usize,
    bound: f32,
    prefix_bound: f64,
}

/// Per-window score/frequency accumulator for the bulk MAXSCORE path. Slot i
/// covers doc id `window_min + i`; `freqs` is laid out clause-major so accept
/// paths can recover (term, freq) pairs of the essential clauses.
struct WindowAccumulator {
    scores: Vec<f32>,
    freqs: Vec<u32>,
    words: Vec<u64>,
    num_clauses: usize,
}

impl WindowAccumulator {
    fn new(num_clauses: usize) -> Self {
        Self {
            scores: vec![0.0; MAXSCORE_INNER_WINDOW],
            freqs: vec![0; num_clauses * MAXSCORE_INNER_WINDOW],
            words: vec![0; MAXSCORE_INNER_WINDOW / 64],
            num_clauses,
        }
    }

    #[inline]
    fn add(&mut self, clause_idx: usize, slot: usize, score: f32, freq: u32) {
        self.scores[slot] += score;
        // Doc-major layout: one slot's clause frequencies share a cache line.
        self.freqs[slot * self.num_clauses + clause_idx] = freq;
        self.words[slot >> 6] |= 1u64 << (slot & 63);
    }

    #[inline]
    fn clause_freq(&self, clause_idx: usize, slot: usize) -> u32 {
        self.freqs[slot * self.num_clauses + clause_idx]
    }

    #[inline]
    fn clear_slot(&mut self, slot: usize) {
        self.scores[slot] = 0.0;
        self.freqs[slot * self.num_clauses..(slot + 1) * self.num_clauses].fill(0);
    }
}

struct HeadPosting {
    // Iterators that are already positioned on or after the next candidate doc.
    // The heap is ordered by smallest doc id so the top element determines
    // the next target doc to consider.
    doc_id: u64,
    posting: Box<PostingIterator>,
}

impl HeadPosting {
    fn new(posting: Box<PostingIterator>) -> Self {
        let doc_id = posting
            .doc()
            .map(|doc| doc.doc_id())
            .unwrap_or(TERMINATED_DOC_ID);
        Self { doc_id, posting }
    }

    fn doc_id(&self) -> u64 {
        self.doc_id
    }
}

impl PartialEq for HeadPosting {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
            && self.posting.approximate_upper_bound().to_bits()
                == other.posting.approximate_upper_bound().to_bits()
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

impl Eq for HeadPosting {}

impl PartialOrd for HeadPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeadPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .doc_id
            .cmp(&self.doc_id)
            .then_with(|| {
                self.posting
                    .approximate_upper_bound()
                    .total_cmp(&other.posting.approximate_upper_bound())
            })
            .then_with(|| self.posting.token_id.cmp(&other.posting.token_id))
            .then_with(|| self.posting.position.cmp(&other.posting.position))
    }
}

struct TailPosting {
    // Iterators that lag behind the current target doc but may still help the
    // target beat the threshold if advanced to that doc.
    upper_bound: f32,
    // Used as a tie-breaker when upper bounds are equal. Lower-cost iterators
    // are cheaper to advance, so they are preferred.
    cost: usize,
    posting: Box<PostingIterator>,
}

impl TailPosting {
    fn new(upper_bound: f32, cost: usize, posting: Box<PostingIterator>) -> Self {
        Self {
            upper_bound,
            cost,
            posting,
        }
    }
}

impl PartialEq for TailPosting {
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound.to_bits() == other.upper_bound.to_bits()
            && self.cost == other.cost
            && self.posting.token_id == other.posting.token_id
            && self.posting.position == other.posting.position
    }
}

#[cfg(test)]
#[derive(Default)]
struct AndWindowStats {
    windows_wide: usize,
    windows_narrow: usize,
    windows_skipped: usize,
    range_blocks_scanned: usize,
    candidates_returned: usize,
    score_first_rejections: usize,
    pairwise_intersections: usize,
    windows_sliced: usize,
    windows_frequency_pruned: usize,
}

impl Eq for TailPosting {}

impl PartialOrd for TailPosting {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TailPosting {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| other.cost.cmp(&self.cost))
            .then_with(|| other.posting.token_id.cmp(&self.posting.token_id))
            .then_with(|| other.posting.position.cmp(&self.posting.position))
    }
}

pub struct Wand<'a, S: Scorer, D: WandDocuments> {
    threshold: f32, // multiple of factor and the minimum score of the top-k documents
    floor_mode: CompetitiveFloorMode,
    operator: Operator,
    num_terms: usize,
    // Posting iterators whose current doc id is >= the next target doc.
    // The heap top gives the smallest current doc id.
    head: BinaryHeap<HeadPosting>,
    #[allow(clippy::vec_box)]
    // Posting iterators that already match the current target doc.
    // Only these iterators participate in scoring / phrase checks for the
    // current candidate.
    lead: Vec<Box<PostingIterator>>,
    // Posting iterators that are behind the current target doc but still kept
    // in play because their score upper bound could affect the decision for the
    // current candidate.
    tail: BinaryHeap<TailPosting>,
    // Conservatively rounded sum of upper bounds for all iterators in `tail`.
    // It is maintained in f64 so candidate checks stay O(1) without allowing
    // repeated f32 add/subtract rounding to underestimate the remaining score.
    tail_max_score: f64,
    // Block-max scores are valid for all candidate docs up to this doc id.
    // `None` means the window has not been initialized yet and the next
    // candidate must refresh block-max state before making pruning decisions.
    up_to: Option<u64>,
    // For conjunctions, this is the maximum attainable score for the current
    // block-max window `[target, up_to]`.
    and_max_score: f32,
    // Last conjunction doc returned to the caller. The next conjunction search
    // resumes strictly after this doc, like Lucene's `nextDoc()/advance()`.
    and_last_doc: Option<u64>,
    // Exact query-order score computed while progressively confirming a
    // conjunction candidate. This is valid only for `and_last_doc`.
    and_candidate_score: Option<f32>,
    // Score-first conjunction pruning is restricted to the current impact
    // format and scorers that expose the BM25 denominator contract.
    score_first_and_enabled: bool,
    #[cfg(test)]
    disable_score_first_and: bool,
    // Suffix sums of clause bounds for the current conjunction window. Entry
    // `i` bounds clauses `[i, num_terms)`, without subtracting prior bounds.
    score_first_and_suffix_bounds: SmallVec<[f64; 8]>,
    score_first_and_bounds_up_to: Option<u64>,
    // Inclusive candidate range where every current posting block is dense.
    // This selects score-first scheduling only; membership is still checked.
    score_first_and_dense_range: Option<(u64, u64)>,
    // Initialized only by a certified score-first candidate; cached `None`
    // avoids retrying unsupported norm tables on later candidates.
    score_first_and_norm_k: OnceCell<Option<NormKCache<'a>>>,
    #[cfg(test)]
    and_window_stats: AndWindowStats,
    // Test-only override for comparing bulk and classic conjunctions without
    // mutating the process-wide environment.
    bulk_and_mode_override: Option<BulkAndMode>,
    #[cfg(test)]
    bulk_and_searches: usize,
    #[cfg(test)]
    maxscore_single_essential_windows: usize,
    #[cfg(test)]
    maxscore_general_windows: usize,
    #[cfg(test)]
    maxscore_essential_blocks_skipped: usize,
    // How many windows took the top2-gap shortcut (wand.rs top2 branch),
    // as opposed to the plain single-essential path. Test-only: like the
    // counters beside it, nothing in a release build reads it.
    #[cfg(test)]
    maxscore_top2_gap_windows: usize,
    // How many times the top2-gap shortcut was declined because its span was
    // empty or sat below the window floor (GUARD 2). Test-only.
    #[cfg(test)]
    maxscore_top2_gap_empty_span: usize,
    #[cfg(test)]
    phrase_position_checks: Cell<usize>,
    documents: &'a D,
    scorer: S,
    // Shared cross-partition top-k floor. Each partition publishes its local
    // k-th score (`atomic_store_max_f32`) and prunes against the running value
    // -- a lower bound on the global k-th, so it never drops a real top-k doc.
    shared_threshold: Option<Arc<AtomicU32>>,
    shared_norm_addends: Option<SharedNormAddends>,
}

/// Monotonically raise an f32 stored in an `AtomicU32` to `val`. CAS loop (not a
/// bit-max) so it stays correct for negative scores -- BM25 idf can go negative.
fn atomic_store_max_f32(slot: &AtomicU32, val: f32) {
    let mut cur = slot.load(Ordering::Relaxed);
    while val > f32::from_bits(cur) {
        match slot.compare_exchange_weak(cur, val.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

fn conservative_score_sum(scores: impl Iterator<Item = f32>) -> f32 {
    let (num_scores, exact) = scores.fold((0, 0.0_f64), |(count, sum), score| {
        (count + 1, sum + f64::from(score))
    });
    let widened = exact * score_sum_upper_bound_factor(num_scores);
    outward_f32_upper_bound(widened)
}

/// Round a wide score bound to the least `f32` that still covers it.
#[inline]
pub(super) fn outward_f32_upper_bound(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) < value {
        next_up_f32(rounded)
    } else {
        rounded
    }
}

fn next_up_f32(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

#[inline]
fn next_up_f64(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[cfg(test)]
mod tests;
