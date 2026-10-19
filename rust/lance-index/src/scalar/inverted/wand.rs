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

mod bulk_and;
mod collector;
mod conjunction;
mod cursor;
mod disjunction;
mod documents;
mod intersection;
mod maxscore;
mod phrase;
mod posting;
mod score_bounds;
mod search;

use collector::TopKCollector;
pub(super) use cursor::WandCursor;
pub use documents::DocCandidate;
#[allow(unused_imports)]
pub(super) use documents::{AllModernDocuments, ModernVisibility};
pub(super) use documents::{
    FlatDocuments, LegacyWandDocuments, ModernWandDocuments, WandDocuments,
};
#[cfg(test)]
use maxscore::{MAXSCORE_INNER_WINDOW, MaxScoreClause};
pub(super) use posting::{GroupedTermScorer, validate_modern_posting_doc_ids};
use score_bounds::{
    CompetitiveFloorMode, conservative_score_sum, exclusive_partial_score_limit, next_up_f64,
    score_sum_cannot_compete,
};
#[cfg(test)]
use score_bounds::{next_down_f32, next_up_f32};
pub(super) use score_bounds::{outward_f32_upper_bound, score_sum_upper_bound_factor};
use search::{ScoreContribution, score_contributions_in_query_order};

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
    // How many windows took the top2-gap shortcut (wand/maxscore.rs top2 branch),
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

#[cfg(test)]
mod tests;
