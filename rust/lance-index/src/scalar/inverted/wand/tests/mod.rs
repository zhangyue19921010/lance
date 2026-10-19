// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::buffer::ScalarBuffer;
use rstest::rstest;

use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::documents::resident_row_address_projection_for_test;
use super::super::impact::build_impact_skip_data;
use super::*;
use crate::scalar::inverted::scorer::{IndexBM25Scorer, MemBM25Scorer};
use crate::{
    metrics::{LocalMetricsCollector, NoOpMetricsCollector},
    scalar::inverted::{
        CompressedPostingList, PlainPostingList, PostingListBuilder, SharedPositionStream,
        builder::PositionRecorder,
        encoding::{
            compress_posting_list, compress_posting_list_with_tail_codec_and_block_size,
            encode_position_stream_block_into,
        },
    },
};

use itertools::Itertools;
use lance_select::RowAddrMask;

use crate::scalar::inverted::documents::DocVisibility;
use crate::scalar::inverted::impact::IMPACT_LEVEL1_BLOCKS;
use crate::scalar::inverted::index::PositionStreamCodec;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::scorer::{BM25_DOC_WEIGHT_UPPER_BOUND, bm25_doc_weight_with_norm};
use crate::scalar::inverted::{CompressedPositionStorage, DocSet};

use super::doc_source::{AllModernDocuments, ModernVisibility};
use super::phrase::{counted_anchor_steps, take_position_cursor_calls};
use super::posting_iterator::{find_next_geq, find_next_geq_in_block};

mod bulk_and;
mod maxscore_search;
mod phrase;
mod posting_iterator;
mod score_first_and;
mod scoring;
mod search;

struct CostOnlyDocuments {
    total_docs: usize,
    visible_cost_upper_bound: usize,
}

impl WandDocuments for CostOnlyDocuments {
    type Candidate = u64;

    fn len(&self) -> usize {
        self.total_docs
    }

    fn visible_cost_upper_bound(&self) -> usize {
        self.visible_cost_upper_bound
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        None
    }

    fn scoring_num_tokens(&self, _doc_id: u32) -> u32 {
        1
    }

    fn doc_length(&self, _doc: &DocInfo) -> u32 {
        1
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        Some(doc.doc_id())
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        Some(u64::from(doc_id))
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        key
    }

    fn flat_documents(&self) -> Option<FlatDocuments<'_>> {
        None
    }

    fn flat_doc_length(&self, _doc_id: u64, _document_key: u64, _compressed: bool) -> u32 {
        1
    }
}

struct UnitScorer;

impl Scorer for UnitScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }
}

struct AdjacentScoreScorer;

impl Scorer for AdjacentScoreScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        match freq {
            1 => f32::from_bits(1.0_f32.to_bits() - 1),
            2 => 1.0,
            _ => unreachable!("test only defines two score buckets"),
        }
    }
}

struct PanicQueryWeightScorer;

impl Scorer for PanicQueryWeightScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        panic!("query_weight should be precomputed before WAND construction");
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }
}

struct InverseDocLengthScorer;

impl Scorer for InverseDocLengthScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        freq as f32 / doc_tokens as f32
    }
}

// Inverse-doc-length scorer that counts scored documents, so a test can
// assert that block-max pruning skipped blocks.
struct CountingScorer {
    scored: Arc<AtomicUsize>,
}

impl Scorer for CountingScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        self.scored.fetch_add(1, Ordering::Relaxed);
        freq as f32 / doc_tokens as f32
    }
}

struct CountingBm25ShapeScorer {
    scored: Arc<AtomicUsize>,
}

impl Scorer for CountingBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        self.scored.fetch_add(1, Ordering::Relaxed);
        bm25_doc_weight_with_norm(freq, 1.2)
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, _doc_tokens: u32) -> Option<f32> {
        Some(1.2)
    }
}

struct CandidateOnlyBm25ShapeScorer {
    norm_calls: Arc<AtomicUsize>,
    unsupported_norm: Option<f32>,
}

impl Scorer for CandidateOnlyBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        bm25_doc_weight_with_norm(freq, 0.4)
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        self.norm_calls.fetch_add(1, Ordering::Relaxed);
        if doc_tokens == 1 {
            Some(0.4)
        } else {
            self.unsupported_norm
        }
    }
}

struct NonFiniteBoundBm25ShapeScorer;

impl Scorer for NonFiniteBoundBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, _freq: u32, _doc_tokens: u32) -> f32 {
        f32::INFINITY
    }

    fn doc_norm(&self, _doc_tokens: u32) -> Option<f32> {
        Some(1.0)
    }
}

struct VariedBm25ShapeScorer;

impl Scorer for VariedBm25ShapeScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        bm25_doc_weight_with_norm(freq, self.doc_norm(doc_tokens).unwrap())
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        Some(0.3 + doc_tokens as f32 * 0.1)
    }
}

#[cfg_attr(coverage, coverage(off))]
fn unit_length_docs(num_docs: usize) -> DocSet {
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(doc_id as u64, 1);
    }
    docs
}

#[cfg_attr(coverage, coverage(off))]
fn unit_length_impact_posting(
    term: u32,
    doc_ids: Vec<u32>,
    frequencies: Vec<u32>,
    num_docs: usize,
) -> PostingIterator {
    let doc_lengths = vec![1; doc_ids.len()];
    PostingIterator::with_query_weight(
        format!("t{term}"),
        term,
        term,
        1.0,
        generate_impact_posting_list_with_freqs_and_block_size(
            doc_ids,
            frequencies,
            doc_lengths,
            MAX_POSTING_BLOCK_SIZE,
        ),
        num_docs,
    )
}

fn generate_posting_list(
    doc_ids: Vec<u32>,
    max_score: f32,
    block_max_scores: Option<Vec<f32>>,
    is_compressed: bool,
) -> PostingList {
    let freqs = vec![1; doc_ids.len()];
    generate_posting_list_with_freqs(doc_ids, freqs, max_score, block_max_scores, is_compressed)
}

fn generate_posting_list_with_freqs(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    max_score: f32,
    block_max_scores: Option<Vec<f32>>,
    is_compressed: bool,
) -> PostingList {
    assert_eq!(doc_ids.len(), freqs.len());
    let block_max_scores = block_max_scores.unwrap_or_else(|| vec![max_score; doc_ids.len()]);
    if is_compressed {
        let blocks = compress_posting_list(
            doc_ids.len(),
            doc_ids.iter(),
            freqs.iter(),
            block_max_scores.into_iter(),
        )
        .unwrap();
        PostingList::Compressed(CompressedPostingList::new(
            blocks,
            max_score,
            doc_ids.len() as u32,
            crate::scalar::inverted::PostingTailCodec::VarintDelta,
            crate::scalar::inverted::LEGACY_BLOCK_SIZE,
            None,
            None,
        ))
    } else {
        PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
            ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
            Some(max_score),
            None,
        ))
    }
}

fn generate_impact_posting_list_with_freqs(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    doc_lengths: Vec<u32>,
) -> PostingList {
    generate_impact_posting_list_with_freqs_and_block_size(
        doc_ids,
        freqs,
        doc_lengths,
        crate::scalar::inverted::LEGACY_BLOCK_SIZE,
    )
}

fn generate_impact_posting_list_with_freqs_and_block_size(
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    doc_lengths: Vec<u32>,
    block_size: usize,
) -> PostingList {
    assert_eq!(doc_ids.len(), freqs.len());
    assert_eq!(doc_ids.len(), doc_lengths.len());
    let block_max_scores = vec![0.0; doc_ids.len().div_ceil(block_size)];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        freqs.iter(),
        block_max_scores.into_iter(),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        block_size,
    )
    .unwrap();
    let impact_blocks = doc_ids
        .chunks(block_size)
        .zip(freqs.chunks(block_size))
        .zip(doc_lengths.chunks(block_size))
        .map(|((doc_ids, freqs), doc_lengths)| {
            doc_ids
                .iter()
                .copied()
                .zip(freqs.iter().copied())
                .zip(doc_lengths.iter().copied())
                .map(|((doc_id, freq), doc_length)| (doc_id, freq, doc_length))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let impacts = build_impact_skip_data(impact_blocks.as_slice()).unwrap();
    PostingList::Compressed(CompressedPostingList::new(
        blocks,
        0.0,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        block_size,
        None,
        Some(impacts),
    ))
}

fn generate_contiguous_impact_posting_list_with_block_size(
    total: usize,
    block_size: usize,
) -> PostingList {
    generate_impact_posting_list_with_freqs_and_block_size(
        (0..total as u32).collect(),
        vec![1; total],
        vec![1; total],
        block_size,
    )
}

fn generate_posting_list_with_positions(
    doc_ids: Vec<u32>,
    positions_by_doc: Vec<Vec<u32>>,
    max_score: f32,
    is_compressed: bool,
) -> PostingList {
    let freqs = positions_by_doc
        .iter()
        .map(|positions| positions.len() as u32)
        .collect::<Vec<_>>();
    if is_compressed {
        let mut builder = PostingListBuilder::new(true);
        for (doc_id, positions) in doc_ids.iter().copied().zip(positions_by_doc) {
            builder.add(doc_id, PositionRecorder::Position(positions.into()));
        }
        let batch = builder
            .to_batch(vec![max_score; doc_ids.len().div_ceil(BLOCK_SIZE)])
            .unwrap();
        PostingList::from_batch(&batch, Some(max_score), Some(doc_ids.len() as u32)).unwrap()
    } else {
        let mut position_builder =
            arrow::array::ListBuilder::new(arrow::array::Int32Builder::new());
        for positions in positions_by_doc {
            for position in positions {
                position_builder.values().append_value(position as i32);
            }
            position_builder.append(true);
        }
        PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from_iter(doc_ids.iter().map(|id| *id as u64)),
            ScalarBuffer::from_iter(freqs.iter().map(|freq| *freq as f32)),
            Some(max_score),
            Some(position_builder.finish()),
        ))
    }
}

fn sorted_candidate_row_ids(candidates: Vec<DocCandidate<u64>>) -> Vec<u64> {
    let mut row_ids = candidates
        .into_iter()
        .map(|candidate| candidate.document)
        .collect::<Vec<_>>();
    row_ids.sort_unstable();
    row_ids
}
