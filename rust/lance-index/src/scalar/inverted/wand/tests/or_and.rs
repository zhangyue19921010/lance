// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[tokio::test]
async fn test_wand(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for i in 0..2 * BLOCK_SIZE {
        docs.append(i as u64, 1);
    }

    // when the pivot is greater than 0, and the first posting list is exhausted after shallow_next
    let postings = vec![
        PostingIterator::new(
            String::from("test"),
            0,
            0,
            generate_posting_list(
                Vec::from_iter(0..=BLOCK_SIZE as u32 + 1),
                1.0,
                None,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("full"),
            1,
            1,
            generate_posting_list(vec![BLOCK_SIZE as u32 + 2], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    // This should trigger the bug when the second posting list becomes empty
    let result = wand
        .search(&FtsSearchParams::default(), &NoOpMetricsCollector)
        .unwrap();
    assert_eq!(result.len(), 0); // Should not panic
}

/// The shared floor prunes partitions that can't reach the global top-k: a
/// high-scoring partition sets the floor and the rest skip their blocks.
/// (Result correctness is covered by the FTS search tests, since sharing is
/// always on.)
#[test]
fn cross_partition_threshold_sharing_prunes() {
    use crate::metrics::MetricsCollector;
    use std::sync::atomic::AtomicUsize;

    #[derive(Default)]
    struct CountComparisons(AtomicUsize);
    impl MetricsCollector for CountComparisons {
        fn record_parts_loaded(&self, _: usize) {}
        fn record_index_loads(&self, _: usize) {}
        fn record_comparisons(&self, n: usize) {
            self.0.fetch_add(n, Ordering::Relaxed);
        }
    }

    let params = FtsSearchParams::default().with_limit(Some(10));
    let part_docs = 4 * BLOCK_SIZE as u32;
    // One high-scoring partition (weight 10) then 7 low-scoring ones.
    let parts: Vec<(f32, std::ops::Range<u32>)> = std::iter::once((10.0, 0..part_docs))
        .chain((1..8).map(|i| (1.0, i * part_docs..(i + 1) * part_docs)))
        .collect();

    let new_floor = || Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));

    // Total comparisons across all partitions. `Some(floor)` makes every
    // partition share that one floor; `None` gives each its own.
    let total_comparisons = |shared_floor: Option<&Arc<AtomicU32>>| -> usize {
        let metrics = CountComparisons::default();
        for (qw, rows) in &parts {
            let mut docs = DocSet::default();
            for d in rows.clone() {
                docs.append(d as u64, 1);
            }
            let postings = vec![PostingIterator::with_query_weight(
                String::from("t"),
                0,
                0,
                *qw,
                generate_posting_list(rows.clone().collect(), *qw, None, false),
                docs.len(),
            )];
            let floor = shared_floor.cloned().unwrap_or_else(new_floor);
            Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer)
                .with_shared_threshold(floor)
                .search(&params, &metrics)
                .unwrap();
        }
        metrics.0.load(Ordering::Relaxed)
    };

    let one_floor = new_floor();
    let with_shared_floor = total_comparisons(Some(&one_floor));
    let with_private_floors = total_comparisons(None);
    assert!(
        with_shared_floor < with_private_floors,
        "shared floor should prune comparisons: \
             shared={with_shared_floor} private={with_private_floors}"
    );
}

#[test]
fn test_posting_iterator_next_compressed_partition_point() {
    let mut docs = DocSet::default();
    let num_docs = (BLOCK_SIZE * 2 + 5) as u32;
    for i in 0..num_docs {
        docs.append(i as u64, 1);
    }

    let doc_ids = (0..num_docs).collect::<Vec<_>>();
    let posting = generate_posting_list(doc_ids, 1.0, None, true);
    let mut iter = PostingIterator::new(String::from("term"), 0, 0, posting, docs.len());

    iter.next(10);
    assert_eq!(iter.doc().unwrap().doc_id(), 10);

    let target = BLOCK_SIZE as u64 + 3;
    iter.next(target);
    assert_eq!(iter.doc().unwrap().doc_id(), target);

    iter.next(num_docs as u64 + 10);
    assert!(iter.doc().is_none());
}

#[test]
fn posting_iterator_fork_restarts_shared_compressed_payload() {
    let num_docs = (BLOCK_SIZE * 2 + 5) as u32;
    let posting = generate_posting_list((0..num_docs).collect(), 1.0, None, true);
    let mut original = PostingIterator::new(String::from("term"), 7, 3, posting, num_docs as usize);

    original.next(BLOCK_SIZE as u64 + 3);
    assert_eq!(
        original.doc().map(|doc| doc.doc_id()),
        Some(BLOCK_SIZE as u64 + 3)
    );

    let mut replay = original.fork_from_start();
    assert_eq!(replay.doc().map(|doc| doc.doc_id()), Some(0));
    replay.next(5);
    assert_eq!(replay.doc().map(|doc| doc.doc_id()), Some(5));
    assert_eq!(
        original.doc().map(|doc| doc.doc_id()),
        Some(BLOCK_SIZE as u64 + 3),
        "fork advancement must not mutate the membership cursor"
    );
}

#[test]
fn test_wand_skip_to_next_block() {
    let mut docs = DocSet::default();
    for i in 0..201 {
        docs.append(i as u64, 1);
    }

    let large_posting_docs1: Vec<u32> = (0..=200).collect();

    let postings = vec![
        PostingIterator::new(
            String::from("full"),
            0,
            0,
            generate_posting_list(large_posting_docs1, 1.0, Some(vec![0.5, 0.5]), true),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("text"),
            1,
            1,
            generate_posting_list(vec![0], 1.0, Some(vec![0.5]), true),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, bm25);

    // set a threshold that the sum of max scores can hit,
    // but the sum of block max scores is less than the threshold,
    wand.threshold = 1.5;

    let result = wand.search(&FtsSearchParams::default(), &NoOpMetricsCollector);

    assert!(result.is_ok());
}

#[test]
fn test_or_single_term_block_skip_matches_and() {
    // Hot docs occupy the middle block; the flanking blocks score far below
    // the threshold. A single-term disjunction must skip them yet return what
    // the conjunctive path returns.
    let total = 3 * BLOCK_SIZE as u32;
    let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 12;

    let mut docs = DocSet::default();
    for row_id in 0..total {
        // hot docs get distinct scores 1/1..1/12; the rest score 0.001
        let doc_tokens = if hot.contains(&row_id) {
            row_id - hot.start + 1
        } else {
            1000
        };
        docs.append(row_id as u64, doc_tokens);
    }

    let params = FtsSearchParams::new().with_limit(Some(10));
    let run = |operator| {
        let scored = Arc::new(AtomicUsize::new(0));
        let posting = PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_posting_list(
                (0..total).collect(),
                1.0,
                Some(vec![0.001, 1.0, 0.001]),
                true,
            ),
            docs.len(),
        );
        let mut wand = Wand::new(
            operator,
            std::iter::once(posting),
            &docs,
            CountingScorer {
                scored: scored.clone(),
            },
        );
        let hits = wand.search(&params, &NoOpMetricsCollector).unwrap();
        let mut row_ids = hits.iter().map(|hit| hit.document).collect::<Vec<_>>();
        row_ids.sort_unstable();
        (row_ids, scored.load(Ordering::Relaxed))
    };

    let (or_hits, or_scored) = run(Operator::Or);
    let (and_hits, _) = run(Operator::And);

    let expected = (hot.start..hot.start + 10)
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(or_hits, expected, "OR must return the top-k");
    assert_eq!(or_hits, and_hits, "OR and AND must agree for a single term");
    // Without pruning OR scores all `total` docs; with it the cold blocks are skipped.
    assert!(
        or_scored <= 2 * BLOCK_SIZE,
        "expected pruning to skip a block, but scored {or_scored} of {total}",
    );
}

#[rstest]
fn test_or_search_records_comparisons(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for row_id in 0..6 {
        docs.append(row_id, 1);
    }

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("alpha"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0, 1, 4], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("beta"),
            1,
            1,
            1.0,
            generate_posting_list(vec![1, 2, 5], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    let metrics = LocalMetricsCollector::default();
    let candidates = wand.search(&FtsSearchParams::default(), &metrics).unwrap();

    assert_eq!(sorted_candidate_row_ids(candidates), vec![0, 1, 2, 4, 5]);
    assert!(metrics.comparisons.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_wand_new_uses_precomputed_query_weight() {
    let mut docs = DocSet::default();
    docs.append(1, 1);

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        2.0,
        generate_posting_list(vec![0], 1.0, None, false),
        docs.len(),
    )];

    let wand = Wand::new(
        Operator::Or,
        postings.into_iter(),
        &docs,
        PanicQueryWeightScorer,
    );
    assert_eq!(wand.head.len(), 1);
}

#[test]
fn test_and_search_terminates_for_disjoint_postings() {
    let mut docs = DocSet::default();
    for i in 0..6 {
        docs.append(i, 1);
    }

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(vec![0, 2, 4], 1.0, None, false),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(vec![1, 3, 5], 1.0, None, false),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    assert!(wand.next().unwrap().is_none());
}

#[test]
fn test_up_to_refreshes_on_first_candidate() {
    let mut docs = DocSet::default();
    for i in 0..=(BLOCK_SIZE as u64 + 1) {
        docs.append(i, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(
            (0..=(BLOCK_SIZE as u32 + 1)).collect(),
            1.0,
            Some(vec![1.0, 1.0]),
            true,
        ),
        docs.len(),
    )];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    assert!(wand.up_to.is_none());
    let _ = wand.next().unwrap();
    assert!(wand.up_to.is_some());
}

#[test]
fn test_or_push_back_lead_uses_current_block_max_for_tail_bound() {
    let total = 2 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list((0..total).collect(), 10.0, Some(vec![1.0, 10.0]), true),
        docs.len(),
    )];
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 1.5;

    wand.update_max_scores(0);
    wand.move_head_doc_to_lead(0);
    assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));

    wand.push_back_leads(1);

    assert_eq!(wand.tail.len(), 1);
    assert!(
        (wand.tail_max_score - 1.0).abs() < 1e-6,
        "tail should use the current block max, got {}",
        wand.tail_max_score
    );
    assert!(wand.head_doc().is_none());
}

#[test]
fn test_or_push_back_lead_falls_back_after_block_window_expires() {
    let total = 2 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let freqs = (0..total)
        .map(|doc_id| if doc_id >= BLOCK_SIZE as u32 { 10 } else { 1 })
        .collect::<Vec<_>>();
    let mut posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list_with_freqs(
            (0..total).collect(),
            freqs,
            10.0,
            Some(vec![1.0, 10.0]),
            true,
        ),
        docs.len(),
    );
    posting.next((BLOCK_SIZE - 1) as u64);
    let mut wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);
    wand.threshold = 1.5;

    let block_end = (BLOCK_SIZE - 1) as u64;
    wand.update_max_scores(block_end);
    wand.move_head_doc_to_lead(block_end);
    assert_eq!(wand.up_to, Some(block_end));

    wand.push_back_leads(BLOCK_SIZE as u64);

    assert!(wand.tail.is_empty());
    assert_eq!(wand.head_doc(), Some(BLOCK_SIZE as u64));
    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
}

#[test]
fn test_non_positive_threshold_advances_without_impact_bound_scoring() {
    let mut docs = DocSet::default();
    for doc_id in 0..3 {
        docs.append(doc_id, 1);
    }
    let make_posting = || {
        PostingIterator::with_query_weight(
            String::from("term"),
            0,
            0,
            1.0,
            generate_impact_posting_list_with_freqs(vec![0, 1, 2], vec![1, 1, 1], vec![1, 1, 1]),
            docs.len(),
        )
    };
    let scored = Arc::new(AtomicUsize::new(0));

    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(make_posting()),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.move_head_before_target_to_tail(1);
    assert_eq!(wand.head_doc(), Some(1));
    assert!(wand.tail.is_empty());
    assert_eq!(scored.load(Ordering::Relaxed), 0);

    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(make_posting()),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.move_head_doc_to_lead(0);
    wand.push_back_leads(1);
    assert_eq!(wand.head_doc(), Some(1));
    assert!(wand.tail.is_empty());
    assert_eq!(scored.load(Ordering::Relaxed), 0);
}

#[test]
fn test_or_plain_tail_does_not_advance_headless_window() {
    let mut docs = DocSet::default();
    for doc_id in 0..4 {
        docs.append(doc_id, 1);
    }

    let postings = vec![PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0, 1, 2, 3], 1.0, None, false),
        docs.len(),
    )];
    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.0;

    wand.update_max_scores(0);
    wand.move_head_doc_to_lead(0);
    wand.push_back_leads(1);

    assert_eq!(wand.tail.len(), 1);
    assert!(wand.head_doc().is_none());
    assert!(!wand.advance_tail_to_next_or_window());
}

#[test]
fn test_or_headless_tail_window_scans_past_final_top_tail() {
    let total = 3 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        docs.append(doc_id as u64, 1);
    }

    let future_docs = (0..BLOCK_SIZE as u32)
        .chain(2 * BLOCK_SIZE as u32..3 * BLOCK_SIZE as u32)
        .collect::<Vec<_>>();
    let mut future_freqs = vec![1; future_docs.len()];
    future_freqs[0] = 4;
    future_freqs[BLOCK_SIZE] = 20;
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("future"),
            0,
            0,
            1.0,
            generate_posting_list_with_freqs(
                future_docs,
                future_freqs,
                20.0,
                Some(vec![4.0, 20.0]),
                true,
            ),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("final_tail"),
            1,
            1,
            1.0,
            generate_posting_list_with_freqs(vec![0], vec![6], 6.0, Some(vec![6.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("booster"),
            2,
            2,
            1.0,
            generate_posting_list_with_freqs(vec![0], vec![7], 7.0, Some(vec![7.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    let result = wand
        .search(
            &FtsSearchParams::new().with_limit(Some(1)),
            &NoOpMetricsCollector,
        )
        .unwrap();

    assert_eq!(
        sorted_candidate_row_ids(result),
        vec![(2 * BLOCK_SIZE) as u64]
    );
}

#[test]
fn test_and_search_prunes_with_threshold_and_keeps_candidate() {
    let mut docs = DocSet::default();
    for i in 0..(2 * BLOCK_SIZE as u64) {
        let doc_tokens = if i < BLOCK_SIZE as u64 { 100 } else { 1 };
        docs.append(i, doc_tokens);
    }
    let all_docs = (0..2 * BLOCK_SIZE as u32).collect::<Vec<_>>();

    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("a"),
            0,
            0,
            1.0,
            generate_posting_list(all_docs.clone(), 1.0, Some(vec![0.02, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("b"),
            1,
            1,
            1.0,
            generate_posting_list(all_docs, 1.0, Some(vec![0.02, 1.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(
        Operator::And,
        postings.into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), BLOCK_SIZE as u64);
}

#[test]
fn test_and_advance_falls_back_to_narrow_when_range_max_loosens_bound() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 10.0, Some(vec![0.1, 10.0, 0.1, 0.1]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 5.0;

    let target = wand.and_advance_target(0);

    assert_eq!(target, BLOCK_SIZE as u64);
    assert_eq!(wand.up_to, Some((2 * BLOCK_SIZE - 1) as u64));
    assert!(
        (wand.and_max_score - 11.0).abs() < 1e-6,
        "expected the second narrow window to include the high follower block, got {}",
        wand.and_max_score
    );
    assert_eq!(wand.and_window_stats.windows_wide, 0);
    assert_eq!(wand.and_window_stats.windows_narrow, 2);
    assert_eq!(wand.and_window_stats.windows_skipped, 1);
}

#[test]
fn test_and_advance_uses_narrow_window_for_candidate_ranges() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 1.0, Some(vec![1.0, 1.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 1.0, Some(vec![1.0, 1.0, 1.0, 1.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 1.5;

    let target = wand.and_advance_target(0);

    assert_eq!(target, 0);
    assert_eq!(wand.up_to, Some((BLOCK_SIZE - 1) as u64));
    assert!((wand.and_max_score - 2.0).abs() < 1e-6);
    assert_eq!(wand.and_window_stats.windows_wide, 0);
    assert_eq!(wand.and_window_stats.windows_narrow, 1);
    assert_eq!(wand.and_window_stats.range_blocks_scanned, 0);
}

#[test]
fn test_and_wide_window_only_skips_and_does_not_return_candidates() {
    let total = 4 * BLOCK_SIZE as u32;
    let mut docs = DocSet::default();
    for i in 0..total {
        docs.append(i as u64, 1);
    }

    let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
    let follower_docs = (0..total).collect::<Vec<_>>();
    let postings = vec![
        PostingIterator::with_query_weight(
            String::from("lead"),
            0,
            0,
            1.0,
            generate_posting_list(lead_docs, 3.0, Some(vec![1.0, 3.0]), true),
            docs.len(),
        ),
        PostingIterator::with_query_weight(
            String::from("follower"),
            1,
            1,
            1.0,
            generate_posting_list(follower_docs, 3.0, Some(vec![0.1, 0.1, 3.0, 3.0]), true),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.0;

    let candidate = wand.next().unwrap().unwrap();

    assert_eq!(candidate.0.doc_id(), (2 * BLOCK_SIZE) as u64);
    assert_eq!(wand.up_to, Some((3 * BLOCK_SIZE - 1) as u64));
    assert_eq!(wand.and_window_stats.windows_wide, 1);
    assert_eq!(wand.and_window_stats.windows_skipped, 1);
    assert_eq!(wand.and_window_stats.windows_narrow, 1);
    assert_eq!(wand.and_window_stats.candidates_returned, 1);
}

#[test]
fn test_and_range_max_preserves_exact_top_k() {
    let total = 4 * BLOCK_SIZE as u32;
    let hot = BLOCK_SIZE as u32..BLOCK_SIZE as u32 + 16;
    let mut docs = DocSet::default();
    for doc_id in 0..total {
        let doc_tokens = if hot.contains(&doc_id) { 1 } else { 1000 };
        docs.append(doc_id as u64, doc_tokens);
    }

    let params = FtsSearchParams::new().with_limit(Some(8));
    let run = |is_compressed: bool| {
        let lead_docs = (0..total).step_by(2).collect::<Vec<_>>();
        let follower_docs = (0..total).collect::<Vec<_>>();
        let lead_scores = is_compressed.then_some(vec![1.0, 0.001]);
        let follower_scores = is_compressed.then_some(vec![0.001, 1.0, 0.001, 0.001]);
        let postings = vec![
            PostingIterator::with_query_weight(
                String::from("lead"),
                0,
                0,
                1.0,
                generate_posting_list(lead_docs, 1.0, lead_scores, is_compressed),
                docs.len(),
            ),
            PostingIterator::with_query_weight(
                String::from("follower"),
                1,
                1,
                1.0,
                generate_posting_list(follower_docs, 1.0, follower_scores, is_compressed),
                docs.len(),
            ),
        ];
        let mut wand = Wand::new(
            Operator::And,
            postings.into_iter(),
            &docs,
            InverseDocLengthScorer,
        );
        sorted_candidate_row_ids(wand.search(&params, &NoOpMetricsCollector).unwrap())
    };

    let compressed = run(true);
    let plain = run(false);
    let expected = hot.step_by(2).map(u64::from).collect::<Vec<_>>();
    assert_eq!(compressed, expected);
    assert_eq!(compressed, plain);
}

#[test]
fn test_block_max_score_up_to_slides_and_expires_old_max() {
    let total = 5 * BLOCK_SIZE as u32;
    let posting = generate_posting_list(
        (0..total).collect(),
        5.0,
        Some(vec![1.0, 4.0, 2.0, 5.0, 3.0]),
        true,
    );
    let mut posting = PostingIterator::new(String::from("term"), 0, 0, posting, total as usize);

    posting.shallow_next(0);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((3 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        4.0
    );

    posting.shallow_next((2 * BLOCK_SIZE) as u64);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((4 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        5.0
    );

    posting.shallow_next((4 * BLOCK_SIZE) as u64);
    assert_eq!(
        posting
            .block_max_score_up_to_with_stats((5 * BLOCK_SIZE - 1) as u64, &UnitScorer)
            .score,
        3.0
    );
}

#[test]
fn test_impact_level1_skip_keeps_boundary_equality_in_group() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS + 1) * block_size;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );
        let target = (IMPACT_LEVEL1_BLOCKS * block_size - 1) as u64;

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, IMPACT_LEVEL1_BLOCKS - 1);

        posting.next(target);
        assert_eq!(posting.block_idx, IMPACT_LEVEL1_BLOCKS - 1);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_impact_level1_skip_handles_partial_final_group() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS + 3) * block_size + 17;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );
        let target = (total - 1) as u64;
        let expected_block = total.div_ceil(block_size) - 1;

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, expected_block);

        posting.next(target);
        assert_eq!(posting.block_idx, expected_block);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_impact_level1_skip_reaches_far_target_doc() {
    for block_size in [crate::scalar::inverted::LEGACY_BLOCK_SIZE, 256] {
        let total = (IMPACT_LEVEL1_BLOCKS * 3 + 5) * block_size;
        let target_block = IMPACT_LEVEL1_BLOCKS * 2 + 2;
        let target = (target_block * block_size + 17) as u64;
        let mut posting = PostingIterator::new(
            String::from("term"),
            0,
            0,
            generate_contiguous_impact_posting_list_with_block_size(total, block_size),
            total,
        );

        posting.shallow_next(target);
        assert_eq!(posting.block_idx, target_block);

        posting.next(target);
        assert_eq!(posting.block_idx, target_block);
        assert_eq!(posting.doc().map(|doc| doc.doc_id()), Some(target));
    }
}

#[test]
fn test_or_impact_level1_window_skips_low_group_with_single_score() {
    let total = (IMPACT_LEVEL1_BLOCKS + 1) * BLOCK_SIZE;
    let target = (IMPACT_LEVEL1_BLOCKS * BLOCK_SIZE) as u64;
    let mut docs = DocSet::default();
    for doc_id in 0..total as u64 {
        docs.append(doc_id, 1);
    }

    let doc_ids = (0..total as u32).collect::<Vec<_>>();
    let freqs = doc_ids
        .iter()
        .map(|doc_id| if u64::from(*doc_id) < target { 1 } else { 10 })
        .collect::<Vec<_>>();
    let posting_list = generate_impact_posting_list_with_freqs(doc_ids, freqs, vec![1; total]);
    let mut probe = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        posting_list.clone(),
        docs.len(),
    );
    probe.shallow_next(0);
    let counting_scorer = CountingScorer {
        scored: Arc::new(AtomicUsize::new(0)),
    };
    let (group_up_to, group_score) = probe.impact_group_bound(&counting_scorer).unwrap();
    assert_eq!(group_up_to, target - 1);
    assert_eq!(group_score, 1.0);
    // A window ending past the current block must answer from level1.
    assert_eq!(
        probe.window_max_score(Some(target - 1), &counting_scorer),
        1.0
    );

    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        posting_list,
        docs.len(),
    );
    let scored = Arc::new(AtomicUsize::new(0));
    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(posting),
        &docs,
        CountingScorer {
            scored: scored.clone(),
        },
    );
    wand.threshold = 2.0;

    let (candidate, score) = wand.next().unwrap().unwrap();
    assert_eq!(candidate.doc_id(), target);
    assert_eq!(score, 10.0);
    // The doc-weight bounds bake exactly once (one doc_weight call per
    // frontier pair across all entries); beyond that only the returned
    // candidate is scored.
    let total_entries = (IMPACT_LEVEL1_BLOCKS + 1) + 2;
    assert!(
        scored.load(Ordering::Relaxed) <= total_entries + 8,
        "bounds should be baked once instead of recomputed per window; scored={}",
        scored.load(Ordering::Relaxed)
    );
}

#[test]
fn test_compressed_impact_block_max_score_memoizes_current_block() {
    let total = 2 * BLOCK_SIZE as u32;
    let doc_ids = (0..total).collect::<Vec<_>>();
    let freqs = doc_ids
        .iter()
        .map(|doc_id| if *doc_id < BLOCK_SIZE as u32 { 1 } else { 2 })
        .collect::<Vec<_>>();
    let doc_lengths = vec![1; total as usize];
    let posting_list = generate_impact_posting_list_with_freqs(doc_ids, freqs, doc_lengths);
    let mut posting =
        PostingIterator::new(String::from("term"), 0, 0, posting_list, total as usize);
    let scored = Arc::new(AtomicUsize::new(0));
    let scorer = CountingScorer {
        scored: scored.clone(),
    };

    let first_score = posting.block_max_score(&scorer);
    assert_eq!(first_score, 1.0);
    // Baking the query-local doc-weight bounds visits every frontier pair
    // once (two level0 entries plus one level1 entry for this list).
    let baked = scored.load(Ordering::Relaxed);
    assert!(baked >= 2);
    {
        let compressed = unsafe { &mut *posting.compressed_state_ptr() };
        assert_eq!(
            compressed.level0_cache,
            Some((0, BLOCK_SIZE as u32 - 1, first_score))
        );
    }

    let second_score = posting.block_max_score(&scorer);
    assert_eq!(second_score, first_score);
    assert_eq!(
        scored.load(Ordering::Relaxed),
        baked,
        "repeated block max scores must not recompute doc weights"
    );

    posting.shallow_next(BLOCK_SIZE as u64);
    let next_block_score = posting.block_max_score(&scorer);
    assert_eq!(next_block_score, 2.0);
    assert_eq!(
        scored.load(Ordering::Relaxed),
        baked,
        "other blocks answer from the baked bounds without rescoring"
    );
}

#[rstest]
#[case(0.0)]
#[case(-1.0)]
fn test_non_positive_query_weight_skips_global_impact_bound(#[case] query_weight: f32) {
    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        query_weight,
        generate_impact_posting_list_with_freqs(vec![0], vec![1], vec![1]),
        1,
    );
    let scored = Arc::new(AtomicUsize::new(0));
    let scorer = CountingScorer {
        scored: scored.clone(),
    };

    assert_eq!(posting.global_upper_bound(&scorer), 0.0);
    assert_eq!(scored.load(Ordering::Relaxed), 0);
}
