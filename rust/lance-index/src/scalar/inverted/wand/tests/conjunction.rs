// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

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
