// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
fn test_exact_phrase_with_repeated_terms(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    docs.append(0, 16);

    let token_a_positions = vec![vec![1_u32, 3, 10]];
    let token_b_positions = vec![vec![2_u32, 11]];
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                token_a_positions.clone(),
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], token_b_positions, 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("a"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], token_a_positions, 1.0, is_compressed),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    assert!(wand.check_exact_positions().unwrap());
    assert!(wand.check_positions(0).unwrap());
}

#[rstest]
fn test_exact_phrase_respects_query_position_gaps(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    docs.append(0, 16);

    let postings = vec![
        PostingIterator::new(
            String::from("want"),
            0,
            0,
            generate_posting_list_with_positions(vec![0], vec![vec![0_u32]], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("apple"),
            1,
            2,
            generate_posting_list_with_positions(vec![0], vec![vec![2_u32]], 1.0, is_compressed),
            docs.len(),
        ),
    ];

    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    assert!(wand.check_exact_positions().unwrap());
    assert!(wand.check_positions(0).unwrap());
}

#[rstest]
fn test_and_phrase_miss_advances_to_next_candidate(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    docs.append(0, 8);
    docs.append(1, 8);

    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0, 1],
                vec![vec![1_u32], vec![10_u32]],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(
                vec![0, 1],
                vec![vec![3_u32], vec![11_u32]],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    let first = wand.next().unwrap().unwrap();
    assert_eq!(first.0.doc_id(), 0);
    assert!(!wand.check_positions(0).unwrap());

    wand.threshold = 1.5;
    let second = wand.next().unwrap().unwrap();
    assert_eq!(second.0.doc_id(), 1);
    assert!(wand.check_positions(0).unwrap());
}

#[rstest]
fn exact_phrase_skips_stopword_decode_when_rare_pair_misses(
    #[values(false, true)] is_compressed: bool,
) {
    let mut docs = DocSet::default();
    docs.append(0, 16);

    let postings = vec![
        PostingIterator::new(
            String::from("the"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                vec![(0..100_u32).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("news"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], vec![vec![50_u32]], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("journal"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], vec![vec![99_u32]], 1.0, is_compressed),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    let _ = take_position_cursor_calls();
    assert!(!wand.check_exact_positions().unwrap());
    assert_eq!(take_position_cursor_calls(), 2);
    assert!(!wand.check_exact_positions_bulk().unwrap());
    assert_eq!(take_position_cursor_calls(), 2);
}

#[rstest]
fn exact_phrase_decodes_stopword_when_rare_pair_aligns(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    docs.append(0, 16);

    let hit_the: Vec<u32> = (0..100).collect();
    let postings = vec![
        PostingIterator::new(
            String::from("the"),
            0,
            0,
            generate_posting_list_with_positions(vec![0], vec![hit_the], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("news"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], vec![vec![11_u32]], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("journal"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], vec![vec![12_u32]], 1.0, is_compressed),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    let _ = take_position_cursor_calls();
    assert!(wand.check_exact_positions().unwrap());
    assert_eq!(take_position_cursor_calls(), 3);
    assert!(wand.check_exact_positions_bulk().unwrap());
    assert_eq!(take_position_cursor_calls(), 3);
}

#[rstest]
fn exact_phrase_rejects_when_stopword_misses_aligned_pair(
    #[values(false, true)] is_compressed: bool,
) {
    let mut docs = DocSet::default();
    docs.append(0, 16);

    // "news journal" aligns at base 10, but "the" has no position 10.
    let the_positions: Vec<u32> = (0..10).chain(11..100).collect();
    let postings = vec![
        PostingIterator::new(
            String::from("the"),
            0,
            0,
            generate_posting_list_with_positions(vec![0], vec![the_positions], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("news"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], vec![vec![11_u32]], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("journal"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], vec![vec![12_u32]], 1.0, is_compressed),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    let _ = take_position_cursor_calls();
    assert!(!wand.check_exact_positions().unwrap());
    assert_eq!(take_position_cursor_calls(), 3);
    assert!(!wand.check_exact_positions_bulk().unwrap());
    assert_eq!(take_position_cursor_calls(), 3);
}

/// The follower sits entirely past the anchor, so the very first probe
/// proves no base can align: the scan must jump instead of walking.
#[rstest]
fn exact_phrase_scan_skips_anchor_past_disjoint_follower(
    #[values(false, true)] is_compressed: bool,
) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                vec![(0..n).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(
                vec![0],
                vec![(2 * n..3 * n).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions().unwrap());
    assert!(!matched, "disjoint lists cannot form a phrase");
    assert!(steps <= 2, "anchor was walked ({steps} steps), not skipped");

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions_bulk().unwrap());
    assert!(!matched, "disjoint lists cannot form a phrase (bulk)");
    assert!(
        steps <= 2,
        "bulk anchor was walked ({steps} steps), not skipped"
    );
}

/// The only match sits at the far end of a long anchor. Reaching it in a
/// couple of steps is what proves the jump lands on the right candidate
/// instead of over-shooting it.
#[rstest]
fn exact_phrase_scan_jumps_to_a_late_match(#[values(false, true)] is_compressed: bool) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                vec![(0..n).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(
                vec![0],
                vec![(n / 2 + 1..n + 1).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions().unwrap());
    assert!(matched, "the late match must be found");
    assert!(
        steps <= 3,
        "late match took {steps} steps; the jump missed it"
    );

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions_bulk().unwrap());
    assert!(matched, "the late match must be found (bulk)");
    assert!(steps <= 3, "bulk late match took {steps} steps");
}

/// Interleaved lists: no bound can skip anything, so the scan must fall back
/// to stepping and still visit every candidate exactly once (and terminate).
#[rstest]
fn exact_phrase_scan_steps_when_no_bound_can_skip(#[values(false, true)] is_compressed: bool) {
    let n = 1024u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                vec![(0..n).map(|k| 3 * k).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(
                vec![0],
                vec![(0..n).map(|k| 3 * k + 2).collect()],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions().unwrap());
    assert!(!matched, "interleaved lists cannot form a phrase");
    assert!(
        (1..=n as usize).contains(&steps),
        "expected a single walk of {n} anchors, got {steps}"
    );
}

/// The anchor jumps over a candidate before the last clause has ever been
/// probed, so that clause is decoded with a target already far into its
/// list: the lazy decode and the monotone probe have to agree.
#[rstest]
fn exact_phrase_scan_decodes_a_follower_after_the_anchor_jumps(
    #[values(false, true)] is_compressed: bool,
) {
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let mut middle = vec![300_u32, 901];
    middle.extend(1000..1050);
    let mut last = vec![902_u32];
    last.extend(3000..3100);
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(
                vec![0],
                vec![vec![100_u32, 200, 900]],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], vec![middle], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("c"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], vec![last], 1.0, is_compressed),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions().unwrap());
    assert!(
        matched,
        "the clause decoded after the jump must still align"
    );
    assert_eq!(
        steps, 2,
        "the anchor must jump straight to 900, got {steps} steps"
    );
    assert_eq!(take_position_cursor_calls(), 3);
}

/// The anchor clause is not the first word of the phrase, so the jump has to
/// account for `anchor_offset`. Without it the bound lands one position
/// short and the scan needs an extra step.
#[rstest]
fn exact_phrase_scan_jumps_with_a_nonzero_anchor_offset(
    #[values(false, true)] is_compressed: bool,
) {
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let mut first = vec![100_u32, 497, 899];
    first.extend(1000..1050);
    let mut last = vec![500_u32];
    last.extend(2000..2050);
    last.push(5000);
    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list_with_positions(vec![0], vec![first], 1.0, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list_with_positions(
                vec![0],
                vec![vec![101_u32, 498, 900]],
                1.0,
                is_compressed,
            ),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("c"),
            2,
            2,
            generate_posting_list_with_positions(vec![0], vec![last], 1.0, is_compressed),
            docs.len(),
        ),
    ];
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);

    let (matched, steps) = counted_anchor_steps(|| wand.check_exact_positions().unwrap());
    assert!(!matched, "the last clause never reaches the phrase");
    assert_eq!(
        steps, 2,
        "the jump must carry anchor_offset (bound 499, not 498), got {steps} steps"
    );
}

/// The compound-phrase request from review: drive the exact-phrase check the
/// way `WandCursor::matches()` does - a phrase leaf inside a compound query -
/// instead of calling the entries directly.
#[rstest]
fn wand_cursor_confirmation_rejects_disjoint_phrase_positions(
    #[values(false, true)] is_compressed: bool,
) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let make_postings = || {
        vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(0..n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(2 * n..3 * n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ]
    };
    let params = FtsSearchParams {
        phrase_slop: Some(0),
        ..Default::default()
    };
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::And,
        make_postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    assert_eq!(cursor.next().unwrap(), Some(0));

    let (matched, steps) = counted_anchor_steps(|| cursor.matches().unwrap());
    assert!(!matched, "disjoint positions cannot form a phrase");
    assert!(
        steps <= 2,
        "the compound path must skip too, took {steps} steps"
    );
}

/// Same path, but the phrase really is present: the confirmation has to
/// survive the jump instead of over-shooting the match.
#[rstest]
fn wand_cursor_confirmation_accepts_aligned_phrase_positions(
    #[values(false, true)] is_compressed: bool,
) {
    let n = 4096u32;
    let mut docs = DocSet::default();
    docs.append(0, 1_000_001);
    let make_postings = || {
        vec![
            PostingIterator::new(
                String::from("a"),
                0,
                0,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(0..n).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
            PostingIterator::new(
                String::from("b"),
                1,
                1,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![(1..n + 1).collect()],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            ),
        ]
    };
    let params = FtsSearchParams {
        phrase_slop: Some(0),
        ..Default::default()
    };
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::And,
        make_postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    assert_eq!(cursor.next().unwrap(), Some(0));

    let (matched, steps) = counted_anchor_steps(|| cursor.matches().unwrap());
    assert!(matched, "the phrase is present at base 0");
    assert!(
        steps <= 2,
        "took {steps} steps to confirm a first-position match"
    );
}

#[rstest]
fn exact_phrase_heap_path_matches_wide_clause_count(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    docs.append(0, 32);
    let num_clauses = 17usize;
    let postings = (0..num_clauses)
        .map(|index| {
            PostingIterator::new(
                format!("t{index}"),
                index as u32,
                index as u32,
                generate_posting_list_with_positions(
                    vec![0],
                    vec![vec![index as u32]],
                    1.0,
                    is_compressed,
                ),
                docs.len(),
            )
        })
        .collect::<Vec<_>>();
    let bm25 = IndexBM25Scorer::new(std::iter::empty());
    let wand = Wand::new(Operator::And, postings.into_iter(), &docs, bm25);
    assert!(wand.check_exact_positions().unwrap());
    assert!(wand.check_exact_positions_bulk().unwrap());
}
