// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// `exclusive_partial_score_limit` must agree with `score_sum_cannot_compete`
/// for every f32 partial score: scores at or below the limit are rejected,
/// scores above it compete. Sweeps ULP neighbourhoods of the boundary plus
/// coarse samples, over floors, remaining bounds and clause counts.
#[test]
fn exclusive_partial_score_limit_matches_predicate() {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut checked = 0usize;
    for case in 0..2_000 {
        let floor = if case % 7 == 0 {
            f32::from_bits((next() as u32) & 0x7F7F_FFFF)
        } else {
            (next() % 10_000) as f32 / 37.0 + 1e-6
        };
        let remaining = if case % 5 == 0 {
            f64::from(f32::from_bits((next() as u32) & 0x7F7F_FFFF))
        } else {
            (next() % 10_000) as f64 / 41.0
        };
        let num_lists = 2 + (next() % 6) as usize;
        let factor = score_sum_upper_bound_factor(num_lists);
        let rejects = |score: f32| {
            score_sum_cannot_compete(
                score,
                remaining,
                floor,
                factor,
                CompetitiveFloorMode::Exclusive,
            )
        };
        let limit = exclusive_partial_score_limit(remaining, floor, factor);
        let mut probes = vec![
            0.0_f32,
            f32::MIN_POSITIVE,
            floor,
            f32::MAX,
            f32::MIN,
            ((f64::from(floor) / factor) - remaining) as f32,
        ];
        if let Some(limit) = limit {
            let mut up = limit;
            let mut down = limit;
            for _ in 0..4 {
                probes.push(up);
                probes.push(down);
                up = next_up_f32(up);
                down = next_down_f32(down);
            }
        }
        for _ in 0..16 {
            probes.push(f32::from_bits((next() as u32) & 0x7F7F_FFFF));
        }
        for score in probes {
            let expected = rejects(score);
            let actual = limit.is_some_and(|limit| score <= limit);
            assert_eq!(
                expected, actual,
                "score={score:?} limit={limit:?} floor={floor:?} remaining={remaining:?} num_lists={num_lists}"
            );
            checked += 1;
        }
    }
    assert!(checked > 50_000);
}

#[test]
fn exclusive_partial_score_limit_handles_non_finite_inputs() {
    let factor = score_sum_upper_bound_factor(3);
    assert_eq!(
        exclusive_partial_score_limit(f64::INFINITY, 1.0, factor),
        None
    );
    assert_eq!(exclusive_partial_score_limit(1.0, f32::NAN, factor), None);
    assert_eq!(
        exclusive_partial_score_limit(1.0, f32::INFINITY, factor),
        None
    );
    // A floor no partial score can fail to beat rejects nothing.
    let limit = exclusive_partial_score_limit(1.0e30, 1.0, factor);
    assert!(limit.is_none_or(|limit| !score_sum_cannot_compete(
        next_up_f32(limit),
        1.0e30,
        1.0,
        factor,
        CompetitiveFloorMode::Exclusive
    )));
}
#[test]
fn wand_cursor_cost_uses_materialized_visibility_upper_bound() {
    let total_docs = 10;
    let selected = DocVisibility::Selected(roaring::RoaringBitmap::from_iter([1, 7]));
    let selected_bound = <&DocVisibility as ModernVisibility>::len(&&selected, total_docs);

    let filtered = DocVisibility::Filtered {
        projection: resident_row_address_projection_for_test((0..total_docs as u64).collect()),
        mask: Arc::new(RowAddrMask::all_rows()),
    };
    let filtered_bound = <&DocVisibility as ModernVisibility>::len(&&filtered, total_docs);
    let all_bound = AllModernDocuments.len(total_docs);

    assert_eq!(selected_bound, 2);
    assert_eq!(filtered_bound, total_docs);
    assert_eq!(all_bound, total_docs);

    let scorer = Arc::new(MemBM25Scorer::new(
        total_docs as u64,
        total_docs,
        std::collections::HashMap::from([("term".to_owned(), 8)]),
    ));
    let posting = || {
        PostingIterator::new(
            "term".to_owned(),
            0,
            0,
            generate_posting_list((0..8).collect(), 1.0, None, true),
            total_docs,
        )
    };
    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;

    let selected_documents = CostOnlyDocuments {
        total_docs,
        visible_cost_upper_bound: selected_bound,
    };
    let selected_cursor = WandCursor::new(
        Operator::Or,
        vec![posting()],
        &selected_documents,
        scorer.clone(),
        &params,
        &metrics,
    );
    assert_eq!(selected_cursor.cost(), 2);

    for visible_cost_upper_bound in [filtered_bound, all_bound] {
        let documents = CostOnlyDocuments {
            total_docs,
            visible_cost_upper_bound,
        };
        let cursor = WandCursor::new(
            Operator::Or,
            vec![posting()],
            &documents,
            scorer.clone(),
            &params,
            &metrics,
        );
        assert_eq!(cursor.cost(), 8);
    }

    let mut complete_docs = DocSet::default();
    for doc_id in 0..total_docs {
        complete_docs.append(doc_id as u64, 1);
    }
    assert_eq!(
        WandDocuments::visible_cost_upper_bound(&complete_docs),
        total_docs
    );
}

#[test]
fn conservative_score_sum_covers_query_order_f32_rounding() {
    let values = [
        f32::from_bits(0x3e65_15bd),
        f32::from_bits(0x34b4_3b11),
        f32::from_bits(0x35e9_48ed),
        f32::from_bits(0x3203_3773),
    ];
    let exact_score = values
        .into_iter()
        .fold(0.0_f32, |score, value| score + value);

    assert_eq!(exact_score.to_bits(), 0x3e65_164a);
    assert!(conservative_score_sum(values.into_iter()) >= exact_score);

    let mut reordered = [
        f32::from_bits(0x3c87_b63e),
        f32::from_bits(0x3d28_d10b),
        f32::from_bits(0x3cc4_29c0),
    ];
    let bound = conservative_score_sum(reordered.into_iter());
    reordered.sort_by(|left, right| right.total_cmp(left));
    let reordered_score = reordered
        .into_iter()
        .fold(0.0_f32, |score, value| score + value);
    assert_eq!(reordered_score.to_bits(), 0x3da7_6086);
    assert!(bound >= reordered_score);
}

#[test]
fn compound_global_bound_rejects_grouped_term_scoring() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let list = generate_posting_list(vec![0], 1.0, None, false);
    let grouped_terms = Arc::<[GroupedTermScorer]>::from([GroupedTermScorer::new(1.0, &list)]);
    let posting =
        PostingIterator::with_query_weight(String::from("term"), 0, 0, 1.0, list, docs.len())
            .with_grouped_terms(grouped_terms);
    let wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);

    assert_eq!(wand.compound_global_score_upper_bound(), None);
}

#[test]
fn compound_global_bound_rejects_late_partial_posting_state() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0], 1.0, None, false),
        docs.len(),
    );
    let mut wand = Wand::new(Operator::Or, std::iter::once(posting), &docs, UnitScorer);

    assert!(wand.next().unwrap().is_some());
    wand.push_back_leads(1);
    assert_eq!(wand.compound_global_score_upper_bound(), None);
}

#[test]
fn test_maxscore_prefix_bound_covers_f32_summation_rounding() {
    let remaining_bounds = [6.286_838_4e-7_f32, 0.015_441_144_f32];
    let essential_score = 2.762_496_2_f32;
    let threshold = f32::from_bits(0x4031_c9bc);

    // Raw f32 prefix accumulation rounds down to an apparent tie, while
    // scoring the clauses individually produces a competitive document.
    let raw_prefix = remaining_bounds.into_iter().sum::<f32>();
    assert_eq!(essential_score + raw_prefix, threshold);
    let actual_score = remaining_bounds
        .into_iter()
        .rev()
        .fold(essential_score, |score, bound| score + bound);
    assert!(actual_score > threshold);

    let prefix_bound = remaining_bounds.into_iter().map(f64::from).sum::<f64>();
    assert!(!score_sum_cannot_compete(
        essential_score,
        prefix_bound,
        threshold,
        score_sum_upper_bound_factor(3),
        CompetitiveFloorMode::Exclusive,
    ));
}

#[test]
fn two_clause_inclusive_bound_rounds_outward_at_f32_boundary() {
    let partial_score = 1.0_f32;
    let remaining_upper_bound = f32::from_bits(0x3380_0001);
    let floor = partial_score + remaining_upper_bound;

    assert_eq!(floor.to_bits(), 0x3f80_0001);
    assert!(!score_sum_cannot_compete(
        partial_score,
        f64::from(remaining_upper_bound),
        floor,
        score_sum_upper_bound_factor(2),
        CompetitiveFloorMode::Inclusive,
    ));
}

#[test]
fn inclusive_floor_keeps_tie_without_admitting_lower_ulp() {
    let posting = PostingIterator::with_query_weight(
        "term".to_owned(),
        0,
        0,
        1.0,
        generate_posting_list_with_freqs(vec![0, 1], vec![1, 2], 1.0, None, false),
        2,
    );
    let mut docs = DocSet::default();
    docs.append(0, 1);
    docs.append(1, 1);
    let mut wand = Wand::new(
        Operator::Or,
        std::iter::once(posting),
        &docs,
        AdjacentScoreScorer,
    )
    .with_floor_mode(CompetitiveFloorMode::Inclusive);
    wand.update_threshold(1.0, 1.0);

    let (doc, score) = wand.next().unwrap().unwrap();
    assert_eq!(doc.doc_id(), 1);
    assert_eq!(score, 1.0);
}

#[test]
fn final_score_uses_query_order_and_keeps_floor_ties() {
    let contributions = [0.002_972_301_6_f32, 0.001_982_450_7_f32, 0.001_882_293_f32];
    // Bounds deliberately make the WAND heap visit positions 2, 0, 1.
    let bounds = [2.0_f32, 1.0, 3.0];
    let postings = || {
        contributions
            .into_iter()
            .zip(bounds)
            .enumerate()
            .map(|(position, (query_weight, max_score))| {
                PostingIterator::with_query_weight(
                    format!("t{position}"),
                    position as u32,
                    position as u32,
                    query_weight,
                    generate_posting_list(vec![0], max_score, None, false),
                    1,
                )
            })
            .collect::<Vec<_>>()
    };
    let mut docs = DocSet::default();
    docs.append(0, 1);
    let mut wand = Wand::new(Operator::Or, postings().into_iter(), &docs, UnitScorer);

    let (_, heap_order_score) = wand.next().unwrap().unwrap();
    let query_order_score = contributions
        .into_iter()
        .fold(0.0_f32, |score, contribution| score + contribution);
    let wide_score = contributions
        .into_iter()
        .fold(0.0_f64, |score, contribution| {
            score + f64::from(contribution)
        }) as f32;

    assert_eq!(query_order_score.to_bits(), 0x3be0_094c);
    assert_eq!(wide_score.to_bits(), 0x3be0_094b);
    assert_eq!(heap_order_score.to_bits(), 0x3be0_094a);
    assert_eq!(wand.score_in_query_order(1), query_order_score);

    let params = FtsSearchParams::default();
    let metrics = NoOpMetricsCollector;
    let mut cursor = WandCursor::new(
        Operator::Or,
        postings(),
        &docs,
        Arc::new(MemBM25Scorer::new(1, 1, std::collections::HashMap::new())),
        &params,
        &metrics,
    );
    // Equal-score candidates remain competitive because row id is the
    // final tie-breaker. The internal heap-order sum is slightly lower.
    cursor.set_min_competitive_score(query_order_score).unwrap();
    assert_eq!(cursor.next().unwrap(), Some(0));
    assert_eq!(cursor.current_score().unwrap(), query_order_score);
}

struct PartialNormScorer;

impl Scorer for PartialNormScorer {
    fn query_weight(&self, _token: &str) -> f32 {
        1.0
    }

    fn doc_weight(&self, freq: u32, _doc_tokens: u32) -> f32 {
        freq as f32
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        (doc_tokens == 0).then_some(0.0)
    }
}

#[test]
fn test_norm_cache_requires_all_norm_codes() {
    let mut docs = DocSet::default();
    docs.append(0, 1);
    docs.set_quantized_scoring(true);
    let wand = Wand::new(Operator::Or, std::iter::empty(), &docs, PartialNormScorer);

    assert!(wand.norm_k_cache().is_none());

    let shared = SharedNormAddends::default();
    for _ in 0..2 {
        let wand = Wand::new(Operator::Or, std::iter::empty(), &docs, PartialNormScorer)
            .with_shared_norm_addends(shared.clone());
        assert!(wand.norm_k_cache().is_none());
    }
}

#[test]
fn test_shared_norm_addends_are_built_once_and_match_local_table() {
    let mut docs = DocSet::default();
    docs.append(0, 7);
    docs.set_quantized_scoring(true);
    let scorer = Arc::new(MemBM25Scorer::new(70, 10, std::collections::HashMap::new()));
    let shared = SharedNormAddends::default();
    let norm_addends = |shared: Option<SharedNormAddends>| {
        let mut wand = Wand::new(Operator::Or, std::iter::empty(), &docs, scorer.clone());
        if let Some(shared) = shared {
            wand = wand.with_shared_norm_addends(shared);
        }
        wand.norm_k_cache().unwrap().1
    };

    let first = norm_addends(Some(shared.clone()));
    let second = norm_addends(Some(shared));
    let local = norm_addends(None);
    assert!(
        Arc::ptr_eq(&first, &second),
        "partitions of one query must share one table"
    );
    assert_eq!((*first).map(f32::to_bits), (*local).map(f32::to_bits));
}

#[test]
fn test_top_k_collector_reuses_frequency_slots() -> Result<()> {
    const LIMIT: usize = 8;
    const NUM_DOCS: usize = 10_000;
    let mut collector = TopKCollector::new(LIMIT, LIMIT);

    for doc in 0..NUM_DOCS {
        let num_terms = doc % 4 + 1;
        let inserted = collector.insert(
            ScoredDoc::new(doc as u64, doc as f32),
            num_terms as u32,
            doc as u64,
            (0..num_terms).map(|term| (term as u32, doc as u32)),
        )?;
        assert!(inserted);
        assert!(collector.num_frequency_slots() <= LIMIT);
    }
    assert_eq!(collector.num_frequency_slots(), LIMIT);

    let mut candidates = collector.into_candidates(|key| key)?;
    candidates.sort_unstable_by_key(|candidate| candidate.posting_doc_id);
    assert_eq!(candidates.len(), LIMIT);
    for (candidate, expected_doc) in candidates.iter().zip(NUM_DOCS - LIMIT..NUM_DOCS) {
        assert_eq!(candidate.posting_doc_id, expected_doc as u64);
        assert_eq!(candidate.document, expected_doc as u64);
        let expected_freqs = (0..expected_doc % 4 + 1)
            .map(|term| (term as u32, expected_doc as u32))
            .collect::<Vec<_>>();
        assert_eq!(candidate.freqs, expected_freqs);
    }
    Ok(())
}

#[rstest]
#[case::auto("auto", Some(BulkAndMode::Auto))]
#[case::auto_case_and_whitespace(" AUTO ", Some(BulkAndMode::Auto))]
#[case::on("on", Some(BulkAndMode::On))]
#[case::on_legacy("1", Some(BulkAndMode::On))]
#[case::off("off", Some(BulkAndMode::Off))]
#[case::off_legacy("0", Some(BulkAndMode::Off))]
#[case::invalid("true", None)]
#[case::empty("", None)]
fn test_bulk_and_mode_parse(#[case] value: &str, #[case] expected: Option<BulkAndMode>) {
    assert_eq!(BulkAndMode::parse(value), expected);
}

#[rstest]
#[case::auto_one(BulkAndMode::Auto, 1, false)]
#[case::auto_two(BulkAndMode::Auto, 2, true)]
#[case::auto_three(BulkAndMode::Auto, 3, true)]
#[case::auto_four(BulkAndMode::Auto, 4, false)]
#[case::on_one(BulkAndMode::On, 1, true)]
#[case::on_five(BulkAndMode::On, 5, true)]
#[case::off_two(BulkAndMode::Off, 2, false)]
#[case::off_five(BulkAndMode::Off, 5, false)]
fn test_bulk_and_mode_enabled_for(
    #[case] mode: BulkAndMode,
    #[case] num_clauses: usize,
    #[case] expected: bool,
) {
    assert_eq!(mode.enabled_for(num_clauses), expected);
}

#[test]
fn find_next_geq_matches_partition_point_for_full_u32_domain() {
    let cases = [
        vec![],
        vec![7],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        (0..32).collect(),
        vec![
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            i32::MAX as u32,
            i32::MAX as u32 + 1,
            i32::MAX as u32 + 2,
            i32::MAX as u32 + 3,
            i32::MAX as u32 + 4,
            i32::MAX as u32 + 5,
            i32::MAX as u32 + 6,
            u32::MAX - 2,
            u32::MAX - 1,
            u32::MAX,
        ],
    ];
    let targets = [
        0,
        1,
        6,
        7,
        8,
        i32::MAX as u32 - 1,
        i32::MAX as u32,
        i32::MAX as u32 + 1,
        u32::MAX - 1,
        u32::MAX,
    ];

    for docs in &cases {
        for pos in 0..=docs.len() {
            for &target in &targets {
                let expected = pos + docs[pos..].partition_point(|&doc_id| doc_id < target);
                let actual = unsafe { find_next_geq(docs.as_ptr(), pos, docs.len(), target) };
                assert_eq!(
                    actual, expected,
                    "docs={docs:?}, pos={pos}, target={target}"
                );
                assert_eq!(
                    find_next_geq_in_block(docs, pos, target),
                    expected,
                    "block search: docs={docs:?}, pos={pos}, target={target}"
                );
            }
        }
    }
}

#[rstest]
#[case::packed_delta(PositionStreamCodec::PackedDelta)]
#[case::varint_doc_delta(PositionStreamCodec::VarintDocDelta)]
fn test_shared_position_cursors_use_independent_scratch(
    #[case] codec: PositionStreamCodec,
) -> Result<()> {
    let mut posting_list =
        generate_posting_list_with_positions(vec![0], vec![vec![1_u32, 3, 10]], 1.0, true);
    let PostingList::Compressed(ref mut list) = posting_list else {
        unreachable!("the helper was asked for a compressed posting list");
    };
    let mut encoded = Vec::new();
    encode_position_stream_block_into(&[1, 3, 10], &[3], codec, &mut encoded)?;
    list.positions = Some(CompressedPositionStorage::SharedStream(
        SharedPositionStream::new(codec, vec![0], bytes::Bytes::from(encoded)),
    ));

    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, 1);
    let first = posting.position_cursor()?;
    let second = posting.position_cursor()?;

    assert_eq!(second.positions.as_slice(), &[1, 3, 10]);
    assert_eq!(first.positions.as_slice(), &[1, 3, 10]);
    assert!(posting.position_scratch.borrow().is_none());
    drop(second);
    assert!(posting.position_scratch.borrow().is_some());
    drop(first);
    assert!(posting.position_scratch.borrow().is_some());
    Ok(())
}

#[test]
fn test_phrase_search_propagates_corrupt_packed_positions() {
    let mut docs = DocSet::default();
    docs.append(0, BLOCK_SIZE as u32 + 1);

    let mut corrupt_list = generate_posting_list_with_positions(
        vec![0],
        vec![(0..BLOCK_SIZE as u32).collect()],
        1.0,
        true,
    );
    let PostingList::Compressed(ref mut list) = corrupt_list else {
        unreachable!("the helper was asked for a compressed posting list");
    };
    // A bit width of one requires a 16-byte payload. Keep only the header
    // to verify malformed on-disk data becomes a search error, not a panic.
    list.positions = Some(CompressedPositionStorage::SharedStream(
        SharedPositionStream::new(
            PositionStreamCodec::PackedDelta,
            vec![0],
            bytes::Bytes::from_static(&[1]),
        ),
    ));

    let postings = vec![
        PostingIterator::new(String::from("corrupt"), 0, 0, corrupt_list, docs.len()),
        PostingIterator::new(
            String::from("valid"),
            1,
            1,
            generate_posting_list_with_positions(vec![0], vec![vec![BLOCK_SIZE as u32]], 1.0, true),
            docs.len(),
        ),
    ];
    let mut wand = Wand::new(Operator::And, postings.into_iter(), &docs, UnitScorer);
    let mut params = FtsSearchParams::default().with_limit(Some(10));
    params.phrase_slop = Some(0);

    let error = wand
        .search(&params, &NoOpMetricsCollector)
        .expect_err("corrupt packed positions should fail the phrase search");
    let message = error.to_string();
    assert!(
        message.contains("packed position group payload"),
        "{message}"
    );
    assert!(message.contains("corrupt"), "{message}");
}

#[rstest]
fn test_wand_batches_lagging_iterators(#[values(false, true)] is_compressed: bool) {
    let mut docs = DocSet::default();
    for i in 0..16 {
        docs.append(i as u64, 1);
    }

    let postings = vec![
        PostingIterator::new(
            String::from("a"),
            0,
            0,
            generate_posting_list(vec![1, 10], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("b"),
            1,
            1,
            generate_posting_list(vec![2, 10], 1.0, None, is_compressed),
            docs.len(),
        ),
        PostingIterator::new(
            String::from("c"),
            2,
            2,
            generate_posting_list(vec![10], 1.0, None, is_compressed),
            docs.len(),
        ),
    ];

    let mut wand = Wand::new(Operator::Or, postings.into_iter(), &docs, UnitScorer);
    wand.threshold = 2.5;

    let candidate = wand.next().unwrap().unwrap();
    assert_eq!(candidate.0.doc_id(), 10);
    assert_eq!(wand.lead.len(), 3);
}

#[test]
fn test_flat_search_or_keeps_masked_docs_in_same_block() {
    let mut docs = DocSet::default();
    for i in 0..=(BLOCK_SIZE as u64 + 1) {
        let doc_tokens = if i == 1 { 100 } else { 1 };
        docs.append(i, doc_tokens);
    }

    let posting = PostingIterator::with_query_weight(
        String::from("term"),
        0,
        0,
        1.0,
        generate_posting_list(
            (1..=(BLOCK_SIZE as u32 + 1)).collect(),
            1.0,
            Some(vec![1.0, 1.0]),
            true,
        ),
        docs.len(),
    );

    let mut wand = Wand::new(
        Operator::Or,
        vec![posting].into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let selected = vec![(1_u64, 1_u64), (2_u64, 2_u64)];
    let result = wand
        .flat_search(
            &FtsSearchParams::default(),
            Box::new(selected.into_iter()),
            &NoOpMetricsCollector,
        )
        .unwrap();

    let matched = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert_eq!(matched, vec![2]);
}

#[test]
fn test_doc_ids_resolves_every_document_a_row_owns() {
    // A list<string> column indexes each element as its own document, so
    // one row id owns several doc ids. row 100 -> {0, 1}, row 101 -> {2}.
    let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
    let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
    let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

    assert_eq!(docs.doc_ids(100).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(docs.doc_ids(101).collect::<Vec<_>>(), vec![2]);
    assert!(docs.doc_ids(999).next().is_none());

    // legacy shape (row id == doc id) still resolves to a single document.
    let mut legacy = DocSet::default();
    legacy.append(7, 1);
    assert_eq!(legacy.doc_ids(7).collect::<Vec<_>>(), vec![7]);
    assert!(legacy.doc_ids(8).next().is_none());
}

#[rstest]
fn test_flat_search_finds_list_row_with_match_at_non_last_position(
    #[values(false, true)] is_compressed: bool,
) {
    // row 100 owns two element-documents (doc 0, doc 1) that share its row
    // id; row 101 owns doc 2. The query term lives only in doc 0 — the
    // *non-last* element of row 100. Resolving the row to a single doc id
    // would evaluate doc 1, miss the term, and drop the row (lancedb#3352).
    let row_id_col = arrow_array::UInt64Array::from(vec![100_u64, 100, 101]);
    let num_tokens_col = arrow_array::UInt32Array::from(vec![1_u32, 1, 1]);
    let docs = DocSet::from_columns(&row_id_col, &num_tokens_col, false, None).unwrap();

    let posting = PostingIterator::with_query_weight(
        String::from("needle"),
        0,
        0,
        1.0,
        generate_posting_list(vec![0], 1.0, None, is_compressed),
        docs.len(),
    );

    let mut wand = Wand::new(
        Operator::Or,
        vec![posting].into_iter(),
        &docs,
        InverseDocLengthScorer,
    );
    wand.threshold = 0.5;

    let selected = docs
        .doc_ids(100)
        .map(|doc_id| (doc_id, 100_u64))
        .collect::<Vec<_>>();
    let result = wand
        .flat_search(
            &FtsSearchParams::default(),
            Box::new(selected.into_iter()),
            &NoOpMetricsCollector,
        )
        .unwrap();

    // The legacy adapter resolves the prefilter to every owned document,
    // while the candidate identity remains row 100.
    let addrs = result
        .into_iter()
        .map(|doc| doc.document)
        .collect::<Vec<_>>();
    assert!(
        addrs.as_slice() == [100],
        "expected exactly row 100, got {addrs:?}"
    );
}

#[test]
fn test_block_max_score_matches_stored_value() {
    let doc_ids = vec![0_u32];
    let block_max_scores = vec![0.7_f32];
    let posting_list = generate_posting_list(doc_ids, 0.7, Some(block_max_scores), true);
    let expected = match &posting_list {
        PostingList::Compressed(list) => list.block_max_score(0),
        PostingList::Plain(_) => unreachable!("expected compressed posting list"),
    };

    let posting = PostingIterator::new(String::from("test"), 0, 0, posting_list, 1);

    let actual = posting.block_max_score(&UnitScorer);
    assert!(
        (actual - expected).abs() < 1e-6,
        "block max score should match stored value"
    );
}

#[test]
fn test_modern_doc_id_validation_checks_layout_and_upper_bound() {
    let compressed = generate_posting_list(vec![0, 4], 1.0, None, true);
    let posting = PostingIterator::new(String::from("term"), 0, 0, compressed, 5);
    posting
        .validate_modern_doc_ids(5)
        .expect("largest DocId is inside the document table");
    let error = posting.validate_modern_doc_ids(4).unwrap_err();
    assert!(error.to_string().contains("DocId 4"));
    assert!(error.to_string().contains("[0, 4)"));

    let plain = generate_posting_list(vec![0], 1.0, None, false);
    let posting = PostingIterator::new(String::from("legacy"), 0, 0, plain, 1);
    let error = posting.validate_modern_doc_ids(1).unwrap_err();
    assert!(error.to_string().contains("legacy row-address layout"));
}

#[test]
fn test_256_document_blocks_without_impacts_use_conservative_quantized_score_bound() {
    let exact_doc_length = 300;
    let quantized_doc_length = crate::scalar::inverted::index::dequantize_doc_length(
        crate::scalar::inverted::index::quantize_doc_length(exact_doc_length),
    );
    assert!(quantized_doc_length < exact_doc_length);

    let scorer = Arc::new(MemBM25Scorer::new(100, 1, Default::default()));
    let stored_exact_score = scorer.doc_weight(1, exact_doc_length);
    let quantized_score = scorer.doc_weight(1, quantized_doc_length);
    assert!(quantized_score > stored_exact_score);

    let doc_ids = [0_u32];
    let frequencies = [1_u32];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        frequencies.iter(),
        std::iter::once(stored_exact_score),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
    )
    .unwrap();
    let posting_list = PostingList::Compressed(CompressedPostingList::new(
        blocks,
        stored_exact_score,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
        None,
        None,
    ));
    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, doc_ids.len());
    let expected_bound = BM25_DOC_WEIGHT_UPPER_BOUND;

    assert_eq!(posting.approximate_upper_bound(), expected_bound);
    assert_eq!(posting.global_upper_bound(&scorer), expected_bound);
    assert_eq!(posting.block_max_score(&scorer), expected_bound);
    assert_eq!(
        posting.block_max_score_up_to_with_stats(0, &scorer).score,
        expected_bound
    );
    assert!(expected_bound >= quantized_score);
}

#[test]
fn test_256_document_blocks_without_impacts_unknown_scorer_uses_infinite_bound() {
    let doc_ids = [0_u32];
    let frequencies = [10_u32];
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        frequencies.iter(),
        std::iter::once(10.0),
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
    )
    .unwrap();
    let posting_list = PostingList::Compressed(CompressedPostingList::new(
        blocks,
        10.0,
        doc_ids.len() as u32,
        crate::scalar::inverted::PostingTailCodec::VarintDelta,
        MAX_POSTING_BLOCK_SIZE,
        None,
        None,
    ));
    let posting = PostingIterator::new(String::from("term"), 0, 0, posting_list, doc_ids.len());

    assert!(posting.global_upper_bound(&UnitScorer).is_infinite());
    assert!(posting.block_max_score(&UnitScorer).is_infinite());
    assert!(
        posting
            .block_max_score_up_to_with_stats(0, &UnitScorer)
            .score
            .is_infinite()
    );
}
