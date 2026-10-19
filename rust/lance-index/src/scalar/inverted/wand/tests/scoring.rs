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
