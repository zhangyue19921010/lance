// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Value-ordered traversal of a [`BTreeIndex`], see
//! [`ScalarIndex::ordered_entries`](crate::scalar::ScalarIndex::ordered_entries).

use std::cmp::Ordering;
use std::ops::Bound;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, cast::AsArray, types::UInt64Type};
use arrow_ord::{ord::make_comparator, partition::partition, sort::sort_to_indices};
use arrow_schema::SortOptions;
use arrow_select::{concat::concat_batches, take::take_record_batch};
use datafusion_common::ScalarValue;
use futures::{
    StreamExt, TryStreamExt,
    stream::{self, BoxStream},
};
use lance_core::{Error, Result};

use super::{BTreeIndex, LazyIndexReader, partition_point};
use crate::metrics::MetricsCollector;
use crate::scalar::IndexEntry;

/// The `(lower, upper)` value range a walk is limited to.
type ValueRange = (Bound<ScalarValue>, Bound<ScalarValue>);

/// The order the index is trained in: ascending with NULLs first. Used both
/// to sort the lookup table by `min` and to sort a cluster's rows by value.
const TRAINING_ORDER: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

/// Whether row `row` of `values` lies beyond `bound`: below it if `bound` is a
/// lower bound (`is_lower`), above it otherwise. Compared in [`TRAINING_ORDER`],
/// as `pages_between` compares the bounds of a range search.
fn beyond(
    values: &dyn Array,
    bound: &Bound<ScalarValue>,
    is_lower: bool,
) -> Result<Box<dyn Fn(usize) -> bool>> {
    let (value, inclusive) = match bound {
        Bound::Unbounded => return Ok(Box::new(|_| false)),
        Bound::Included(value) => (value, true),
        Bound::Excluded(value) => (value, false),
    };
    let cmp = make_comparator(values, value.to_array_of_size(1)?.as_ref(), TRAINING_ORDER)?;
    Ok(Box::new(move |row| match cmp(row, 0) {
        Ordering::Equal => !inclusive,
        ordering => (ordering == Ordering::Less) == is_lower,
    }))
}

/// Lookup rows sorted by `min`, without the pages that hold no value in the
/// walk's range, and the cluster boundaries over that order
/// (`starts[i]..starts[i + 1]` is cluster `i`).
///
/// A cluster is a connected component of pages whose value ranges strictly
/// overlap; its pages must be loaded together to yield their values in order.
/// A single-stream build never overlaps pages, so every cluster is one page;
/// only a legacy range-partitioned merge produces multi-page clusters.
struct Plan {
    order: Vec<u32>,
    starts: Vec<usize>,
}

fn plan(lookup: &RecordBatch, range: &ValueRange) -> Result<Plan> {
    let mins = lookup.column(0);
    let maxs = lookup.column(1);
    // The lookup is only sorted by `min` for a single-stream build; a legacy
    // range-partitioned merge concatenates per-partition tables, so sort here.
    let sorted = sort_to_indices(mins.as_ref(), Some(TRAINING_ORDER), None)?;
    // An all-null page has a null `max` and holds no value to yield. A page
    // whose `max` is below the range, or whose `min` is above it, holds none in
    // range; a straddle page's NULL `min` sorts first and is never above it.
    let max_below = beyond(maxs.as_ref(), &range.0, true)?;
    let min_above = beyond(mins.as_ref(), &range.1, false)?;
    let order: Vec<u32> = sorted
        .values()
        .iter()
        .copied()
        .filter(|&row| {
            let row = row as usize;
            !maxs.is_null(row) && !max_below(row) && !min_above(row)
        })
        .collect();
    let mut starts = Vec::new();
    let Some(&first) = order.first() else {
        return Ok(Plan { order, starts });
    };
    let max_vs_min = make_comparator(maxs.as_ref(), mins.as_ref(), TRAINING_ORDER)?;
    let max_vs_max = make_comparator(maxs.as_ref(), maxs.as_ref(), TRAINING_ORDER)?;
    starts.push(0);
    let mut running_max = first as usize;
    for (pos, &row) in order.iter().enumerate().skip(1) {
        let row = row as usize;
        // Pages whose ranges strictly overlap must be sorted together. A page
        // whose `min` merely equals the running `max` starts a new cluster: the
        // shared value is then yielded once per cluster, which the contract
        // allows. A null `min` (straddle page: NULLs then values) compares
        // below everything and therefore always joins.
        if max_vs_min(running_max, row) == Ordering::Greater {
            if max_vs_max(row, running_max) == Ordering::Greater {
                running_max = row;
            }
        } else {
            starts.push(pos);
            running_max = row;
        }
    }
    starts.push(order.len());
    Ok(Plan { order, starts })
}

/// Cursor over the clusters of a [`Plan`], consumed one cluster per step.
struct Walk<'a> {
    index: &'a BTreeIndex,
    reader: LazyIndexReader,
    metrics: &'a dyn MetricsCollector,
    /// Only values in this range are yielded.
    range: ValueRange,
    descending: bool,
    /// Built on the first poll so a failure surfaces as the first stream item.
    plan: Option<Plan>,
    /// Clusters already consumed, counted from the end the walk starts at.
    done: usize,
}

impl Walk<'_> {
    /// Page numbers of the next cluster in walk order; `None` once exhausted.
    fn next_cluster(&mut self) -> Result<Option<Vec<u32>>> {
        let plan = match self.plan.take() {
            Some(plan) => plan,
            None => plan(&self.index.page_lookup.batch, &self.range)?,
        };
        let plan = self.plan.insert(plan);
        let num_clusters = plan.starts.len().saturating_sub(1);
        if self.done >= num_clusters {
            return Ok(None);
        }
        let cluster = if self.descending {
            num_clusters - 1 - self.done
        } else {
            self.done
        };
        self.done += 1;
        // Lookup rows are compared by row number but loaded by `page_idx`.
        let page_numbers = self.index.page_lookup.page_numbers()?;
        Ok(Some(
            plan.order[plan.starts[cluster]..plan.starts[cluster + 1]]
                .iter()
                .map(|&row| page_numbers.value(row as usize))
                .collect(),
        ))
    }

    /// Non-null rows of every page in the cluster, jointly sorted by value.
    async fn cluster_rows(&self, pages: &[u32]) -> Result<RecordBatch> {
        let mut batches = Vec::with_capacity(pages.len());
        for &page_number in pages {
            let page = self
                .index
                .lookup_page(page_number, self.reader.clone(), self.metrics)
                .await?;
            batches.push(page.non_null_rows()?);
        }
        let batch = match batches.as_slice() {
            [single] => single.clone(),
            [first, ..] => concat_batches(&first.schema(), &batches)?,
            [] => return Err(Error::internal("btree cluster without pages")),
        };
        let indices = sort_to_indices(batch.column(0).as_ref(), Some(TRAINING_ORDER), None)?;
        Ok(take_record_batch(&batch, &indices)?)
    }

    /// The rows of `rows`, sorted by value, that lie in the walk's range.
    fn in_range(&self, rows: &RecordBatch) -> Result<RecordBatch> {
        let values = rows.column(0).as_ref();
        let below = beyond(values, &self.range.0, true)?;
        let above = beyond(values, &self.range.1, false)?;
        let start = partition_point(0, rows.num_rows(), below);
        let end = partition_point(start, rows.num_rows(), |row| !above(row));
        Ok(rows.slice(start, end - start))
    }

    /// One entry per run of equal values within the cluster, in walk order.
    fn groups(&self, rows: &RecordBatch) -> Result<Vec<IndexEntry>> {
        let values = rows.column(0);
        let ids = rows
            .column(1)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| Error::internal("btree page ids column must be UInt64"))?
            .values();
        let mut ranges = partition(std::slice::from_ref(values))?.ranges();
        if self.descending {
            ranges.reverse();
        }
        ranges
            .into_iter()
            .map(|range| {
                Ok(IndexEntry {
                    value: ScalarValue::try_from_array(values.as_ref(), range.start)?,
                    row_ids: Arc::new(ids[range].iter().copied().collect()),
                })
            })
            .collect()
    }

    /// Entries of the next cluster; `None` once every cluster is consumed.
    async fn next_entries(mut self) -> Result<Option<(Vec<IndexEntry>, Self)>> {
        loop {
            let Some(pages) = self.next_cluster()? else {
                return Ok(None);
            };
            let rows = self.cluster_rows(&pages).await?;
            let rows = self.in_range(&rows)?;
            let entries = self.groups(&rows)?;
            if !entries.is_empty() {
                return Ok(Some((entries, self)));
            }
            // Every row of the cluster was NULL or outside the range (a boundary
            // cluster, or a straddle page whose non-null rows were all dropped by
            // a row id remap): keep going.
        }
    }
}

/// Entries of `index` within `range` in value order; see
/// [`ScalarIndex::ordered_entries`](crate::scalar::ScalarIndex::ordered_entries).
pub(super) fn entries<'a>(
    index: &'a BTreeIndex,
    range: ValueRange,
    descending: bool,
    metrics: &'a dyn MetricsCollector,
) -> BoxStream<'a, Result<IndexEntry>> {
    let walk = Walk {
        index,
        reader: LazyIndexReader::new(index.store.clone(), index.ranges_to_files.clone()),
        metrics,
        range,
        descending,
        plan: None,
        done: 0,
    };
    stream::try_unfold(walk, Walk::next_entries)
        .map_ok(|entries| stream::iter(entries.into_iter().map(Ok::<_, Error>)))
        .try_flatten()
        .boxed()
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;
    use std::ops::Bound::{Excluded, Included, Unbounded};
    use std::sync::{Arc, atomic::Ordering::Relaxed};

    use arrow_array::{ArrayRef, Float64Array, Int32Array, StringArray, UInt64Array, record_batch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::cache::LanceCache;
    use lance_io::object_store::ObjectStore;
    use lance_select::{RowAddrTreeMap, RowSetOps};
    use object_store::path::Path;

    use super::*;
    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::progress::noop_progress;
    use crate::scalar::btree::{
        merge_metadata_files, part_lookup_file_path, part_page_data_file_path, train_btree_index,
    };
    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::{SargableQuery, ScalarIndex, SearchResult};

    const ALL: ValueRange = (Unbounded, Unbounded);

    fn int_range(lower: Bound<i32>, upper: Bound<i32>) -> ValueRange {
        let int = |value| ScalarValue::Int32(Some(value));
        (lower.map(int), upper.map(int))
    }

    fn memory_store() -> Arc<LanceIndexStore> {
        Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::memory()),
            Path::default(),
            Arc::new(LanceCache::no_cache()),
        ))
    }

    /// `nulls` NULLs followed by `values`: the order the index is trained in.
    fn int32(nulls: usize, values: &[i32]) -> ArrayRef {
        let values = values.iter().map(|&value| Some(value));
        Arc::new(Int32Array::from_iter(repeat_n(None, nulls).chain(values)))
    }

    /// A training stream over `values` and `row_ids`. An empty table trains
    /// from an empty stream, not from a batch of zero rows.
    fn training_stream(values: ArrayRef, row_ids: Vec<u64>) -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", values.data_type().clone(), true),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        let ids: ArrayRef = Arc::new(UInt64Array::from(row_ids));
        let batch = RecordBatch::try_new(schema.clone(), vec![values, ids]).unwrap();
        let batches = (batch.num_rows() > 0).then_some(Ok(batch));
        Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)))
    }

    /// Train a single-stream btree with `page_size` rows per page; row `i`
    /// gets row id `100 + i`.
    async fn train(values: ArrayRef, page_size: u64, cache: &LanceCache) -> Arc<BTreeIndex> {
        let store = memory_store();
        let row_ids = (100..100 + values.len() as u64).collect();
        train_btree_index(
            training_stream(values, row_ids),
            store.as_ref(),
            page_size,
            None,
            None,
        )
        .await
        .unwrap();
        BTreeIndex::load(store, None, cache).await.unwrap()
    }

    fn row_ids(set: &RowAddrTreeMap) -> Vec<u64> {
        let mut ids: Vec<u64> = set.row_addrs().unwrap().map(u64::from).collect();
        ids.sort_unstable();
        ids
    }

    /// Every entry in walk order as `(value, sorted row ids)`.
    async fn collect(
        index: &dyn ScalarIndex,
        range: ValueRange,
        descending: bool,
        metrics: &dyn MetricsCollector,
    ) -> Vec<(ScalarValue, Vec<u64>)> {
        let entries: Vec<IndexEntry> = index
            .ordered_entries(range, descending, metrics)
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        entries
            .into_iter()
            .map(|entry| (entry.value, row_ids(&entry.row_ids)))
            .collect()
    }

    /// The rows an exact `IS NULL` search selects, with none left undecided.
    fn null_rows(result: SearchResult) -> Vec<u64> {
        let SearchResult::Exact(nulls) = result else {
            panic!("IS NULL must be exact: {result:?}");
        };
        assert!(nulls.null_rows().is_empty());
        row_ids(&nulls.true_rows())
    }

    #[tokio::test]
    async fn test_entries_skip_nulls_and_allow_adjacent_duplicates() {
        // Pages of 3: [N, N, 1] (a straddle page), [1, 2, 3], [3, 3, 5].
        let values = int32(2, &[1, 1, 2, 3, 3, 3, 5]);
        let index = train(values, 3, &LanceCache::no_cache()).await;

        // MIN on a low-cardinality column loads only the first page.
        let metrics = LocalMetricsCollector::default();
        let mut entries = index.ordered_entries(ALL, false, &metrics).unwrap();
        let first = entries.try_next().await.unwrap().unwrap();
        assert_eq!(first.value, ScalarValue::Int32(Some(1)));
        assert_eq!(row_ids(&first.row_ids), vec![102]);
        assert_eq!(metrics.parts_loaded.load(Relaxed), 1);

        // A value shared by two pages is yielded once per page; equal values
        // within a page are merged.
        let entry = |value: i32, ids: &[u64]| (ScalarValue::Int32(Some(value)), ids.to_vec());
        let expected = vec![
            entry(1, &[102]),
            entry(1, &[103]),
            entry(2, &[104]),
            entry(3, &[105]),
            entry(3, &[106, 107]),
            entry(5, &[108]),
        ];
        assert_eq!(
            collect(index.as_ref(), ALL, false, &NoOpMetricsCollector).await,
            expected
        );
        let mut reversed = expected;
        reversed.reverse();
        assert_eq!(
            collect(index.as_ref(), ALL, true, &NoOpMetricsCollector).await,
            reversed
        );

        // An equality-like range reaches the value in both pages holding it.
        let range = int_range(Included(3), Included(3));
        assert_eq!(
            collect(index.as_ref(), range, false, &NoOpMetricsCollector).await,
            vec![entry(3, &[105]), entry(3, &[106, 107])]
        );
    }

    #[tokio::test]
    async fn test_leading_all_null_pages_and_null_set() {
        // Pages of 2: [N, N], [N, N], ["b", "c"]; variable-width values.
        let values = repeat_n(None, 4).chain([Some("b"), Some("c")]);
        let values: ArrayRef = Arc::new(StringArray::from_iter(values));
        let index = train(values, 2, &LanceCache::no_cache()).await;

        let utf8 = |s: &str| ScalarValue::Utf8(Some(s.to_string()));
        assert_eq!(
            collect(index.as_ref(), ALL, false, &NoOpMetricsCollector).await,
            vec![(utf8("b"), vec![104]), (utf8("c"), vec![105])]
        );

        // The NULL rows are reachable through an IS NULL search instead.
        let result = index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(null_rows(result), vec![100, 101, 102, 103]);
    }

    #[tokio::test]
    async fn test_empty_and_all_null_index() {
        let cache = LanceCache::no_cache();
        let empty = train(int32(0, &[]), 2, &cache).await;
        let all_null = train(int32(2, &[]), 2, &cache).await;
        for index in [empty, all_null] {
            for descending in [false, true] {
                assert!(
                    collect(index.as_ref(), ALL, descending, &NoOpMetricsCollector)
                        .await
                        .is_empty()
                );
            }
        }
    }

    #[tokio::test]
    async fn test_first_entry_loads_one_page() {
        // 1000 values in 16 pages of 64.
        let values: Vec<i32> = (0..1000).collect();
        let cache = LanceCache::with_capacity(64 * 1024 * 1024);
        let index = train(int32(0, &values), 64, &cache).await;

        for (descending, value, row_id) in [(false, 0, 100), (true, 999, 1099)] {
            let metrics = LocalMetricsCollector::default();
            let mut entries = index.ordered_entries(ALL, descending, &metrics).unwrap();
            let first = entries.try_next().await.unwrap().unwrap();
            assert_eq!(first.value, ScalarValue::Int32(Some(value)));
            assert_eq!(row_ids(&first.row_ids), vec![row_id]);
            assert_eq!(metrics.parts_loaded.load(Relaxed), 1);
            assert_eq!(metrics.single_page_reads(), 1);
        }

        // A full walk over the same cache finds those two pages already loaded.
        let metrics = LocalMetricsCollector::default();
        let all = collect(index.as_ref(), ALL, false, &metrics).await;
        assert_eq!(all.len(), 1000);
        assert_eq!(all[0], (ScalarValue::Int32(Some(0)), vec![100]));
        assert_eq!(all[999], (ScalarValue::Int32(Some(999)), vec![1099]));
        assert!(metrics.index_cache_hits() >= 1);
    }

    #[tokio::test]
    async fn test_range_starts_at_bound_and_prunes_pages() {
        // 1000 values in 16 pages of 64: page p holds 64p..=64p + 63.
        let values: Vec<i32> = (0..1000).collect();
        let index = train(int32(0, &values), 64, &LanceCache::no_cache()).await;
        let int = |value: i32| ScalarValue::Int32(Some(value));

        // The first entry is the bound's nearest value and needs one page.
        for (lower, upper, descending, first) in [
            (Included(500), Unbounded, false, 500),
            (Unbounded, Excluded(500), true, 499),
            (Excluded(127), Unbounded, false, 128),
        ] {
            let metrics = LocalMetricsCollector::default();
            let range = int_range(lower, upper);
            let mut entries = index.ordered_entries(range, descending, &metrics).unwrap();
            let entry = entries.try_next().await.unwrap().unwrap();
            assert_eq!(entry.value, int(first));
            assert_eq!(metrics.parts_loaded.load(Relaxed), 1);
        }

        // Open bounds on page edges prune both neighbouring pages.
        let range = int_range(Excluded(127), Excluded(192));
        let metrics = LocalMetricsCollector::default();
        let mut expected: Vec<_> = (128..192)
            .map(|value| (int(value), vec![100 + value as u64]))
            .collect();
        assert_eq!(
            collect(index.as_ref(), range.clone(), false, &metrics).await,
            expected
        );
        assert_eq!(metrics.parts_loaded.load(Relaxed), 1);
        expected.reverse();
        assert_eq!(
            collect(index.as_ref(), range, true, &NoOpMetricsCollector).await,
            expected
        );

        // A range past every value loads nothing; an inverted one loads at
        // most the page spanning both bounds, and yields nothing either way.
        for (lower, upper, pages) in [
            (Included(1000), Unbounded, 0),
            (Unbounded, Excluded(0), 0),
            (Included(10), Excluded(5), 1),
        ] {
            for descending in [false, true] {
                let metrics = LocalMetricsCollector::default();
                let range = int_range(lower, upper);
                let entries = collect(index.as_ref(), range, descending, &metrics).await;
                assert!(entries.is_empty());
                assert_eq!(metrics.parts_loaded.load(Relaxed), pages);
            }
        }
    }

    #[tokio::test]
    async fn test_range_within_straddle_page_matches_search() {
        // Pages of 3: [N, -0.0, 0.0] (a straddle page) and [1.0, NaN]. Floats
        // compare in totalOrder: -0.0 < 0.0, and NaN sorts after every number.
        let values = [None, Some(-0.0), Some(0.0), Some(1.0), Some(f64::NAN)];
        let values: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
        let index = train(values, 3, &LanceCache::no_cache()).await;
        let float = |value: f64| ScalarValue::Float64(Some(value));

        let nan = f64::NAN;
        for (lower, upper, expected) in [
            (Excluded(-0.0), Excluded(nan), vec![(0.0, 102), (1.0, 103)]),
            (Unbounded, Included(-0.0), vec![(-0.0, 101)]),
            (Included(nan), Included(nan), vec![(nan, 104)]),
        ] {
            let (lower, upper) = (lower.map(float), upper.map(float));
            let range = (lower.clone(), upper.clone());
            let mut expected: Vec<_> = expected
                .into_iter()
                .map(|(value, row_id)| (float(value), vec![row_id]))
                .collect();
            assert_eq!(
                collect(index.as_ref(), range.clone(), false, &NoOpMetricsCollector).await,
                expected
            );
            // The walk selects exactly the rows a range search selects.
            let query = SargableQuery::Range(lower, upper);
            let result = index.search(&query, &NoOpMetricsCollector).await;
            let Ok(SearchResult::Exact(rows)) = result else {
                panic!("a btree range search must be exact: {result:?}");
            };
            let walked: Vec<u64> = expected.iter().flat_map(|(_, ids)| ids.clone()).collect();
            assert_eq!(row_ids(&rows.true_rows()), walked);
            expected.reverse();
            assert_eq!(
                collect(index.as_ref(), range, true, &NoOpMetricsCollector).await,
                expected
            );
        }
    }

    /// The clusters `plan()` builds over `lookup`, each as sorted lookup rows.
    fn clusters(lookup: &RecordBatch, range: &ValueRange) -> Vec<Vec<u32>> {
        let built = plan(lookup, range).unwrap();
        built
            .starts
            .windows(2)
            .map(|bounds| {
                let mut cluster = built.order[bounds[0]..bounds[1]].to_vec();
                cluster.sort_unstable();
                cluster
            })
            .collect()
    }

    #[test]
    fn test_plan_clusters_overlapping_and_touching_pages() {
        // Page 0 straddles (null min), page 1 is entirely null, pages 3 and 4
        // overlap (9..12 and 9..15), page 2 merely touches page 3 at 9.
        let lookup = record_batch!(
            ("min", Int32, [None, None, Some(7), Some(9), Some(9)]),
            ("max", Int32, [Some(5), None, Some(9), Some(12), Some(15)]),
            ("null_count", UInt32, [2, 3, 0, 0, 0]),
            ("page_idx", UInt32, [0, 1, 2, 3, 4])
        )
        .unwrap();
        assert_eq!(clusters(&lookup, &ALL), vec![vec![0], vec![2], vec![3, 4]]);
        // Pages wholly outside a range are pruned; a straddle page is only
        // bounded by its `max`.
        let range = int_range(Included(10), Included(12));
        assert_eq!(clusters(&lookup, &range), vec![vec![3, 4]]);
        let range = int_range(Unbounded, Excluded(7));
        assert_eq!(clusters(&lookup, &range), vec![vec![0]]);

        // A straddle page only bounds its values from above, so a page whose
        // `min` lies below that bound is sorted together with it.
        let lookup = record_batch!(
            ("min", Int32, [None, Some(3), Some(10)]),
            ("max", Int32, [Some(5), Some(8), Some(12)]),
            ("null_count", UInt32, [1, 0, 0]),
            ("page_idx", UInt32, [0, 1, 2])
        )
        .unwrap();
        assert_eq!(clusters(&lookup, &ALL), vec![vec![0, 1], vec![2]]);
    }

    #[tokio::test]
    async fn test_range_partitioned_overlapping_pages() {
        // Two legacy range partitions whose pages interleave: partition 0 holds
        // the odd values, partition 1 the even ones, each with a NULL in front.
        // The merged lookup thus has two straddle pages and every other page
        // range crosses one from the other partition. The walk sorts the lookup
        // itself, so it handles this layout; `search` assumes it sorted. Row id = value.
        let store = memory_store();
        for (range_id, null_row_id, first) in [(0u32, 200u64, 1i32), (1, 201, 2)] {
            let values: Vec<i32> = (first..=32).step_by(2).collect();
            let row_ids = std::iter::once(null_row_id)
                .chain(values.iter().map(|&value| value as u64))
                .collect();
            train_btree_index(
                training_stream(int32(1, &values), row_ids),
                store.as_ref(),
                4,
                None,
                Some(range_id),
            )
            .await
            .unwrap();
        }
        let partitions = [0u64, 1 << 32];
        merge_metadata_files(
            store.as_ref(),
            &partitions.map(part_page_data_file_path),
            &partitions.map(part_lookup_file_path),
            Some(1),
            noop_progress(),
        )
        .await
        .unwrap();
        let index = BTreeIndex::load(store, None, &LanceCache::no_cache())
            .await
            .unwrap();

        let expected: Vec<_> = (1..=32)
            .map(|value| (ScalarValue::Int32(Some(value)), vec![value as u64]))
            .collect();
        assert_eq!(
            collect(index.as_ref(), ALL, false, &NoOpMetricsCollector).await,
            expected
        );
        let mut reversed = expected;
        reversed.reverse();
        assert_eq!(
            collect(index.as_ref(), ALL, true, &NoOpMetricsCollector).await,
            reversed
        );

        // A range cuts into both partitions' pages.
        let range = int_range(Included(10), Included(20));
        let mut within = reversed[12..23].to_vec();
        assert_eq!(
            collect(index.as_ref(), range.clone(), true, &NoOpMetricsCollector).await,
            within
        );
        within.reverse();
        assert_eq!(
            collect(index.as_ref(), range, false, &NoOpMetricsCollector).await,
            within
        );

        let result = index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(null_rows(result), vec![200, 201]);
    }
}
