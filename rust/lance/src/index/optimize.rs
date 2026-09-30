// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index optimization split into plan, execute and commit steps, so that the
//! work of merging index segments can be distributed the same way compaction
//! is (see [`crate::dataset::optimize`]).
//!
//! A [`plan`](plan_index_optimization) inspects one version of the table and
//! produces independent [`IndexOptimizeTask`]s. Each task names, for one
//! logical index, the candidate segments it may replace and the unindexed
//! fragments it indexes; executing it writes one new segment and reports which
//! candidates that segment replaces. The results of every task of a plan are
//! then committed together by [`commit_index_optimization`].
//!
//! A task whose new data can be built in parallel (`shardable`) can be split
//! with [`IndexOptimizeTask::shard`]: each shard indexes a subset of the
//! fragments on its own, and [`IndexOptimizeTask::merge`] then merges the
//! shard outputs with the candidate segments instead of scanning the
//! fragments again. A shard's result is itself a valid task result, so it may
//! also be committed as is.
//!
//! Two planners are provided. [`DeltaMergePlanner`] mirrors what
//! `optimize_indices` does in a single process: one task per index that
//! merges the most recent `num_indices_to_merge` segments with the new data.
//! [`SizeTieredPlanner`] packs segments below a row budget, together with the
//! new data, into bins of at most that many rows; each bin is one task, so an
//! index's merge work runs as several tasks in parallel and large segments
//! are never rewritten.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, noop_progress};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::append::{
    NewIndexData, is_definition_only_segment, merge_indices_impl, metadata_is_vector_index,
};
use super::scalar::fetch_index_details;
use super::vector::ivf::{IVFIndex as LegacyIvfIndex, vector_model_mismatch};
use super::{DatasetIndexInternalExt, eligible_index_groups, load_all_indices};
use crate::Dataset;
use crate::dataset::optimize::{FragmentMetrics, collect_metrics};
use crate::dataset::transaction::{Operation, TransactionBuilder};
use crate::io::commit::detect_overlapping_fragments;

/// The row budget [`IndexOptimizeStrategy::SizeTiered`] uses when none is given.
pub const DEFAULT_MAX_ROWS_PER_SEGMENT: u64 = 1_000_000_000;

/// Options for planning an index optimization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizePlanOptions {
    /// The index names to plan for. `None` plans for every index.
    pub index_names: Option<Vec<String>>,
    /// How segments are chosen for merging. The two strategies are mutually
    /// exclusive; bindings that expose them as two optional parameters should
    /// refuse both and fall back to the default when neither is given.
    pub strategy: IndexOptimizeStrategy,
}

/// How a plan chooses which segments to merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexOptimizeStrategy {
    /// Merge the most recent delta segments with the new data: exactly what a
    /// single-process `optimize_indices` does, one task per index. The fields
    /// mean the same as in [`OptimizeOptions`].
    DeltaMerge {
        num_indices_to_merge: Option<usize>,
        retrain: bool,
    },
    /// Size-tiered: segments holding fewer than `max_rows_per_segment` rows
    /// and the unindexed fragments are packed, in order, into bins of at most
    /// `max_rows_per_segment` rows; every bin is one task and every segment
    /// at or above the budget is left alone.
    SizeTiered { max_rows_per_segment: u64 },
}

impl Default for IndexOptimizeStrategy {
    fn default() -> Self {
        Self::SizeTiered {
            max_rows_per_segment: DEFAULT_MAX_ROWS_PER_SEGMENT,
        }
    }
}

/// Produces an [`IndexOptimizePlan`] for one version of a dataset. The two
/// built-in planners are [`DeltaMergePlanner`] and [`SizeTieredPlanner`]; an
/// engine may implement its own grouping.
#[async_trait]
pub trait IndexOptimizePlanner: Send + Sync {
    async fn plan(&self, dataset: &Dataset) -> Result<IndexOptimizePlan>;
}

/// Plan an index optimization with the planner `options.strategy` selects.
pub async fn plan_index_optimization(
    dataset: &Dataset,
    options: &IndexOptimizePlanOptions,
) -> Result<IndexOptimizePlan> {
    match &options.strategy {
        IndexOptimizeStrategy::DeltaMerge {
            num_indices_to_merge,
            retrain,
        } => {
            DeltaMergePlanner::new(options.index_names.clone(), *num_indices_to_merge, *retrain)
                .plan(dataset)
                .await
        }
        IndexOptimizeStrategy::SizeTiered {
            max_rows_per_segment,
        } => {
            SizeTieredPlanner::new(options.index_names.clone(), *max_rows_per_segment)?
                .plan(dataset)
                .await
        }
    }
}

/// The tasks of one optimization, all planned at `read_version`. Tasks are
/// independent of each other and their results are committed together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizePlan {
    pub read_version: u64,
    pub tasks: Vec<IndexOptimizeTask>,
}

/// An unindexed fragment and its live row count, as counted when the plan was
/// made. The count is a sizing hint for splitting a task into shards; executing
/// the task does not read it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragmentRows {
    pub id: u32,
    pub num_rows: u64,
}

/// One unit of index optimization work: produce one new segment of
/// `index_name` from some of `segments` and all of `fragments`, and report
/// which of `segments` that new segment replaces.
///
/// The same type serves as a shard (see [`Self::shard`]): a task whose only
/// segment provides the model and whose `num_indices_to_merge` is `Some(0)`,
/// so that it replaces nothing and covers exactly its fragments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizeTask {
    /// The dataset version the task was planned at; it is executed at that
    /// version.
    pub read_version: u64,
    pub index_name: String,
    /// The candidate segments, in manifest order. The task may replace some of
    /// them: the trailing `num_indices_to_merge` (or as many as the executing
    /// code decides for `None`). With `Some(0)` none is replaced and the last
    /// one only provides the model for the new data.
    pub segments: Vec<Uuid>,
    /// The unindexed fragments this task indexes, by ascending id.
    pub fragments: Vec<FragmentRows>,
    /// Passed through to the merge; same meaning as in [`OptimizeOptions`].
    pub num_indices_to_merge: Option<usize>,
    /// Passed through to the merge; same meaning as in [`OptimizeOptions`].
    pub retrain: bool,
    /// Whether the new data may be built in parallel with [`Self::shard`].
    /// Decided by the planner from the index type and state; false for a
    /// shard itself.
    pub shardable: bool,
}

/// The outcome of executing an [`IndexOptimizeTask`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizeResult {
    pub read_version: u64,
    pub index_name: String,
    /// The segment to commit, or `None` when the task had nothing to do.
    #[serde(with = "segment_serde")]
    pub new_segment: Option<IndexMetadata>,
    /// The task's candidate segments that `new_segment` replaces.
    pub removed_segments: Vec<Uuid>,
}

impl IndexOptimizeTask {
    /// Execute the task. Equivalent to [`Self::execute_with_progress`] with no
    /// progress reporting.
    ///
    /// `dataset` is checked out at the task's `read_version` if it is at
    /// another version.
    pub async fn execute(&self, dataset: &Dataset) -> Result<IndexOptimizeResult> {
        self.execute_with_progress(dataset, noop_progress()).await
    }

    /// Execute the task, reporting index build progress to `progress`.
    pub async fn execute_with_progress(
        &self,
        dataset: &Dataset,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<IndexOptimizeResult> {
        let dataset = self.checkout(dataset).await?;
        let segments = self.resolve_segments(&dataset).await?;
        let fragments = self.resolve_fragments(&dataset)?;
        self.run(
            dataset,
            segments,
            NewIndexData::Fragments(&fragments),
            progress,
        )
        .await
    }

    /// A shard of this task over a subset of its fragments: a task that indexes
    /// just those fragments with the model of this task's last segment, and
    /// replaces nothing. Executing every shard of a partition of the fragments
    /// and passing the results to [`Self::merge`] is equivalent to executing
    /// this task.
    ///
    /// `fragment_ids` must be non-empty, without duplicates, and part of this
    /// task's fragments. How the fragments are partitioned is up to the
    /// caller; [`FragmentRows::num_rows`] is there to balance the shards. Shard
    /// results may also be committed without a merge, as delta segments; in
    /// that case, prefer shards over runs of consecutive fragment ids, since
    /// compaction only rewrites neighbouring fragments that the same segments
    /// cover.
    pub fn shard(&self, fragment_ids: &[u32]) -> Result<Self> {
        if !self.shardable {
            return Err(Error::invalid_input(format!(
                "index optimize task for '{}' cannot be sharded",
                self.index_name
            )));
        }
        if fragment_ids.is_empty() {
            return Err(Error::invalid_input(
                "an index optimize shard needs at least one fragment",
            ));
        }
        let mut wanted = HashSet::with_capacity(fragment_ids.len());
        for id in fragment_ids {
            if !wanted.insert(*id) {
                return Err(Error::invalid_input(format!(
                    "fragment {id} is listed twice for the index optimize shard"
                )));
            }
        }
        let fragments: Vec<FragmentRows> = self
            .fragments
            .iter()
            .filter(|fragment| wanted.contains(&fragment.id))
            .cloned()
            .collect();
        if fragments.len() != wanted.len() {
            let known: HashSet<u32> = self.fragments.iter().map(|f| f.id).collect();
            let unknown: Vec<u32> = wanted.difference(&known).copied().collect();
            return Err(Error::invalid_input(format!(
                "fragments {unknown:?} are not part of the index optimize task for '{}'",
                self.index_name
            )));
        }
        let reference = *self.segments.last().ok_or_else(|| {
            Error::invalid_input(format!(
                "index optimize task for '{}' has no segment to take a model from",
                self.index_name
            ))
        })?;
        Ok(Self {
            read_version: self.read_version,
            index_name: self.index_name.clone(),
            segments: vec![reference],
            fragments,
            num_indices_to_merge: Some(0),
            retrain: false,
            shardable: false,
        })
    }

    /// Execute this task with its new data taken from `shard_results`, the
    /// results of executing shards of this task, instead of scanning the
    /// fragments.
    ///
    /// Every shard result must belong to this task (same index name and read
    /// version), have replaced nothing, and the shard segments together must
    /// cover exactly this task's fragments, without overlap.
    pub async fn merge(
        &self,
        dataset: &Dataset,
        shard_results: Vec<IndexOptimizeResult>,
    ) -> Result<IndexOptimizeResult> {
        let dataset = self.checkout(dataset).await?;
        let segments = self.resolve_segments(&dataset).await?;
        let reference = segments.last().ok_or_else(|| {
            Error::invalid_input(format!(
                "index optimize task for '{}' has no segment",
                self.index_name
            ))
        })?;

        let expected: RoaringBitmap = self.fragments.iter().map(|f| f.id).collect();
        let mut covered = RoaringBitmap::new();
        let mut new_segments = Vec::with_capacity(shard_results.len());
        for result in shard_results {
            if result.index_name != self.index_name || result.read_version != self.read_version {
                return Err(Error::invalid_input(format!(
                    "shard result for '{}' at version {} does not belong to the task for '{}' \
                     at version {}",
                    result.index_name, result.read_version, self.index_name, self.read_version
                )));
            }
            if !result.removed_segments.is_empty() {
                return Err(Error::invalid_input(format!(
                    "shard result for '{}' replaced segments {:?}; a shard must replace nothing",
                    self.index_name, result.removed_segments
                )));
            }
            let Some(segment) = result.new_segment else {
                return Err(Error::invalid_input(format!(
                    "shard result for '{}' carries no segment",
                    self.index_name
                )));
            };
            if segment.name != self.index_name || segment.fields != reference.fields {
                return Err(Error::invalid_input(format!(
                    "shard segment {} is '{}' on fields {:?}, expected '{}' on {:?}",
                    segment.uuid, segment.name, segment.fields, self.index_name, reference.fields
                )));
            }
            let bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                Error::invalid_input(format!(
                    "shard segment {} has no fragment coverage",
                    segment.uuid
                ))
            })?;
            if !covered.is_disjoint(bitmap) {
                return Err(Error::invalid_input(format!(
                    "shard segment {} covers fragments {:?} another shard already covers",
                    segment.uuid,
                    covered.clone() & bitmap
                )));
            }
            if !bitmap.is_subset(&expected) {
                return Err(Error::invalid_input(format!(
                    "shard segment {} covers fragments {:?} outside the task's {:?}",
                    segment.uuid,
                    bitmap.clone() - &expected,
                    expected
                )));
            }
            covered |= bitmap;
            new_segments.push(segment);
        }
        if covered != expected {
            return Err(Error::invalid_input(format!(
                "shard results cover fragments {covered:?}, the task's are {expected:?}"
            )));
        }

        self.run(
            dataset,
            segments,
            NewIndexData::Segments(&new_segments),
            noop_progress(),
        )
        .await
    }

    async fn checkout(&self, dataset: &Dataset) -> Result<Dataset> {
        if dataset.manifest.version == self.read_version {
            Ok(dataset.clone())
        } else {
            dataset.checkout_version(self.read_version).await
        }
    }

    /// The task's candidate segments as recorded at its version, in manifest
    /// order. The order in the task is not trusted: the merge picks the
    /// trailing segments and the model of the last one by manifest order.
    async fn resolve_segments(&self, dataset: &Dataset) -> Result<Vec<IndexMetadata>> {
        let stored = load_all_indices(dataset).await?;
        let wanted: HashSet<Uuid> = self.segments.iter().copied().collect();
        if wanted.len() != self.segments.len() {
            return Err(Error::invalid_input(format!(
                "index optimize task for '{}' lists a segment twice",
                self.index_name
            )));
        }
        let segments: Vec<IndexMetadata> = stored
            .iter()
            .filter(|segment| wanted.contains(&segment.uuid))
            .cloned()
            .collect();
        if segments.len() != wanted.len() {
            let found: HashSet<Uuid> = segments.iter().map(|s| s.uuid).collect();
            let missing: Vec<Uuid> = wanted.difference(&found).copied().collect();
            return Err(Error::invalid_input(format!(
                "segments {missing:?} do not exist at version {}",
                dataset.manifest.version
            )));
        }
        if let Some(other) = segments.iter().find(|s| s.name != self.index_name) {
            return Err(Error::invalid_input(format!(
                "segment {} belongs to index '{}', not '{}'",
                other.uuid, other.name, self.index_name
            )));
        }
        Ok(segments)
    }

    /// The task's fragments as recorded at its version, by ascending id.
    fn resolve_fragments(&self, dataset: &Dataset) -> Result<Vec<Fragment>> {
        let mut ids: Vec<u32> = self.fragments.iter().map(|f| f.id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.iter()
            .map(|id| {
                dataset
                    .get_fragment(*id as usize)
                    .map(|fragment| fragment.metadata().clone())
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "fragment {id} does not exist at version {}",
                            dataset.manifest.version
                        ))
                    })
            })
            .collect()
    }

    async fn run(
        &self,
        dataset: Dataset,
        segments: Vec<IndexMetadata>,
        new_data: NewIndexData<'_>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<IndexOptimizeResult> {
        let mut options = OptimizeOptions::default();
        options.num_indices_to_merge = self.num_indices_to_merge;
        options.retrain = self.retrain;
        options.progress = progress;
        let refs: Vec<&IndexMetadata> = segments.iter().collect();
        let Some(merged) = merge_indices_impl(Arc::new(dataset), &refs, new_data, &options).await?
        else {
            return Ok(IndexOptimizeResult {
                read_version: self.read_version,
                index_name: self.index_name.clone(),
                new_segment: None,
                removed_segments: Vec::new(),
            });
        };

        // The merge may only replace what the task offered it.
        let offered: HashSet<Uuid> = segments.iter().map(|s| s.uuid).collect();
        if let Some(stray) = merged
            .removed_indices
            .iter()
            .find(|removed| !offered.contains(&removed.uuid))
        {
            return Err(Error::internal(format!(
                "index optimize task for '{}' replaced segment {}, which it was not given",
                self.index_name, stray.uuid
            )));
        }

        let last = segments
            .last()
            .expect("merge_indices_impl refuses an empty segment list");
        let new_segment = IndexMetadata {
            uuid: merged.new_uuid,
            name: last.name.clone(),
            fields: last.fields.clone(),
            covering_fields: last.covering_fields.clone(),
            dataset_version: merged.new_dataset_version,
            fragment_bitmap: Some(merged.new_fragment_bitmap),
            index_details: Some(Arc::new(merged.new_index_details)),
            index_version: merged.new_index_version,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(merged.files),
        };
        Ok(IndexOptimizeResult {
            read_version: self.read_version,
            index_name: self.index_name.clone(),
            new_segment: Some(new_segment),
            removed_segments: merged.removed_indices.iter().map(|s| s.uuid).collect(),
        })
    }
}

/// Commit the results of one plan's tasks as a single `CreateIndex`
/// transaction.
///
/// All results must carry the same `read_version` (be from one plan), and are
/// validated against the manifest at that version: every replaced segment must
/// exist there under the result's index name, no segment may be replaced by
/// two results, and the segments left after the replacement must not overlap
/// in coverage. What changed on the table since the plan is not judged here:
/// the transaction is anchored at the plan's version, so the commit's conflict
/// resolution sees every later transaction and reports a retryable conflict
/// when one of them replaced the same segments or rewrote the covered
/// fragments (see the compaction commit for the same reasoning), in which
/// case the caller plans again.
///
/// Results without a segment are skipped. When no result has one the commit
/// is skipped too, unless the table's MemWAL catch-up position would advance,
/// which needs an (empty) commit to be recorded.
pub async fn commit_index_optimization(
    dataset: &mut Dataset,
    results: Vec<IndexOptimizeResult>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
) -> Result<()> {
    let read_versions: HashSet<u64> = results.iter().map(|r| r.read_version).collect();
    if read_versions.len() > 1 {
        return Err(Error::invalid_input(format!(
            "index optimize results were planned at different versions {:?}; \
             commit the results of one plan at a time",
            read_versions
        )));
    }
    let read_version = read_versions
        .into_iter()
        .next()
        .unwrap_or(dataset.manifest.version);

    let produced: Vec<&IndexOptimizeResult> = results
        .iter()
        .filter(|result| result.new_segment.is_some())
        .collect();
    if produced.is_empty() {
        let indices = load_all_indices(dataset).await?;
        if !dataset.mem_wal_catch_up_would_advance(&indices)? {
            return Ok(());
        }
    }

    // Validate against the version the results were built from: that is the
    // coordinate system their coverage and replacements are expressed in.
    let snapshot = if dataset.manifest.version == read_version {
        dataset.clone()
    } else {
        dataset.checkout_version(read_version).await?
    };
    let stored = load_all_indices(&snapshot).await?;
    let by_uuid: HashMap<Uuid, &IndexMetadata> = stored.iter().map(|s| (s.uuid, s)).collect();
    let names: HashSet<&str> = produced.iter().map(|r| r.index_name.as_str()).collect();
    let mut removed_indices = Vec::new();
    let mut removed_uuids = HashSet::new();
    for result in &produced {
        for uuid in &result.removed_segments {
            let segment = by_uuid
                .get(uuid)
                .filter(|segment| segment.name == result.index_name)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "segment {uuid} is not a segment of '{}' at version {read_version}",
                        result.index_name
                    ))
                })?;
            if !removed_uuids.insert(*uuid) {
                return Err(Error::invalid_input(format!(
                    "segment {uuid} of '{}' is replaced by two results",
                    result.index_name
                )));
            }
            removed_indices.push((*segment).clone());
        }
    }
    let new_indices: Vec<IndexMetadata> = produced
        .iter()
        .map(|result| result.new_segment.clone().expect("filtered to Some"))
        .collect();
    let mut projected: Vec<IndexMetadata> = stored
        .iter()
        .filter(|segment| {
            names.contains(segment.name.as_str()) && !removed_uuids.contains(&segment.uuid)
        })
        .cloned()
        .collect();
    projected.extend(new_indices.iter().cloned());
    if let Err(overlap) = detect_overlapping_fragments(&projected) {
        return Err(Error::invalid_input(format!(
            "index optimize results would leave overlapping coverage at version {read_version}: {:?}",
            overlap.bad_indices
        )));
    }

    let transaction = TransactionBuilder::new(
        read_version,
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        },
    )
    .transaction_properties(transaction_properties)
    .build();
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
}

/// Serde adapter for an [`IndexMetadata`](lance_table::format::IndexMetadata)
/// carried inside a task result.
///
/// `IndexMetadata` has no serde implementation; its protobuf form is the
/// serialization the manifest already uses, so a result carries the protobuf
/// bytes as a hex string.
mod segment_serde {
    use lance_table::format::{IndexMetadata, pb};
    use prost::Message;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        segment: &Option<IndexMetadata>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match segment {
            Some(segment) => {
                let bytes = pb::IndexMetadata::from(segment).encode_to_vec();
                serializer.serialize_some(&hex::encode(bytes))
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<IndexMetadata>, D::Error> {
        let encoded: Option<String> = Option::deserialize(deserializer)?;
        encoded
            .map(|encoded| {
                let bytes = hex::decode(encoded).map_err(serde::de::Error::custom)?;
                let proto = pb::IndexMetadata::decode(bytes.as_slice())
                    .map_err(serde::de::Error::custom)?;
                IndexMetadata::try_from(proto).map_err(serde::de::Error::custom)
            })
            .transpose()
    }
}

// ---------------------------------------------------------------------------
// Planners
// ---------------------------------------------------------------------------

/// What the planners need to know about an index family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexKind {
    /// `legacy` is the v1 IVF format, which cannot merge new data segments.
    Vector {
        legacy: bool,
    },
    BTree,
    NGram,
    /// `legacy` inverted segments are rebuilt from the table on every merge.
    Inverted {
        legacy: bool,
    },
    /// A scalar family that reads one old segment and scans the new rows.
    Other,
}

/// One logical index the planners may act on, with everything both need.
struct IndexGroup {
    name: String,
    /// Every segment of the name, in manifest order.
    segments: Vec<IndexMetadata>,
    /// The unindexed fragments with their live row counts, by ascending id.
    unindexed: Vec<FragmentRows>,
    kind: IndexKind,
    /// A segment with stored rows but no live coverage; a merge replaces it
    /// along with whatever it produces.
    has_dormant: bool,
    /// A segment that is only a definition (no files, no coverage); a merge
    /// trains from it.
    has_definition_only: bool,
}

impl IndexGroup {
    fn is_vector(&self) -> bool {
        matches!(self.kind, IndexKind::Vector { .. })
    }

    /// Whether the new data can be built as shards and merged from segments.
    fn shardable(&self, no_uncommitted_segments: bool) -> bool {
        if no_uncommitted_segments {
            return false;
        }
        match self.kind {
            IndexKind::Vector { legacy } => {
                !legacy && !self.has_dormant && !self.has_definition_only
            }
            IndexKind::BTree | IndexKind::NGram => true,
            IndexKind::Inverted { legacy } => !legacy,
            IndexKind::Other => false,
        }
    }

    /// Whether the merge would rebuild the whole index (or remove segments
    /// beyond the ones it merges), so that the group must go into one task.
    fn needs_whole_task(&self) -> bool {
        self.has_dormant
            || self.has_definition_only
            || matches!(
                self.kind,
                IndexKind::Vector { legacy: true } | IndexKind::Inverted { legacy: true }
            )
    }
}

/// Live row counts and physical row counts of every fragment.
async fn fragment_metrics(dataset: &Dataset) -> Result<HashMap<u32, FragmentMetrics>> {
    futures::stream::iter(dataset.get_fragments())
        .map(|fragment| async move {
            let metrics = collect_metrics(&fragment).await?;
            Ok::<_, Error>((fragment.id() as u32, metrics))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await
}

/// Whether the table's fragment reuse history is the tagged (v1) format,
/// under which a segment the manifest does not list cannot be opened, so a
/// shard's output could not be merged. Decided from the manifest entry's
/// version, not from what the reader would do with it.
async fn has_tagged_fragment_reuse_history(dataset: &Dataset) -> Result<bool> {
    Ok(load_all_indices(dataset)
        .await?
        .iter()
        .any(|index| index.name == FRAG_REUSE_INDEX_NAME && index.index_version != 0))
}

async fn index_kind(dataset: &Dataset, segments: &[IndexMetadata]) -> Result<IndexKind> {
    let last = segments.last().expect("a group has at least one segment");
    let field_path = dataset.schema().field_path(last.fields[0])?;
    if metadata_is_vector_index(dataset, last).await? {
        let legacy = if is_definition_only_segment(last) {
            false
        } else {
            dataset
                .open_vector_index_from_metadata(&field_path, last, &NoOpMetricsCollector)
                .await?
                .as_any()
                .is::<LegacyIvfIndex>()
        };
        return Ok(IndexKind::Vector { legacy });
    }
    let details = fetch_index_details(dataset, &field_path, last).await?;
    let type_url = details.type_url.as_str();
    if type_url.ends_with("BTreeIndexDetails") {
        Ok(IndexKind::BTree)
    } else if type_url.ends_with("NGramIndexDetails") {
        Ok(IndexKind::NGram)
    } else if type_url.ends_with("InvertedIndexDetails") {
        let details = lance_index::pbold::InvertedIndexDetails::decode(details.value.as_slice())
            .map_err(|error| {
                Error::io(format!(
                    "failed to decode InvertedIndexDetails payload: {error}"
                ))
            })?;
        let granularity = lance_index::scalar::inverted::DocumentGranularity::try_from(
            details.document_granularity,
        )?;
        let resolved = super::scalar::inverted::resolve_fts_field_by_id(
            dataset.schema(),
            last.fields[0],
            granularity,
        )?;
        let index = super::scalar::open_scalar_index(
            dataset,
            &resolved.canonical_path,
            last,
            &NoOpMetricsCollector,
        )
        .await?;
        Ok(IndexKind::Inverted {
            legacy: index.update_criteria().requires_old_data,
        })
    } else {
        Ok(IndexKind::Other)
    }
}

/// The groups both planners start from, with their unindexed fragments sized.
async fn index_groups(
    dataset: &Dataset,
    index_names: Option<&[String]>,
    metrics: &HashMap<u32, FragmentMetrics>,
) -> Result<Vec<IndexGroup>> {
    let mut groups = Vec::new();
    for (name, segments) in eligible_index_groups(dataset, index_names).await? {
        let mut unindexed: Vec<FragmentRows> = dataset
            .unindexed_fragments(&name)
            .await?
            .iter()
            .map(|fragment| {
                let id = fragment.id as u32;
                let num_rows = metrics
                    .get(&id)
                    .map(|metrics| metrics.num_rows() as u64)
                    .ok_or_else(|| {
                        Error::internal(format!("no metrics for unindexed fragment {id}"))
                    })?;
                Ok(FragmentRows { id, num_rows })
            })
            .collect::<Result<_>>()?;
        unindexed.sort_by_key(|fragment| fragment.id);
        let kind = index_kind(dataset, &segments).await?;
        let has_dormant = segments.iter().any(|segment| {
            segment
                .fragment_bitmap
                .as_ref()
                .is_some_and(|bitmap| !bitmap.is_empty())
                && segment
                    .effective_fragment_bitmap(&dataset.fragment_bitmap)
                    .is_some_and(|bitmap| bitmap.is_empty())
        });
        let has_definition_only = segments.iter().any(is_definition_only_segment);
        groups.push(IndexGroup {
            name,
            segments,
            unindexed,
            kind,
            has_dormant,
            has_definition_only,
        });
    }
    Ok(groups)
}

/// The planner that reproduces a single-process `optimize_indices`: one task
/// per index with every segment as a candidate, so the merge picks the
/// trailing `num_indices_to_merge` exactly as it does today.
#[derive(Debug, Clone)]
pub struct DeltaMergePlanner {
    index_names: Option<Vec<String>>,
    num_indices_to_merge: Option<usize>,
    retrain: bool,
}

impl DeltaMergePlanner {
    pub fn new(
        index_names: Option<Vec<String>>,
        num_indices_to_merge: Option<usize>,
        retrain: bool,
    ) -> Self {
        Self {
            index_names,
            num_indices_to_merge,
            retrain,
        }
    }
}

#[async_trait]
impl IndexOptimizePlanner for DeltaMergePlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<IndexOptimizePlan> {
        let read_version = dataset.manifest.version;
        let metrics = fragment_metrics(dataset).await?;
        let tagged = has_tagged_fragment_reuse_history(dataset).await?;
        let mut tasks = Vec::new();
        for group in index_groups(dataset, self.index_names.as_deref(), &metrics).await? {
            // Scalar indices have no rebalance concept, so a scalar group with
            // every fragment covered has nothing to do unless the caller asked
            // for a retrain or an explicit delta merge. Vector groups may still
            // rebalance a partition, which the merge decides.
            if !self.retrain
                && self.num_indices_to_merge.is_none_or(|n| n == 0)
                && !group.is_vector()
                && group.unindexed.is_empty()
            {
                continue;
            }
            let shardable = !self.retrain && !group.unindexed.is_empty() && group.shardable(tagged);
            tasks.push(IndexOptimizeTask {
                read_version,
                index_name: group.name,
                segments: group.segments.iter().map(|s| s.uuid).collect(),
                fragments: group.unindexed,
                num_indices_to_merge: self.num_indices_to_merge,
                retrain: self.retrain,
                shardable,
            });
        }
        Ok(IndexOptimizePlan {
            read_version,
            tasks,
        })
    }
}

/// The size-tiered planner: segments below `max_rows_per_segment` rows and
/// the unindexed fragments are packed, in order, into bins of at most that
/// many rows, one task per bin. Segments at or above the budget are never
/// rewritten. Vector segments are packed per shared model, since only
/// segments sharing a model can be merged physically; the new data joins the
/// model of the newest segment.
///
/// A segment's size is the physical row count of the live fragments it
/// covers (a merge drops the rows of retired fragments but, under
/// address-style row ids, copies deleted rows of live fragments); a fragment's
/// size is its live row count. Both come from the manifest and the deletion
/// files, without opening the index.
#[derive(Debug, Clone)]
pub struct SizeTieredPlanner {
    index_names: Option<Vec<String>>,
    max_rows_per_segment: u64,
}

impl SizeTieredPlanner {
    pub fn new(index_names: Option<Vec<String>>, max_rows_per_segment: u64) -> Result<Self> {
        if max_rows_per_segment == 0 {
            return Err(Error::invalid_input(
                "max_rows_per_segment must be at least 1",
            ));
        }
        Ok(Self {
            index_names,
            max_rows_per_segment,
        })
    }

    /// The row count a segment contributes to a merge, or `None` when its
    /// coverage is unknown (a segment older than fragment bitmaps).
    fn segment_rows(
        segment: &IndexMetadata,
        dataset: &Dataset,
        metrics: &HashMap<u32, FragmentMetrics>,
    ) -> Option<u64> {
        let effective = segment.effective_fragment_bitmap(&dataset.fragment_bitmap)?;
        Some(
            effective
                .iter()
                .map(|id| {
                    metrics
                        .get(&id)
                        .map(|metrics| metrics.physical_rows as u64)
                        .unwrap_or(0)
                })
                .sum(),
        )
    }

    /// The segments of a vector group by shared model, each class in manifest
    /// order, in order of first appearance. A scalar group is one class.
    async fn model_classes(
        &self,
        dataset: &Dataset,
        group: &IndexGroup,
    ) -> Result<Vec<Vec<usize>>> {
        if !group.is_vector() {
            return Ok(vec![(0..group.segments.len()).collect()]);
        }
        let field_path = dataset.schema().field_path(group.segments[0].fields[0])?;
        let logical = dataset
            .open_logical_vector_index(&field_path, &group.name)
            .await?;
        let opened: HashMap<Uuid, _> = logical
            .iter()
            .map(|(metadata, index)| (metadata.uuid, index.clone()))
            .collect();
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for (position, segment) in group.segments.iter().enumerate() {
            let index = opened.get(&segment.uuid).ok_or_else(|| {
                Error::index(format!(
                    "logical vector index '{}' does not contain segment {}",
                    group.name, segment.uuid
                ))
            })?;
            let class = classes.iter().position(|class| {
                let representative = &opened[&group.segments[class[0]].uuid];
                vector_model_mismatch(&[representative.clone(), index.clone()]).is_none()
            });
            match class {
                Some(class) => classes[class].push(position),
                None => classes.push(vec![position]),
            }
        }
        Ok(classes)
    }
}

/// One item packed by the size-tiered planner.
enum BinItem {
    Segment(usize),
    Fragment(FragmentRows),
}

#[async_trait]
impl IndexOptimizePlanner for SizeTieredPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<IndexOptimizePlan> {
        let read_version = dataset.manifest.version;
        let budget = self.max_rows_per_segment;
        let metrics = fragment_metrics(dataset).await?;
        let tagged = has_tagged_fragment_reuse_history(dataset).await?;
        let mut tasks = Vec::new();
        for group in index_groups(dataset, self.index_names.as_deref(), &metrics).await? {
            let uuids: Vec<Uuid> = group.segments.iter().map(|s| s.uuid).collect();
            let reference = *uuids.last().expect("a group has at least one segment");

            // A rebuild, or a replacement that reaches beyond the merged
            // segments, cannot be split; hand the whole group to one task
            // with the single-process default options.
            if group.needs_whole_task() {
                if group.unindexed.is_empty() && !group.is_vector() {
                    continue;
                }
                tasks.push(IndexOptimizeTask {
                    read_version,
                    index_name: group.name,
                    segments: uuids,
                    fragments: group.unindexed,
                    num_indices_to_merge: None,
                    retrain: false,
                    shardable: false,
                });
                continue;
            }

            let shardable = group.shardable(tagged);
            let classes = self.model_classes(dataset, &group).await?;
            let reference_class = classes
                .iter()
                .position(|class| class.contains(&(group.segments.len() - 1)))
                .expect("the last segment is in a class");
            for (class_index, class) in classes.iter().enumerate() {
                // Candidates are the segments below the budget, in manifest
                // order; the new data follows them in the reference class.
                let mut items: Vec<(BinItem, u64)> = class
                    .iter()
                    .filter_map(|&position| {
                        let rows =
                            Self::segment_rows(&group.segments[position], dataset, &metrics)?;
                        (rows < budget).then_some((BinItem::Segment(position), rows))
                    })
                    .collect();
                if class_index == reference_class {
                    items.extend(
                        group.unindexed.iter().map(|fragment| {
                            (BinItem::Fragment(fragment.clone()), fragment.num_rows)
                        }),
                    );
                }

                // Greedy packing in order: close the bin when the next item
                // would exceed the budget. A single item over the budget (a
                // fragment; segments over it are not candidates) gets a bin
                // of its own.
                let mut bins: Vec<Vec<BinItem>> = Vec::new();
                let mut bin: Vec<BinItem> = Vec::new();
                let mut bin_rows = 0u64;
                for (item, rows) in items {
                    if !bin.is_empty() && bin_rows.saturating_add(rows) > budget {
                        bins.push(std::mem::take(&mut bin));
                        bin_rows = 0;
                    }
                    bin.push(item);
                    bin_rows = bin_rows.saturating_add(rows);
                }
                if !bin.is_empty() {
                    bins.push(bin);
                }

                for bin in bins {
                    let mut segments = Vec::new();
                    let mut fragments = Vec::new();
                    for item in bin {
                        match item {
                            BinItem::Segment(position) => segments.push(uuids[position]),
                            BinItem::Fragment(fragment) => fragments.push(fragment),
                        }
                    }
                    // One segment and nothing to add: rewriting it to itself.
                    if segments.len() == 1 && fragments.is_empty() {
                        continue;
                    }
                    let (segments, num_indices_to_merge) = if segments.is_empty() {
                        (vec![reference], Some(0))
                    } else {
                        let count = segments.len();
                        (segments, Some(count))
                    };
                    tasks.push(IndexOptimizeTask {
                        read_version,
                        index_name: group.name.clone(),
                        segments,
                        shardable: shardable && !fragments.is_empty(),
                        fragments,
                        num_indices_to_merge,
                        retrain: false,
                    });
                }
            }
        }
        Ok(IndexOptimizePlan {
            read_version,
            tasks,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_table::format::{IndexFile, IndexMetadata};
    use roaring::RoaringBitmap;
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Carrier {
        #[serde(with = "segment_serde")]
        segment: Option<IndexMetadata>,
    }

    #[test]
    fn segment_serde_round_trips_through_protobuf() {
        let segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "vector_idx".to_string(),
            fields: vec![1],
            covering_fields: vec![],
            dataset_version: 7,
            fragment_bitmap: Some(RoaringBitmap::from_iter([1u32, 2, 3])),
            index_details: Some(Arc::new(crate::index::vector_index_details_default())),
            index_version: 3,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(vec![IndexFile {
                path: "index.idx".to_string(),
                size_bytes: 10,
            }]),
        };
        let json = serde_json::to_string(&Carrier {
            segment: Some(segment.clone()),
        })
        .unwrap();
        let decoded: Carrier = serde_json::from_str(&json).unwrap();
        let decoded = decoded.segment.unwrap();
        assert_eq!(decoded.uuid, segment.uuid);
        assert_eq!(decoded.name, segment.name);
        assert_eq!(decoded.fields, segment.fields);
        assert_eq!(decoded.dataset_version, segment.dataset_version);
        assert_eq!(decoded.fragment_bitmap, segment.fragment_bitmap);
        assert_eq!(decoded.index_details, segment.index_details);
        assert_eq!(decoded.index_version, segment.index_version);
        assert_eq!(decoded.files, segment.files);
        // Protobuf keeps millisecond precision.
        assert_eq!(
            decoded.created_at.map(|t| t.timestamp_millis()),
            segment.created_at.map(|t| t.timestamp_millis())
        );

        let json = serde_json::to_string(&Carrier { segment: None }).unwrap();
        let decoded: Carrier = serde_json::from_str(&json).unwrap();
        assert!(decoded.segment.is_none());
    }
}
