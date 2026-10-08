// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index optimization as plan, execute and commit steps, so the work can run
//! on several workers the way compaction does (see [`crate::dataset::optimize`]).
//! The single-process `optimize_indices` runs the same steps with the same
//! [`OptimizeOptions`].

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, noop_progress};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::system_index::frag_reuse::metadata::is_tagged;
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::append::{
    NewIndexData, is_definition_only_segment, merge_indices_impl, metadata_is_vector_index,
    tagged_segment_coverage,
};
use super::scalar::fetch_index_details;
use super::vector::ivf::{IVFIndex as LegacyIvfIndex, vector_model_mismatch};
use super::{DatasetIndexInternalExt, eligible_index_groups, load_all_indices};
use crate::Dataset;
use crate::dataset::fragment::FileFragment;
use crate::dataset::optimize::collect_metrics;
use crate::dataset::transaction::{Operation, TransactionBuilder};
use crate::io::commit::detect_overlapping_fragments;

/// Produces an [`IndexOptimizePlan`] for one version of a dataset.
#[async_trait]
pub(crate) trait IndexOptimizePlanner: Send + Sync {
    async fn plan(&self, dataset: &Dataset) -> Result<IndexOptimizePlan>;
}

/// Plan with the strategy `options` select: `max_rows_per_segment` packs
/// segments and new fragments size-tiered, otherwise the trailing
/// `num_indices_to_merge` segments are merged (the single-process behavior).
/// The two are mutually exclusive. A caller with its own grouping builds the
/// [`IndexOptimizePlan`] directly; its tasks are plain data.
pub async fn plan_index_optimization(
    dataset: &Dataset,
    options: &OptimizeOptions,
) -> Result<IndexOptimizePlan> {
    let index_names = options.index_names.clone();
    match options.max_rows_per_segment {
        Some(max_rows_per_segment) => {
            if options.num_indices_to_merge.is_some() || options.retrain {
                return Err(Error::invalid_input(
                    "max_rows_per_segment cannot be combined with num_indices_to_merge or retrain",
                ));
            }
            SizeTieredPlanner::new(index_names, max_rows_per_segment)?
                .plan(dataset)
                .await
        }
        None => {
            DeltaMergePlanner::new(index_names, options.num_indices_to_merge, options.retrain)
                .plan(dataset)
                .await
        }
    }
}

/// Independent tasks planned at one version; their results commit together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizePlan {
    pub read_version: u64,
    pub tasks: Vec<IndexOptimizeTask>,
}

/// An unindexed fragment and its live row count, a hint for sizing shards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragmentRows {
    pub id: u32,
    pub num_rows: u64,
}

/// Produce one new segment of `index_name` from some of `segments` and all
/// of `fragments`, and report which of `segments` it replaces. A shard (see
/// [`Self::shard`]) is the same type with `Some(0)` and one segment for the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizeTask {
    pub read_version: u64,
    pub index_name: String,
    /// Candidate segments in manifest order; the merge replaces the trailing
    /// `num_indices_to_merge` of them (`Some(0)`: none, the last lends its model).
    pub segments: Vec<Uuid>,
    /// The unindexed fragments to index, by ascending id.
    pub fragments: Vec<FragmentRows>,
    pub num_indices_to_merge: Option<usize>,
    pub retrain: bool,
    /// Whether [`Self::shard`] may split the new data; false for a shard.
    pub shardable: bool,
}

/// The outcome of executing an [`IndexOptimizeTask`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexOptimizeResult {
    pub read_version: u64,
    pub index_name: String,
    /// `None` when the task had nothing to do.
    #[serde(with = "segment_serde")]
    pub new_segment: Option<IndexMetadata>,
    pub removed_segments: Vec<Uuid>,
}

impl IndexOptimizeTask {
    /// Execute at the task's `read_version`, checking `dataset` out if needed.
    pub async fn execute(&self, dataset: &Dataset) -> Result<IndexOptimizeResult> {
        self.execute_with_progress(dataset, noop_progress()).await
    }

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

    /// A task indexing just `fragment_ids` with the last segment's model and
    /// replacing nothing. Executing every shard of a partition of the fragments
    /// and passing the results to [`Self::merge`] equals executing this task; a
    /// shard's result may also be committed as is, in which case runs of
    /// consecutive fragment ids keep the segments compaction-friendly.
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

    /// Execute with the new data read from `shard_results` instead of the
    /// fragments. The shard segments must replace nothing and together cover
    /// exactly this task's fragments.
    pub async fn merge(
        &self,
        dataset: &Dataset,
        shard_results: Vec<IndexOptimizeResult>,
    ) -> Result<IndexOptimizeResult> {
        self.merge_with_progress(dataset, shard_results, noop_progress())
            .await
    }

    pub async fn merge_with_progress(
        &self,
        dataset: &Dataset,
        shard_results: Vec<IndexOptimizeResult>,
        progress: Arc<dyn IndexBuildProgress>,
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
            progress,
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

    /// The candidate segments in manifest order (the task's order is not
    /// trusted: the merge picks the trailing ones and the last one's model).
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

/// Commit the results of one plan as a single `CreateIndex` transaction
/// anchored at the plan's version, after validating them against the manifest
/// at that version. Changes on the table since the plan are left to the
/// commit's conflict resolution (a rewrite of covered fragments or a concurrent
/// optimize is a retryable conflict). Results without a segment are skipped;
/// with none left, nothing is committed unless the MemWAL catch-up needs it.
/// New segments are appended in the order of the segments they replace, with
/// pure additions last, so the manifest's last segment stays the newest one
/// whatever order the results arrive in.
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

    let snapshot = if dataset.manifest.version == read_version {
        dataset.clone()
    } else {
        dataset.checkout_version(read_version).await?
    };
    let stored = load_all_indices(&snapshot).await?;
    let by_uuid: HashMap<Uuid, &IndexMetadata> = stored.iter().map(|s| (s.uuid, s)).collect();
    let position: HashMap<Uuid, usize> = stored
        .iter()
        .enumerate()
        .map(|(i, s)| (s.uuid, i))
        .collect();
    let mut produced = produced;
    produced.sort_by_key(|result| {
        let last_replaced = result
            .removed_segments
            .iter()
            .filter_map(|u| position.get(u))
            .max();
        last_replaced.map_or((1, 0), |p| (0, *p))
    });
    let names: HashSet<&str> = produced.iter().map(|r| r.index_name.as_str()).collect();
    let mut removed_indices = Vec::new();
    let mut removed_uuids = HashSet::new();
    for result in &produced {
        let segment = result.new_segment.as_ref().expect("filtered to Some");
        if segment.name != result.index_name {
            return Err(Error::invalid_input(format!(
                "result for '{}' carries a segment named '{}'",
                result.index_name, segment.name
            )));
        }
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

/// `IndexMetadata` has no serde; carry its protobuf bytes as a hex string.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexFamily {
    Vector,
    BTree,
    Bitmap,
    NGram,
    Inverted,
    Other,
}

struct IndexGroup {
    name: String,
    segments: Vec<IndexMetadata>,
    unindexed: Vec<FragmentRows>,
    /// Each segment's live coverage: derived through the lineage on a tagged
    /// table, stored coverage intersected with the live fragments elsewhere.
    /// Absent for a segment without stored coverage.
    live_coverage: HashMap<Uuid, RoaringBitmap>,
    family: IndexFamily,
    /// A segment with stored rows but no live coverage.
    has_dormant: bool,
    /// A segment that is only a definition: no files, no coverage.
    has_definition_only: bool,
}

impl IndexGroup {
    fn is_vector(&self) -> bool {
        self.family == IndexFamily::Vector
    }

    /// Whether the family can take its new data from shards; a legacy format
    /// (see [`Self::legacy_format`]) still rules it out.
    fn family_shardable(&self) -> bool {
        match self.family {
            IndexFamily::Vector => !self.has_dormant && !self.has_definition_only,
            IndexFamily::BTree | IndexFamily::Bitmap | IndexFamily::NGram => true,
            IndexFamily::Inverted => true,
            IndexFamily::Other => false,
        }
    }

    /// Whether the last segment is in a format that cannot merge from
    /// segments (v1 IVF, legacy inverted) and is rebuilt whole. Opens it
    /// through the maintenance entry, as the merge does, so a segment the
    /// tagged reader excludes opens as empty; `None` when it cannot be opened
    /// at all, which the merge reports and skips as well.
    async fn legacy_format(&self, dataset: &Dataset) -> Result<Option<bool>> {
        let last = self
            .segments
            .last()
            .expect("a group has at least one segment");
        let field_path = dataset.schema().field_path(last.fields[0])?;
        let opened = match self.family {
            IndexFamily::Vector if is_definition_only_segment(last) => return Ok(Some(false)),
            IndexFamily::Vector => dataset
                .open_vector_index_for_maintenance(&field_path, &last.uuid, &NoOpMetricsCollector)
                .await
                .map(|index| index.as_any().is::<LegacyIvfIndex>()),
            IndexFamily::Inverted => {
                let details = fetch_index_details(dataset, &field_path, last).await?;
                let details =
                    lance_index::pbold::InvertedIndexDetails::decode(details.value.as_slice())
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
                dataset
                    .open_scalar_index_for_maintenance(
                        &resolved.canonical_path,
                        &last.uuid,
                        &NoOpMetricsCollector,
                    )
                    .await
                    .map(|index| index.update_criteria().requires_old_data)
            }
            _ => return Ok(Some(false)),
        };
        match opened {
            Ok(legacy) => Ok(Some(legacy)),
            Err(error) => {
                log::warn!(
                    "Skipping optimization of index '{}': cannot open segment {}: {error}",
                    last.name,
                    last.uuid
                );
                Ok(None)
            }
        }
    }
}

/// The index family, from metadata alone.
async fn index_family(dataset: &Dataset, last: &IndexMetadata) -> Result<IndexFamily> {
    if metadata_is_vector_index(dataset, last).await? {
        return Ok(IndexFamily::Vector);
    }
    let field_path = dataset.schema().field_path(last.fields[0])?;
    let details = fetch_index_details(dataset, &field_path, last).await?;
    let type_url = details.type_url.as_str();
    Ok(if type_url.ends_with("BTreeIndexDetails") {
        IndexFamily::BTree
    } else if type_url.ends_with("BitmapIndexDetails") {
        IndexFamily::Bitmap
    } else if type_url.ends_with("NGramIndexDetails") {
        IndexFamily::NGram
    } else if type_url.ends_with("InvertedIndexDetails") {
        IndexFamily::Inverted
    } else {
        IndexFamily::Other
    })
}

async fn count_rows<Fut>(
    dataset: &Dataset,
    ids: HashSet<u32>,
    count: impl Fn(FileFragment) -> Fut,
) -> Result<HashMap<u32, u64>>
where
    Fut: Future<Output = Result<u64>>,
{
    let count = &count;
    futures::stream::iter(ids)
        .map(|id| async move {
            let fragment = dataset
                .get_fragment(id as usize)
                .ok_or_else(|| Error::internal(format!("fragment {id} is not in the manifest")))?;
            Ok::<_, Error>((id, count(fragment).await?))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await
}

/// Live rows (physical rows minus deletions) of `ids`; the deletion count is
/// read from the deletion file when the manifest does not carry it.
async fn live_rows(dataset: &Dataset, ids: HashSet<u32>) -> Result<HashMap<u32, u64>> {
    count_rows(dataset, ids, |fragment| async move {
        Ok(collect_metrics(&fragment).await?.num_rows() as u64)
    })
    .await
}

/// Physical rows of `ids`, from the manifest where it records them.
async fn physical_rows(dataset: &Dataset, ids: HashSet<u32>) -> Result<HashMap<u32, u64>> {
    count_rows(dataset, ids, |fragment| async move {
        Ok(fragment.physical_rows().await? as u64)
    })
    .await
}

/// A tagged (v1) fragment reuse history cannot open uncommitted segments, so
/// shard outputs could not be merged. Decided from the entry's version, never
/// from what the reader does with it.
async fn has_tagged_fragment_reuse_history(dataset: &Dataset) -> Result<bool> {
    Ok(load_all_indices(dataset).await?.iter().any(is_tagged))
}

/// The eligible groups with their unindexed fragments sized; nothing is
/// opened and only the unindexed fragments are counted.
async fn index_groups(
    dataset: &Dataset,
    index_names: Option<&[String]>,
) -> Result<Vec<IndexGroup>> {
    let mut groups = Vec::new();
    for (name, segments) in eligible_index_groups(dataset, index_names).await? {
        let mut unindexed: Vec<FragmentRows> = dataset
            .unindexed_fragments(&name)
            .await?
            .iter()
            .map(|fragment| FragmentRows {
                id: fragment.id as u32,
                num_rows: 0,
            })
            .collect();
        unindexed.sort_by_key(|fragment| fragment.id);
        let refs: Vec<&IndexMetadata> = segments.iter().collect();
        let tagged = tagged_segment_coverage(dataset, &refs, None).await?;
        let is_tagged = tagged.is_some();
        let live_coverage: HashMap<Uuid, RoaringBitmap> = match tagged {
            Some(derived) => derived,
            None => segments
                .iter()
                .filter_map(|s| {
                    Some((
                        s.uuid,
                        s.effective_fragment_bitmap(&dataset.fragment_bitmap)?,
                    ))
                })
                .collect(),
        };
        // Stored rows but no live coverage: the merge replaces such a segment
        // (and rebuilds when none is live), as it does in a single process.
        let dormant = |segment: &IndexMetadata| {
            let has_stored_rows = if is_tagged {
                !is_definition_only_segment(segment)
            } else {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|b| !b.is_empty())
            };
            has_stored_rows
                && live_coverage
                    .get(&segment.uuid)
                    .is_some_and(|b| b.is_empty())
        };
        let last = segments.last().expect("a group has at least one segment");
        let family = index_family(dataset, last).await?;
        let has_dormant = segments.iter().any(dormant);
        let has_definition_only = segments.iter().any(is_definition_only_segment);
        groups.push(IndexGroup {
            name,
            segments,
            unindexed,
            live_coverage,
            family,
            has_dormant,
            has_definition_only,
        });
    }
    let ids: HashSet<u32> = groups
        .iter()
        .flat_map(|group| group.unindexed.iter().map(|f| f.id))
        .collect();
    let rows = live_rows(dataset, ids).await?;
    for fragment in groups
        .iter_mut()
        .flat_map(|group| group.unindexed.iter_mut())
    {
        fragment.num_rows = rows[&fragment.id];
    }
    Ok(groups)
}

/// One task per index with every segment as a candidate: the single-process
/// `optimize_indices` behavior.
#[derive(Debug, Clone)]
pub(crate) struct DeltaMergePlanner {
    index_names: Option<Vec<String>>,
    num_indices_to_merge: Option<usize>,
    retrain: bool,
}

impl DeltaMergePlanner {
    pub(crate) fn new(
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
        let tagged = has_tagged_fragment_reuse_history(dataset).await?;
        let mut tasks = Vec::new();
        for group in index_groups(dataset, self.index_names.as_deref()).await? {
            // A fully covered scalar index has nothing to do unless a retrain or
            // an explicit merge was asked for; a vector one may still rebalance.
            if !self.retrain
                && self.num_indices_to_merge.is_none_or(|n| n == 0)
                && !group.is_vector()
                && group.unindexed.is_empty()
            {
                continue;
            }
            let mut shardable =
                !self.retrain && !group.unindexed.is_empty() && !tagged && group.family_shardable();
            // Only a task that could be sharded needs the segment's format.
            if shardable && matches!(group.family, IndexFamily::Vector | IndexFamily::Inverted) {
                match group.legacy_format(dataset).await? {
                    Some(legacy) => shardable = !legacy,
                    None => continue,
                }
            }
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

/// Packs segments under `max_rows_per_segment` rows and the new fragments, in
/// order and per shared vector model, into bins of at most that many rows. A
/// segment's size is the physical row count of its live fragments, a
/// fragment's its live row count; both come from the manifest and deletion
/// files alone.
#[derive(Debug, Clone)]
pub(crate) struct SizeTieredPlanner {
    index_names: Option<Vec<String>>,
    max_rows_per_segment: u64,
}

impl SizeTieredPlanner {
    pub(crate) fn new(index_names: Option<Vec<String>>, max_rows_per_segment: u64) -> Result<Self> {
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

    /// The physical rows of a segment's live fragments; `None` when unknown.
    fn segment_rows(
        live_coverage: Option<&RoaringBitmap>,
        physical_rows: &HashMap<u32, u64>,
    ) -> Option<u64> {
        Some(
            live_coverage?
                .iter()
                .map(|id| physical_rows.get(&id).copied().unwrap_or(0))
                .sum(),
        )
    }

    /// Segment positions grouped by shared model, each class in manifest order.
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

enum BinItem {
    Segment(usize),
    Fragment(FragmentRows),
}

#[async_trait]
impl IndexOptimizePlanner for SizeTieredPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<IndexOptimizePlan> {
        let read_version = dataset.manifest.version;
        let budget = self.max_rows_per_segment;
        let tagged = has_tagged_fragment_reuse_history(dataset).await?;
        let mut physical: HashMap<u32, u64> = HashMap::new();
        let mut tasks = Vec::new();
        for group in index_groups(dataset, self.index_names.as_deref()).await? {
            let uuids: Vec<Uuid> = group.segments.iter().map(|s| s.uuid).collect();
            let reference = *uuids.last().expect("a group has at least one segment");

            // A rebuild, or a replacement that reaches beyond the merged
            // segments, cannot be split: one task with the single-process defaults.
            let mut whole = group.has_dormant || group.has_definition_only;
            if !whole && matches!(group.family, IndexFamily::Vector | IndexFamily::Inverted) {
                match group.legacy_format(dataset).await? {
                    Some(legacy) => whole = legacy,
                    None => continue,
                }
            }
            if whole {
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

            let shardable = !tagged && group.family_shardable();
            let missing: HashSet<u32> = group
                .live_coverage
                .values()
                .flat_map(|bitmap| bitmap.iter())
                .filter(|id| !physical.contains_key(id))
                .collect();
            physical.extend(physical_rows(dataset, missing).await?);
            let classes = self.model_classes(dataset, &group).await?;
            let reference_class = classes
                .iter()
                .position(|class| class.contains(&(group.segments.len() - 1)))
                .expect("the last segment is in a class");
            for (class_index, class) in classes.iter().enumerate() {
                // Candidates in manifest order, then (in the reference class,
                // whose model the new data takes) the new fragments.
                let mut items: Vec<(BinItem, u64)> = class
                    .iter()
                    .filter_map(|&position| {
                        let live = group.live_coverage.get(&group.segments[position].uuid);
                        let rows = Self::segment_rows(live, &physical)?;
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

                // Greedy: close the bin when the next item would exceed the
                // budget; an item over the budget gets a bin of its own.
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
    use std::collections::{BTreeMap, HashSet};
    use std::ops::Bound;
    use std::sync::Arc;

    use arrow::datatypes::{Float32Type, UInt8Type, UInt32Type, UInt64Type};
    use arrow_array::cast::AsArray;
    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
        StringArray, UInt32Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, SargableQuery,
        ScalarIndexParams, SearchResult,
    };
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::storage::transpose;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::{PQ_CODE_COLUMN, SQ_CODE_COLUMN, VectorIndex};
    use lance_io::stream::RecordBatchStreamAdapter;
    use lance_linalg::distance::MetricType;
    use lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    use lance_testing::datagen::generate_random_array_with_seed;
    use rstest::rstest;

    use super::*;
    use crate::dataset::WriteParams;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::vector::builder::ExistingIndex;
    use crate::index::vector::ivf::optimize_vector_indices_v2;
    use crate::index::vector::{IndexFileVersion, VectorIndexParams};
    use crate::index::{CreateIndexBuilder, DatasetIndexExt};

    // ---- fixtures ----------------------------------------------------------

    const DIM: usize = 16;
    const NAMES: [&str; 4] = ["vector_idx", "id_idx", "text_idx", "ngram_idx"];
    type Plan = IndexOptimizePlan;
    type Task = IndexOptimizeTask;
    type Shape = (Vec<Uuid>, Vec<u32>, Option<usize>, bool);

    fn schema() -> Arc<Schema> {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("vector", DataType::FixedSizeList(item, DIM as i32), true),
            Field::new("text", DataType::Utf8, false),
            Field::new("ngram_text", DataType::Utf8, false),
        ]))
    }

    fn batch(start: u32, rows: usize) -> RecordBatch {
        let seed = [(start % 251) as u8; 32];
        let vectors = generate_random_array_with_seed::<Float32Type>(rows * DIM, seed);
        let ids = start..start + rows as u32;
        let words = ids.clone().map(|i| format!("word{} common", i % 7));
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(UInt32Array::from_iter_values(ids)),
                Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap()),
                Arc::new(StringArray::from_iter_values(words.clone())),
                Arc::new(StringArray::from_iter_values(words)),
            ],
        )
        .unwrap()
    }

    /// One fragment per entry of `sizes`, ids continuing from `next_id`.
    async fn append_rows(dataset: &mut Dataset, sizes: &[usize], next_id: &mut u32) {
        for &size in sizes {
            let reader = RecordBatchIterator::new(vec![Ok(batch(*next_id, size))], schema());
            let params = WriteParams {
                max_rows_per_file: size,
                ..Default::default()
            };
            dataset.append(reader, Some(params)).await.unwrap();
            *next_id += size as u32;
        }
    }

    async fn write_dataset(uri: &str, stable: bool, sizes: &[usize], next_id: &mut u32) -> Dataset {
        let reader = RecordBatchIterator::new(vec![Ok(batch(0, sizes[0]))], schema());
        let params = WriteParams {
            max_rows_per_file: sizes[0],
            enable_stable_row_ids: stable,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, uri, Some(params)).await.unwrap();
        *next_id = sizes[0] as u32;
        append_rows(&mut dataset, &sizes[1..], next_id).await;
        dataset
    }

    /// `vector_idx`, `id_idx` (BTree), `text_idx` (inverted), `ngram_idx` (NGram).
    async fn create_indices(dataset: &mut Dataset, vector: &VectorIndexParams) {
        let btree = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let ngram = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
        let inverted = InvertedIndexParams::default();
        let specs: [(&str, IndexType, &str, &dyn lance_index::IndexParams); 4] = [
            ("vector", IndexType::Vector, "vector_idx", vector),
            ("id", IndexType::BTree, "id_idx", &btree),
            ("text", IndexType::Inverted, "text_idx", &inverted),
            ("ngram_text", IndexType::NGram, "ngram_idx", &ngram),
        ];
        for (column, index_type, name, params) in specs {
            let name = Some(name.to_string());
            dataset
                .create_index(&[column], index_type, name, params, true)
                .await
                .unwrap();
        }
    }

    /// Four indexed fragments of 256 rows and `new` unindexed ones.
    async fn indexed_dataset(uri: &str, vector: &VectorIndexParams, new: usize) -> Dataset {
        let mut next_id = 0;
        let mut dataset = write_dataset(uri, false, &[256; 4], &mut next_id).await;
        create_indices(&mut dataset, vector).await;
        append_rows(&mut dataset, &vec![256; new], &mut next_id).await;
        dataset
    }

    fn ivf_pq() -> VectorIndexParams {
        VectorIndexParams::ivf_pq(4, 8, 4, MetricType::L2, 10)
    }

    fn ivf_hnsw_sq() -> VectorIndexParams {
        let (ivf, hnsw, sq) = (
            IvfBuildParams::new(4),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        VectorIndexParams::with_ivf_hnsw_sq_params(MetricType::L2, ivf, hnsw, sq)
    }

    fn delta_merge(num_indices_to_merge: Option<usize>) -> OptimizeOptions {
        OptimizeOptions::default().num_indices_to_merge(num_indices_to_merge)
    }

    fn size_tiered(max_rows_per_segment: u64) -> OptimizeOptions {
        OptimizeOptions::default().max_rows_per_segment(max_rows_per_segment)
    }

    fn only(name: &str, options: OptimizeOptions) -> OptimizeOptions {
        options.index_names(vec![name.to_string()])
    }

    async fn plan(dataset: &Dataset, options: &OptimizeOptions) -> Plan {
        plan_index_optimization(dataset, options).await.unwrap()
    }

    fn task_for<'a>(plan: &'a Plan, name: &str) -> &'a Task {
        let task = plan.tasks.iter().find(|task| task.index_name == name);
        task.unwrap_or_else(|| panic!("no task for {name} in {plan:?}"))
    }

    fn fragment_ids(task: &Task) -> Vec<u32> {
        task.fragments.iter().map(|f| f.id).collect()
    }

    /// (segments, fragment ids, num_indices_to_merge, shardable) of every task.
    fn shape(plan: &Plan) -> Vec<Shape> {
        let shape = |t: &Task| {
            (
                t.segments.clone(),
                fragment_ids(t),
                t.num_indices_to_merge,
                t.shardable,
            )
        };
        plan.tasks.iter().map(shape).collect()
    }

    async fn segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
        dataset.load_indices_by_name(name).await.unwrap()
    }

    async fn uuids(dataset: &Dataset, name: &str) -> Vec<Uuid> {
        segments(dataset, name)
            .await
            .iter()
            .map(|s| s.uuid)
            .collect()
    }

    /// Sorted per-segment coverage of `name`.
    async fn coverage(dataset: &Dataset, name: &str) -> Vec<Vec<u32>> {
        let bitmap = |s: &IndexMetadata| s.fragment_bitmap.as_ref().unwrap().iter().collect();
        let mut out: Vec<Vec<u32>> = segments(dataset, name).await.iter().map(bitmap).collect();
        out.sort();
        out
    }

    /// `name` over `column` committed as one segment per fragment group.
    async fn commit_grouped(
        dataset: &mut Dataset,
        name: &str,
        column: &str,
        kind: IndexType,
        params: &dyn lance_index::IndexParams,
        groups: &[Vec<u32>],
    ) {
        let mut staged = Vec::new();
        for group in groups {
            let mut builder = CreateIndexBuilder::new(dataset, &[column], kind, params)
                .name(name.to_string())
                .fragments(group.clone());
            staged.push(builder.execute_uncommitted().await.unwrap());
        }
        dataset
            .commit_existing_index_segments(name, column, staged)
            .await
            .unwrap();
    }

    async fn commit_btree_segments(dataset: &mut Dataset, name: &str, groups: &[Vec<u32>]) {
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        commit_grouped(dataset, name, "id", IndexType::BTree, &params, groups).await;
    }

    async fn commit_segments(
        dataset: &Dataset,
        new: Vec<IndexMetadata>,
        removed: Vec<IndexMetadata>,
    ) -> Dataset {
        let mut committed = dataset.clone();
        let operation = Operation::CreateIndex {
            new_indices: new,
            removed_indices: removed,
        };
        let transaction = TransactionBuilder::new(dataset.manifest.version, operation).build();
        let (write, commit) = (Default::default(), Default::default());
        committed
            .apply_commit(transaction, &write, &commit)
            .await
            .unwrap();
        committed
    }

    async fn open_vector(dataset: &Dataset, segment: &IndexMetadata) -> Arc<dyn VectorIndex> {
        let opened =
            dataset.open_vector_index_from_metadata("vector", segment, &NoOpMetricsCollector);
        opened.await.unwrap()
    }

    /// (row id -> code bytes) per partition; PQ codes are stored transposed.
    async fn partition_contents(index: &Arc<dyn VectorIndex>) -> Vec<BTreeMap<u64, Vec<u8>>> {
        let mut out = Vec::new();
        for part in 0..index.ivf_model().num_partitions() {
            let mut rows = BTreeMap::new();
            if index.partition_size(part) > 0 {
                let mut reader = index
                    .partition_reader(part, true, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                while let Some(batch) = reader.try_next().await.unwrap() {
                    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                    let fields = batch.schema().fields().clone();
                    let is_code = |f: &Arc<Field>| {
                        [PQ_CODE_COLUMN, SQ_CODE_COLUMN].contains(&f.name().as_str())
                    };
                    let code_idx = fields.iter().position(is_code).unwrap();
                    let mut codes = batch.column(code_idx).as_fixed_size_list().clone();
                    if fields[code_idx].name() == PQ_CODE_COLUMN {
                        let values = codes.values().as_primitive::<UInt8Type>();
                        let width = values.len() / batch.num_rows();
                        let original = transpose(values, width, batch.num_rows());
                        codes = FixedSizeListArray::try_new_from_values(original, width as i32)
                            .unwrap();
                    }
                    for i in 0..batch.num_rows() {
                        let bytes = codes.value(i).as_primitive::<UInt8Type>().values().to_vec();
                        assert!(rows.insert(row_ids.value(i), bytes).is_none());
                    }
                }
            }
            out.push(rows);
        }
        out
    }

    async fn row_ids(dataset: &Dataset, segment: &IndexMetadata) -> HashSet<u64> {
        let contents = partition_contents(&open_vector(dataset, segment).await).await;
        contents
            .into_iter()
            .flat_map(|rows| rows.into_keys())
            .collect()
    }

    /// Recall of the index search against exact search over the first eight rows.
    async fn recall(dataset: &Dataset) -> f32 {
        let mut scan = dataset.scan();
        scan.project(&["vector"])
            .unwrap()
            .limit(Some(8), None)
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let vectors = batch["vector"].as_fixed_size_list();
        let queries: Vec<ArrayRef> = (0..vectors.len()).map(|i| vectors.value(i)).collect();
        let mut hits = 0;
        for query in &queries {
            let mut exact = dataset.scan();
            exact
                .with_row_id()
                .nearest("vector", query.as_ref(), 10)
                .unwrap()
                .use_index(false);
            let mut approx = dataset.scan();
            approx
                .with_row_id()
                .nearest("vector", query.as_ref(), 10)
                .unwrap()
                .nprobes(4)
                .ef(64);
            let ids = |b: RecordBatch| b[ROW_ID].as_primitive::<UInt64Type>().values().to_vec();
            let exact = ids(exact.try_into_batch().await.unwrap());
            let approx = ids(approx.try_into_batch().await.unwrap());
            hits += approx.iter().filter(|id| exact.contains(id)).count();
        }
        hits as f32 / (queries.len() * 10) as f32
    }

    /// Same partitions (row ids and codes), and the same recall once committed.
    async fn assert_vector_equivalent(dataset: &Dataset, a: &IndexMetadata, b: &IndexMetadata) {
        let (ia, ib) = (open_vector(dataset, a).await, open_vector(dataset, b).await);
        assert_eq!(partition_contents(&ia).await, partition_contents(&ib).await);
        let with_a = commit_segments(
            dataset,
            vec![a.clone()],
            segments(dataset, "vector_idx").await,
        )
        .await;
        let with_b = commit_segments(&with_a, vec![b.clone()], vec![a.clone()]).await;
        assert!((recall(&with_a).await - recall(&with_b).await).abs() <= 0.1);
    }

    /// Ids a query spanning old and new rows returns for `name`.
    async fn query_ids(dataset: &Dataset, name: &str, use_index: bool) -> Vec<u32> {
        let mut scan = dataset.scan();
        scan.project(&["id"]).unwrap().use_scalar_index(use_index);
        match name {
            "id_idx" => scan.filter("id >= 250 AND id < 1300").unwrap(),
            "ngram_idx" => scan.filter("contains(ngram_text, 'word3')").unwrap(),
            _ => scan
                .full_text_search(FullTextSearchQuery::new("word3".into()))
                .unwrap(),
        };
        let batch = scan.try_into_batch().await.unwrap();
        let mut ids = batch["id"].as_primitive::<UInt32Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    /// Sorted row addresses a bitmap segment returns for ids in [250, 1300).
    async fn bitmap_rows(dataset: &Dataset, segment: &IndexMetadata) -> Vec<u64> {
        let range = SargableQuery::Range(
            Bound::Included(ScalarValue::UInt32(Some(250))),
            Bound::Excluded(ScalarValue::UInt32(Some(1300))),
        );
        let index =
            crate::index::scalar::open_scalar_index(dataset, "id", segment, &NoOpMetricsCollector);
        let SearchResult::Exact(rows) = index
            .await
            .unwrap()
            .search(&range, &NoOpMetricsCollector)
            .await
            .unwrap()
        else {
            panic!("bitmap search must be exact");
        };
        let mut rows: Vec<u64> = rows
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        rows.sort_unstable();
        rows
    }

    async fn tag_table(dataset: &mut Dataset) {
        let indices = load_all_indices(dataset).await.unwrap().as_ref().clone();
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        crate::index::frag_reuse_reader::tests::persist_fixture(dataset, indices).await;
    }

    async fn compact(
        dataset: &mut Dataset,
        target_rows_per_fragment: usize,
        defer_index_remap: bool,
    ) {
        let options = CompactionOptions {
            target_rows_per_fragment,
            defer_index_remap,
            ..Default::default()
        };
        compact_files(dataset, options, None).await.unwrap();
    }

    fn invalid(err: Error) {
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    fn retryable(err: Error) {
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
    }

    // ---- planners ----------------------------------------------------------

    #[tokio::test]
    async fn delta_merge_plans_like_optimize_indices() {
        let dir = TempStrDir::default();
        let dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 2).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        assert_eq!(planned.read_version, dataset.manifest.version);
        assert_eq!(planned.tasks.len(), 4);
        for name in NAMES {
            let task = task_for(&planned, name);
            assert_eq!(task.segments, uuids(&dataset, name).await);
            assert_eq!(fragment_ids(task), [4, 5]);
            assert!(task.fragments.iter().all(|f| f.num_rows == 256));
            assert!(task.num_indices_to_merge == Some(1) && !task.retrain && task.shardable);
        }
        let planned = plan(&dataset, &only("id_idx", delta_merge(None))).await;
        assert!(planned.tasks.len() == 1 && planned.tasks[0].num_indices_to_merge.is_none());

        // Nothing unindexed: scalar groups are skipped under `None` / `Some(0)`.
        let dir = TempStrDir::default();
        let dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 0).await;
        for num in [None, Some(0)] {
            let planned = plan(&dataset, &delta_merge(num)).await;
            assert_eq!(planned.tasks.len(), 1, "{num:?}");
            assert!(planned.tasks[0].index_name == "vector_idx" && !planned.tasks[0].shardable);
        }
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        assert!(planned.tasks.len() == 4 && planned.tasks.iter().all(|t| !t.shardable));
    }

    /// `s1` covers four fragments, `s2` / `s3` one each, two fragments are new:
    /// with a budget of three fragments `[s2, s3, f6, f7]` packs as `[s2, s3, f6]`, `[f7]`.
    #[tokio::test]
    async fn size_tiered_packs_in_order() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[64; 6], &mut next_id).await;
        commit_btree_segments(
            &mut dataset,
            "id_seg",
            &[vec![0, 1, 2, 3], vec![4], vec![5]],
        )
        .await;
        let no_new_data = dataset.clone();
        append_rows(&mut dataset, &[64, 64], &mut next_id).await;
        let s = uuids(&dataset, "id_seg").await;

        let expected = [
            (vec![s[1], s[2]], vec![6], Some(2), true),
            (vec![s[2]], vec![7], Some(0), true),
        ];
        assert_eq!(shape(&plan(&dataset, &size_tiered(192)).await), expected);
        let expected = [(s.clone(), vec![6, 7], Some(3), true)];
        assert_eq!(shape(&plan(&dataset, &size_tiered(1_000)).await), expected);
        // Without new data the small segments still merge; a lone one is no task.
        let expected = [(vec![s[1], s[2]], vec![], Some(2), false)];
        assert_eq!(
            shape(&plan(&no_new_data, &size_tiered(192)).await),
            expected
        );
        assert!(plan(&no_new_data, &size_tiered(65)).await.tasks.is_empty());
    }

    #[tokio::test]
    async fn size_tiered_packs_vector_segments_per_model() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[128; 4], &mut next_id).await;
        // The base segment trains its own centroids; the tail shares fixed ones.
        let own = VectorIndexParams::ivf_flat(2, MetricType::L2);
        let centroids = Float32Array::from_iter_values((0..2 * DIM).map(|i| (i / DIM) as f32));
        let centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf = IvfBuildParams::try_with_centroids(2, Arc::new(centroids)).unwrap();
        let shared = VectorIndexParams::with_ivf_flat_params(MetricType::L2, ivf);
        let specs: [(&dyn lance_index::IndexParams, Vec<u32>); 3] =
            [(&own, vec![0, 1]), (&shared, vec![2]), (&shared, vec![3])];
        let mut staged = Vec::new();
        for (params, fragments) in specs {
            let mut builder =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, params)
                    .name("vector_idx".to_string())
                    .fragments(fragments);
            staged.push(builder.execute_uncommitted().await.unwrap());
        }
        let ids: Vec<Uuid> = staged.iter().map(|s| s.uuid).collect();
        dataset
            .commit_existing_index_segments("vector_idx", "vector", staged)
            .await
            .unwrap();
        append_rows(&mut dataset, &[128, 128], &mut next_id).await;

        let planned = plan(&dataset, &size_tiered(100_000)).await;
        assert_eq!(
            shape(&planned),
            [(ids[1..].to_vec(), vec![4, 5], Some(2), true)]
        );
        let result = planned.tasks[0].execute(&dataset).await.unwrap();
        commit_index_optimization(&mut dataset, vec![result], None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(uuids(&dataset, "vector_idx").await[0], ids[0]);
        assert_eq!(
            coverage(&dataset, "vector_idx").await,
            [vec![0, 1], vec![2, 3, 4, 5]]
        );
    }

    /// Definition-only and legacy groups become one unsharded task.
    #[tokio::test]
    async fn size_tiered_hands_rebuilding_groups_to_one_task() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[64], &mut next_id).await;
        create_indices(&mut dataset, &ivf_pq()).await; // too few rows to train PQ
        let segment = segments(&dataset, "vector_idx").await.remove(0);
        assert!(is_definition_only_segment(&segment));
        append_rows(&mut dataset, &[256, 256], &mut next_id).await;
        let planned = plan(&dataset, &size_tiered(64)).await;
        let task = task_for(&planned, "vector_idx");
        assert_eq!(fragment_ids(task), [0, 1, 2], "a definition covers nothing");
        assert!(task.segments == [segment.uuid] && task.num_indices_to_merge.is_none());
        assert!(!task.shardable && fragment_ids(task_for(&planned, "id_idx")) == [1]);

        let dir = TempStrDir::default();
        let mut dataset = write_dataset(dir.as_str(), false, &[256; 2], &mut next_id).await;
        let mut legacy = VectorIndexParams::ivf_pq(2, 8, 4, MetricType::L2, 10);
        legacy.version(IndexFileVersion::Legacy);
        let name = Some("vector_idx".to_string());
        let created = dataset.create_index(&["vector"], IndexType::Vector, name, &legacy, true);
        created.await.unwrap();
        append_rows(&mut dataset, &[256], &mut next_id).await;
        let task = &plan(&dataset, &size_tiered(1_000)).await.tasks[0];
        assert!(!task.shardable && fragment_ids(task) == [2]);
        let task = &plan(&dataset, &delta_merge(Some(1))).await.tasks[0];
        assert!(!task.shardable && task.num_indices_to_merge == Some(1));
    }

    /// Segments count physical rows of live fragments, new fragments their live rows.
    #[tokio::test]
    async fn size_tiered_sizes_segments_and_fragments() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[64, 64], &mut next_id).await;
        commit_btree_segments(&mut dataset, "id_seg", &[vec![0], vec![1]]).await;
        dataset.delete("id < 32").await.unwrap();
        append_rows(&mut dataset, &[64], &mut next_id).await;
        dataset.delete("id >= 128 AND id < 160").await.unwrap();
        let s = uuids(&dataset, "id_seg").await;
        // Sizes 64, 64, 32 under a budget of 128: the segments fill one bin and
        // the fragment starts another; counting live rows would pack all three.
        let planned = plan(&dataset, &size_tiered(128)).await;
        let expected = [
            (s.clone(), vec![], Some(2), false),
            (vec![s[1]], vec![2], Some(0), true),
        ];
        assert_eq!(shape(&planned), expected);
        assert_eq!(planned.tasks[1].fragments[0].num_rows, 32);
        // A retired fragment left in the bitmap adds nothing; no bitmap, no size.
        let rows = physical_rows(&dataset, [0, 1, 2].into()).await.unwrap();
        let mut segment = segments(&dataset, "id_seg").await.remove(0);
        segment.fragment_bitmap = Some(RoaringBitmap::from_iter([0u32, 1, 7]));
        let live = segment.effective_fragment_bitmap(&dataset.fragment_bitmap);
        assert_eq!(
            SizeTieredPlanner::segment_rows(live.as_ref(), &rows),
            Some(128)
        );
        assert_eq!(SizeTieredPlanner::segment_rows(None, &rows), None);
    }

    // ---- shardable and shard ----------------------------------------------

    #[tokio::test]
    async fn shardable_follows_the_index_family_and_the_reuse_history() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[256; 4], &mut next_id).await;
        create_indices(&mut dataset, &ivf_pq()).await;
        let extra = [
            ("id_bitmap", IndexType::Bitmap, BuiltinIndexType::Bitmap),
            ("id_zonemap", IndexType::ZoneMap, BuiltinIndexType::ZoneMap),
        ];
        for (name, kind, builtin) in extra {
            let params = ScalarIndexParams::for_builtin(builtin);
            let name = Some(name.to_string());
            dataset
                .create_index(&["id"], kind, name, &params, true)
                .await
                .unwrap();
        }
        append_rows(&mut dataset, &[256], &mut next_id).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        assert!(NAMES.iter().all(|name| task_for(&planned, name).shardable));
        assert!(task_for(&planned, "id_bitmap").shardable);
        assert!(!task_for(&planned, "id_zonemap").shardable);

        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[128, 128], &mut next_id).await;
        create_indices(&mut dataset, &ivf_pq()).await;
        tag_table(&mut dataset).await;
        compact(&mut dataset, 256, true).await;
        assert!(has_tagged_fragment_reuse_history(&dataset).await.unwrap());
        append_rows(&mut dataset, &[256], &mut next_id).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        assert!(planned.tasks.len() == 4 && planned.tasks.iter().all(|t| !t.shardable));
    }

    #[tokio::test]
    async fn shard_validates_its_fragments_and_builds_an_append_task() {
        let dir = TempStrDir::default();
        let dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 2).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        let task = task_for(&planned, "vector_idx");
        let shard = task.shard(&[5]).unwrap();
        assert!(shard.read_version == task.read_version && shard.index_name == "vector_idx");
        assert_eq!(shard.segments, [*task.segments.last().unwrap()]);
        assert_eq!(
            shard.fragments,
            [FragmentRows {
                id: 5,
                num_rows: 256
            }]
        );
        assert!(shard.num_indices_to_merge == Some(0) && !shard.retrain && !shard.shardable);
        assert_eq!(fragment_ids(&task.shard(&[5, 4]).unwrap()), [4, 5]);
        for ids in [vec![], vec![4, 4], vec![4, 9]] {
            invalid(task.shard(&ids).unwrap_err());
        }
        invalid(shard.shard(&[5]).unwrap_err());
    }

    // ---- map + reduce ------------------------------------------------------

    #[rstest]
    #[case::ivf_pq("vector_idx", ivf_pq())]
    #[case::ivf_hnsw_sq("vector_idx", ivf_hnsw_sq())]
    #[case::btree("id_idx", ivf_pq())]
    #[case::bitmap("id_bitmap", ivf_pq())]
    #[case::ngram("ngram_idx", ivf_pq())]
    #[case::inverted("text_idx", ivf_pq())]
    #[tokio::test]
    async fn merging_shard_results_matches_executing_the_task(
        #[case] name: &str,
        #[case] params: VectorIndexParams,
    ) {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[256; 4], &mut next_id).await;
        create_indices(&mut dataset, &params).await;
        if name == "id_bitmap" {
            let bitmap = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
            let name = Some(name.to_string());
            dataset
                .create_index(&["id"], IndexType::Bitmap, name, &bitmap, true)
                .await
                .unwrap();
        }
        append_rows(&mut dataset, &[256; 2], &mut next_id).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        let task = task_for(&planned, name);
        let direct = task.execute(&dataset).await.unwrap();
        let mut shard_results = Vec::new();
        for id in [4, 5] {
            let result = task.shard(&[id]).unwrap().execute(&dataset).await.unwrap();
            let bitmap = result.new_segment.as_ref().unwrap().fragment_bitmap.clone();
            assert!(
                result.removed_segments.is_empty()
                    && bitmap == Some(RoaringBitmap::from_iter([id]))
            );
            shard_results.push(result);
        }
        let reduced = task.merge(&dataset, shard_results).await.unwrap();
        assert_eq!(reduced.removed_segments, direct.removed_segments);
        assert_eq!(reduced.removed_segments, task.segments);
        let (a, b) = (
            direct.new_segment.clone().unwrap(),
            reduced.new_segment.unwrap(),
        );
        assert_eq!(a.fragment_bitmap, b.fragment_bitmap);
        if name == "vector_idx" {
            assert_vector_equivalent(&dataset, &a, &b).await;
        } else if name == "id_bitmap" {
            // Two indices share the column, so ask the segments themselves.
            let (ra, rb) = (
                bitmap_rows(&dataset, &a).await,
                bitmap_rows(&dataset, &b).await,
            );
            assert!(ra.len() == 1050 && ra == rb);
        } else {
            let expected = query_ids(&dataset, name, false).await;
            let mut with_direct = dataset.clone();
            commit_index_optimization(&mut with_direct, vec![direct], None)
                .await
                .unwrap();
            assert_eq!(query_ids(&with_direct, name, true).await, expected);
            let with_reduced = commit_segments(&with_direct, vec![b], vec![a]).await;
            assert_eq!(query_ids(&with_reduced, name, true).await, expected);
        }
    }

    /// Retired fragments' rows are dropped; rows deleted inside a live fragment
    /// survive under address-style row ids and are dropped under stable ones.
    #[rstest]
    #[case::address(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn merging_shard_results_filters_stale_rows_like_the_task(#[case] stable: bool) {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), stable, &[128, 128, 512], &mut next_id).await;
        create_indices(&mut dataset, &ivf_pq()).await;
        dataset
            .delete("id < 16 OR (id >= 256 AND id < 272)")
            .await
            .unwrap();
        compact(&mut dataset, 256, false).await;
        assert!(dataset.fragment_bitmap.contains(2) && !dataset.fragment_bitmap.contains(0));
        append_rows(&mut dataset, &[256; 2], &mut next_id).await;

        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        let task = task_for(&planned, "vector_idx");
        let direct = task.execute(&dataset).await.unwrap().new_segment.unwrap();
        let mut shard_results = Vec::new();
        for id in fragment_ids(task) {
            shard_results.push(task.shard(&[id]).unwrap().execute(&dataset).await.unwrap());
        }
        let reduced = task
            .merge(&dataset, shard_results)
            .await
            .unwrap()
            .new_segment
            .unwrap();
        assert_vector_equivalent(&dataset, &direct, &reduced).await;

        let stored = row_ids(&dataset, &reduced).await;
        let mut scan = dataset.scan();
        scan.with_row_id().project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let live: HashSet<u64> = batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect();
        let stale: Vec<u64> = stored.difference(&live).copied().collect();
        if stable {
            assert_eq!(stored, live);
        } else {
            assert!(
                stored
                    .iter()
                    .all(|row| dataset.fragment_bitmap.contains((row >> 32) as u32))
            );
            assert!(
                stale.len() == 16 && stale.iter().all(|row| row >> 32 == 2),
                "{stale:?}"
            );
        }
    }

    /// Shard segments given to the builder as new data sources are merged
    /// whatever the candidate selection is, and never counted as merged.
    #[tokio::test]
    async fn new_data_sources_are_merged_but_not_counted() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[512, 128, 128], &mut next_id).await;
        let flat = VectorIndexParams::ivf_flat(2, MetricType::L2);
        commit_grouped(
            &mut dataset,
            "vector_idx",
            "vector",
            IndexType::Vector,
            &flat,
            &[vec![0]],
        )
        .await;
        let committed = segments(&dataset, "vector_idx").await.remove(0);
        let live = dataset.fragment_bitmap.as_ref();
        let (effective, deleted) = (
            committed.effective_fragment_bitmap(live),
            committed.deleted_fragment_bitmap(live),
        );
        let candidate = open_vector(&dataset, &committed).await;
        let candidates = vec![ExistingIndex::with_coverage(
            candidate,
            dataset.clone(),
            effective.unwrap(),
            deleted.unwrap(),
        )];
        let task = task_for(&plan(&dataset, &delta_merge(Some(1))).await, "vector_idx").clone();
        let mut shards = Vec::new();
        for id in [1, 2] {
            let result = task.shard(&[id]).unwrap().execute(&dataset).await.unwrap();
            shards.push(ExistingIndex::unfiltered(
                open_vector(&dataset, &result.new_segment.unwrap()).await,
            ));
        }
        // (options, target partition size) -> (candidates merged, output rows, split happened)
        let cases = [
            (OptimizeOptions::merge(1), None, 1, 768, false),
            (OptimizeOptions::new(), Some(256), 0, 256, false), // ~384 rows per partition: no split, no join
            (OptimizeOptions::new(), Some(64), 1, 768, true), // over 4 * 64: a split merges every candidate
        ];
        for (options, target, merged, rows, split) in cases {
            let none = Option::<
                RecordBatchStreamAdapter<futures::stream::Empty<Result<RecordBatch>>>,
            >::None;
            let sources = shards.clone();
            let built = optimize_vector_indices_v2(
                &dataset,
                none,
                "vector",
                &candidates,
                sources,
                &options,
                target,
            );
            let (uuid, indices_merged, files) = built.await.unwrap();
            let output = IndexMetadata {
                uuid,
                files: Some(files),
                ..committed.clone()
            };
            let index = open_vector(&dataset, &output).await;
            let partitions = index.ivf_model().num_partitions();
            assert_eq!(
                (indices_merged, index.num_rows() as usize, partitions > 2),
                (merged, rows, split)
            );
        }
    }

    /// One bin runs as shards then merge, the other as a shard committed as is.
    #[tokio::test]
    async fn bins_run_in_parallel_and_commit_together() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[64; 6], &mut next_id).await;
        commit_btree_segments(
            &mut dataset,
            "id_idx",
            &[vec![0, 1, 2, 3], vec![4], vec![5]],
        )
        .await;
        append_rows(&mut dataset, &[64, 64], &mut next_id).await;
        let before = uuids(&dataset, "id_idx").await;
        let expected = query_ids(&dataset, "id_idx", false).await;
        assert_eq!(expected, (250..512).collect::<Vec<u32>>());

        let planned = plan(&dataset, &size_tiered(192)).await;
        let merged = async {
            let task = &planned.tasks[0];
            let shard_result = task.shard(&[6]).unwrap().execute(&dataset).await.unwrap();
            task.merge(&dataset, vec![shard_result]).await.unwrap()
        };
        let delta = async {
            planned.tasks[1]
                .shard(&[7])
                .unwrap()
                .execute(&dataset)
                .await
                .unwrap()
        };
        let (merged, delta) = futures::join!(merged, delta);
        assert_eq!(merged.removed_segments, [before[1], before[2]]);
        assert!(delta.removed_segments.is_empty());
        let new_uuid = |r: &IndexOptimizeResult| r.new_segment.as_ref().unwrap().uuid;
        let (merged_uuid, delta_uuid) = (new_uuid(&merged), new_uuid(&delta));
        // Results in any order: the manifest keeps the pure addition last.
        commit_index_optimization(&mut dataset, vec![delta, merged], None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(
            uuids(&dataset, "id_idx").await,
            [before[0], merged_uuid, delta_uuid]
        );
        assert_eq!(
            coverage(&dataset, "id_idx").await,
            [vec![0, 1, 2, 3], vec![4, 5, 6], vec![7]]
        );
        assert_eq!(query_ids(&dataset, "id_idx", true).await, expected);
    }

    // ---- commit ------------------------------------------------------------

    #[tokio::test]
    async fn commit_validates_results_and_anchors_at_the_plan_version() {
        let dir = TempStrDir::default();
        let mut dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 2).await;
        let version = dataset.manifest.version;
        let planned = plan(&dataset, &only("id_idx", delta_merge(Some(1)))).await;
        let task = &planned.tasks[0];
        let result = task.execute(&dataset).await.unwrap();
        let shard = task.shard(&[4]).unwrap().execute(&dataset).await.unwrap();
        let mut other_plan = result.clone();
        other_plan.read_version += 1;
        let mut foreign = result.clone();
        foreign.removed_segments = vec![Uuid::new_v4()];
        let mut renamed = result.clone();
        renamed.new_segment.as_mut().unwrap().name = "text_idx".to_string();
        let overlapping = vec![result.clone(), shard];
        for results in [
            vec![result.clone(), other_plan],
            vec![result.clone(), result.clone()],
            vec![foreign],
            vec![renamed],
            overlapping,
        ] {
            invalid(
                commit_index_optimization(&mut dataset, results, None)
                    .await
                    .unwrap_err(),
            );
        }
        let nothing = IndexOptimizeResult {
            new_segment: None,
            removed_segments: vec![],
            ..result.clone()
        };
        commit_index_optimization(&mut dataset, vec![nothing], None)
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, version, "nothing to commit");

        // From a handle that moved on through an unrelated append.
        let mut latest = dataset.clone();
        append_rows(&mut latest, &[64], &mut (6 * 256)).await;
        commit_index_optimization(&mut latest, vec![result.clone()], None)
            .await
            .unwrap();
        latest.validate().await.unwrap();
        assert_eq!(coverage(&latest, "id_idx").await, [vec![0, 1, 2, 3, 4, 5]]);
        assert_eq!(latest.unindexed_fragments("id_idx").await.unwrap().len(), 1);

        // Another writer optimized the same index: a retryable conflict, from either handle.
        let dir = TempStrDir::default();
        let mut dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 2).await;
        let planned = plan(&dataset, &only("id_idx", delta_merge(Some(1)))).await;
        let result = planned.tasks[0].execute(&dataset).await.unwrap();
        let mut other = dataset.clone();
        let mut options = OptimizeOptions::merge(1);
        options.index_names = Some(vec!["id_idx".to_string()]);
        other.optimize_indices(&options).await.unwrap();
        retryable(
            commit_index_optimization(&mut other, vec![result.clone()], None)
                .await
                .unwrap_err(),
        );
        retryable(
            commit_index_optimization(&mut dataset, vec![result], None)
                .await
                .unwrap_err(),
        );
    }

    /// An append or delete after the plan commits; a compaction of covered
    /// fragments conflicts unless a tagged history records it; NGram always conflicts.
    #[rstest]
    #[case::append("append")]
    #[case::delete("delete")]
    #[case::plain_compaction("compact")]
    #[case::tagged_compaction("tagged")]
    #[case::ngram_compaction("ngram")]
    #[tokio::test]
    async fn commit_after_the_table_moved_on(#[case] drift: &str) {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[128; 3], &mut next_id).await;
        if drift == "tagged" {
            tag_table(&mut dataset).await;
        }
        let (name, column, kind) = match drift {
            "ngram" => ("ngram_idx", "ngram_text", IndexType::NGram),
            _ => ("id_idx", "id", IndexType::BTree),
        };
        let params = ScalarIndexParams::for_builtin(kind.try_into().unwrap());
        dataset
            .create_index(&[column], kind, Some(name.into()), &params, true)
            .await
            .unwrap();
        append_rows(&mut dataset, &[128], &mut next_id).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        let result = task_for(&planned, name).execute(&dataset).await.unwrap();

        let mut latest = dataset.clone();
        match drift {
            "append" => append_rows(&mut latest, &[128], &mut next_id).await,
            "delete" => drop(latest.delete("id < 10").await.unwrap()),
            _ => compact(&mut latest, 1024, drift != "compact").await,
        }
        let committed = commit_index_optimization(&mut latest, vec![result], None).await;
        if matches!(drift, "append" | "delete" | "tagged") {
            committed.unwrap();
            latest.validate().await.unwrap();
            assert_eq!(segments(&latest, name).await.len(), 1);
        } else {
            retryable(committed.unwrap_err());
        }
    }

    /// On a tagged table a merged segment keeps its sources' provenance
    /// coordinates and the reader derives the live coverage, so a result is
    /// never checked against stored coverage; the commit validates and lands.
    #[tokio::test]
    async fn tagged_merge_keeps_provenance_coverage() {
        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[128, 128], &mut next_id).await;
        create_indices(
            &mut dataset,
            &VectorIndexParams::ivf_flat(2, MetricType::L2),
        )
        .await;
        tag_table(&mut dataset).await;
        compact(&mut dataset, 256, true).await; // fragments 0 and 1 become 2
        append_rows(&mut dataset, &[128], &mut next_id).await; // fragment 3
        let stored = load_all_indices(&dataset).await.unwrap();
        let stored = stored.iter().find(|s| s.name == "vector_idx").unwrap();
        assert_eq!(
            stored.fragment_bitmap,
            Some(RoaringBitmap::from_iter([0u32, 1]))
        );

        let planned = plan(&dataset, &only("vector_idx", delta_merge(None))).await;
        let result = planned.tasks[0].execute(&dataset).await.unwrap();
        assert_eq!(result.removed_segments, [stored.uuid]);
        let bitmap = result
            .new_segment
            .as_ref()
            .unwrap()
            .fragment_bitmap
            .clone()
            .unwrap();
        assert_eq!(
            bitmap.iter().collect::<Vec<u32>>(),
            [0, 1, 3],
            "provenance plus new"
        );
        commit_index_optimization(&mut dataset, vec![result], None)
            .await
            .unwrap();
        assert_eq!(
            coverage(&dataset, "vector_idx").await,
            [vec![2, 3]],
            "derived"
        );
    }

    /// An in-place rewrite of an indexed column on a tagged table withdraws
    /// that index's coverage. The planner still plans the other indices, hands
    /// the withdrawn one to a whole task, and `optimize_indices` rebuilds it.
    #[tokio::test]
    #[serial_test::serial(frag_reuse_maintenance)]
    async fn withdrawn_segment_is_rebuilt_instead_of_failing_the_plan() {
        use crate::dataset::{
            MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched,
        };

        let dir = TempStrDir::default();
        let mut next_id = 0;
        let mut dataset = write_dataset(dir.as_str(), false, &[128, 128], &mut next_id).await;
        create_indices(
            &mut dataset,
            &VectorIndexParams::ivf_flat(2, MetricType::L2),
        )
        .await;
        tag_table(&mut dataset).await;
        compact(&mut dataset, 256, true).await;
        // One row's text rewritten in place: the whole transition is withdrawn from `text_idx`.
        let schema = dataset.schema().project(&["id", "text"]).unwrap();
        let source = RecordBatch::try_new(
            Arc::new(Schema::from(&schema)),
            vec![
                Arc::new(UInt32Array::from(vec![5u32])),
                Arc::new(StringArray::from(vec!["rewritten"])),
            ],
        )
        .unwrap();
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_batches(vec![source])
            .await
            .unwrap();
        let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
        let stored = load_all_indices(&dataset).await.unwrap();
        assert!(
            stored.iter().any(|s| s.name == "text_idx"),
            "the withdrawn segment stays registered"
        );
        assert!(
            segments(&dataset, "text_idx")
                .await
                .iter()
                .all(|s| s.fragment_bitmap.as_ref().is_none_or(|b| b.is_empty()))
        );

        let planned = plan(&dataset, &delta_merge(None)).await;
        let task = task_for(&planned, "text_idx");
        assert!(task.segments.len() == 1 && !task.shardable);
        assert!(task_for(&planned, "vector_idx").segments.len() == 1);
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let live: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();
        assert_eq!(coverage(&dataset, "text_idx").await, [live]);
        let mut scan = dataset.scan();
        scan.project(&["id"])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("rewritten".into()))
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<UInt32Type>().values().to_vec(),
            [5]
        );
    }

    // ---- serialization -----------------------------------------------------

    #[tokio::test]
    async fn strategies_are_mutually_exclusive_and_default_to_delta_merge() {
        let dir = TempStrDir::default();
        let dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 1).await;
        let mut retrain = size_tiered(100);
        retrain.retrain = true;
        for options in [
            size_tiered(100).num_indices_to_merge(Some(1)),
            retrain,
            size_tiered(0),
        ] {
            invalid(
                plan_index_optimization(&dataset, &options)
                    .await
                    .unwrap_err(),
            );
        }
        let planned = plan(&dataset, &OptimizeOptions::default()).await;
        assert_eq!(planned.tasks.len(), 4);
        assert!(
            planned
                .tasks
                .iter()
                .all(|t| t.num_indices_to_merge.is_none() && !t.retrain)
        );
        assert!(SizeTieredPlanner::new(None, 0).is_err());
    }

    #[tokio::test]
    async fn plan_task_and_result_round_trip_through_json() {
        let dir = TempStrDir::default();
        let dataset = indexed_dataset(dir.as_str(), &ivf_pq(), 2).await;
        let planned = plan(&dataset, &delta_merge(Some(1))).await;
        let json = serde_json::to_string(&planned).unwrap();
        assert_eq!(serde_json::from_str::<Plan>(&json).unwrap(), planned);
        let json = serde_json::to_string(task_for(&planned, "id_idx")).unwrap();
        let result = serde_json::from_str::<Task>(&json)
            .unwrap()
            .execute(&dataset)
            .await
            .unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let decoded: IndexOptimizeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(
            (decoded.read_version, &decoded.removed_segments),
            (result.read_version, &result.removed_segments)
        );
        let (a, b) = (
            decoded.new_segment.clone().unwrap(),
            result.new_segment.unwrap(),
        );
        assert!(a.uuid == b.uuid && a.fragment_bitmap == b.fragment_bitmap && a.files == b.files);
        assert_eq!(a.index_details, b.index_details);
        let none = IndexOptimizeResult {
            new_segment: None,
            ..decoded
        };
        let json = serde_json::to_string(&none).unwrap();
        assert!(
            serde_json::from_str::<IndexOptimizeResult>(&json)
                .unwrap()
                .new_segment
                .is_none()
        );
    }
}
