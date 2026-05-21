// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::redundant_pub_crate)]

//! BTree-specific helpers for the segmented index workflow.
//!
//! Each fragment-scoped `execute_uncommitted()` call for a BTree index writes
//! `part_<partition_id>_page_data.lance` and `part_<partition_id>_page_lookup.lance`
//! under its own segment UUID. Those single-partition files are directly
//! loadable through [`crate::index::scalar::open_scalar_index`] (via
//! `BTreeIndex::load`'s single-partition fallback), so the segment-level
//! planner and finalizer here only need to validate input, surface a 1:1 plan,
//! and assemble a commit-ready `IndexSegment` per source.

use std::sync::Arc;

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::ROW_ID;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold::BTreeIndexDetails;
use lance_index::progress::noop_progress;
use lance_index::scalar::btree::BTreeIndex;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result,
    dataset::index::LanceIndexStoreExt,
    index::api::{IndexSegment, IndexSegmentPlan},
};

/// Build an empty update stream for the BTree merge API.
///
/// `BTreeIndex::merge_segments` is shaped as "merge N old segments plus a
/// `new_data` stream", so a pure segment merge still needs a stream with the
/// `(<column>, _rowid)` schema. The stream intentionally contains no batches —
/// it participates in the N+1-way union as an empty input and contributes no
/// rows to the final merged training data.
fn empty_btree_update_stream(
    dataset: &Dataset,
    field_id: i32,
) -> Result<SendableRecordBatchStream> {
    let field = dataset.schema().field_by_id(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "merge_existing_index_segments: field id {} does not exist",
            field_id
        ))
    })?;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(VALUE_COLUMN_NAME, field.data_type(), true),
        ArrowField::new(ROW_ID, arrow_schema::DataType::UInt64, false),
    ]));
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::empty(),
    )))
}

fn ensure_btree_details(segment: &IndexMetadata) -> Result<Arc<prost_types::Any>> {
    let details = segment.index_details.as_ref().ok_or_else(|| {
        Error::index(format!(
            "Segment '{}' is missing index details",
            segment.uuid
        ))
    })?;
    if !details.type_url.ends_with("BTreeIndexDetails") {
        return Err(Error::invalid_input(format!(
            "Segment '{}' is not a BTree segment (details type_url = '{}')",
            segment.uuid, details.type_url
        )));
    }
    Ok(details.clone())
}

/// Plan physical segments for staged BTree-index outputs.
///
/// Without `target_segment_bytes`, each staged BTree root maps 1:1 to its own
/// output segment (no merge work). When `target_segment_bytes` is set and
/// multiple sources are supplied, sources are packed greedily — in input
/// order — into groups whose summed `estimated_bytes` does not exceed the
/// target. Each group of size > 1 will be merged into a single output by
/// [`build_segment`] / [`merge_segments`] (which delegate to
/// [`BTreeIndex::merge_segments`] for a k-way page-data merge — no dataset
/// scan).
pub(crate) fn plan_segments(
    segments: &[IndexMetadata],
    target_segment_bytes: Option<u64>,
) -> Result<Vec<IndexSegmentPlan>> {
    if let Some(0) = target_segment_bytes {
        return Err(Error::invalid_input(
            "target_segment_bytes must be greater than zero".to_string(),
        ));
    }

    // Reject overlapping fragment coverage at planning time so the planner
    // contract matches the commit-time validation in
    // `commit_existing_index_segments`. Catching it here gives callers a
    // clear, build-time error rather than a confusing failure deep inside
    // the commit transaction.
    let mut covered = RoaringBitmap::new();
    for segment in segments {
        let fragment_bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
            Error::index(format!(
                "Segment '{}' is missing fragment coverage",
                segment.uuid
            ))
        })?;
        if covered.intersection_len(fragment_bitmap) > 0 {
            return Err(Error::invalid_input(format!(
                "BTree segment builder received overlapping fragment coverage \
                 (segment '{}')",
                segment.uuid
            )));
        }
        covered |= fragment_bitmap.clone();
    }

    // Group sources by target size. Without a target, every source is its own
    // group of one (matches the original 1:1 contract). With a target,
    // accumulate sources greedily; a source that alone exceeds the target
    // still gets its own group (we never split a source).
    let groups: Vec<Vec<&IndexMetadata>> = match target_segment_bytes {
        None => segments.iter().map(|s| vec![s]).collect(),
        Some(target) => {
            let mut groups: Vec<Vec<&IndexMetadata>> = Vec::new();
            let mut current: Vec<&IndexMetadata> = Vec::new();
            let mut current_bytes: u64 = 0;
            for segment in segments {
                let segment_bytes: u64 = segment
                    .files
                    .as_ref()
                    .map(|files| files.iter().map(|file| file.size_bytes).sum())
                    .unwrap_or(0);
                if !current.is_empty() && current_bytes.saturating_add(segment_bytes) > target {
                    groups.push(std::mem::take(&mut current));
                    current_bytes = 0;
                }
                current.push(segment);
                current_bytes = current_bytes.saturating_add(segment_bytes);
            }
            if !current.is_empty() {
                groups.push(current);
            }
            groups
        }
    };

    groups
        .into_iter()
        .map(|group| {
            debug_assert!(!group.is_empty());
            let mut group_bitmap = RoaringBitmap::new();
            let mut estimated_bytes: u64 = 0;
            for source in &group {
                let fragment_bitmap = source.fragment_bitmap.as_ref().ok_or_else(|| {
                    Error::index(format!(
                        "Segment '{}' is missing fragment coverage",
                        source.uuid
                    ))
                })?;
                group_bitmap |= fragment_bitmap;
                ensure_btree_details(source)?;
                if let Some(files) = source.files.as_ref() {
                    estimated_bytes = estimated_bytes
                        .saturating_add(files.iter().map(|f| f.size_bytes).sum::<u64>());
                }
            }

            // For singleton groups, preserve the source UUID and version so
            // `build_segment` can fast-path validate the existing files.
            // For multi-source groups, allocate a fresh UUID; the real
            // version is decided by `merge_segments` at build time and will
            // overwrite this placeholder in `build_segment`.
            let (placeholder_uuid, version, details) = if group.len() == 1 {
                let only = group[0];
                (only.uuid, only.index_version, ensure_btree_details(only)?)
            } else {
                (
                    Uuid::new_v4(),
                    group[0].index_version,
                    ensure_btree_details(group[0])?,
                )
            };

            let built_segment =
                IndexSegment::new(placeholder_uuid, group_bitmap.iter(), details, version);
            let source_metadata: Vec<IndexMetadata> = group.into_iter().cloned().collect();
            Ok(IndexSegmentPlan::new(
                built_segment,
                source_metadata,
                estimated_bytes,
                Some(IndexType::BTree),
            ))
        })
        .collect()
}

/// Finalize one staged BTree root into a commit-ready physical segment.
///
/// For singleton groups, per-fragment BTree training already produced
/// self-contained `part_<partition_id>_*.lance` files inside the segment UUID
/// directory and `BTreeIndex::load` opens them via its single-partition
/// fallback — so finalization is a no-op once we validate inputs and confirm
/// the staged files exist on disk.
///
/// For multi-source groups, the staged segments are merged into a single new
/// physical segment via [`merge_segments`], which delegates to
/// [`BTreeIndex::merge_segments`] for a k-way merge over already-sorted page
/// data (no dataset scan).
pub(crate) async fn build_segment(
    dataset: &Dataset,
    segment_plan: &IndexSegmentPlan,
) -> Result<IndexSegment> {
    let source_segments = segment_plan.segments();
    if source_segments.is_empty() {
        return Err(Error::invalid_input(
            "BTree segment builder received an empty source group".to_string(),
        ));
    }

    // Multi-source group → k-way merge via the inherent `merge_segments` Layer 2.
    if source_segments.len() > 1 {
        let merged_metadata = merge_segments(dataset, source_segments.to_vec()).await?;
        let fragment_bitmap = merged_metadata.fragment_bitmap.ok_or_else(|| {
            Error::internal(
                "merge_segments returned metadata without fragment coverage".to_string(),
            )
        })?;
        let index_details = merged_metadata.index_details.ok_or_else(|| {
            Error::internal("merge_segments returned metadata without index details".to_string())
        })?;
        return Ok(IndexSegment::new(
            merged_metadata.uuid,
            fragment_bitmap.iter(),
            index_details,
            merged_metadata.index_version,
        ));
    }

    // Singleton group → validate the staged files and pass through.
    let built_segment = segment_plan.segment().clone();
    let source_segment = &source_segments[0];
    if source_segment.uuid != built_segment.uuid() {
        return Err(Error::invalid_input(
            "BTree segment builder requires the built segment UUID to match the staged source UUID"
                .to_string(),
        ));
    }
    ensure_btree_details(source_segment)?;

    // Verify the staged directory actually contains a BTree lookup file
    // (canonical `page_lookup.lance` or a `part_*_page_lookup.lance` written
    // by per-fragment training). Without this, a malformed segment passed
    // through the builder would become commit-ready and only fail later at
    // open/query time.
    let index_dir = dataset.indices_dir().join(source_segment.uuid.to_string());
    let mut list_stream = dataset.object_store.list(Some(index_dir.clone()));
    let mut has_lookup = false;
    while let Some(meta) = futures::StreamExt::next(&mut list_stream).await {
        let file_name = meta?.location.filename().unwrap_or_default().to_string();
        if file_name == "page_lookup.lance"
            || (file_name.starts_with("part_") && file_name.ends_with("_page_lookup.lance"))
        {
            has_lookup = true;
            break;
        }
    }
    if !has_lookup {
        return Err(Error::invalid_input(format!(
            "BTree segment '{}' has no BTree lookup file under {}",
            source_segment.uuid, index_dir
        )));
    }

    Ok(built_segment)
}

/// Merge one caller-defined group of source BTree segments into a single
/// physical segment.
///
/// The merge replays each source segment's already-sorted `(value, _rowid)`
/// page data as one stream and performs an N-way `SortPreservingMerge` via
/// [`BTreeIndex::merge_segments`], then re-trains a fresh BTree over the
/// merged stream. No dataset scan is performed — only the existing index
/// files are read. The output lives under a new UUID and uses the canonical
/// `page_data.lance` / `page_lookup.lance` layout, so the resulting segment
/// is interchangeable with a non-distributed BTree build.
pub(crate) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    for segment in &segments {
        ensure_btree_details(segment)?;
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    // Intersect each segment's stored bitmap with the dataset's current
    // fragments so we don't claim coverage on IDs that compaction or pruning
    // has already retired. The merged segment must only claim live coverage.
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    let mut fragment_bitmap = RoaringBitmap::new();
    let mut source_indices = Vec::with_capacity(segments.len());
    for segment in &segments {
        if segment.fragment_bitmap.is_none() {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            )));
        }
        if let Some(effective) = segment.effective_fragment_bitmap(dataset_fragments) {
            fragment_bitmap |= effective;
        }

        let scalar_index =
            super::open_scalar_index(dataset, &field_path, segment, &NoOpMetricsCollector).await?;
        let btree = scalar_index
            .as_any()
            .downcast_ref::<BTreeIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected BTree segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(Arc::new(btree.clone()));
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid.to_string())?;
    let empty_new_data = empty_btree_update_stream(dataset, field_id)?;
    let created_index = BTreeIndex::merge_segments(
        &source_indices,
        empty_new_data,
        &new_store,
        None,
        noop_progress(),
    )
    .await?;

    // Layer 1 must return BTreeIndexDetails, but be defensive in case the
    // contract is ever broken.
    if !created_index
        .index_details
        .type_url
        .ends_with("BTreeIndexDetails")
    {
        return Err(Error::internal(format!(
            "merge_existing_index_segments: BTree merge produced unexpected details type_url '{}'",
            created_index.index_details.type_url
        )));
    }
    debug_assert_eq!(
        created_index.index_details,
        prost_types::Any::from_msg(&BTreeIndexDetails::default()).unwrap(),
    );

    Ok(IndexMetadata {
        uuid: new_uuid,
        fields: vec![field_id],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: created_index.files,
        ..segments[0].clone()
    })
}
