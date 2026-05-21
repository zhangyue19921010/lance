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

use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold::BTreeIndexDetails;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::{ScalarIndex, ScalarIndexParams};
use lance_index::scalar::btree::BTreeIndex;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result,
    index::api::{IndexSegment, IndexSegmentPlan},
};

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
/// Each staged BTree root remains its own physical segment for now; merging
/// multiple staged roots into a single physical segment is exposed through
/// [`merge_segments`] instead.
pub(crate) fn plan_segments(
    segments: &[IndexMetadata],
    target_segment_bytes: Option<u64>,
) -> Result<Vec<IndexSegmentPlan>> {
    if let Some(0) = target_segment_bytes {
        return Err(Error::invalid_input(
            "target_segment_bytes must be greater than zero".to_string(),
        ));
    }
    if target_segment_bytes.is_some() && segments.len() > 1 {
        // TODO: support N:1 grouping here. The natural implementation is to
        // read each source segment's existing data via
        // `BTreeIndex::into_data_stream`, do a k-way sorted merge across
        // them, and feed the merged stream into `train_btree_index` — that
        // avoids a dataset scan and reuses already-sorted page data, which
        // matters when distributed builds produce many small staged
        // segments. Until then, callers that want N:1 must go through
        // `merge_existing_index_segments(...)`, which rebuilds from the
        // dataset.
        return Err(Error::invalid_input(
            "BTree segment builder does not yet support merging multiple source segments; \
             use merge_existing_index_segments(...) instead"
                .to_string(),
        ));
    }

    segments
        .iter()
        .map(|segment| {
            let fragment_bitmap = segment.fragment_bitmap.as_ref().ok_or_else(|| {
                Error::index(format!(
                    "Segment '{}' is missing fragment coverage",
                    segment.uuid
                ))
            })?;
            let index_details = ensure_btree_details(segment)?;
            let built_segment = IndexSegment::new(
                segment.uuid,
                fragment_bitmap.iter(),
                index_details,
                segment.index_version,
            );
            let estimated_bytes = segment
                .files
                .as_ref()
                .map(|files| files.iter().map(|file| file.size_bytes).sum())
                .unwrap_or(0);
            Ok(IndexSegmentPlan::new(
                built_segment,
                vec![segment.clone()],
                estimated_bytes,
                Some(IndexType::BTree),
            ))
        })
        .collect()
}

/// Finalize one staged BTree root into a commit-ready physical segment.
///
/// Per-fragment BTree training already produces self-contained
/// `part_<partition_id>_*.lance` files inside the segment UUID directory and
/// `BTreeIndex::load` knows how to open them via its single-partition
/// fallback, so finalization is a no-op once we have validated the inputs.
pub(crate) async fn build_segment(
    _dataset: &Dataset,
    segment_plan: &IndexSegmentPlan,
) -> Result<IndexSegment> {
    let built_segment = segment_plan.segment().clone();
    let source_segments = segment_plan.segments();
    if source_segments.len() != 1 {
        // TODO: support N:1 grouping here. See the matching TODO in
        // `plan_segments` for the recommended approach (`into_data_stream` +
        // k-way merge + `train_btree_index`).
        return Err(Error::invalid_input(
            "BTree segment builder does not yet support merging multiple source segments; \
             use merge_existing_index_segments(...) instead"
                .to_string(),
        ));
    }
    let source_segment = &source_segments[0];
    if source_segment.uuid != built_segment.uuid() {
        return Err(Error::invalid_input(
            "BTree segment builder requires the built segment UUID to match the staged source UUID"
                .to_string(),
        ));
    }
    ensure_btree_details(source_segment)?;
    Ok(built_segment)
}

/// Merge one caller-defined group of source BTree segments into a single
/// physical segment.
///
/// The merge re-trains a fresh BTree over the union of fragments covered by
/// the inputs by reusing the original training parameters from one of the
/// source segments. The output lives under a new UUID and uses the canonical
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

    let mut fragment_bitmap = RoaringBitmap::new();
    for segment in &segments {
        fragment_bitmap |= segment.fragment_bitmap.as_ref().cloned().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
    }

    // Derive training parameters (e.g. zone_size) from one of the source
    // segments so the merged segment keeps the caller's original tuning.
    let reference_index =
        super::open_scalar_index(dataset, &field_path, &segments[0], &NoOpMetricsCollector).await?;
    let reference_btree = reference_index
        .as_any()
        .downcast_ref::<BTreeIndex>()
        .ok_or_else(|| {
            Error::index(format!(
                "merge_existing_index_segments: expected BTree segment {}, got {:?}",
                segments[0].uuid,
                reference_index.index_type()
            ))
        })?;
    let params: ScalarIndexParams = reference_btree.derive_index_params()?;

    let new_uuid = Uuid::new_v4();
    let union_fragment_ids: Vec<u32> = fragment_bitmap.iter().collect();
    let created_index = super::build_scalar_index(
        dataset,
        &field_path,
        &new_uuid.to_string(),
        &params,
        true,
        Some(union_fragment_ids),
        None,
        Arc::new(NoopIndexBuildProgress),
    )
    .await?;

    // Plugins are required to return BTreeIndexDetails here, but be defensive
    // so the merge surfaces a clear error if that contract is ever broken.
    if !created_index.index_details.type_url.ends_with("BTreeIndexDetails") {
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
