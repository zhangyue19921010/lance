// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ngram::NGramIndex;
use lance_index::scalar::{CreatedIndex, IndexStore, OldIndexDataFilter};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Open the given segments and collect the stores that back their canonical
/// posting files.
///
/// The NGram merge primitive reads each segment's `ngram_postings.lance`
/// directly, so we only need the backing stores — not the full (potentially
/// large `tokens` map) index objects.
async fn open_ngram_segment_stores(
    dataset: &Dataset,
    field_path: &str,
    segments: &[IndexMetadata],
) -> Result<Vec<Arc<dyn IndexStore>>> {
    let mut stores: Vec<Arc<dyn IndexStore>> = Vec::with_capacity(segments.len());
    for segment in segments {
        let scalar_index =
            super::open_scalar_index(dataset, field_path, segment, &NoOpMetricsCollector).await?;
        let ngram_index = scalar_index
            .as_any()
            .downcast_ref::<NGramIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected ngram segment {}, got {:?}",
                    segment.uuid,
                    scalar_index.index_type()
                ))
            })?;
        stores.push(ngram_index.store().clone());
    }
    Ok(stores)
}

/// Merge one caller-defined group of source NGram segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    // All source segments must belong to the same column.
    let reference_fields = segments[0].fields.as_slice();
    for segment in segments.iter().skip(1) {
        if segment.fields.as_slice() != reference_fields {
            return Err(Error::invalid_input(format!(
                "NGram merge_segments: segment {} has fields {:?}, expected {:?}",
                segment.uuid, segment.fields, reference_fields,
            )));
        }
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    // Drop fragments that compaction/deletion retired: the merged segment must
    // cover only live fragments, and its posting lists must not carry rows from
    // retired ones (mirroring the btree merge behavior). The merged segment is
    // committed at the post-compaction version, so frag-reuse remapping is not
    // applied to it at read time — stale rows would otherwise surface.
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    let mut effective_old_frags = RoaringBitmap::new();
    let mut deleted_old_frags = RoaringBitmap::new();
    for segment in &segments {
        if segment.fragment_bitmap.is_none() {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            )));
        }
        if let Some(effective) = segment.effective_fragment_bitmap(dataset_fragments) {
            effective_old_frags |= effective;
        }
        if let Some(deleted) = segment.deleted_fragment_bitmap(dataset_fragments) {
            deleted_old_frags |= deleted;
        }
    }

    let fragment_bitmap = effective_old_frags.clone();
    let old_data_filter = crate::index::append::build_old_data_filter(
        dataset,
        &effective_old_frags,
        &deleted_old_frags,
    )
    .await?;

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    // Pure segment consolidation: no dataset scan, so there is no new data and
    // the merge is driven entirely by the source posting lists.
    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    let created_index = open_and_merge_segments(
        dataset,
        &field_path,
        &segment_refs,
        None,
        &new_store,
        old_data_filter,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        name: segments[0].name.clone(),
        fields: vec![field_id],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(created_index.files),
    })
}

/// Merge the given NGram segments with optionally newly appended data into a
/// single canonical segment, used by the optimize/append decision tree.
///
/// Pass `None` for `new_data` to do a pure consolidation of the source segments.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    field_path: &str,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filter: Option<OldIndexDataFilter>,
) -> Result<CreatedIndex> {
    let segments = segments.iter().map(|&s| s.clone()).collect::<Vec<_>>();
    let segment_stores = open_ngram_segment_stores(dataset, field_path, &segments).await?;
    NGramIndex::merge_segments(&segment_stores, new_data, new_store, old_data_filter).await
}
