// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use datafusion::physical_plan::SendableRecordBatchStream;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::ngram::NGramIndex;
use lance_index::scalar::{CreatedIndex, IndexStore, OldIndexDataFilter};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::{
    Dataset, Error, Result, dataset::index::LanceIndexStoreExt, index::DatasetIndexInternalExt,
};

async fn collect_ngram_segment_stores(
    dataset: &Dataset,
    segments: &[IndexMetadata],
) -> Result<Vec<Arc<dyn IndexStore>>> {
    let mut stores: Vec<Arc<dyn IndexStore>> = Vec::with_capacity(segments.len());
    for segment in segments {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        stores.push(Arc::new(store));
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

    let (fragment_bitmap, old_data_filters) =
        crate::index::append::effective_coverage_and_filters(dataset, &segments).await?;

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    // Pure consolidation: no new data, driven entirely by the source posting lists.
    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    let created_index =
        open_and_merge_segments(dataset, &segment_refs, None, &new_store, &old_data_filters).await?;

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

/// Merge the given NGram segments with optional newly appended data into a single
/// canonical segment. Pass `None` for `new_data` for a pure consolidation.
pub(in crate::index) async fn open_and_merge_segments(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
    new_data: Option<SendableRecordBatchStream>,
    new_store: &LanceIndexStore,
    old_data_filters: &[Option<OldIndexDataFilter>],
) -> Result<CreatedIndex> {
    let segments = segments.iter().map(|&s| s.clone()).collect::<Vec<_>>();
    let segment_stores = collect_ngram_segment_stores(dataset, &segments).await?;
    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    NGramIndex::merge_segments(
        &segment_stores,
        new_data,
        new_store,
        old_data_filters,
        frag_reuse_index,
    )
    .await
}
