// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance_core::utils::tempfile::TempStrDir;
use lance::dataset::{optimize::CompactionOptions, Dataset, WriteParams};
use lance_file::version::LanceFileVersion;

fn make_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
    ]));
    let ints = Int64Array::from_iter_values(0..rows as i64);
    let floats = Float32Array::from_iter_values((0..rows).map(|i| i as f32 * std::f32::consts::PI));
    RecordBatch::try_new(schema, vec![Arc::new(ints), Arc::new(floats)]).unwrap()
}

async fn create_dataset_fs(version: LanceFileVersion, num_frags: usize, rows_per_frag: usize) -> (Dataset, TempStrDir) {
    let batches = (0..num_frags)
        .map(|_| Ok(make_batch(rows_per_frag)))
        .collect::<Vec<_>>();
    let reader = RecordBatchIterator::new(batches, make_batch(0).schema());
    let dir = TempStrDir::default();
    let ds = Dataset::write(
        reader,
        &*dir,
        Some(WriteParams {
            data_storage_version: Some(version),
            max_rows_per_file: rows_per_frag,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    (ds, dir)
}

fn bench_compaction_binary_copy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Use enough small fragments to trigger compaction
    let num_frags = 200;
    let rows_per_frag = 1_000;

    for &version in &[LanceFileVersion::V2_0, LanceFileVersion::V2_1, LanceFileVersion::V2_2] {
        let title_prefix = match version {
            LanceFileVersion::V2_0 => "V2_0",
            LanceFileVersion::V2_1 => "V2_1",
            LanceFileVersion::V2_2 => "V2_2",
            _ => "Other",
        };

        // Baseline: full read/write compaction
        c.bench_function(&format!("{title_prefix} compaction_full"), |b| {
            b.to_async(&rt).iter(|| async {
                let (mut dataset, _dir) = create_dataset_fs(version, num_frags, rows_per_frag).await;
                let options = CompactionOptions {
                    target_rows_per_fragment: 10_000,
                    enable_binary_copy: false,
                    ..Default::default()
                };
                let _ = lance::dataset::optimize::compact_files(&mut dataset, options, None)
                    .await
                    .unwrap();
            })
        });

        // Binary copy compaction
        c.bench_function(&format!("{title_prefix} compaction_binary_copy"), |b| {
            b.to_async(&rt).iter(|| async {
                let (mut dataset, _dir) = create_dataset_fs(version, num_frags, rows_per_frag).await;
                let options = CompactionOptions {
                    target_rows_per_fragment: 10_000,
                    enable_binary_copy: true,
                    ..Default::default()
                };
                let _ = lance::dataset::optimize::compact_files(&mut dataset, options, None)
                    .await
                    .unwrap();
            })
        });
    }
}

criterion_group!(benches, bench_compaction_binary_copy);
criterion_main!(benches);
