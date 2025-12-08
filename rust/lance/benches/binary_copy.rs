// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::{optimize::CompactionOptions, Dataset, WriteParams};
use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
use tempfile::TempDir;

const ROW_NUM: usize = 5_000_000;

fn bench_binary_copy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let temp = rt.block_on(prepare_dataset_on_disk(ROW_NUM));
    let dataset_path = temp.path().join("binary-copy-bench.lance");
    let dataset = rt.block_on(async { Dataset::open(dataset_path.to_str().unwrap()).await.unwrap() });
    let dataset = Arc::new(dataset);

    let mut group = c.benchmark_group("binary_copy_compaction");
    group.sample_size(1);
    group.measurement_time(Duration::from_secs(600));

    group.bench_function("full_compaction", |b| {
        let dataset = dataset.clone();
        b.to_async(&rt).iter(move || {
            let dataset = dataset.clone();
            async move {
                let mut ds = dataset.checkout_version(1).await.unwrap();
                ds.restore().await.unwrap();
                let options = CompactionOptions { enable_binary_copy: false, ..Default::default() };
                let _metrics = lance::dataset::optimize::compact_files(&mut ds, options, None)
                    .await
                    .unwrap();
            }
        });
    });

    group.bench_function("binary_copy_compaction", |b| {
        let dataset = dataset.clone();
        b.to_async(&rt).iter(move || {
            let dataset = dataset.clone();
            async move {
                let mut ds = dataset.checkout_version(1).await.unwrap();
                ds.restore().await.unwrap();
                let options = CompactionOptions { enable_binary_copy: true, ..Default::default() };
                let _metrics = lance::dataset::optimize::compact_files(&mut ds, options, None)
                    .await
                    .unwrap();
            }
        });
    });

    group.finish();
}

async fn prepare_dataset_on_disk(row_num: usize) -> TempDir {
    let inner_fields = Fields::from(vec![
        Field::new("x", DataType::UInt32, true),
        Field::new("y", DataType::LargeUtf8, true),
    ]);
    let nested_fields = Fields::from(vec![
        Field::new("inner", DataType::Struct(inner_fields.clone()), true),
        Field::new("fsb", DataType::FixedSizeBinary(16), true),
        Field::new("bin", DataType::Binary, true),
    ]);
    let event_fields = Fields::from(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("payload", DataType::Binary, true),
    ]);

    let reader = gen_batch()
        .col("vec1", array::rand_vec::<Float32Type>(Dimension::from(12)))
        .col("vec2", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .col("i32", array::step::<Int32Type>())
        .col("i64", array::step::<Int64Type>())
        .col("f32", array::rand::<Float32Type>())
        .col("f64", array::rand::<Float64Type>())
        .col("bool", array::rand_boolean())
        .col("date32", array::rand_date32())
        .col("date64", array::rand_date64())
        .col(
            "ts_ms",
            array::rand_timestamp(&DataType::Timestamp(TimeUnit::Millisecond, None)),
        )
        .col("utf8", array::rand_utf8(lance_datagen::ByteCount::from(16), false))
        .col("large_utf8", array::random_sentence(1, 6, true))
        .col("bin", array::rand_fixedbin(lance_datagen::ByteCount::from(24), false))
        .col("large_bin", array::rand_fixedbin(lance_datagen::ByteCount::from(24), true))
        .col(
            "varbin",
            array::rand_varbin(
                lance_datagen::ByteCount::from(8),
                lance_datagen::ByteCount::from(32),
            ),
        )
        .col("fsb16", array::rand_fsb(16))
        .col("struct_simple", array::rand_struct(inner_fields.clone()))
        .col("struct_nested", array::rand_struct(nested_fields))
        .col(
            "events",
            array::rand_list_any(array::rand_struct(event_fields.clone()), true),
        )
        .into_reader_rows(RowCount::from(row_num as u64), BatchCount::from(10));

    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("binary-copy-bench.lance");
    let uri = path.to_str().unwrap();

    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: (row_num / 100) as usize,
            ..Default::default()
        }),
    )
    .await
    .expect("failed to write dataset");

    tmp
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_binary_copy);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_binary_copy);
criterion_main!(benches);
