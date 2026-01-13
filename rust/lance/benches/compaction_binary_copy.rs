// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::optimize::{compact_files, CompactionOptions};
use lance::dataset::{Dataset, WriteParams};
use lance_datagen::{array, gen_batch, BatchCount, ByteCount, Dimension, RowCount};

async fn create_dataset(uri: &str, row_num: usize) -> Dataset {
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

    let reader_full = gen_batch()
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
        .col("utf8", array::rand_utf8(ByteCount::from(16), false))
        .col("large_utf8", array::random_sentence(1, 6, true))
        .col("bin", array::rand_fixedbin(ByteCount::from(24), false))
        .col("large_bin", array::rand_fixedbin(ByteCount::from(24), true))
        .col(
            "varbin",
            array::rand_varbin(ByteCount::from(8), ByteCount::from(32)),
        )
        .col("fsb16", array::rand_fsb(16))
        .col(
            "fsl4",
            array::cycle_vec(array::rand::<Float32Type>(), Dimension::from(4)),
        )
        .col("struct_simple", array::rand_struct(inner_fields.clone()))
        .col("struct_nested", array::rand_struct(nested_fields))
        .col(
            "events",
            array::rand_list_any(array::rand_struct(event_fields.clone()), true),
        )
        .into_reader_rows(RowCount::from(row_num as u64), BatchCount::from(10));

    Dataset::write(
        reader_full,
        uri,
        Some(WriteParams {
            max_rows_per_file: 1_000,
            max_rows_per_group: 1_024,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn bench_compaction_binary_copy(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let row_num = 100_000usize;

    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("dataset");
    let uri = uri.to_str().unwrap().to_string();
    let dataset = rt.block_on(create_dataset(&uri, row_num));
    let base_version = dataset.manifest.version;

    c.bench_function("compaction/common_complex_schema", |b| {
        let dataset = dataset.clone();
        b.to_async(&rt).iter(|| async {
            let mut ds = dataset.checkout_version(base_version).await.unwrap();
            ds.restore().await.unwrap();
            let options = CompactionOptions {
                target_rows_per_fragment: 1_000_000,
                enable_binary_copy: false,
                ..Default::default()
            };
            compact_files(&mut ds, options, None).await.unwrap();
        })
    });

    c.bench_function("compaction/binary_copy_complex_schema", |b| {
        let dataset = dataset.clone();
        b.to_async(&rt).iter(|| async {
            let mut ds = dataset.checkout_version(base_version).await.unwrap();
            ds.restore().await.unwrap();
            let options = CompactionOptions {
                target_rows_per_fragment: 1_000_000,
                enable_binary_copy: true,
                enable_binary_copy_force: true,
                ..Default::default()
            };
            compact_files(&mut ds, options, None).await.unwrap();
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(5);
    targets = bench_compaction_binary_copy
);
criterion_main!(benches);
