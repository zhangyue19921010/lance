// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_io::object_store::ObjectStoreParams;
use std::collections::HashMap;
use std::env;
use std::time::Instant;

fn perf_s3_config_from_env() -> Option<(String, ObjectStoreParams)> {
    let base_uri = env::var("LANCE_PERF_S3_URI").ok()?;
    let key = env::var("LANCE_PERF_S3_ACCESS_KEY")
        .or_else(|_| env::var("AWS_ACCESS_KEY_ID"))
        .ok()?;
    let secret = env::var("LANCE_PERF_S3_SECRET_KEY")
        .or_else(|_| env::var("AWS_SECRET_ACCESS_KEY"))
        .ok()?;
    let region = env::var("LANCE_PERF_S3_REGION")
        .or_else(|_| env::var("AWS_DEFAULT_REGION"))
        .ok()?;
    let endpoint = env::var("LANCE_PERF_S3_ENDPOINT")
        .or_else(|_| env::var("AWS_ENDPOINT"))
        .ok()?;

    let mut storage_options = HashMap::from([
        ("access_key_id".to_string(), key),
        ("secret_access_key".to_string(), secret),
        ("aws_region".to_string(), region),
        ("aws_endpoint".to_string(), endpoint.clone()),
        (
            "virtual_hosted_style_request".to_string(),
            "true".to_string(),
        ),
    ]);
    if endpoint.starts_with("http://") {
        storage_options.insert("allow_http".to_string(), "true".to_string());
    }

    Some((
        base_uri,
        ObjectStoreParams {
            storage_options: Some(storage_options),
            ..Default::default()
        },
    ))
}

#[tokio::test]
async fn test_binary_copy_merge_small_files() {
    for version in LanceFileVersion::iter_non_legacy() {
        do_test_binary_copy_merge_small_files(version).await;
    }
}

async fn do_test_binary_copy_merge_small_files(version: LanceFileVersion) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let reader2 = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 2_500,
        max_rows_per_group: 1_000,
        data_storage_version: Some(version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    dataset.append(reader2, Some(write_params)).await.unwrap();

    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000_000,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_added >= 1);
    assert_eq!(
        dataset.count_rows(None).await.unwrap() as usize,
        before.num_rows()
    );
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_with_defer_remap() {
    for version in LanceFileVersion::iter_non_legacy() {
        do_test_binary_copy_with_defer_remap(version).await;
    }
}

async fn do_test_binary_copy_with_defer_remap(version: LanceFileVersion) {
    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
    use std::sync::Arc;

    let fixed_list_dt =
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4);

    let meta_fields = Fields::from(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", fixed_list_dt.clone(), true),
    ]);

    let inner_fields = Fields::from(vec![
        Field::new("x", DataType::UInt32, true),
        Field::new("y", DataType::LargeUtf8, true),
    ]);
    let nested_fields = Fields::from(vec![
        Field::new("inner", DataType::Struct(inner_fields.clone()), true),
        Field::new("fsb", DataType::FixedSizeBinary(8), true),
    ]);

    let event_fields = Fields::from(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("payload", DataType::Binary, true),
    ]);

    let reader = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .col("i", array::step::<Int32Type>())
        .col("meta", array::rand_struct(meta_fields))
        .col("nested", array::rand_struct(nested_fields))
        .col(
            "events",
            array::rand_list_any(array::rand_struct(event_fields), true),
        )
        .into_reader_rows(RowCount::from(6_000), BatchCount::from(1));

    let mut dataset = Dataset::write(
        reader,
        "memory://test/binary_copy_nested",
        Some(WriteParams {
            max_rows_per_file: 1_000,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let before_batch = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        defer_index_remap: true,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after_batch = dataset.scan().try_into_batch().await.unwrap();

    assert_eq!(before_batch, after_batch);
}

#[tokio::test]
async fn test_binary_copy_preserves_stable_row_ids() {
    for version in LanceFileVersion::iter_non_legacy() {
        do_binary_copy_preserves_stable_row_ids(version).await;
    }
}

async fn do_binary_copy_preserves_stable_row_ids(version: LanceFileVersion) {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(8).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(4_000),
        format!("memory://test/binary_copy_stable_row_ids_{}", version).as_str(),
        Some(WriteParams {
            enable_stable_row_ids: true,
            data_storage_version: Some(version),
            max_rows_per_file: 500,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vector".into()),
            &params,
            false,
        )
        .await
        .unwrap();

    async fn index_set(dataset: &Dataset) -> HashSet<Uuid> {
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.uuid)
            .collect()
    }
    let indices = index_set(&dataset).await;

    async fn vector_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        let query = Float32Array::from(vec![0.0f32; 8]);
        scanner
            .nearest("vec", &query, 10)
            .unwrap()
            .project(&["i"])
            .unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    async fn scalar_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    let before_vec_result = vector_query(&dataset).await;
    let before_scalar_result = scalar_query(&dataset).await;

    let before_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let current_indices = index_set(&dataset).await;
    assert_eq!(indices, current_indices);

    let after_vec_result = vector_query(&dataset).await;
    assert_eq!(before_vec_result, after_vec_result);

    let after_scalar_result = scalar_query(&dataset).await;
    assert_eq!(before_scalar_result, after_scalar_result);

    let after_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let before_idx = arrow_ord::sort::sort_to_indices(
        before_batch.column_by_name(lance_core::ROW_ID).unwrap(),
        None,
        None,
    )
    .unwrap();
    let after_idx = arrow_ord::sort::sort_to_indices(
        after_batch.column_by_name(lance_core::ROW_ID).unwrap(),
        None,
        None,
    )
    .unwrap();
    let before = arrow::compute::take_record_batch(&before_batch, &before_idx).unwrap();
    let after = arrow::compute::take_record_batch(&after_batch, &after_idx).unwrap();

    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_remaps_unstable_row_ids() {
    for version in LanceFileVersion::iter_non_legacy() {
        do_binary_copy_remaps_unstable_row_ids(version).await;
    }
}

async fn do_binary_copy_remaps_unstable_row_ids(version: LanceFileVersion) {
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(8).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(4_000),
        "memory://test/binary_copy_no_stable",
        Some(WriteParams {
            enable_stable_row_ids: false,
            data_storage_version: Some(version),
            max_rows_per_file: 500,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vector".into()),
            &params,
            false,
        )
        .await
        .unwrap();

    async fn vector_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        let query = Float32Array::from(vec![0.0f32; 8]);
        scanner
            .nearest("vec", &query, 10)
            .unwrap()
            .project(&["i"])
            .unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    async fn scalar_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    let before_vec_result = vector_query(&dataset).await;
    let before_scalar_result = scalar_query(&dataset).await;
    let before_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after_vec_result = vector_query(&dataset).await;
    assert_eq!(before_vec_result, after_vec_result);

    let after_scalar_result = scalar_query(&dataset).await;
    assert_eq!(before_scalar_result, after_scalar_result);

    let after_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before_batch, after_batch);
}

#[tokio::test]
async fn test_binary_copy_preserves_zonemap_queries() {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    let mut data_gen = BatchGenerator::new()
        .col(Box::new(IncrementingInt32::new().named("a".to_owned())))
        .col(Box::new(IncrementingInt32::new().named("b".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(5_000),
        "memory://test/binary_copy_zonemap",
        Some(WriteParams {
            max_rows_per_file: 500,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let zonemap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(
            &["a"],
            IndexType::Scalar,
            Some("zonemap".into()),
            &zonemap_params,
            false,
        )
        .await
        .unwrap();

    let predicate = "a >= 2500 AND b < 4000";
    let before = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_preserves_bloom_filter_queries() {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    let mut data_gen = BatchGenerator::new()
        .col(Box::new(IncrementingInt32::new().named("id".to_owned())))
        .col(Box::new(IncrementingInt32::new().named("val".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(6_000),
        "memory://test/binary_copy_bloom",
        Some(WriteParams {
            max_rows_per_file: 500,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    #[derive(serde::Serialize)]
    struct BloomParams {
        number_of_items: u64,
        probability: f64,
    }
    let bloom_params =
        ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter).with_params(&BloomParams {
            number_of_items: 500,
            probability: 0.01,
        });
    dataset
        .create_index(
            &["val"],
            IndexType::Scalar,
            Some("bloom".into()),
            &bloom_params,
            false,
        )
        .await
        .unwrap();

    let predicate = "val IN (123, 124, 125, 126)";
    let before = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_fallback_to_common_compaction() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 500,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    dataset.delete("a < 100").await.unwrap();

    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        enable_binary_copy: true,
        ..Default::default()
    };

    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(!can_use_binary_copy(&dataset, &options, &frags));

    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
}

#[tokio::test]
async fn test_can_use_binary_copy_schema_consistency_ok() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader1 = RecordBatchIterator::new(vec![Ok(data.slice(0, 5_000))], data.schema());
    let reader2 = RecordBatchIterator::new(vec![Ok(data.slice(5_000, 5_000))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader1, test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    dataset.append(reader2, Some(write_params)).await.unwrap();

    let options = CompactionOptions {
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };
    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(can_use_binary_copy(&dataset, &options, &frags));
}

#[tokio::test]
async fn test_can_use_binary_copy_schema_mismatch() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let options = CompactionOptions {
        enable_binary_copy: true,
        ..Default::default()
    };
    let mut frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    // Introduce a column index mismatch in the first data file
    if let Some(df) = frags.get_mut(0).and_then(|f| f.files.get_mut(0)) {
        if let Some(first) = df.column_indices.get_mut(0) {
            *first = -*first - 1;
        } else {
            df.column_indices.push(-1);
        }
    }
    assert!(!can_use_binary_copy(&dataset, &options, &frags));

    // Also introduce a version mismatch and ensure rejection
    if let Some(df) = frags.get_mut(0).and_then(|f| f.files.get_mut(0)) {
        df.file_minor_version = if df.file_minor_version == 1 { 2 } else { 1 };
    }
    assert!(!can_use_binary_copy(&dataset, &options, &frags));
}

#[tokio::test]
async fn test_can_use_binary_copy_version_mismatch() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 500,
        data_storage_version: Some(LanceFileVersion::V2_0),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Append additional data and then mark its files as a newer format version (v2.1).
    let reader_append = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    dataset.append(reader_append, None).await.unwrap();

    let options = CompactionOptions {
        enable_binary_copy: true,
        ..Default::default()
    };
    let mut frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(
        frags.len() >= 2,
        "expected multiple fragments for version mismatch test"
    );

    // Simulate mixed file versions by marking the second fragment as v2.1.
    let (v21_major, v21_minor) = LanceFileVersion::V2_1.to_numbers();
    for file in &mut frags[1].files {
        file.file_major_version = v21_major;
        file.file_minor_version = v21_minor;
    }

    assert!(!can_use_binary_copy(&dataset, &options, &frags));
}

#[tokio::test]
async fn test_can_use_binary_copy_reject_deletions() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    dataset.delete("a < 10").await.unwrap();

    let options = CompactionOptions {
        enable_binary_copy: true,
        ..Default::default()
    };
    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(!can_use_binary_copy(&dataset, &options, &frags));
}

#[tokio::test]
async fn test_binary_copy_compaction_with_complex_schema() {
    for version in LanceFileVersion::iter_non_legacy() {
        do_test_binary_copy_compaction_with_complex_schema(version).await;
    }
}

async fn do_test_binary_copy_compaction_with_complex_schema(version: LanceFileVersion) {
    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};

    let row_num = 1_000;

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
        .col(
            "utf8",
            array::rand_utf8(lance_datagen::ByteCount::from(16), false),
        )
        .col("large_utf8", array::random_sentence(1, 6, true))
        .col(
            "bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), false),
        )
        .col(
            "large_bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), true),
        )
        .col(
            "varbin",
            array::rand_varbin(
                lance_datagen::ByteCount::from(8),
                lance_datagen::ByteCount::from(32),
            ),
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
        .into_reader_rows(RowCount::from(row_num), BatchCount::from(10));

    let full_dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        reader_full,
        &*full_dir,
        Some(WriteParams {
            enable_stable_row_ids: true,
            data_storage_version: Some(version),
            max_rows_per_file: (row_num / 100) as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let opt_full = CompactionOptions {
        enable_binary_copy: false,
        ..Default::default()
    };
    let opt_binary = CompactionOptions {
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };

    let _ = compact_files(&mut dataset, opt_full, None).await.unwrap();
    let before = dataset.count_rows(None).await.unwrap();
    let batch_before = dataset.scan().try_into_batch().await.unwrap();

    let mut dataset = dataset.checkout_version(1).await.unwrap();

    // rollback and trigger another binary copy compaction
    dataset.restore().await.unwrap();
    let _ = compact_files(&mut dataset, opt_binary, None).await.unwrap();
    let after = dataset.count_rows(None).await.unwrap();
    let batch_after = dataset.scan().try_into_batch().await.unwrap();

    assert_eq!(before, after);
    assert_eq!(batch_before, batch_after);
}

async fn measure_point_and_scan(
    dataset: &Dataset,
    point_value: i64,
) -> ((usize, std::time::Duration), (usize, std::time::Duration)) {
    let filter = format!("i64 = {}", point_value);

    let mut point_scanner = dataset.scan();
    point_scanner.filter(&filter).unwrap();
    point_scanner.project(&["i64"]).unwrap();
    let point_start = Instant::now();
    let point_batch = point_scanner.try_into_batch().await.unwrap();
    let point_elapsed = point_start.elapsed();

    let mut scan_scanner = dataset.scan();
    scan_scanner.project(&["i64"]).unwrap();
    let scan_start = Instant::now();
    let scan_batch = scan_scanner.try_into_batch().await.unwrap();
    let scan_elapsed = scan_start.elapsed();

    (
        (point_batch.num_rows(), point_elapsed),
        (scan_batch.num_rows(), scan_elapsed),
    )
}

#[tokio::test]
async fn test_perf_binary_copy_vs_full() {
    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};

    let row_num = 1_000_000;

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
        .col(
            "utf8",
            array::rand_utf8(lance_datagen::ByteCount::from(16), false),
        )
        .col("large_utf8", array::random_sentence(1, 6, true))
        .col(
            "bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), false),
        )
        .col(
            "large_bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), true),
        )
        .col(
            "varbin",
            array::rand_varbin(
                lance_datagen::ByteCount::from(8),
                lance_datagen::ByteCount::from(32),
            ),
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
        .into_reader_rows(RowCount::from(row_num), BatchCount::from(10));

    let _local_dir = TempStrDir::default();
    let (base_uri, store_params) = perf_s3_config_from_env()
        .map(|(uri, params)| (uri, Some(params)))
        .unwrap_or_else(|| (_local_dir.to_string(), None));
    println!("perf dataset uri: {}", base_uri);

    let mut dataset = Dataset::write(
        reader_full,
        &base_uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            data_storage_version: Some(LanceFileVersion::V2_1),
            max_rows_per_file: (row_num / 100) as usize,
            store_params: store_params.clone(),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let opt_binary = CompactionOptions {
        enable_binary_copy: false,
        ..Default::default()
    };
    let opt_full = CompactionOptions {
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };

    let target_point = (row_num / 2) as i64;
    let before_rows = dataset.count_rows(None).await.unwrap();
    let ((point_before_rows, point_before), (scan_before_rows, scan_before)) =
        measure_point_and_scan(&dataset, target_point).await;

    let t0 = Instant::now();
    let _ = compact_files(&mut dataset, opt_full, None).await.unwrap();
    let d_full = t0.elapsed();
    let ((point_full_rows, point_full), (scan_full_rows, scan_full)) =
        measure_point_and_scan(&dataset, target_point).await;
    let after_full = dataset.count_rows(None).await.unwrap();

    let mut dataset = dataset.checkout_version(1).await.unwrap();
    dataset.restore().await.unwrap();
    let t1 = Instant::now();
    let _ = compact_files(&mut dataset, opt_binary, None).await.unwrap();
    let d_bin = t1.elapsed();
    let ((point_bin_rows, point_bin), (scan_bin_rows, scan_bin)) =
        measure_point_and_scan(&dataset, target_point).await;
    let after_bin = dataset.count_rows(None).await.unwrap();

    println!(
        "perf: full_compaction={:?}, binary_copy={:?}, speedup={:.2}x",
        d_full,
        d_bin,
        (d_full.as_secs_f64() / d_bin.as_secs_f64())
    );

    println!(
        "point query (before/full/bin): {:?} / {:?} / {:?}, scan (before/full/bin): {:?} / {:?} / {:?}, point speedup (full/bin): {:.2}x / {:.2}x, scan speedup (full/bin): {:.2}x / {:.2}x",
        point_before, point_full, point_bin,
        scan_before, scan_full, scan_bin,
        point_before.as_secs_f64() / point_full.as_secs_f64(),
        point_before.as_secs_f64() / point_bin.as_secs_f64(),
        scan_before.as_secs_f64() / scan_full.as_secs_f64(),
        scan_before.as_secs_f64() / scan_bin.as_secs_f64(),
    );

    assert_eq!(point_before_rows, 1);
    assert_eq!(point_full_rows, 1);
    assert_eq!(point_bin_rows, 1);
    assert_eq!(scan_before_rows, before_rows);
    assert_eq!(scan_full_rows, before_rows);
    assert_eq!(scan_bin_rows, before_rows);
    assert_eq!(before_rows, after_full);
    assert_eq!(before_rows, after_bin);
}

#[tokio::test]
async fn do_compact_binary_copy() {
    let dataset = Dataset::open("/home/zhangyue.1010/binarycopytest").await.unwrap();
    let mut dataset = dataset.checkout_version(1).await.unwrap();
    dataset.restore().await.unwrap();

    let opt = CompactionOptions {
        enable_binary_copy: true,
        enable_binary_copy_force: true,
        ..Default::default()
    };

    let _ = compact_files(&mut dataset, opt, None).await.unwrap();
}
