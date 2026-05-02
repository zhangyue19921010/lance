// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::Arc;

use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::CommitBuilder;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment};
use object_store::path::Path;
use url::Url;

use crate::cli::{
    LanceDatasetLocateDataFileArgs, LanceDatasetRepairManifestArgs, LanceDatasetRestoreArgs,
    LanceDatasetVerifyDataFilesArgs,
};

#[derive(Debug, Clone)]
struct ReferencedDataFile {
    fragment_id: u64,
    path: String,
    data_file: DataFile,
    base_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct LocatedDataFileManifest {
    version: u64,
    data_files: Vec<ReferencedDataFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DataFileStatus {
    Ok,
    Missing(String),
    Corrupt(String),
    Error(String),
}

impl DataFileStatus {
    fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Missing(_) => "missing",
            Self::Corrupt(_) => "corrupt",
            Self::Error(_) => "error",
        }
    }

    fn detail(&self) -> Option<&str> {
        match self {
            Self::Ok => None,
            Self::Missing(s) | Self::Corrupt(s) | Self::Error(s) => Some(s.as_str()),
        }
    }
}

/// I/O buffer budget for reading Lance file metadata in deep verify mode.
const DEEP_VERIFY_IO_BUFFER_SIZE: u64 = 4 * 1024 * 1024;

fn normalize_path(path: &str) -> String {
    let path = Url::parse(path)
        .ok()
        .filter(|url| url.scheme().len() > 1)
        .map(|url| url.path().to_string())
        .unwrap_or_else(|| path.to_string());
    path.trim_start_matches('/').replace('\\', "/")
}

fn parse_storage_options(storage_options: Option<&str>) -> Result<HashMap<String, String>> {
    let parsed = storage_options
        .map(|options| {
            serde_json::from_str::<HashMap<String, String>>(options).map_err(|err| {
                Error::invalid_input(format!(
                    "storage_options must be a JSON object with string keys and string values: {}",
                    err
                ))
            })
        })
        .transpose()?;
    Ok(parsed.unwrap_or_default())
}

async fn open_dataset(source: &str, storage_options: Option<&str>) -> Result<Dataset> {
    let storage_options = parse_storage_options(storage_options)?;
    let mut builder = DatasetBuilder::from_uri(source);
    if !storage_options.is_empty() {
        builder = builder.with_storage_options(storage_options);
    }
    builder.load().await
}

fn path_matches(candidate: &str, query: &str) -> bool {
    let candidate = normalize_path(candidate);
    let query = normalize_path(query);
    candidate == query
        || candidate.ends_with(&format!("/{query}"))
        || query.ends_with(&format!("/{candidate}"))
}

fn error_message_is_not_found(message: &str) -> bool {
    message.contains("not found") || message.contains("No such file or directory")
}

fn source_is_not_found(source: &(dyn std::error::Error + 'static)) -> bool {
    source
        .downcast_ref::<object_store::Error>()
        .is_some_and(|err| matches!(err, object_store::Error::NotFound { .. }))
        || source
            .downcast_ref::<std::io::Error>()
            .is_some_and(|err| err.kind() == std::io::ErrorKind::NotFound)
        || source
            .downcast_ref::<Error>()
            .is_some_and(is_not_found_error)
        || source.source().is_some_and(source_is_not_found)
}

fn is_not_found_error(error: &Error) -> bool {
    match error {
        Error::NotFound { .. } => true,
        Error::Cloned { message, .. } => error_message_is_not_found(message),
        Error::IO { source, .. } | Error::Wrapped { error: source, .. } => {
            source_is_not_found(source.as_ref()) || error_message_is_not_found(&source.to_string())
        }
        _ => false,
    }
}

/// Return all data files referenced by a fragment manifest entry.
async fn fragment_data_file_refs(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Vec<ReferencedDataFile>> {
    let mut data_files = Vec::with_capacity(fragment.files.len());
    for data_file in &fragment.files {
        let (_, path) = dataset.resolve_data_file_location(data_file).await?;
        data_files.push(ReferencedDataFile {
            fragment_id: fragment.id,
            path: path.to_string(),
            data_file: data_file.clone(),
            base_id: data_file.base_id,
        });
    }
    Ok(data_files)
}

/// Return all data files referenced by the checked-out dataset manifest.
async fn dataset_data_file_refs(dataset: &Dataset) -> Result<Vec<ReferencedDataFile>> {
    let mut data_files = Vec::new();
    for fragment in dataset.fragments().iter() {
        data_files.extend(fragment_data_file_refs(dataset, fragment).await?);
    }
    Ok(data_files)
}

/// Match a user-supplied data file path against manifest data file references.
async fn matching_data_files(
    dataset: &Dataset,
    data_file_path: &str,
) -> Result<Vec<ReferencedDataFile>> {
    Ok(dataset_data_file_refs(dataset)
        .await?
        .into_iter()
        .filter(|candidate| path_matches(&candidate.path, data_file_path))
        .collect())
}

fn format_data_file(data_file: &ReferencedDataFile) -> String {
    format!(
        "fragment={} path={} base_id={}",
        data_file.fragment_id,
        data_file.path,
        data_file
            .base_id
            .map(|base_id| base_id.to_string())
            .unwrap_or_else(|| "default".to_string())
    )
}

/// Verify that a data file exists and, in deep mode, has readable Lance metadata.
async fn verify_data_file_ref(
    dataset: &Dataset,
    data_file: &ReferencedDataFile,
    deep: bool,
) -> DataFileStatus {
    let (store, path) = match dataset
        .resolve_data_file_location(&data_file.data_file)
        .await
    {
        Ok(store_and_path) => store_and_path,
        Err(err) => return DataFileStatus::Error(err.to_string()),
    };

    if deep {
        let scan_scheduler =
            ScanScheduler::new(store, SchedulerConfig::new(DEEP_VERIFY_IO_BUFFER_SIZE));
        match scan_scheduler
            .open_file(&path, &data_file.data_file.file_size_bytes)
            .await
        {
            Ok(file_scheduler) => match FileReader::read_all_metadata(&file_scheduler).await {
                Ok(_) => DataFileStatus::Ok,
                Err(err) if is_not_found_error(&err) => DataFileStatus::Missing(path.to_string()),
                Err(err) => DataFileStatus::Corrupt(err.to_string()),
            },
            Err(err) if is_not_found_error(&err) => DataFileStatus::Missing(path.to_string()),
            Err(err) => DataFileStatus::Error(err.to_string()),
        }
    } else {
        match store.exists(&path).await {
            Ok(true) => DataFileStatus::Ok,
            Ok(false) => DataFileStatus::Missing(path.to_string()),
            Err(err) => DataFileStatus::Error(err.to_string()),
        }
    }
}

/// Locate the first manifest in the latest reverse scan run that references a data file.
async fn locate_first_data_file_manifest(
    dataset: &Dataset,
    data_file_path: &str,
    max_version: Option<u64>,
) -> Result<Option<LocatedDataFileManifest>> {
    let mut versions = dataset.versions().await?;
    if let Some(max_version) = max_version {
        versions.retain(|version| version.version <= max_version);
    }
    versions.sort_by(|left, right| right.version.cmp(&left.version));

    let mut first_manifest = None;
    for version in versions {
        let dataset = dataset.checkout_version(version.version).await?;
        let data_files = matching_data_files(&dataset, data_file_path).await?;
        if data_files.is_empty() {
            if first_manifest.is_some() {
                break;
            }
            continue;
        }
        first_manifest = Some(LocatedDataFileManifest {
            version: version.version,
            data_files,
        });
    }
    Ok(first_manifest)
}

/// Locate the first manifest in reverse scan order and optionally restore before it.
pub(crate) async fn locate_data_file(
    mut writer: impl Write,
    args: &LanceDatasetLocateDataFileArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.source, args.storage_options.as_deref()).await?;
    let located = locate_first_data_file_manifest(&dataset, &args.data_file, args.version).await?;
    let Some(located) = located else {
        writeln!(writer, "data_file: {}", args.data_file)?;
        writeln!(writer, "first_manifest_version: none")?;
        return Ok(());
    };

    writeln!(writer, "data_file: {}", args.data_file)?;
    writeln!(writer, "first_manifest_version: {}", located.version)?;
    writeln!(writer, "matched_data_files: {}", located.data_files.len())?;
    writeln!(
        writer,
        "affected_fragments: {}",
        located
            .data_files
            .iter()
            .map(|data_file| data_file.fragment_id)
            .collect::<BTreeSet<_>>()
            .len()
    )?;
    if located.version > 1 {
        writeln!(writer, "rollback_version: {}", located.version - 1)?;
    } else {
        writeln!(writer, "rollback_version: none")?;
    }

    if args.restore {
        if located.version == 1 {
            return Err(Error::invalid_input(format!(
                "data file {} is first referenced in version 1; there is no previous version to restore",
                args.data_file
            )));
        }
        let restore_version = located.version - 1;
        let mut restore_dataset = dataset.checkout_version(restore_version).await?;
        restore_dataset.restore().await?;
        writeln!(
            writer,
            "restored_version: {} new_latest_version: {}",
            restore_version,
            restore_dataset.version_id()
        )?;
    }

    Ok(())
}

/// Verify all data files referenced by a dataset manifest.
pub(crate) async fn verify_data_files(
    mut writer: impl Write,
    args: &LanceDatasetVerifyDataFilesArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.source, args.storage_options.as_deref()).await?;
    let dataset = match args.version {
        Some(version) => dataset.checkout_version(version).await?,
        None => dataset,
    };
    let data_files = dataset_data_file_refs(&dataset).await?;
    let checked_data_files = data_files.len();
    let mut bad_data_files = Vec::new();

    for data_file in data_files {
        let status = verify_data_file_ref(&dataset, &data_file, args.deep).await;
        if !status.is_ok() {
            bad_data_files.push((data_file, status));
        }
    }

    writeln!(writer, "version: {}", dataset.version_id())?;
    writeln!(writer, "checked_data_files: {}", checked_data_files)?;
    writeln!(writer, "bad_data_files: {}", bad_data_files.len())?;
    for (data_file, status) in &bad_data_files {
        writeln!(
            writer,
            "{} {} detail={}",
            status.label(),
            format_data_file(data_file),
            status.detail().unwrap_or("")
        )?;
    }

    if args.fail && !bad_data_files.is_empty() {
        return Err(Error::corrupt_file(
            Path::from(args.source.as_str()),
            format!(
                "found {} missing or corrupt data files",
                bad_data_files.len()
            ),
        ));
    }

    Ok(())
}

/// Commit a new manifest version with the requested repair actions applied.
///
/// Currently supported repairs:
/// - `--remove-data-files`: remove fragments whose manifest entries reference
///   any of the provided data file paths or suffixes.
pub(crate) async fn repair_manifest(
    mut writer: impl Write,
    args: &LanceDatasetRepairManifestArgs,
) -> Result<()> {
    let dataset = open_dataset(&args.source, args.storage_options.as_deref()).await?;
    let mut affected_fragment_ids = BTreeSet::new();
    let mut matched_data_file_count = 0;

    for data_file_path in &args.remove_data_files {
        let matches = matching_data_files(&dataset, data_file_path).await?;
        if matches.is_empty() {
            return Err(Error::invalid_input(format!(
                "data file {} was not found in dataset version {}",
                data_file_path,
                dataset.version_id()
            )));
        }
        for matched in &matches {
            affected_fragment_ids.insert(matched.fragment_id);
        }
        matched_data_file_count += matches.len();
    }

    writeln!(writer, "read_version: {}", dataset.version_id())?;
    writeln!(
        writer,
        "remove_data_files: {}",
        args.remove_data_files.len()
    )?;
    writeln!(writer, "matched_data_files: {}", matched_data_file_count)?;
    writeln!(
        writer,
        "affected_fragments: {}",
        affected_fragment_ids.len()
    )?;

    if args.dry_run {
        writeln!(writer, "dry_run: true")?;
        return Ok(());
    }

    let operation = Operation::Delete {
        updated_fragments: vec![],
        deleted_fragment_ids: affected_fragment_ids.iter().copied().collect(),
        predicate: format!(
            "repair: removed fragments containing {}",
            args.remove_data_files.join(", ")
        ),
    };
    let transaction = Transaction::new(dataset.version_id(), operation, None);
    let committed = CommitBuilder::new(Arc::new(dataset))
        .execute(transaction)
        .await?;
    writeln!(writer, "new_version: {}", committed.version_id())?;
    Ok(())
}

/// Restore a previous dataset version as the latest version.
pub(crate) async fn restore(mut writer: impl Write, args: &LanceDatasetRestoreArgs) -> Result<()> {
    let dataset = open_dataset(&args.source, args.storage_options.as_deref()).await?;
    let mut restore_dataset = dataset.checkout_version(args.version).await?;
    restore_dataset.restore().await?;
    writeln!(
        writer,
        "restored_version: {} new_latest_version: {}",
        args.version,
        restore_dataset.version_id()
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance::dataset::{WriteMode, WriteParams};
    use lance_table::format::BasePath;
    use tempfile::TempDir;

    fn test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]))
    }

    fn test_batch(values: Vec<i32>, schema: Arc<ArrowSchema>) -> RecordBatch {
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    async fn write_two_version_dataset() -> (TempDir, String, Dataset) {
        let temp_dir = tempfile::tempdir().unwrap();
        let dataset_uri = temp_dir.path().to_str().unwrap().to_string();
        let schema = test_schema();
        let batch = test_batch(vec![1, 2, 3], schema.clone());
        let reader = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, &dataset_uri, Some(WriteParams::default()))
            .await
            .unwrap();

        let batch = test_batch(vec![4, 5], schema.clone());
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        dataset.append(reader, None).await.unwrap();

        (temp_dir, dataset_uri, dataset)
    }

    async fn data_file_for_fragment(dataset: &Dataset, fragment_id: u64) -> ReferencedDataFile {
        dataset_data_file_refs(dataset)
            .await
            .unwrap()
            .into_iter()
            .find(|data_file| data_file.fragment_id == fragment_id)
            .unwrap()
    }

    fn output_string(output: Vec<u8>) -> String {
        String::from_utf8(output).unwrap()
    }

    #[tokio::test]
    async fn test_locate_data_file_finds_latest_first_manifest_and_restores() {
        let (_temp_dir, dataset_uri, dataset) = write_two_version_dataset().await;
        let data_file = data_file_for_fragment(&dataset, 1).await;
        let reintroduced_fragment = dataset
            .fragments()
            .iter()
            .find(|fragment| fragment.id == data_file.fragment_id)
            .unwrap()
            .clone();

        repair_manifest(
            Vec::new(),
            &LanceDatasetRepairManifestArgs {
                source: dataset_uri.clone(),
                storage_options: None,
                remove_data_files: vec![data_file.path.clone()],
                dry_run: false,
            },
        )
        .await
        .unwrap();

        let repaired = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(repaired.version_id(), 3);
        assert!(
            matching_data_files(&repaired, &data_file.path)
                .await
                .unwrap()
                .is_empty()
        );

        let transaction = Transaction::new(
            repaired.version_id(),
            Operation::Append {
                fragments: vec![reintroduced_fragment],
            },
            None,
        );
        CommitBuilder::new(Arc::new(repaired))
            .execute(transaction)
            .await
            .unwrap();

        let mut output = Vec::new();
        locate_data_file(
            &mut output,
            &LanceDatasetLocateDataFileArgs {
                source: dataset_uri.clone(),
                storage_options: None,
                data_file: data_file.path.clone(),
                version: None,
                restore: true,
            },
        )
        .await
        .unwrap();
        let output = output_string(output);
        assert!(output.contains("first_manifest_version: 4"));
        assert!(output.contains("matched_data_files: 1"));
        assert!(output.contains("affected_fragments: 1"));
        assert!(output.contains("rollback_version: 3"));
        assert!(output.contains("restored_version: 3 new_latest_version: 5"));
        assert!(!output.contains("fragment="));
        assert!(!output.contains("version=2"));

        let restored = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(restored.version_id(), 5);
        assert_eq!(restored.count_rows(None).await.unwrap(), 3);
        assert!(
            matching_data_files(&restored, &data_file.path)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_repair_manifest_removes_data_file_fragment_and_commits() {
        let (_temp_dir, dataset_uri, dataset) = write_two_version_dataset().await;
        let data_file = data_file_for_fragment(&dataset, 1).await;
        let mut output = Vec::new();

        repair_manifest(
            &mut output,
            &LanceDatasetRepairManifestArgs {
                source: dataset_uri.clone(),
                storage_options: None,
                remove_data_files: vec![data_file.path.clone()],
                dry_run: false,
            },
        )
        .await
        .unwrap();

        let output = output_string(output);
        assert!(output.contains("new_version: 3"));
        assert!(output.contains("remove_data_files: 1"));
        assert!(output.contains("matched_data_files: 1"));
        assert!(output.contains("affected_fragments: 1"));
        assert!(!output.contains("input="));

        let repaired = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(repaired.version_id(), 3);
        assert_eq!(repaired.count_rows(None).await.unwrap(), 3);
        assert!(
            matching_data_files(&repaired, &data_file.path)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(dataset_data_file_refs(&repaired).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_verify_data_files_reports_missing_file_and_fail_errors() {
        let (_temp_dir, dataset_uri, dataset) = write_two_version_dataset().await;
        let data_file = data_file_for_fragment(&dataset, 1).await;
        let (store, path) = dataset
            .resolve_data_file_location(&data_file.data_file)
            .await
            .unwrap();
        store.delete(&path).await.unwrap();

        let mut output = Vec::new();
        verify_data_files(
            &mut output,
            &LanceDatasetVerifyDataFilesArgs {
                source: dataset_uri.clone(),
                storage_options: None,
                version: None,
                deep: true,
                fail: false,
            },
        )
        .await
        .unwrap();
        let output = output_string(output);
        assert!(output.contains("checked_data_files: 2"));
        assert!(output.contains("missing fragment=1"));
        assert!(output.contains("bad_data_files: 1"));

        let err = verify_data_files(
            Vec::new(),
            &LanceDatasetVerifyDataFilesArgs {
                source: dataset_uri,
                storage_options: None,
                version: None,
                deep: false,
                fail: true,
            },
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("found 1 missing or corrupt data files")
        );
    }
}
