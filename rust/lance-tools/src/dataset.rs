// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::Arc;

use lance::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::write::CommitBuilder;
use lance_core::{Error, Result};
use lance_file::reader::FileReader;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
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
    relative_path: String,
    base_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct DataFileMatch {
    version: u64,
    data_file: ReferencedDataFile,
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

    fn message(&self) -> Option<&str> {
        match self {
            Self::Ok => None,
            Self::Missing(message) | Self::Corrupt(message) | Self::Error(message) => {
                Some(message.as_str())
            }
        }
    }
}

fn normalize_path(path: &str) -> String {
    let path = Url::parse(path)
        .ok()
        .filter(|url| url.scheme().len() > 1)
        .map(|url| url.path().to_string())
        .unwrap_or_else(|| path.to_string());
    path.trim_start_matches('/').replace('\\', "/")
}

fn path_matches(candidate: &str, query: &str) -> bool {
    let candidate = normalize_path(candidate);
    let query = normalize_path(query);
    candidate == query
        || candidate.ends_with(&format!("/{query}"))
        || query.ends_with(&format!("/{candidate}"))
}

fn data_file_relative_path(data_file: &DataFile) -> String {
    format!("data/{}", data_file.path)
}

/// Return all data files referenced by a fragment manifest entry.
fn fragment_data_file_refs(fragment: &Fragment) -> Vec<ReferencedDataFile> {
    fragment
        .files
        .iter()
        .map(|data_file| ReferencedDataFile {
            fragment_id: fragment.id,
            relative_path: data_file_relative_path(data_file),
            base_id: data_file.base_id,
        })
        .collect()
}

/// Return all data files referenced by the checked-out dataset manifest.
fn dataset_data_file_refs(dataset: &Dataset) -> Vec<ReferencedDataFile> {
    dataset
        .manifest
        .fragments
        .iter()
        .flat_map(fragment_data_file_refs)
        .collect()
}

/// Match a user-supplied data file path against manifest data file references.
///
/// Operators often copy a full object-store key from an error message. This
/// intentionally accepts the full key, `data/foo.lance`, or the bare file name.
fn matching_data_files(dataset: &Dataset, data_file_path: &str) -> Vec<ReferencedDataFile> {
    dataset_data_file_refs(dataset)
        .into_iter()
        .filter(|candidate| {
            path_matches(&candidate.relative_path, data_file_path)
                || candidate
                    .relative_path
                    .rsplit_once('/')
                    .map(|(_, filename)| path_matches(filename, data_file_path))
                    .unwrap_or(false)
        })
        .collect()
}

async fn checkout_dataset_version(dataset: &Dataset, version: Option<u64>) -> Result<Dataset> {
    match version {
        Some(version) => dataset.checkout_version(version).await,
        None => Ok(dataset.clone()),
    }
}

fn format_data_file(data_file: &ReferencedDataFile) -> String {
    format!(
        "fragment={} path={} base_id={}",
        data_file.fragment_id,
        data_file.relative_path,
        data_file
            .base_id
            .map(|base_id| base_id.to_string())
            .unwrap_or_else(|| "default".to_string())
    )
}

/// Resolve the object store and object-store path for a manifest data file.
async fn object_store_for_data_file(
    dataset: &Dataset,
    data_file: &ReferencedDataFile,
) -> Result<(Arc<ObjectStore>, Path)> {
    let Some(base_id) = data_file.base_id else {
        let object_path = dataset
            .data_dir()
            .child(data_file.relative_path.trim_start_matches("data/"));
        return Ok((Arc::new(dataset.object_store().clone()), object_path));
    };

    let base_path = dataset.manifest.base_paths.get(&base_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "base_id {} for data file {} was not found in manifest base_paths",
            base_id, data_file.relative_path
        ))
    })?;
    let registry = Arc::new(ObjectStoreRegistry::default());
    let store_params = ObjectStoreParams::default();
    let (store, base_object_path) =
        ObjectStore::from_uri_and_params(registry, &base_path.path, &store_params).await?;
    let object_path = if base_path.is_dataset_root {
        base_object_path
            .child("data")
            .child(data_file.relative_path.trim_start_matches("data/"))
    } else {
        base_object_path.child(data_file.relative_path.trim_start_matches("data/"))
    };
    Ok((store, object_path))
}

/// Verify that a data file exists and, in deep mode, has readable Lance metadata.
async fn verify_data_file_ref(
    dataset: &Dataset,
    data_file: &ReferencedDataFile,
    deep: bool,
) -> DataFileStatus {
    let (store, path) = match object_store_for_data_file(dataset, data_file).await {
        Ok(store_and_path) => store_and_path,
        Err(err) => return DataFileStatus::Error(err.to_string()),
    };

    match store.inner.head(&path).await {
        Ok(meta) => {
            if deep {
                let scan_scheduler =
                    ScanScheduler::new(store, SchedulerConfig::new(2 * 1024 * 1024 * 1024));
                let file_size = CachedFileSize::new(meta.size);
                match scan_scheduler.open_file(&path, &file_size).await {
                    Ok(file_scheduler) => {
                        match FileReader::read_all_metadata(&file_scheduler).await {
                            Ok(_) => DataFileStatus::Ok,
                            Err(err) => DataFileStatus::Corrupt(err.to_string()),
                        }
                    }
                    Err(err) => DataFileStatus::Error(err.to_string()),
                }
            } else {
                DataFileStatus::Ok
            }
        }
        Err(object_store::Error::NotFound { .. }) => DataFileStatus::Missing(path.to_string()),
        Err(err) => DataFileStatus::Error(err.to_string()),
    }
}

/// Search dataset history for versions whose manifests reference a data file.
async fn locate_data_file_matches(
    dataset: &Dataset,
    data_file_path: &str,
    max_version: Option<u64>,
) -> Result<Vec<DataFileMatch>> {
    let mut versions = dataset.versions().await?;
    if let Some(max_version) = max_version {
        versions.retain(|version| version.version <= max_version);
    }

    let mut matches = Vec::new();
    for version in versions {
        let dataset = dataset.checkout_version(version.version).await?;
        matches.extend(
            matching_data_files(&dataset, data_file_path)
                .into_iter()
                .map(|data_file| DataFileMatch {
                    version: version.version,
                    data_file,
                }),
        );
    }
    Ok(matches)
}

/// Return versions where a data file reference first appears in contiguous runs.
fn introduced_versions(matches: &[DataFileMatch]) -> Vec<u64> {
    let versions = matches
        .iter()
        .map(|matched| matched.version)
        .collect::<BTreeSet<_>>();
    let mut introduced = Vec::new();
    let mut previous_version = None;
    for version in versions {
        if previous_version
            .map(|prev| prev + 1 != version)
            .unwrap_or(true)
        {
            introduced.push(version);
        }
        previous_version = Some(version);
    }
    introduced
}

/// Locate the versions that reference a data file and optionally restore before it.
pub(crate) async fn locate_data_file(
    mut writer: impl Write,
    args: &LanceDatasetLocateDataFileArgs,
) -> Result<()> {
    let dataset = Dataset::open(&args.source).await?;
    let matches = locate_data_file_matches(&dataset, &args.data_file, args.version).await?;
    if matches.is_empty() {
        writeln!(writer, "data_file: {}", args.data_file)?;
        writeln!(writer, "introduced_versions: none")?;
        return Ok(());
    }

    let introduced = introduced_versions(&matches);
    let first_introduced = introduced[0];
    writeln!(writer, "data_file: {}", args.data_file)?;
    writeln!(
        writer,
        "introduced_versions: {}",
        introduced
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(writer, "first_introduced_version: {first_introduced}")?;
    if first_introduced > 1 {
        writeln!(writer, "rollback_version: {}", first_introduced - 1)?;
    } else {
        writeln!(writer, "rollback_version: none")?;
    }
    writeln!(writer, "matches:")?;
    for matched in &matches {
        writeln!(
            writer,
            "  version={} {}",
            matched.version,
            format_data_file(&matched.data_file)
        )?;
    }

    if args.restore {
        if first_introduced == 1 {
            return Err(Error::invalid_input(format!(
                "data file {} first appears in version 1; there is no previous version to restore",
                args.data_file
            )));
        }
        let restore_version = first_introduced - 1;
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
    let dataset = Dataset::open(&args.source).await?;
    let dataset = checkout_dataset_version(&dataset, args.version).await?;
    let data_files = dataset_data_file_refs(&dataset);
    let mut bad_data_files = Vec::new();

    writeln!(writer, "version: {}", dataset.version_id())?;
    writeln!(writer, "checked_data_files: {}", data_files.len())?;
    for data_file in data_files {
        let status = verify_data_file_ref(&dataset, &data_file, args.deep).await;
        if !status.is_ok() {
            writeln!(
                writer,
                "{} {} message={}",
                status.label(),
                format_data_file(&data_file),
                status.message().unwrap_or("")
            )?;
            bad_data_files.push((data_file, status));
        }
    }
    writeln!(writer, "bad_data_files: {}", bad_data_files.len())?;

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
    let dataset = Dataset::open(&args.source).await?;
    let mut affected_fragment_ids = BTreeSet::new();
    let mut matched_by_input: HashMap<&str, Vec<ReferencedDataFile>> = HashMap::new();

    for data_file_path in &args.remove_data_files {
        let matches = matching_data_files(&dataset, data_file_path);
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
        matched_by_input.insert(data_file_path.as_str(), matches);
    }

    writeln!(writer, "read_version: {}", dataset.version_id())?;
    writeln!(
        writer,
        "affected_fragments: {}",
        affected_fragment_ids
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(writer, "matched_data_files:")?;
    for data_file_path in &args.remove_data_files {
        if let Some(matches) = matched_by_input.get(data_file_path.as_str()) {
            for matched in matches {
                writeln!(
                    writer,
                    "  input={} {}",
                    data_file_path,
                    format_data_file(matched)
                )?;
            }
        }
    }

    if args.dry_run {
        writeln!(writer, "dry_run: true")?;
        return Ok(());
    }

    let operation = Operation::Delete {
        updated_fragments: vec![],
        deleted_fragment_ids: affected_fragment_ids.iter().copied().collect(),
        predicate: format!(
            "repaired by lance-tools with --remove-data-files: {}",
            args.remove_data_files.join(",")
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
    let dataset = Dataset::open(&args.source).await?;
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
    use lance_table::format::DataFile;

    #[test]
    fn test_path_matches_full_key_suffix_and_filename() {
        assert!(path_matches(
            "data/abc.lance",
            "default.db/table.lance/data/abc.lance"
        ));
        assert!(path_matches("data/abc.lance", "abc.lance"));
        assert!(!path_matches("data/abc.lance", "data/def.lance"));
    }

    #[test]
    fn test_matching_data_files_accepts_data_prefix_or_filename() {
        let mut fragment = Fragment::new(7);
        fragment
            .files
            .push(DataFile::new_legacy_from_fields("abc.lance", vec![0], None));
        let data_files = fragment_data_file_refs(&fragment);
        assert_eq!(data_files.len(), 1);
        assert!(path_matches(&data_files[0].relative_path, "data/abc.lance"));
        assert!(path_matches(&data_files[0].relative_path, "abc.lance"));
    }

    #[test]
    fn test_introduced_versions_tracks_reintroduced_data_file() {
        let data_file = ReferencedDataFile {
            fragment_id: 1,
            relative_path: "data/abc.lance".to_string(),
            base_id: None,
        };
        let matches = vec![
            DataFileMatch {
                version: 2,
                data_file: data_file.clone(),
            },
            DataFileMatch {
                version: 3,
                data_file: data_file.clone(),
            },
            DataFileMatch {
                version: 5,
                data_file,
            },
        ];
        assert_eq!(introduced_versions(&matches), vec![2, 5]);
    }
}
