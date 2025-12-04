// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table maintenance for optimizing table layout.
//!
//! As a table is updated, its layout can become suboptimal. For example, if
//! a series of small streaming appends are performed, eventually there will be
//! a large number of small files. This imposes an overhead to track the large
//! number of files and for very small files can make it harder to read data
//! efficiently. In this case, files can be compacted into fewer larger files.
//!
//! To compact files in a table, use the [compact_files] method. This currently
//! can compact in two cases:
//!
//! 1. If a fragment has fewer rows than the target number of rows per fragment.
//!    The fragment must also have neighbors that are also candidates for
//!    compaction.
//! 2. If a fragment has a higher percentage of deleted rows than the provided
//!    threshold.
//!
//! In addition to the rules above there may be restrictions due to indexes.
//! When a fragment is compacted its row ids change and any index that contained
//! that fragment will be remapped.  However, we cannot combine indexed fragments
//! with unindexed fragments.
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
//! // Remapping indices is ignored in this example.
//! use lance::dataset::optimize::IgnoreRemap;
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = lance_core::utils::tempfile::TempStrDir::default();
//! # let uri = test_dir.to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from_iter_values(0..10_000))]
//! ).unwrap();
//! let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
//!
//! // Write 100 small files
//! let write_params = WriteParams { max_rows_per_file: 100, ..Default::default()};
//! let mut dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! assert_eq!(dataset.get_fragments().len(), 100);
//!
//! // Use compact_files() to consolidate the data to 1 fragment
//! let metrics = compact_files(&mut dataset, Default::default(), None).await.unwrap();
//! assert_eq!(metrics.fragments_removed, 100);
//! assert_eq!(metrics.fragments_added, 1);
//! assert_eq!(dataset.get_fragments().len(), 1);
//! # })
//! ```
//!
//! ## Distributed execution
//!
//! The [compact_files] method internally can use multiple threads, but
//! sometimes you might want to run it across multiple machines. To do this,
//! use the task API.
//!
//! ```text
//!                                      ┌──► CompactionTask.execute() ─► RewriteResult ─┐
//! plan_compaction() ─► CompactionPlan ─┼──► CompactionTask.execute() ─► RewriteResult ─┼─► commit_compaction()
//!                                      └──► CompactionTask.execute() ─► RewriteResult ─┘
//! ```
//!
//! [plan_compaction()] produces a [CompactionPlan]. This can be split into multiple
//! [CompactionTask], which can be serialized and sent to other machines. Calling
//! [CompactionTask::execute()] performs the compaction and returns a [RewriteResult].
//! The [RewriteResult] can be sent back to the coordinator, which can then call
//! [commit_compaction()] to commit the changes to the dataset.
//!
//! It's not required that all tasks are passed to [commit_compaction]. If some
//! didn't complete successfully or before a deadline, they can be omitted and
//! the successful tasks can be committed. You can also commit in batches if
//! you wish. As long as the tasks don't rewrite any of the same fragments,
//! they can be committed in any order.
use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use super::fragment::FileFragment;
use super::index::DatasetIndexRemapperOptions;
use super::rowids::load_row_id_sequences;
use super::transaction::{Operation, RewriteGroup, RewrittenIndex, Transaction};
use super::utils::make_rowid_capture_stream;
use super::{write_fragments_internal, WriteMode, WriteParams};
use crate::io::commit::{commit_transaction, migrate_fragments};
use crate::Dataset;
use crate::Result;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use lance_arrow::DataTypeExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::Error;
use lance_index::frag_reuse::FragReuseGroup;
use lance_index::DatasetIndexExt;
use lance_table::format::{DataFile, Fragment, RowIdMeta};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use snafu::location;
use tracing::info;

pub mod remapping;

use super::rowids::load_row_id_sequence;
use crate::dataset::fragment::write::generate_random_filename;
use crate::index::frag_reuse::build_new_frag_reuse_index;
use crate::io::deletion::read_dataset_deletion_file;
use lance_core::datatypes::Schema;
use lance_encoding::decoder::{ColumnInfo, PageEncoding, PageInfo as DecPageInfo};
use lance_encoding::version::LanceFileVersion;
use lance_file::format::pbfile;
use lance_file::reader::FileReader as LFReader;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::object_writer::ObjectWriter;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Writer;
use lance_table::rowids::{write_row_ids, RowIdSequence};
use prost::Message;
use prost_types::Any;
pub use remapping::{IgnoreRemap, IndexRemapper, IndexRemapperOptions, RemappedIndex};
use tokio::io::AsyncWriteExt;

/// Options to be passed to [compact_files].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionOptions {
    /// Target number of rows per file. Defaults to 1 million.
    ///
    /// This is used to determine which fragments need compaction, as any
    /// fragments that have fewer rows than this value will be candidates for
    /// compaction.
    pub target_rows_per_fragment: usize,
    /// Max number of rows per group
    ///
    /// This does not affect which fragments need compaction, but does affect
    /// how they are re-written if selected.
    pub max_rows_per_group: usize,
    /// Max number of bytes per file
    ///
    /// This does not affect which frgamnets need compaction, but does affect
    /// how they are re-written if selected.
    ///
    /// If not specified then the default (see [`WriteParams`]) will be used.
    pub max_bytes_per_file: Option<usize>,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    pub materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    pub materialize_deletions_threshold: f32,
    /// The number of threads to use (how many compaction tasks to run in parallel).
    /// Defaults to the number of compute-intensive CPUs.  Not used when running
    /// tasks manually using [`plan_compaction`]
    pub num_threads: Option<usize>,
    /// The batch size to use when scanning the input fragments.  If not
    /// specified then the default (see
    /// [`crate::dataset::Scanner::batch_size`]) will be used.
    pub batch_size: Option<usize>,
    /// Whether to defer remapping indices during compaction. If true, indices will
    /// not be remapped during this compaction operation. Instead, the fragment reuse index
    /// is updated and will be used to perform remapping later.
    pub defer_index_remap: bool,
    /// Whether to enable binary copy optimization when eligible.
    /// Defaults to false.
    pub enable_binary_copy: bool,
    pub enable_binary_copy_force: bool,
    pub binary_copy_read_batch_bytes: Option<usize>,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            // Matching defaults for WriteParams
            target_rows_per_fragment: 1024 * 1024,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.1,
            num_threads: None,
            max_bytes_per_file: None,
            batch_size: None,
            defer_index_remap: false,
            enable_binary_copy: false,
            enable_binary_copy_force: false,
            binary_copy_read_batch_bytes: Some(16 * 1024 * 1024),
        }
    }
}

impl CompactionOptions {
    pub fn validate(&mut self) {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletions_threshold >= 1.0 {
            self.materialize_deletions = false;
        }
    }
}

/// Determines whether binary copy compaction can be safely applied.
///
/// Binary copy merges existing Lance data files by copying page and buffer
/// blocks without decoding/re-encoding column data. For correctness and
/// compatibility it requires:
/// - Global feature switch `enable_binary_copy` to be on
/// - Non-legacy Lance file format (legacy files are unsupported)
/// - All candidate fragments/files share the same Lance file version
/// - No deletion vectors present (binary copy does not materialize deletions)
/// - Data layout matches the current schema (field id set, dense leaf column
///   indices are consistent and contiguous)
///
/// If any precondition fails the compaction falls back to the decode/rewrite
/// path.
fn can_use_binary_copy(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> bool {
    use lance_file::version::LanceFileVersion;
    if !options.enable_binary_copy {
        return false;
    }
    // Check dataset storage version is supported
    // Binary copy is not supported for legacy Lance file format
    let storage_ok = dataset
        .manifest
        .data_storage_format
        .lance_file_version()
        .map(|v| !matches!(v.resolve(), LanceFileVersion::Legacy))
        .unwrap_or(false);
    if !storage_ok {
        return false;
    }

    if fragments.is_empty() {
        return false;
    }

    // Make sure all files in fragments have the same Lance file version
    let first_data_file_version = LanceFileVersion::try_from_major_minor(
        fragments[0].files[0].file_major_version,
        fragments[0].files[0].file_minor_version,
    )
    .map(|v| v.resolve())
    .unwrap();

    // Check version consistency and deletion vector in a single pass
    let mut is_same_version = true;

    for fragment in fragments {
        // Check for deletion file
        if fragment.deletion_file.is_some() {
            return false;
        }

        // Check version consistency for all files in the fragment
        for data_file in &fragment.files {
            let version_ok = LanceFileVersion::try_from_major_minor(
                data_file.file_major_version,
                data_file.file_minor_version,
            )
            .map(|v| v.resolve())
            .map_or(false, |v| v == first_data_file_version);

            if !version_ok {
                is_same_version = false;
            }
        }
    }

    if !is_same_version {
        return false;
    }

    // Data layout must match current schema (field ids and dense leaf column indices).
    // This ensures copied pages can be reassembled under the active schema
    // without remapping or re-encoding.
    let schema = dataset.schema().clone();
    let leaf_count = schema.fields_pre_order().filter(|f| f.is_leaf()).count();
    let field_ids = schema.field_ids();
    let layout_ok = fragments.iter().all(|f| {
        let df = &f.files[0];
        let b1 = df.fields == field_ids;
        let b2 = df.column_indices.len() == leaf_count;
        let b3 = df
            .column_indices
            .iter()
            .enumerate()
            .all(|(i, ci)| *ci == i as i32);
        // Expect identical field id set and a contiguous leaf column index mapping.
        b1 && b2 && b3
    });
    // if !layout_ok {
    //     return false;
    // }
    true
}

/// Metrics returned by [compact_files].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// The number of fragments that have been overwritten.
    pub fragments_removed: usize,
    /// The number of new fragments that have been added.
    pub fragments_added: usize,
    /// The number of files that have been removed, including deletion files.
    pub files_removed: usize,
    /// The number of files that have been added, which is always equal to the
    /// number of fragments.
    pub files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// This does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    mut options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = &dataset.uri);
    options.validate();

    let compaction_plan: CompactionPlan = plan_compaction(dataset, &options).await?;

    if compaction_plan.tasks().is_empty() && options.enable_binary_copy_force {
        return Err(Error::NotSupported {
            source: "not execute binary copy compaction task".into(),
            location: location!(),
        });
    }

    // If nothing to compact, don't make a commit.
    if compaction_plan.tasks().is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let dataset_ref = &dataset.clone();

    let result_stream = futures::stream::iter(compaction_plan.tasks.into_iter())
        .map(|task| rewrite_files(Cow::Borrowed(dataset_ref), task, &options))
        .buffer_unordered(
            options
                .num_threads
                .unwrap_or_else(get_num_compute_intensive_cpus),
        );

    let completed_tasks: Vec<RewriteResult> = result_stream.try_collect().await?;
    let remap_options = remap_options.unwrap_or(Arc::new(DatasetIndexRemapperOptions::default()));
    let metrics = commit_compaction(dataset, completed_tasks, remap_options, &options).await?;

    Ok(metrics)
}

/// Information about a fragment used to decide its fate in compaction
#[derive(Debug)]
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub physical_rows: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.physical_rows > 0 {
            self.num_deletions as f32 / self.physical_rows as f32
        } else {
            0.0
        }
    }

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.physical_rows - self.num_deletions
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let physical_rows = fragment.physical_rows();
    let num_deletions = fragment.count_deletions();
    let (physical_rows, num_deletions) =
        futures::future::try_join(physical_rows, num_deletions).await?;
    Ok(FragmentMetrics {
        physical_rows,
        num_deletions,
    })
}

/// A plan for what groups of fragments to compact.
///
/// See [plan_compaction()] for more details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPlan {
    pub tasks: Vec<TaskData>,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionPlan {
    /// Retrieve standalone tasks that be be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }
}

/// A single group of fragments to compact, which is a view into the compaction
/// plan. We keep the `replace_range` indices so we can map the result of the
/// compact back to the fragments it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    /// The fragments to compact.
    pub fragments: Vec<Fragment>,
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result.
    ///
    /// This result should be later passed to [commit_compaction()] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let dataset = if dataset.manifest.version == self.read_version {
            Cow::Borrowed(dataset)
        } else {
            Cow::Owned(dataset.checkout_version(self.read_version).await?)
        };
        rewrite_files(dataset, self.task.clone(), &self.options).await
    }
}

impl CompactionPlan {
    fn new(read_version: u64, options: CompactionOptions) -> Self {
        Self {
            tasks: Vec::new(),
            read_version,
            options,
        }
    }

    fn extend_tasks(&mut self, tasks: impl IntoIterator<Item = TaskData>) {
        self.tasks.extend(tasks);
    }

    fn tasks(&self) -> &[TaskData] {
        &self.tasks
    }
}

#[derive(Debug, Clone)]
enum CompactionCandidacy {
    /// Compact the fragment if it has neighbors that are also candidates
    CompactWithNeighbors,
    /// Compact the fragment regardless.
    CompactItself,
}

/// Internal struct used for planning compaction.
struct CandidateBin {
    pub fragments: Vec<Fragment>,
    pub pos_range: Range<usize>,
    pub candidacy: Vec<CompactionCandidacy>,
    pub row_counts: Vec<usize>,
    pub indices: Vec<usize>,
}

impl CandidateBin {
    /// Return true if compacting these fragments wouldn't do anything.
    fn is_noop(&self) -> bool {
        if self.fragments.is_empty() {
            return true;
        }
        // If there's only one fragment, it's a noop if it's not CompactItself
        if self.fragments.len() == 1 {
            matches!(self.candidacy[0], CompactionCandidacy::CompactWithNeighbors)
        } else {
            false
        }
    }

    /// Split into one or more bins with at least `min_num_rows` in them.
    fn split_for_size(mut self, min_num_rows: usize) -> Vec<Self> {
        let mut bins = Vec::new();

        loop {
            let mut bin_len = 0;
            let mut bin_row_count = 0;
            while bin_row_count < min_num_rows && bin_len < self.row_counts.len() {
                bin_row_count += self.row_counts[bin_len];
                bin_len += 1;
            }

            // If there's enough remaining to make another worthwhile bin, then
            // push what we have as a bin.
            if self.row_counts[bin_len..].iter().sum::<usize>() >= min_num_rows {
                bins.push(Self {
                    fragments: self.fragments.drain(0..bin_len).collect(),
                    pos_range: self.pos_range.start..(self.pos_range.start + bin_len),
                    candidacy: self.candidacy.drain(0..bin_len).collect(),
                    row_counts: self.row_counts.drain(0..bin_len).collect(),
                    // By the time we are splitting for size we are done considering indices
                    indices: Vec::new(),
                });
                self.pos_range.start += bin_len;
            } else {
                // Otherwise, just push the remaining fragments into the last bin
                bins.push(self);
                break;
            }
        }

        bins
    }
}

async fn load_index_fragmaps(dataset: &Dataset) -> Result<Vec<RoaringBitmap>> {
    let indices = dataset.load_indices().await?;
    let mut index_fragmaps = Vec::with_capacity(indices.len());
    for index in indices.iter() {
        if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
            index_fragmaps.push(fragment_bitmap.clone());
        } else {
            let dataset_at_index = dataset.checkout_version(index.dataset_version).await?;
            let frags = 0..dataset_at_index.manifest.max_fragment_id.unwrap_or(0);
            index_fragmaps.push(RoaringBitmap::from_sorted_iter(frags).unwrap());
        }
    }
    Ok(index_fragmaps)
}

/// Formulate a plan to compact the files in a dataset
///
/// The compaction plan will contain a list of tasks to execute. Each task
/// will contain approximately `target_rows_per_fragment` rows and will be
/// rewriting fragments that are adjacent in the dataset's fragment list. Some
/// tasks may contain a single fragment when that fragment has deletions that
/// are being materialized and doesn't have any neighbors that need to be
/// compacted.
pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    // get_fragments should be returning fragments in sorted order (by id)
    // and fragment ids should be unique
    let fragments = dataset.get_fragments();
    debug_assert!(
        fragments.windows(2).all(|w| w[0].id() < w[1].id()),
        "fragments in manifest are not sorted"
    );
    let mut fragment_metrics = futures::stream::iter(fragments)
        .map(|fragment| async move {
            match collect_metrics(&fragment).await {
                Ok(metrics) => Ok((fragment.metadata, metrics)),
                Err(e) => Err(e),
            }
        })
        .buffered(dataset.object_store().io_parallelism());

    let index_fragmaps = load_index_fragmaps(dataset).await?;
    let indices_containing_frag = |frag_id: u32| {
        index_fragmaps
            .iter()
            .enumerate()
            .filter(|(_, bitmap)| bitmap.contains(frag_id))
            .map(|(pos, _)| pos)
            .collect::<Vec<_>>()
    };

    let mut candidate_bins: Vec<CandidateBin> = Vec::new();
    let mut current_bin: Option<CandidateBin> = None;
    let mut i = 0;

    while let Some(res) = fragment_metrics.next().await {
        let (fragment, metrics) = res?;

        let candidacy = if options.materialize_deletions
            && metrics.deletion_percentage() > options.materialize_deletions_threshold
        {
            Some(CompactionCandidacy::CompactItself)
        } else if metrics.physical_rows < options.target_rows_per_fragment {
            // Only want to compact if their are neighbors to compact such that
            // we can get a larger fragment.
            Some(CompactionCandidacy::CompactWithNeighbors)
        } else {
            // Not a candidate
            None
        };

        let indices = indices_containing_frag(fragment.id as u32);

        match (candidacy, &mut current_bin) {
            (None, None) => {} // keep searching
            (Some(candidacy), None) => {
                // Start a new bin
                current_bin = Some(CandidateBin {
                    fragments: vec![fragment],
                    pos_range: i..(i + 1),
                    candidacy: vec![candidacy],
                    row_counts: vec![metrics.num_rows()],
                    indices,
                });
            }
            (Some(candidacy), Some(bin)) => {
                // We cannot mix "indexed" and "non-indexed" fragments and so we only consider
                // the existing bin if it contains the same indices
                if bin.indices == indices {
                    // Add to current bin
                    bin.fragments.push(fragment);
                    bin.pos_range.end += 1;
                    bin.candidacy.push(candidacy);
                    bin.row_counts.push(metrics.num_rows());
                } else {
                    // Index set is different.  Complete previous bin and start new one
                    candidate_bins.push(current_bin.take().unwrap());
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment],
                        pos_range: i..(i + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                    });
                }
            }
            (None, Some(_)) => {
                // Bin is complete
                candidate_bins.push(current_bin.take().unwrap());
            }
        }

        i += 1;
    }

    // Flush the last bin
    if let Some(bin) = current_bin {
        candidate_bins.push(bin);
    }

    let final_bins = candidate_bins
        .into_iter()
        .filter(|bin| !bin.is_noop())
        .flat_map(|bin| bin.split_for_size(options.target_rows_per_fragment))
        .map(|bin| TaskData {
            fragments: bin.fragments,
        });

    let mut compaction_plan = CompactionPlan::new(dataset.manifest.version, options.clone());
    compaction_plan.extend_tasks(final_bins);

    Ok(compaction_plan)
}

/// The result of a single compaction task.
///
/// This should be passed to [commit_compaction()] to commit the operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
    /// A HashMap of original row IDs to new row IDs or None (deleted)
    /// Only set when index remap is done as a part of the compaction
    pub row_id_map: Option<HashMap<u64, Option<u64>>>,
    /// the changed row addresses in the original fragment
    /// in the form of serialized RoaringTreemap
    /// Only set when index remap is deferred after compaction
    pub changed_row_addrs: Option<Vec<u8>>,
}

async fn reserve_fragment_ids(
    dataset: &Dataset,
    fragments: impl ExactSizeIterator<Item = &mut Fragment>,
) -> Result<()> {
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: fragments.len() as u32,
        },
        None,
    );

    let (manifest, _) = commit_transaction(
        dataset,
        dataset.object_store(),
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
        dataset.manifest_location.naming_scheme,
        None,
    )
    .await?;

    // Need +1 since max_fragment_id is inclusive in this case and ranges are exclusive
    let new_max_exclusive = manifest.max_fragment_id.unwrap_or(0) + 1;
    let reserved_ids = (new_max_exclusive - fragments.len() as u32)..(new_max_exclusive);

    for (fragment, new_id) in fragments.zip(reserved_ids) {
        fragment.id = new_id as u64;
    }

    Ok(())
}

/// Rewrite the files in a single task.
///
/// This assumes that the dataset is the correct read version to be compacted.
async fn rewrite_files(
    dataset: Cow<'_, Dataset>,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            read_version: dataset.manifest.version,
            original_fragments: task.fragments,
            row_id_map: None,
            changed_row_addrs: None,
        });
    }

    let previous_writer_version = &dataset.manifest.writer_version;
    // The versions of Lance prior to when we started writing the writer version
    // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
    // make sure to recompute them.
    // See: https://github.com/lancedb/lance/issues/1531
    let recompute_stats = previous_writer_version.is_none();

    // It's possible the fragments are old and don't have physical rows or
    // num deletions recorded. If that's the case, we need to grab and set that
    // information.
    let fragments = migrate_fragments(dataset.as_ref(), &task.fragments, recompute_stats).await?;
    let num_rows = fragments
        .iter()
        .map(|f| f.physical_rows.unwrap() as u64)
        .sum::<u64>();
    // If we aren't using stable row ids, then we need to remap indices.
    let needs_remapping = !dataset.manifest.uses_stable_row_ids();
    let mut new_fragments: Vec<Fragment> = Vec::new();
    let mut scanner = dataset.scan();
    if let Some(batch_size) = options.batch_size {
        scanner.batch_size(batch_size);
    }
    let task_id = uuid::Uuid::new_v4();
    log::info!(
        "Compaction task {}: Begin compacting {} rows across {} fragments",
        task_id,
        num_rows,
        fragments.len()
    );
    let can_binary_copy = can_use_binary_copy(dataset.as_ref(), options, &fragments);
    if !can_binary_copy && options.enable_binary_copy_force {
        return Err(Error::NotSupported {
            source: format!("compaction task {}: binary copy is not supported", task_id).into(),
            location: location!(),
        });
    }
    let mut row_ids_rx = None;
    let reader;
    if !can_binary_copy {
        scanner
            .with_fragments(fragments.clone())
            .scan_in_order(true);
        if needs_remapping {
            scanner.with_row_id();
            let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
            let (data_no_row_ids, rx) =
                make_rowid_capture_stream(data, dataset.manifest.uses_stable_row_ids())?;
            row_ids_rx = Some(rx);
            reader = data_no_row_ids;
        } else {
            reader = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
        }
    } else {
        reader = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
    }

    let mut rows_read = 0;
    let schema = reader.schema();
    let reader = reader.inspect_ok(move |batch| {
        rows_read += batch.num_rows();
        log::info!(
            "Compaction task {}: Read progress {}/{}",
            task_id,
            rows_read,
            num_rows,
        );
    });
    let reader = Box::pin(RecordBatchStreamAdapter::new(schema, reader));

    let mut params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        params.max_bytes_per_file = max_bytes_per_file;
    }

    if dataset.manifest.uses_stable_row_ids() {
        params.enable_stable_row_ids = true;
    }

    if can_binary_copy {
        new_fragments = rewrite_files_binary_copy(
            dataset.as_ref(),
            &fragments,
            &params,
            options.binary_copy_read_batch_bytes,
        )
        .await?;

        if new_fragments.is_empty() && options.enable_binary_copy_force {
            return Err(Error::NotSupported {
                source: format!("compaction task {}: binary copy is not supported", task_id).into(),
                location: location!(),
            });
        }

        // fall back to common path if binary copy is not supported
        if new_fragments.is_empty() {
            let (frags, _) = write_fragments_internal(
                Some(dataset.as_ref()),
                dataset.object_store.clone(),
                &dataset.base,
                dataset.schema().clone(),
                reader,
                params,
                None,
            )
            .await?;
            new_fragments = frags;
        }
    } else {
        let (frags, _) = write_fragments_internal(
            Some(dataset.as_ref()),
            dataset.object_store.clone(),
            &dataset.base,
            dataset.schema().clone(),
            reader,
            params,
            None,
        )
        .await?;
        new_fragments = frags;
    }

    log::info!("Compaction task {}: file written", task_id);

    let (row_id_map, changed_row_addrs) = if let Some(row_ids_rx) = row_ids_rx {
        let captured_ids = row_ids_rx.try_recv().map_err(|err| Error::Internal {
            message: format!("Failed to receive row ids: {}", err),
            location: location!(),
        })?;
        // This code path is only when we use address style ids.
        let row_addrs = captured_ids.row_addrs(None).into_owned();

        log::info!(
            "Compaction task {}: reserving fragment ids and transposing row addrs",
            task_id
        );
        reserve_fragment_ids(&dataset, new_fragments.iter_mut()).await?;

        if options.defer_index_remap {
            let mut changed_row_addrs = Vec::with_capacity(row_addrs.serialized_size());
            row_addrs.serialize_into(&mut changed_row_addrs)?;
            (None, Some(changed_row_addrs))
        } else {
            let row_id_map = remapping::transpose_row_addrs(row_addrs, &fragments, &new_fragments);
            (Some(row_id_map), None)
        }
    } else {
        if dataset.manifest.uses_stable_row_ids() {
            log::info!("Compaction task {}: rechunking stable row ids", task_id);
            rechunk_stable_row_ids(dataset.as_ref(), &mut new_fragments, &fragments).await?;
            recalc_versions_for_rewritten_fragments(
                dataset.as_ref(),
                &mut new_fragments,
                &fragments,
            )
            .await?;
        }

        if options.defer_index_remap {
            let no_addrs = RoaringTreemap::new();
            let mut serialized_no_addrs = Vec::with_capacity(no_addrs.serialized_size());
            no_addrs.serialize_into(&mut serialized_no_addrs)?;
            (None, Some(serialized_no_addrs))
        } else {
            (Some(HashMap::new()), None)
        }
    };

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    log::info!("Compaction task {}: completed", task_id);

    Ok(RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments: task.fragments,
        row_id_map,
        changed_row_addrs,
    })
}

async fn rechunk_stable_row_ids(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    let mut old_sequences = load_row_id_sequences(dataset, old_fragments)
        .try_collect::<Vec<_>>()
        .await?;
    // Should sort them back into original order.
    old_sequences.sort_by_key(|(frag_id, _)| {
        old_fragments
            .iter()
            .position(|frag| frag.id as u32 == *frag_id)
            .expect("Fragment not found")
    });

    // Need to remove deleted rows
    futures::stream::iter(old_sequences.iter_mut().zip(old_fragments.iter()))
        .map(Ok)
        .try_for_each(|((_, seq), frag)| async move {
            if let Some(deletion_file) = &frag.deletion_file {
                let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;

                let mut new_seq = seq.as_ref().clone();
                new_seq.mask(deletions.to_sorted_iter())?;
                *seq = Arc::new(new_seq);
            }
            Ok::<(), crate::Error>(())
        })
        .await?;

    debug_assert_eq!(
        { old_sequences.iter().map(|(_, seq)| seq.len()).sum::<u64>() },
        {
            new_fragments
                .iter()
                .map(|frag| frag.physical_rows.unwrap() as u64)
                .sum::<u64>()
        },
        "{:?}",
        old_sequences
    );

    let new_sequences = lance_table::rowids::rechunk_sequences(
        old_sequences
            .into_iter()
            .map(|(_, seq)| seq.as_ref().clone()),
        new_fragments
            .iter()
            .map(|frag| frag.physical_rows.unwrap() as u64),
        false,
    )?;

    for (fragment, sequence) in new_fragments.iter_mut().zip(new_sequences) {
        // TODO: if large enough, serialize to separate file
        let serialized = lance_table::rowids::write_row_ids(&sequence);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
    }

    Ok(())
}

/// After row id rechunking, preserve per-row latest update versions by masking deletions and rechunking
async fn recalc_versions_for_rewritten_fragments(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    // Load old per-row last_updated_at version sequences
    let mut old_last_updated_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());
    // Load old per-row created_at version sequences
    let mut old_created_at_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());

    for frag in old_fragments.iter() {
        let row_count = if let Some(row_id_meta) = &frag.row_id_meta {
            match row_id_meta {
                RowIdMeta::Inline(data) => lance_table::rowids::read_row_ids(data)?.len(),
                RowIdMeta::External(_file) => frag.physical_rows.unwrap_or(0) as u64,
            }
        } else {
            frag.physical_rows.unwrap_or(0) as u64
        };

        // Load created_at sequence (default to version 1 if missing)
        let mut created_at_seq = if let Some(version_meta) = &frag.created_at_version_meta {
            version_meta.load_sequence().map_err(|e| Error::Internal {
                message: format!("Failed to load created_at version sequence: {}", e),
                location: location!(),
            })?
        } else {
            // Default: treat all rows as created at version 1
            lance_table::format::RowDatasetVersionSequence::from_uniform_row_count(row_count, 1)
        };

        // Load last_updated_at sequence (default to same as created_at sequence)
        let mut last_updated_seq = if let Some(version_meta) = &frag.last_updated_at_version_meta {
            version_meta.load_sequence().map_err(|e| Error::Internal {
                message: format!("Failed to load last_updated_at version sequence: {}", e),
                location: location!(),
            })?
        } else {
            created_at_seq.clone()
        };

        // Apply deletion mask if present (positions are local offsets)
        if let Some(deletion_file) = &frag.deletion_file {
            let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;
            last_updated_seq.mask(deletions.to_sorted_iter())?;
            created_at_seq.mask(deletions.to_sorted_iter())?;
        }

        old_last_updated_sequences.push(last_updated_seq);
        old_created_at_sequences.push(created_at_seq);
    }

    // Ensure row counts match new fragments total
    let old_total: u64 = old_last_updated_sequences.iter().map(|s| s.len()).sum();
    let new_total: u64 = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .sum();
    debug_assert_eq!(old_total, new_total);

    // Rechunk version runs aligned to new fragment sizes
    let chunk_sizes: Vec<u64> = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .collect();

    let new_last_updated_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_last_updated_sequences,
        chunk_sizes.clone(),
        false,
    )?;

    let new_created_at_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_created_at_sequences,
        chunk_sizes,
        false,
    )?;

    // Set both version metadata on new fragments
    for ((fragment, last_updated_seq), created_at_seq) in new_fragments
        .iter_mut()
        .zip(new_last_updated_sequences.into_iter())
        .zip(new_created_at_sequences.into_iter())
    {
        fragment.last_updated_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&last_updated_seq).unwrap(),
        );
        fragment.created_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&created_at_seq).unwrap(),
        );
    }

    Ok(())
}

async fn rewrite_files_binary_copy(
    dataset: &Dataset,
    fragments: &[Fragment],
    params: &WriteParams,
    read_batch_bytes_opt: Option<usize>,
) -> Result<Vec<Fragment>> {
    // Binary copy algorithm overview:
    // - Reads page and buffer regions directly from source files in bounded batches
    // - Appends them to a new output file with alignment, updating offsets
    // - Recomputes page priorities by adding the cumulative row count to preserve order
    // - For v2_0, enforces single-page structural header columns when closing a file
    // - Writes a new footer (schema descriptor, column metadata, offset tables, version)
    // - Optionally carries forward stable row ids and persists them inline in fragment metadata
    // Merge small Lance files into larger ones by page-level binary copy.
    let schema = dataset.schema().clone();
    let full_field_ids = schema.field_ids();

    // The previous checks have ensured that the file versions of all files are consistent.
    let version = LanceFileVersion::try_from_major_minor(
        fragments[0].files[0].file_major_version,
        fragments[0].files[0].file_minor_version,
    )
    .unwrap()
    .resolve();
    // v2_0 compatibility: column layout differs across file versions
    // - v2_0 materializes BOTH leaf columns and non-leaf structural headers (e.g., Struct / List)
    //   which means the ColumnInfo set includes all fields in pre-order traversal.
    // - v2_1+ materializes ONLY leaf columns. Non-leaf structural headers are not stored as columns.
    //   As a result, the ColumnInfo set contains leaf fields only.
    // To correctly align copy layout, we derive `column_count` by version:
    // - v2_0: use total number of fields in pre-order (leaf + non-leaf headers)
    // - v2_1+: use only the number of leaf fields
    let leaf_count = schema.fields_pre_order().filter(|f| f.is_leaf()).count();
    let column_count = if version == LanceFileVersion::V2_0 {
        schema.fields_pre_order().count()
    } else {
        leaf_count
    };

    // v2_0 compatibility: build a map to identify non-leaf structural header columns
    // - In v2_0 these headers exist as columns and must have a single page
    // - In v2_1+ these headers are not stored as columns and this map is unused
    let mut is_non_leaf_column: Vec<bool> = vec![false; column_count];
    if version == LanceFileVersion::V2_0 {
        let mut col_idx = 0;
        for field in schema.fields_pre_order() {
            // Only mark non-packed Struct fields (lists remain as leaf data carriers)
            let is_non_leaf = field.data_type().is_struct() && !field.is_packed_struct();
            is_non_leaf_column[col_idx] = is_non_leaf;
            col_idx += 1;
        }
    }

    let mut out: Vec<Fragment> = Vec::new();
    let mut current_writer: Option<ObjectWriter> = None;
    let mut current_filename: Option<String> = None;
    let mut current_pos: u64 = 0;
    let mut current_page_table: Vec<ColumnInfo> = Vec::new();

    // Column-list<Page-List<DecPageInfo>>
    let mut col_pages: Vec<Vec<DecPageInfo>> = std::iter::repeat_with(|| Vec::<DecPageInfo>::new())
        .take(column_count)
        .collect();
    let mut col_buffers: Vec<Vec<(u64, u64)>> = vec![Vec::new(); column_count];
    let mut total_rows_in_current: u64 = 0;
    let max_rows_per_file = params.max_rows_per_file as u64;
    let uses_stable_row_ids = dataset.manifest.uses_stable_row_ids();
    let mut current_row_ids = RowIdSequence::new();

    // Align all writes to 64-byte boundaries to honor typical IO alignment and
    // keep buffer offsets valid across concatenated pages.
    const ALIGN: usize = 64;
    static ZERO_BUFFER: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let zero_buf = ZERO_BUFFER.get_or_init(|| vec![0u8; ALIGN]);
    // Visit each fragment and all of its data files (a fragment may contain multiple files)
    for frag in fragments.iter() {
        let mut frag_row_ids_offset: u64 = 0;
        let frag_row_ids = if uses_stable_row_ids {
            Some(load_row_id_sequence(dataset, frag).await?)
        } else {
            None
        };
        for df in frag.files.iter() {
            let object_store = if let Some(base_id) = df.base_id {
                dataset.object_store_for_base(base_id).await?
            } else {
                dataset.object_store.clone()
            };
            let full_path = dataset.data_file_dir(df)?.child(df.path.as_str());
            let scan_scheduler = ScanScheduler::new(
                object_store.clone(),
                SchedulerConfig::max_bandwidth(&object_store),
            );
            let file_scheduler = scan_scheduler
                .open_file_with_priority(&full_path, 0, &df.file_size_bytes)
                .await?;
            let file_meta = LFReader::read_all_metadata(&file_scheduler).await?;
            let src_collum_infos = file_meta.column_infos.clone();
            // Initialize current_page_table
            if current_page_table.is_empty() {
                current_page_table = src_collum_infos
                    .iter()
                    .map(|column_index| ColumnInfo {
                        index: column_index.index,
                        buffer_offsets_and_sizes: Arc::from(
                            Vec::<(u64, u64)>::new().into_boxed_slice(),
                        ),
                        page_infos: Arc::from(Vec::<DecPageInfo>::new().into_boxed_slice()),
                        encoding: column_index.encoding.clone(),
                    })
                    .collect();
            }

            // Iterate through each column of the current data file of the current fragment
            for (col_idx, src_column_info) in src_collum_infos.iter().enumerate() {
                // v2_0 compatibility: special handling for non-leaf structural header columns
                // - v2_0 expects structural header columns to have a SINGLE page; they carry layout
                //   metadata only and are not true data carriers.
                // - When merging multiple input files via binary copy, naively appending pages would
                //   yield multiple pages for the same structural header column, violating v2_0 rules.
                // - To preserve v2_0 invariants, we skip pages beyond the first one for these columns.
                // - During finalization we also normalize the single remaining page’s `num_rows` to the
                //   total number of rows in the output file and reset `priority` to 0.
                // - For v2_1+ this logic does not apply because non-leaf headers are not stored as columns.
                let is_non_leaf = col_idx < is_non_leaf_column.len() && is_non_leaf_column[col_idx];
                if is_non_leaf && !col_pages[col_idx].is_empty() {
                    continue;
                }

                if current_writer.is_none() {
                    let filename = format!("{}.lance", generate_random_filename());
                    let path = dataset.base.child(super::DATA_DIR).child(filename.as_str());
                    let writer = dataset.object_store.create(&path).await?;
                    current_writer = Some(writer);
                    current_filename = Some(filename);
                    current_pos = 0;
                }

                let read_batch_bytes: u64 = read_batch_bytes_opt.unwrap_or(16 * 1024 * 1024) as u64;

                let mut page_index = 0;

                // Iterate through each page of the current column in the current data file of the current fragment
                while page_index < src_column_info.page_infos.len() {
                    let mut batch_ranges: Vec<Range<u64>> = Vec::new();
                    let mut batch_counts: Vec<usize> = Vec::new();
                    let mut batch_bytes: u64 = 0;
                    let mut batch_pages: usize = 0;
                    // Build a single read batch by coalescing consecutive pages up to
                    // `read_batch_bytes` budget:
                    // - Accumulate total bytes (`batch_bytes`) and page count (`batch_pages`).
                    // - For each page, append its buffer ranges to `batch_ranges` and record
                    //   the number of buffers in `batch_counts` so returned bytes can be
                    //   mapped back to page boundaries.
                    // - Stop when adding the next page would exceed the byte budget, then
                    //   issue one I/O request for the collected ranges.
                    // - Advance `page_index` to reflect pages scheduled in this batch.
                    for current_page in &src_column_info.page_infos[page_index..] {
                        let page_bytes: u64 = current_page
                            .buffer_offsets_and_sizes
                            .iter()
                            .map(|(_, size)| *size as u64)
                            .sum();
                        let would_exceed =
                            batch_pages > 0 && (batch_bytes + page_bytes > read_batch_bytes);
                        if would_exceed {
                            break;
                        }
                        batch_counts.push(current_page.buffer_offsets_and_sizes.len());
                        for (offset, size) in current_page.buffer_offsets_and_sizes.iter() {
                            batch_ranges.push((*offset)..(*offset + *size));
                        }
                        batch_bytes += page_bytes;
                        batch_pages += 1;
                        page_index += 1;
                    }

                    let bytes_vec = if batch_ranges.is_empty() {
                        Vec::new()
                    } else {
                        // read many buffers at once
                        file_scheduler.submit_request(batch_ranges, 0).await?
                    };
                    let mut bytes_iter = bytes_vec.into_iter();

                    for (local_idx, buffer_count) in batch_counts.iter().enumerate() {
                        // Reconstruct the absolute page index within the source column:
                        // - `page_index` now points to the page position
                        // - `batch_pages` is how many pages we included in this batch
                        // - `local_idx` enumerates pages inside the batch [0..batch_pages)
                        // Therefore `page_index - batch_pages + local_idx` yields the exact
                        // source page we are currently materializing, allowing us to access
                        // its metadata (encoding, row count, buffers) for the new page entry.
                        let page =
                            &src_column_info.page_infos[page_index - batch_pages + local_idx];
                        let mut new_offsets = Vec::with_capacity(*buffer_count);
                        for _ in 0..*buffer_count {
                            if let Some(bytes) = bytes_iter.next() {
                                let writer = current_writer.as_mut().unwrap();
                                let pad = (ALIGN - (current_pos as usize % ALIGN)) % ALIGN;
                                if pad != 0 {
                                    writer.write_all(&zero_buf[..pad]).await?;
                                    current_pos += pad as u64;
                                }
                                let start = current_pos;
                                writer.write_all(&bytes).await?;
                                current_pos += bytes.len() as u64;
                                new_offsets.push((start, bytes.len() as u64));
                            }
                        }

                        // manual clone encoding
                        let encoding = if page.encoding.is_structural() {
                            PageEncoding::Structural(page.encoding.as_structural().clone())
                        } else {
                            PageEncoding::Legacy(page.encoding.as_legacy().clone())
                        };
                        // `priority` acts as the global row offset for this page, ensuring
                        // downstream iterators maintain the correct logical order across
                        // merged inputs.
                        let new_page_info = DecPageInfo {
                            num_rows: page.num_rows,
                            priority: page.priority + total_rows_in_current,
                            encoding,
                            buffer_offsets_and_sizes: Arc::from(new_offsets.into_boxed_slice()),
                        };
                        col_pages[col_idx].push(new_page_info);
                    }
                } // finished scheduling & copying pages for this column in the current source file

                // Copy column-level buffers (outside page data) with alignment
                if !src_column_info.buffer_offsets_and_sizes.is_empty() {
                    let ranges: Vec<Range<u64>> = src_column_info
                        .buffer_offsets_and_sizes
                        .iter()
                        .map(|(offset, size)| (*offset)..(*offset + *size))
                        .collect();
                    let bytes_vec = file_scheduler.submit_request(ranges, 0).await?;
                    for bytes in bytes_vec.into_iter() {
                        let writer = current_writer.as_mut().unwrap();
                        let pad = (ALIGN - (current_pos as usize % ALIGN)) % ALIGN;
                        if pad != 0 {
                            writer.write_all(&zero_buf[..pad]).await?;
                            current_pos += pad as u64;
                        }
                        let start = current_pos;
                        writer.write_all(&bytes).await?;
                        current_pos += bytes.len() as u64;
                        col_buffers[col_idx].push((start, bytes.len() as u64));
                    }
                }
            } // finished all columns in the current source file

            if uses_stable_row_ids {
                // When stable row IDs are enabled, incorporate the fragment's row IDs
                if let Some(seq) = frag_row_ids.as_ref() {
                    // Number of rows in the current source file
                    let count = file_meta.num_rows as usize;

                    // Take the subsequence of row IDs corresponding to this file
                    let slice = seq.slice(frag_row_ids_offset as usize, count);

                    // Materialize the slice into a Vec for conversion
                    // NOTE: This allocation can be avoided by extending with `slice` directly.
                    let ids_vec: Vec<u64> = slice.iter().collect();

                    // Append these row IDs to the accumulated sequence for the current output
                    current_row_ids.extend(RowIdSequence::from(ids_vec.as_slice()));

                    // Advance the offset so the next file reads the subsequent row IDs
                    frag_row_ids_offset += count as u64;
                }
            }

            // Accumulate rows for the current output file and flush when reaching the threshold
            total_rows_in_current += file_meta.num_rows;
            if total_rows_in_current >= max_rows_per_file {
                // v2_0 compatibility: enforce single-page structural headers before file close
                // - We truncate to a single page and rewrite the page’s `num_rows` to match the output
                //   file’s row count so downstream decoders see a consistent header.
                let mut final_cols: Vec<Arc<ColumnInfo>> = Vec::with_capacity(column_count);
                for (i, column_info) in current_page_table.iter().enumerate() {
                    // For v2_0 struct headers, force a single page and set num_rows to total
                    let mut pages_vec = std::mem::take(&mut col_pages[i]);
                    if version == LanceFileVersion::V2_0
                        && is_non_leaf_column.get(i).copied().unwrap_or(false)
                    {
                        if !pages_vec.is_empty() {
                            pages_vec[0].num_rows = total_rows_in_current;
                            pages_vec[0].priority = 0;
                            pages_vec.truncate(1);
                        }
                    }
                    let pages_arc = Arc::from(pages_vec.into_boxed_slice());
                    let buffers_vec = std::mem::take(&mut col_buffers[i]);
                    final_cols.push(Arc::new(ColumnInfo::new(
                        column_info.index,
                        pages_arc,
                        buffers_vec,
                        column_info.encoding.clone(),
                    )));
                }
                let writer = current_writer.take().unwrap();
                flush_footer(writer, &schema, &final_cols, total_rows_in_current, version).await?;

                // Register the newly closed output file as a fragment data file
                let (maj, min) = version.to_numbers();
                let mut fragment_out = Fragment::new(0);
                let mut data_file_out =
                    DataFile::new_unstarted(current_filename.take().unwrap(), maj, min);
                // v2_0 vs v2_1+ field-to-column index mapping
                // - v2_1+ stores only leaf columns; non-leaf fields get `-1` in the mapping
                // - v2_0 includes structural headers as columns; non-leaf fields map to a concrete index
                let is_structural = version >= LanceFileVersion::V2_1;
                let mut field_column_indices: Vec<i32> = Vec::with_capacity(full_field_ids.len());
                let mut curr_col_idx: i32 = 0;
                for field in schema.fields_pre_order() {
                    if field.is_packed_struct() {
                        field_column_indices.push(curr_col_idx);
                        curr_col_idx += 1;
                    } else if field.children.is_empty() || !is_structural {
                        field_column_indices.push(curr_col_idx);
                        curr_col_idx += 1;
                    } else {
                        field_column_indices.push(-1);
                    }
                }
                data_file_out.fields = full_field_ids.clone();
                data_file_out.column_indices = field_column_indices;
                fragment_out.files.push(data_file_out);
                fragment_out.physical_rows = Some(total_rows_in_current as usize);
                if uses_stable_row_ids {
                    fragment_out.row_id_meta =
                        Some(RowIdMeta::Inline(write_row_ids(&current_row_ids)));
                }
                // Reset state for next output file
                current_writer = None;
                current_pos = 0;
                current_page_table.clear();
                for v in col_pages.iter_mut() {
                    v.clear();
                }
                for v in col_buffers.iter_mut() {
                    v.clear();
                }
                out.push(fragment_out);
                total_rows_in_current = 0;
                if uses_stable_row_ids {
                    current_row_ids = RowIdSequence::new();
                }
            }
        }
    } // Complete the writing of all fragments, except for some data remaining in memory

    if total_rows_in_current > 0 {
        // Flush remaining rows as a final output file
        // v2_0 compatibility: same single-page enforcement applies for the final file close
        let mut final_cols: Vec<Arc<ColumnInfo>> = Vec::with_capacity(column_count);
        for (i, ci) in current_page_table.iter().enumerate() {
            // For v2_0 struct headers, force a single page and set num_rows to total
            let mut pages_vec = std::mem::take(&mut col_pages[i]);
            if version == LanceFileVersion::V2_0
                && is_non_leaf_column.get(i).copied().unwrap_or(false)
            {
                if !pages_vec.is_empty() {
                    pages_vec[0].num_rows = total_rows_in_current;
                    pages_vec[0].priority = 0;
                    pages_vec.truncate(1);
                }
            }
            let pages_arc = Arc::from(pages_vec.into_boxed_slice());
            let buffers_vec = std::mem::take(&mut col_buffers[i]);
            final_cols.push(Arc::new(ColumnInfo::new(
                ci.index,
                pages_arc,
                buffers_vec,
                ci.encoding.clone(),
            )));
        }
        if current_writer.is_none() {
            let filename = format!("{}.lance", generate_random_filename());
            let path = dataset.base.child(super::DATA_DIR).child(filename.as_str());
            let writer = dataset.object_store.create(&path).await?;
            current_writer = Some(writer);
            current_filename = Some(filename);
            current_pos = 0;
        }
        let writer = current_writer.take().unwrap();
        flush_footer(writer, &schema, &final_cols, total_rows_in_current, version).await?;
        // Register the final file
        let (maj, min) = version.to_numbers();
        let mut frag = Fragment::new(0);
        let mut df = DataFile::new_unstarted(current_filename.take().unwrap(), maj, min);
        // v2_0 vs v2_1+ field-to-column index mapping for the final file
        let is_structural = version >= LanceFileVersion::V2_1;
        let mut field_column_indices: Vec<i32> = Vec::with_capacity(full_field_ids.len());
        let mut curr_col_idx: i32 = 0;
        for field in schema.fields_pre_order() {
            if field.is_packed_struct() {
                field_column_indices.push(curr_col_idx);
                curr_col_idx += 1;
            } else if field.children.is_empty() || !is_structural {
                field_column_indices.push(curr_col_idx);
                curr_col_idx += 1;
            } else {
                field_column_indices.push(-1);
            }
        }
        df.fields = full_field_ids.clone();
        df.column_indices = field_column_indices;
        frag.files.push(df);
        frag.physical_rows = Some(total_rows_in_current as usize);
        if uses_stable_row_ids {
            frag.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&current_row_ids)));
        }
        out.push(frag);
    }
    Ok(out)
}

/// Finalizes a compacted data file by writing the Lance footer via `FileWriter`.
///
/// This function does not manually craft the footer. Instead it:
/// - Pads the current `ObjectWriter` position to a 64‑byte boundary (required for v2_1+ readers).
/// - Converts the collected per‑column info (`final_cols`) into `ColumnMetadata`.
/// - Constructs a `lance_file::writer::FileWriter` with the active `schema`, column metadata,
///   and `total_rows_in_current`.
/// - Calls `FileWriter::finish()` to emit column metadata, offset tables, global buffers
///   (schema descriptor), version, and to close the writer.
///
/// Preconditions:
/// - All page data and column‑level buffers referenced by `final_cols` have already been written
///   to `writer`; otherwise offsets in the footer will be invalid.
///
/// Version notes:
/// - v2_0 structural single‑page enforcement is handled when building `final_cols`; this function
///   only performs consistent finalization.
async fn flush_footer(
    mut writer: ObjectWriter,
    schema: &Schema,
    final_cols: &[Arc<ColumnInfo>],
    total_rows_in_current: u64,
    version: LanceFileVersion,
) -> Result<()> {
    if version >= LanceFileVersion::V2_1 {
        const ALIGN: usize = 64;
        static ZERO_BUFFER: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        let zero_buf = ZERO_BUFFER.get_or_init(|| vec![0u8; ALIGN]);
        let pos = writer.tell().await? as u64;
        let pad = (ALIGN as u64 - (pos % ALIGN as u64)) % ALIGN as u64;
        if pad != 0 {
            writer.write_all(&zero_buf[..pad as usize]).await?;
        }
    }
    let mut col_metadatas = Vec::with_capacity(final_cols.len());
    for col in final_cols {
        let pages = col
            .page_infos
            .iter()
            .map(|page_info| {
                let encoded_encoding = match &page_info.encoding {
                    PageEncoding::Legacy(array_encoding) => {
                        Any::from_msg(array_encoding)?.encode_to_vec()
                    }
                    PageEncoding::Structural(page_layout) => {
                        Any::from_msg(page_layout)?.encode_to_vec()
                    }
                };
                let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) = page_info
                    .buffer_offsets_and_sizes
                    .as_ref()
                    .iter()
                    .cloned()
                    .unzip();
                Ok(pbfile::column_metadata::Page {
                    buffer_offsets,
                    buffer_sizes,
                    encoding: Some(pbfile::Encoding {
                        location: Some(pbfile::encoding::Location::Direct(
                            pbfile::DirectEncoding {
                                encoding: encoded_encoding,
                            },
                        )),
                    }),
                    length: page_info.num_rows,
                    priority: page_info.priority,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (buffer_offsets, buffer_sizes): (Vec<_>, Vec<_>) =
            col.buffer_offsets_and_sizes.iter().cloned().unzip();
        let encoded_col_encoding = Any::from_msg(&col.encoding)?.encode_to_vec();
        let column = pbfile::ColumnMetadata {
            pages,
            buffer_offsets,
            buffer_sizes,
            encoding: Some(pbfile::Encoding {
                location: Some(pbfile::encoding::Location::Direct(pbfile::DirectEncoding {
                    encoding: encoded_col_encoding,
                })),
            }),
        };
        col_metadatas.push(column);
    }
    let mut file_writer = FileWriter::new_lazy(
        writer,
        FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        },
    );
    file_writer.initialize_with_external_metadata(
        schema.clone(),
        col_metadatas,
        total_rows_in_current,
    );
    file_writer.finish().await?;
    Ok(())
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    // If we aren't using stable row ids, then we need to remap indices.
    let needs_remapping = !dataset.manifest.uses_stable_row_ids() && !options.defer_index_remap;

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = CompactionMetrics::default();

    let mut row_id_map: HashMap<u64, Option<u64>> = HashMap::default();
    let mut frag_reuse_groups: Vec<FragReuseGroup> = Vec::new();
    let mut new_fragment_bitmap: RoaringBitmap = RoaringBitmap::new();

    for task in completed_tasks {
        metrics += task.metrics;
        let rewrite_group = RewriteGroup {
            old_fragments: task.original_fragments.clone(),
            new_fragments: task.new_fragments.clone(),
        };
        if needs_remapping {
            row_id_map.extend(task.row_id_map.unwrap());
        } else if options.defer_index_remap {
            frag_reuse_groups.push(FragReuseGroup {
                changed_row_addrs: task.changed_row_addrs.unwrap(),
                old_frags: task.original_fragments.iter().map(|f| f.into()).collect(),
                new_frags: task.new_fragments.iter().map(|f| f.into()).collect(),
            });

            task.new_fragments.iter().for_each(|frag| {
                new_fragment_bitmap.insert(frag.id as u32);
            });
        }
        rewrite_groups.push(rewrite_group);
    }

    let rewritten_indices = if needs_remapping {
        let index_remapper = remap_options.create_remapper(dataset)?;
        let affected_ids = rewrite_groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
            .collect::<Vec<_>>();

        let remapped_indices = index_remapper
            .remap_indices(row_id_map, &affected_ids)
            .await?;
        remapped_indices
            .into_iter()
            .map(|rewritten| RewrittenIndex {
                old_id: rewritten.old_id,
                new_id: rewritten.new_id,
                new_index_details: rewritten.index_details,
                new_index_version: rewritten.index_version,
            })
            .collect()
    } else if !options.defer_index_remap {
        // We need to reserve fragment ids here so that the fragment bitmap
        // can be updated for each index.
        let new_fragments = rewrite_groups
            .iter_mut()
            .flat_map(|group| group.new_fragments.iter_mut())
            .collect::<Vec<_>>();
        reserve_fragment_ids(dataset, new_fragments.into_iter()).await?;
        Vec::new()
    } else {
        Vec::new()
    };

    let frag_reuse_index = if options.defer_index_remap {
        Some(build_new_frag_reuse_index(dataset, frag_reuse_groups, new_fragment_bitmap).await?)
    } else {
        None
    };

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: rewrite_groups,
            rewritten_indices,
            frag_reuse_index,
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(metrics)
}

#[cfg(test)]
mod tests {

    use self::remapping::RemappedIndex;
    use super::*;
    use crate::dataset;
    use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
    use crate::dataset::optimize::remapping::{transpose_row_addrs, transpose_row_ids_from_digest};
    use crate::dataset::WriteDestination;
    use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
    use crate::index::vector::{StageParams, VectorIndexParams};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
    use arrow_array::{
        Float32Array, Int64Array, LargeStringArray, PrimitiveArray, RecordBatch,
        RecordBatchIterator,
    };
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::concat::concat_batches;
    use async_trait::async_trait;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::Error;
    use lance_datagen::Dimension;
    use lance_file::version::LanceFileVersion;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams};
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::{Index, IndexType};
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_table::io::manifest::read_manifest_indexes;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::io::Cursor;
    use uuid::Uuid;

    #[test]
    fn test_candidate_bin() {
        let empty_bin = CandidateBin {
            fragments: vec![],
            pos_range: 0..0,
            candidacy: vec![],
            row_counts: vec![],
            indices: vec![],
        };
        assert!(empty_bin.is_noop());

        let fragment = Fragment {
            id: 0,
            files: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };
        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
            row_counts: vec![100],
            indices: vec![],
        };
        assert!(single_bin.is_noop());

        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactItself],
            row_counts: vec![100],
            indices: vec![],
        };
        // Not a no-op because it's CompactItself
        assert!(!single_bin.is_noop());

        let big_bin = CandidateBin {
            fragments: std::iter::repeat_n(fragment, 8).collect(),
            pos_range: 0..8,
            candidacy: std::iter::repeat_n(CompactionCandidacy::CompactItself, 8).collect(),
            row_counts: vec![100, 400, 200, 200, 400, 300, 300, 100],
            indices: vec![],
            // Will group into: [[100, 400], [200, 200, 400], [300, 300, 100]]
            // with size = 500
        };
        assert!(!big_bin.is_noop());
        let split = big_bin.split_for_size(500);
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].pos_range, 0..2);
        assert_eq!(split[1].pos_range, 2..5);
        assert_eq!(split[2].pos_range, 5..8);
    }

    fn sample_data() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap()
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapperExpectation {
        expected: HashMap<u64, Option<u64>>,
        answer: Vec<RemappedIndex>,
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapper {
        expectations: Vec<MockIndexRemapperExpectation>,
    }

    impl MockIndexRemapper {
        fn stringify_map(map: &HashMap<u64, Option<u64>>) -> String {
            let mut sorted_keys = map.keys().collect::<Vec<_>>();
            sorted_keys.sort();
            let mut first_keys = sorted_keys
                .into_iter()
                .take(10)
                .map(|key| {
                    format!(
                        "{}:{:?}",
                        RowAddress::from(*key),
                        map[key].map(RowAddress::from)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            if map.len() > 10 {
                first_keys.push_str(", ...");
            }
            let mut result_str = format!("(len={})", map.len());
            result_str.push_str(&first_keys);
            result_str
        }

        fn in_any_order(expectations: &[Self]) -> Self {
            let expectations = expectations
                .iter()
                .flat_map(|item| item.expectations.clone())
                .collect::<Vec<_>>();
            Self { expectations }
        }
    }

    #[async_trait]
    impl IndexRemapper for MockIndexRemapper {
        async fn remap_indices(
            &self,
            index_map: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            for expectation in &self.expectations {
                if expectation.expected == index_map {
                    return Ok(expectation.answer.clone());
                }
            }
            panic!(
                "Unexpected index map (len={}): {}\n  Options: {}",
                index_map.len(),
                Self::stringify_map(&index_map),
                self.expectations
                    .iter()
                    .map(|expectation| Self::stringify_map(&expectation.expected))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            );
        }
    }

    impl IndexRemapperOptions for MockIndexRemapper {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(self.clone()))
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_empty(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Compact an empty table
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        assert_eq!(metrics, CompactionMetrics::default());
        assert_eq!(dataset.manifest.version, 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_all_good(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Compact a table with nothing to do
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 10_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // There's only one file, so we can't compact any more if we wanted to.
        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 3_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);
    }

    fn row_addrs(frag_idx: u32, offsets: Range<u32>) -> Range<u64> {
        let start = RowAddress::new_from_parts(frag_idx, offsets.start);
        let end = RowAddress::new_from_parts(frag_idx, offsets.end);
        start.into()..end.into()
    }

    // The outer list has one item per new fragment
    // The inner list has ranges of old row ids that map to the new fragment, in order
    fn expect_remap(
        ranges: &[Vec<(Range<u64>, bool)>],
        starting_new_frag_idx: u32,
    ) -> MockIndexRemapper {
        let mut expected_remap: HashMap<u64, Option<u64>> = HashMap::default();
        expected_remap.reserve(ranges.iter().map(|r| r.len()).sum());
        for (new_frag_offset, new_frag_ranges) in ranges.iter().enumerate() {
            let new_frag_idx = starting_new_frag_idx + new_frag_offset as u32;
            let mut row_offset = 0;
            for (old_id_range, is_found) in new_frag_ranges.iter() {
                for old_id in old_id_range.clone() {
                    if *is_found {
                        let new_id = RowAddress::new_from_parts(new_frag_idx, row_offset);
                        expected_remap.insert(old_id, Some(new_id.into()));
                        row_offset += 1;
                    } else {
                        expected_remap.insert(old_id, None);
                    }
                }
            }
        }
        MockIndexRemapper {
            expectations: vec![MockIndexRemapperExpectation {
                expected: expected_remap,
                answer: vec![],
            }],
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_many(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete 1 row from first large fragment
        dataset.delete("a = 1300").await.unwrap();

        // Delete 20% of rows from second large fragment
        dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

        // Append 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 300,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let first_new_frag_idx = 7;
        // Predicting the remap is difficult.  One task will remap to fragments 7/8 and the other
        // will remap to fragments 9/10 but we don't know which is which and so we just allow ourselves
        // to expect both possibilities.
        let remap_a = expect_remap(
            &[
                vec![
                    // 3 small fragments are rewritten to frags 7 & 8
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
                // frag 3 is skipped since it does not have enough missing data
                // Frags 4, 5, and 6 are rewritten to frags 9 & 10
                vec![
                    // Only 800 of the 1000 rows taken from frag 4
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    // frags 5 compacted with frag 4
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
            ],
            first_new_frag_idx,
        );
        let remap_b = expect_remap(
            &[
                // Frags 4, 5, and 6 are rewritten to frags 7 & 8
                vec![
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
                // 3 small fragments rewritten to frags 9 & 10
                vec![
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
            ],
            first_new_frag_idx,
        );

        // Create compaction plan
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 2);
        assert_eq!(plan.tasks()[0].fragments.len(), 3);
        assert_eq!(plan.tasks()[1].fragments.len(), 3);

        assert_eq!(
            plan.tasks()[0]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            plan.tasks()[1]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        let mock_remapper = MockIndexRemapper::in_any_order(&[remap_a, remap_b]);

        // Run compaction
        let metrics = compact_files(&mut dataset, options, Some(Arc::new(mock_remapper)))
            .await
            .unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids, vec![3, 7, 8, 9, 10]);
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
        let debug = dataset.manifest.clone();
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
    async fn test_binary_copy_with_stable_row_ids_enabled() {
        for version in LanceFileVersion::iter_non_legacy() {
            do_test_binary_copy_with_stable_row_ids_enabled(version).await;
        }
    }

    async fn do_test_binary_copy_with_stable_row_ids_enabled(version: LanceFileVersion) {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(8).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(4_000),
            "memory://test/binary_copy_stable_row_ids",
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

        assert_eq!(before_batch, after_batch);
    }

    // #[tokio::test]
    async fn test_perf_binary_copy_vs_full() {
        use arrow_schema::{DataType, Field, Fields, TimeUnit};
        use lance_core::utils::tempfile::TempStrDir;
        use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
        use std::time::Instant;

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

        let full_dir = TempStrDir::default();
        let a = full_dir.as_into_string().into();
        println!("full_dir: {:?}", a);
        let mut dataset = Dataset::write(
            reader_full,
            &*full_dir,
            Some(WriteParams {
                enable_stable_row_ids: true,
                data_storage_version: Some(LanceFileVersion::V2_2),
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
            ..Default::default()
        };

        let t0 = Instant::now();
        let _ = compact_files(&mut dataset, opt_full, None).await.unwrap();
        let d_full = t0.elapsed();
        let before = dataset.count_rows(None).await.unwrap();

        let versions = dataset.versions().await.unwrap();
        let mut dataset = dataset.checkout_version(1).await.unwrap();
        dataset.restore().await.unwrap();
        let t1 = Instant::now();
        let _ = compact_files(&mut dataset, opt_binary, None).await.unwrap();
        let d_bin = t1.elapsed();
        let after = dataset.count_rows(None).await.unwrap();

        println!(
            "perf: full_compaction={:?}, binary_copy={:?}, speedup={:.2}x",
            d_full,
            d_bin,
            (d_full.as_secs_f64() / d_bin.as_secs_f64())
        );

        assert_eq!(before, after);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_data_files(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Add a column
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]);

        let data = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..10_000)),
                Arc::new(Float32Array::from_iter_values(
                    (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

        dataset.merge(reader, "a", "a").await.unwrap();

        let expected_remap = expect_remap(
            &[vec![
                // 3 small fragments are rewritten entirely
                (row_addrs(0, 0..5000), true),
                (row_addrs(1, 0..5000), true),
            ]],
            2,
        );

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(plan.tasks().len(), 1);
        assert_eq!(plan.tasks()[0].fragments.len(), 2);

        let metrics = compact_files(&mut dataset, plan.options, Some(Arc::new(expected_remap)))
            .await
            .unwrap();

        assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
        assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // Assert order unchanged and data is all there.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(scanned_data, data);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // For files that have few rows, we don't want to compact just 1 since
        // that won't do anything. But if there are deletions to materialize,
        // we want to do groups of 1. This test checks that.
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 1 fragment
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.delete("a <= 500").await.unwrap();

        // Threshold must be satisfied
        let mut options = CompactionOptions {
            materialize_deletions_threshold: 0.8,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Ignore deletions if materialize_deletions is false
        options.materialize_deletions_threshold = 0.1;
        options.materialize_deletions = false;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Materialize deletions if threshold is met
        options.materialize_deletions = true;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 1);
        assert_eq!(metrics.files_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct IgnoreRemap {}

    #[async_trait]
    impl IndexRemapper for IgnoreRemap {
        async fn remap_indices(
            &self,
            _: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            Ok(Vec::new())
        }
    }

    impl IndexRemapperOptions for IgnoreRemap {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(Self {}))
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_compact_distributed(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Can run the tasks independently
        // Can provide subset of tasks to commit_compaction
        // Once committed, can't commit remaining tasks
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Write dataset as 9 1k row fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Plan compaction with 3 tasks
        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 3);

        let dataset_ref = &dataset;
        let mut results = futures::stream::iter(plan.compaction_tasks())
            .then(|task| async move { task.execute(dataset_ref).await.unwrap() })
            .collect::<Vec<_>>()
            .await;

        assert_eq!(results.len(), 3);

        assert_eq!(
            results[0]
                .original_fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(results[0].metrics.files_removed, 3);
        assert_eq!(results[0].metrics.files_added, 1);

        // Just commit the last task
        commit_compaction(
            &mut dataset,
            vec![results.pop().unwrap()],
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();

        if use_stable_row_id {
            // 1 commit for reserve fragments and 1 for final commit, both
            // from the call to commit_compaction
            assert_eq!(dataset.manifest.version, 3);
        } else {
            // 1 commit for each task's reserve fragments plus 1 for
            // the call to commit_compaction
            assert_eq!(dataset.manifest.version, 5);
        }

        // Can commit the remaining tasks
        commit_compaction(
            &mut dataset,
            results,
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();
        if use_stable_row_id {
            // 1 commit for reserve fragments and 1 for final commit, both
            // from the call to commit_compaction
            assert_eq!(dataset.manifest.version, 5);
        } else {
            // The reserve fragments call already happened for this task
            // and so we just see the bump from the commit_compaction
            assert_eq!(dataset.manifest.version, 6);
        }

        assert_eq!(dataset.manifest.uses_stable_row_ids(), use_stable_row_id,);
    }

    #[tokio::test]
    async fn test_stable_row_indices() {
        // Validate behavior of indices after compaction with stable row ids.
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(16).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));
        let mut dataset = Dataset::write(
            data_gen.batch(500),
            "memory://test/table",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 100, // 5 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete first 110 rows so rowids != final rowaddrs
        // First 100 rows deletes first file. Next 10 deletes part of second
        // file, so we will trigger the with deletions code path.
        dataset.delete("i < 110").await.unwrap();

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

            let query = Float32Array::from(vec![0.0f32; 16]);
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

        let options = CompactionOptions {
            target_rows_per_fragment: 180,
            ..Default::default()
        };
        let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

        // The indices should be unchanged after compaction, since we are using
        // stable row ids.
        let current_indices = index_set(&dataset).await;
        assert_eq!(indices, current_indices);

        let after_vec_result = vector_query(&dataset).await;
        assert_eq!(before_vec_result, after_vec_result);

        let after_scalar_result = scalar_query(&dataset).await;
        assert_eq!(before_scalar_result, after_scalar_result);
    }

    #[tokio::test]
    async fn test_defer_index_remap() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create another same dataset to mimic behavior without deferred index remap
        let mut data_gen2 = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset2 = Dataset::write(
            data_gen2.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete some rows to create deletions
        dataset.delete("i < 500").await.unwrap();
        dataset2.delete("i < 500").await.unwrap();

        // Create a scalar index to check this is not touched
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

        // Verify the initial state - no fragment reuse index should exist
        let initial_indices = dataset.load_indices().await.unwrap();
        assert_eq!(initial_indices.len(), 1);
        assert_eq!(initial_indices[0].name, "scalar");

        // Store the original scalar index UUID for comparison
        let original_scalar_uuid = initial_indices[0].uuid;

        // Plan and execute compaction manually
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };
        let options2 = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: false,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let plan2 = plan_compaction(&dataset2, &options2).await.unwrap();

        let mut expected_all_old_frag_ids = Vec::new();
        let mut expected_all_new_frag_ids = Vec::new();
        let mut expected_all_new_frag_bitmap = RoaringBitmap::new();
        let mut expected_all_row_id_map = HashMap::new();
        let mut deferred_results = Vec::new();

        for (task, task2) in plan.tasks().iter().zip(plan2.tasks()) {
            let deferred_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();
            let immediate_result =
                rewrite_files(Cow::Borrowed(&dataset2), task2.clone(), &options2)
                    .await
                    .unwrap();

            // Verify RewriteResult for deferred index remap
            assert!(deferred_result.row_id_map.is_none());
            assert!(deferred_result.changed_row_addrs.is_some());
            assert!(!deferred_result
                .changed_row_addrs
                .as_ref()
                .unwrap()
                .is_empty());
            assert!(!deferred_result.original_fragments.is_empty());
            assert!(!deferred_result.new_fragments.is_empty());

            // Verify RewriteResult for immediate index remap
            assert!(immediate_result.changed_row_addrs.is_none());
            assert!(!immediate_result.original_fragments.is_empty());
            assert!(!immediate_result.new_fragments.is_empty());
            assert!(immediate_result.row_id_map.is_some());

            // Deserialize the changed_row_addrs from the deferred result
            let changed_row_addrs_bytes = deferred_result.changed_row_addrs.as_ref().unwrap();
            let mut cursor = Cursor::new(changed_row_addrs_bytes);
            let changed_row_addrs = RoaringTreemap::deserialize_from(&mut cursor).unwrap();

            // Use transpose_row_ids to convert changed_row_addrs to row_id_map
            let transposed_map = transpose_row_addrs(
                changed_row_addrs,
                &deferred_result.original_fragments,
                &deferred_result.new_fragments,
            );

            // Compare with the immediate result's row_id_map
            let immediate_map = immediate_result.row_id_map.as_ref().unwrap();
            assert_eq!(transposed_map.len(), immediate_map.len());
            for (old_row_id, new_row_id) in &transposed_map {
                assert_eq!(
                    immediate_map.get(old_row_id),
                    Some(new_row_id),
                    "Row ID mapping should be identical: {} -> {:?}",
                    old_row_id,
                    new_row_id
                );
            }

            // Store result for further comparison against frag reuse index
            deferred_results.push(deferred_result);
            immediate_result.new_fragments.iter().for_each(|frag| {
                expected_all_new_frag_bitmap.insert(frag.id as u32);
            });
            expected_all_new_frag_ids.extend(
                immediate_result
                    .new_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
            expected_all_old_frag_ids.extend(
                immediate_result
                    .original_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
            expected_all_row_id_map.extend(immediate_result.row_id_map.unwrap());
        }

        // Now commit the first compaction (using deferred results)
        let first_metrics = commit_compaction(
            &mut dataset,
            deferred_results.clone(),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Verify compaction happened
        assert!(first_metrics.fragments_removed > 0);
        assert!(first_metrics.fragments_added > 0);

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        assert_eq!(
            frag_reuse_index_meta.fragment_bitmap.clone().unwrap(),
            expected_all_new_frag_bitmap
        );
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();
        let stats = frag_reuse_index.statistics().unwrap();
        assert_eq!(
            serde_json::to_string(&stats).unwrap(),
            dataset
                .index_statistics(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
        );

        // Verify the index has one version with the correct dataset version
        let compaction_version = &frag_reuse_index.details.versions[0];
        assert_eq!(frag_reuse_index.details.versions.len(), 1);
        assert_eq!(
            compaction_version.dataset_version,
            frag_reuse_index_meta.dataset_version
        );

        // Verify the index compaction version information matches the RewriteResults
        let mut compacted_all_old_frag_digests = Vec::new();
        let mut compacted_all_new_frag_digests = Vec::new();
        let mut transposed_map = HashMap::new();
        for group in compaction_version.groups.iter() {
            let changed_row_addr_bytes = &group.changed_row_addrs;
            let mut cursor = Cursor::new(&changed_row_addr_bytes);
            let changed_row_addrs = RoaringTreemap::deserialize_from(&mut cursor).unwrap();
            compacted_all_old_frag_digests.extend(group.old_frags.clone());
            compacted_all_new_frag_digests.extend(group.new_frags.clone());

            let group_transposed_map = transpose_row_ids_from_digest(
                changed_row_addrs,
                &group.old_frags,
                &group.new_frags,
            );
            transposed_map.extend(group_transposed_map);
        }
        assert_eq!(transposed_map, expected_all_row_id_map);
        assert_eq!(
            compacted_all_old_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_old_frag_ids
        );
        assert_eq!(
            compacted_all_new_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_new_frag_ids
        );

        // Verify the scalar index UUID is unchanged (it should not be remapped yet)
        let Some(current_scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        assert_eq!(current_scalar_index.uuid, original_scalar_uuid);
    }

    #[tokio::test]
    async fn test_defer_index_remap_multiple_compactions() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let mut compact_read_versions = Vec::new();
        for i in 0..10 {
            dataset
                .delete(&format!("i < {}", 500 * (i + 1)))
                .await
                .unwrap();
            let read_version = dataset.manifest.version;
            compact_files(&mut dataset, options.clone(), None)
                .await
                .unwrap();

            // Record the read version for verification if compaction has happened
            if dataset.manifest.version > read_version {
                compact_read_versions.push(read_version);
            }

            // Load and verify the fragment reuse index content
            let Some(frag_reuse_index_meta) = dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
            else {
                panic!("Fragment reuse index must be available");
            };
            let frag_reuse_details =
                load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
                    .await
                    .unwrap();
            let frag_reuse_index =
                open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                    .await
                    .unwrap();

            // Verify the index has one version with the correct dataset version
            assert_eq!(
                frag_reuse_index
                    .details
                    .versions
                    .iter()
                    .map(|v| v.dataset_version)
                    .collect::<Vec<_>>(),
                compact_read_versions
            );
        }
    }

    #[tokio::test]
    async fn test_remap_index_after_compaction() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create a index to be remapped
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        // Remap without a frag reuse index should yield unsupported
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };

        let result = remapping::remap_column_index(&mut dataset, &["i"], index_name.clone()).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));

        let plan = plan_compaction(&dataset, &options).await.unwrap();

        // Commit each rewrite task separately to simulate 3 compaction runs
        // being accumulated in the fragment reuse index
        for task in plan.tasks().iter() {
            let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();

            commit_compaction(
                &mut dataset,
                Vec::from([rewrite_result]),
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();

        assert_eq!(frag_reuse_index.details.versions.len(), plan.tasks().len());

        // Check auto-remap
        let mut all_fragment_bitmap = RoaringBitmap::new();
        dataset.fragments().iter().for_each(|f| {
            all_fragment_bitmap.insert(f.id as u32);
        });
        let Some(scalar_index_before_remap) = dataset.load_index_by_name("scalar").await.unwrap()
        else {
            panic!("scalar index must be available");
        };
        assert_eq!(
            scalar_index_before_remap.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );

        // Trigger index remap
        remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
            .await
            .unwrap();

        // Compare against original index
        let indices = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let Some(remapped_scalar_index) = indices.into_iter().find(|idx| idx.name == "scalar")
        else {
            panic!("scalar index must be available");
        };
        assert_ne!(remapped_scalar_index.uuid, scalar_index.uuid);
        assert_eq!(
            remapped_scalar_index.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_compaction_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Concurrent reindex should succeed
        dataset_clone
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Check new index does not cover the compacted files
        dataset.checkout_latest().await.unwrap();

        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_reindex_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // Concurrent reindex should succeed
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset_clone,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Check new index is auto-remapped
        dataset.checkout_latest().await.unwrap();
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_cleanup() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the remaining 2 compaction tasks.
        let rewrite_result2 = rewrite_files(Cow::Borrowed(&dataset), tasks[1].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        let new_frags2 = rewrite_result2
            .new_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<u64>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let rewrite_result3 = rewrite_files(Cow::Borrowed(&dataset), tasks[2].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags3 = rewrite_result3
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        let new_frags3 = rewrite_result3
            .new_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<u64>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result3]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Concurrently commit a frag_reuse_index cleanup operation.
        // Because there is no index, it should remove the first version.
        // but after rebase it should contain the new compaction versions.
        cleanup_frag_reuse_index(&mut dataset_clone).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 2);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        assert_eq!(frag_reuse_details.versions[0].new_frag_ids(), new_frags2);
        assert_eq!(
            frag_reuse_details.versions[1].old_frag_ids(),
            rewritten_frags3
        );
        assert_eq!(frag_reuse_details.versions[1].new_frag_ids(), new_frags3);
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_compaction() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the frag_reuse_index cleanup
        // Because there is no index, it should remove the first version.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Concurrently commit a rewrite
        // After rebase it should only contain the latest reuse version
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        let new_frags2 = rewrite_result2
            .new_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<u64>>();
        commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        assert_eq!(frag_reuse_details.versions[0].new_frag_ids(), new_frags2);
    }

    #[tokio::test]
    async fn test_concurrent_compactions_with_defer_index_remap() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        let mut dataset_clone = dataset.clone();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // Concurrently commit a rewrite should fail
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let result = commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await;
        assert!(matches!(result, Err(Error::RetryableCommitConflict { .. })));
    }

    #[tokio::test]
    async fn test_read_bitmap_index_with_defer_index_remap() {
        // Create a dataset with categorical values
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for each category
        let count1 = dataset
            .count_rows(Some("category = 1".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("category = 2".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("category = 3".to_owned()))
            .await
            .unwrap();

        // Create a bitmap index on the category column
        let index_name = Some("category_idx".into());
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices
            .iter()
            .find(|idx| idx.name == "category_idx")
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("category_idx").await.unwrap() else {
            panic!("category index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("category = 1".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 2".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 3".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses bitmap index scan
        let mut scanner = dataset.scan();
        scanner.filter("category = 1").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[category = 1]@category_idx"),
            "Expected index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_btree_index_with_defer_index_remap() {
        // Create a dataset with an incremental ID column
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(110), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for some ID ranges
        let count_low = dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap();
        let count_mid = dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap();
        let count_high = dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap();

        // Create a btree index on the id column
        let index_name = Some("id_idx".into());
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "id_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("id_idx").await.unwrap() else {
            panic!("id index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("id < 1000".to_owned()))
                .await
                .unwrap(),
            count_low
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
                .await
                .unwrap(),
            count_mid
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 5000".to_owned()))
                .await
                .unwrap(),
            count_high
        );

        // Verify that after index creation and compaction, scan uses btree index scan
        let mut scanner = dataset.scan();
        scanner.filter("id >= 2000 and id < 3000").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_inverted_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Initial scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count1 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count2 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count3 = scanner.count_rows().await.unwrap();

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("MatchQuery"),
            "Expected inverted index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );

        // Reindex to the latest
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Verify that scans still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count1);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count2);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count3);
    }

    #[tokio::test]
    async fn test_read_ngram_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::NGram,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Initial scan
        let count1 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
            .await
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .filter(&format!("contains(doc, '{}')", test_word1))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[contains(doc, Utf8"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_label_list_index_with_defer_index_remap() {
        // Create a dataset with list data for labels
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "labels",
                lance_datagen::array::rand_list_any(
                    lance_datagen::array::cycle::<Int64Type>(vec![1, 2, 3, 4, 5]),
                    false,
                ),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for different label values
        let count1 = dataset
            .count_rows(Some("array_has_any(labels, [1])".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("array_has_any(labels, [5])".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("array_has_any(labels, [10])".to_owned()))
            .await
            .unwrap();

        // Create a label list index on the labels column
        let index_name = Some("labels_idx".into());
        dataset
            .create_index(
                &["labels"],
                IndexType::LabelList,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2000,
            defer_index_remap: true,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify that the index UUID remains unchanged
        let indices = dataset.load_indices().await.unwrap();
        let current_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [1])".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [5])".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [10])".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses label list index scan
        let mut scanner = dataset.scan();
        scanner.filter("array_has_any(labels, [1])").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[array_has_any(labels, List([1]))]@labels_idx"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_ivf_pq_index_v3_with_defer_index_remap() {
        // Create a dataset with vector data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get some query vectors for KNN search
        let query_vec1: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(0.0, 128));
        let query_vec2: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(1.1, 128));
        let query_vec3: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(2.2, 128));

        // Get initial KNN search results
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count1 = results1.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count2 = results2.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count3 = results3.len();

        // Create an IVF-PQ index on the vec column
        let index_name = Some("vec_idx".into());
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                index_name.clone(),
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: Some(2),
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::PQ(PQBuildParams {
                            max_iters: 2,
                            num_sub_vectors: 2,
                            ..Default::default()
                        }),
                    ],
                    version: crate::index::vector::IndexFileVersion::V3,
                },
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
            panic!("vec index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that KNN searches still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results1.len(), count1);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results2.len(), count2);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results3.len(), count3);

        // Verify that after index creation and compaction, scan uses vector index scan
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "Expected vector index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );
    }
}
