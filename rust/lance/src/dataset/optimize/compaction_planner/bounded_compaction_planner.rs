// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::dataset::fragment::FileFragment;
use crate::dataset::optimize::{
    build_compaction_candidacy, collect_metrics, finalize_candidate_bins,
    get_indices_containing_frag, load_index_fragmaps, CandidateBin,
};
use crate::Error;

use lance_table::format::Fragment;
use serde::{Deserialize, Serialize};
use snafu::location;

use crate::Result;
use crate::{
    dataset::optimize::{CompactionOptions, CompactionPlan, CompactionPlanner},
    Dataset,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedCompactionPlannerOptions {
    pub max_compaction_size: Option<usize>,
    pub max_compaction_rows: Option<usize>,
}

impl From<&CompactionOptions> for BoundedCompactionPlannerOptions {
    fn from(options: &CompactionOptions) -> Self {
        Self {
            max_compaction_size: options.max_compaction_bytes,
            max_compaction_rows: options.max_compaction_rows,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct BoundUsage {
    input_bytes: u64,
    input_rows: usize,
}

impl BoundedCompactionPlannerOptions {
    fn validate(&mut self) -> Result<()> {
        if self.max_compaction_size.is_none() && self.max_compaction_rows.is_none() {
            return Err(Error::InvalidInput {
                source: "max_compaction_size and max_compaction_rows cannot be both None when using BoundedCompactionPlanner".into(),
                location: location!(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BoundedCompactionPlanner {
    options: CompactionOptions,
    bounded_compaction_planner_options: BoundedCompactionPlannerOptions,
}

impl BoundedCompactionPlanner {
    pub fn new(
        mut options: CompactionOptions,
        mut bounded_compaction_planner_options: BoundedCompactionPlannerOptions,
    ) -> Self {
        options.validate();
        let _ = bounded_compaction_planner_options.validate();

        Self {
            options,
            bounded_compaction_planner_options,
        }
    }
}

#[async_trait::async_trait]
impl CompactionPlanner for BoundedCompactionPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
        let index_fragmaps = load_index_fragmaps(dataset).await?;

        let mut usage = BoundUsage::default();
        let mut candidate_bins: Vec<CandidateBin> = Vec::new();
        let mut current_bin: Option<CandidateBin> = None;
        let dataset_arc = Arc::new(dataset.clone());

        for (position, fragment) in dataset.manifest.fragments.iter().enumerate() {
            let file_fragment = FileFragment::new(dataset_arc.clone(), fragment.clone());
            let metrics = collect_metrics(&file_fragment).await?;
            let candidacy = build_compaction_candidacy(&self.options, &metrics);
            let indices = get_indices_containing_frag(&index_fragmaps, fragment.id as u32);

            match (candidacy, &mut current_bin) {
                (None, None) => {}
                (Some(candidacy), None) => {
                    self.update_usage(&mut usage, fragment);
                    if !candidate_bins.is_empty() && self.out_of_usage(&mut usage) {
                        break;
                    }
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment.clone()],
                        pos_range: position..(position + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                    });
                }
                (Some(candidacy), Some(bin)) => {
                    if bin.indices == indices {
                        self.update_usage(&mut usage, fragment);
                        if self.out_of_usage(&mut usage)  {
                            // ignore current loop and return
                            candidate_bins.push(current_bin.take().unwrap());
                            break;
                        }
                        bin.fragments.push(fragment.clone());
                        bin.pos_range.end += 1;
                        bin.candidacy.push(candidacy);
                        bin.row_counts.push(metrics.num_rows());
                    } else {
                        // Index set is different.  Complete previous bin and start new one
                        candidate_bins.push(current_bin.take().unwrap());

                        self.update_usage(&mut usage, fragment);
                        if self.out_of_usage(&mut usage)  {
                            break;
                        }
                        current_bin = Some(CandidateBin {
                            fragments: vec![fragment.clone()],
                            pos_range: position..(position + 1),
                            candidacy: vec![candidacy],
                            row_counts: vec![metrics.num_rows()],
                            indices,
                        });
                    }
                }
                (None, Some(_)) => {
                    // current bin is completed
                    let bin = current_bin.take().unwrap();
                    if !bin.is_noop() {
                        candidate_bins.push(bin);
                    }
                }
            }
        }

        // Flush the last bin
        if let Some(bin) = current_bin {
            candidate_bins.push(bin);
        }

        let mut compaction_plan =
            CompactionPlan::new(dataset.manifest.version, self.options.clone());
        compaction_plan.extend_tasks(finalize_candidate_bins(
            candidate_bins,
            self.options.target_rows_per_fragment,
        ));

        Ok(compaction_plan)
    }
}

impl BoundedCompactionPlanner {
    /// Check if the usage exceeds the max compaction size or max compaction rows.
    fn out_of_usage(&self, usage: &mut BoundUsage) -> bool {
        match (
            self.bounded_compaction_planner_options.max_compaction_size,
            self.bounded_compaction_planner_options.max_compaction_rows,
        ) {
            (None, None) => false,
            (Some(max_compaction_size), None) => usage.input_bytes > max_compaction_size as u64,
            (None, Some(max_compaction_rows)) => usage.input_rows > max_compaction_rows,
            (Some(max_compaction_size), Some(max_compaction_rows)) => {
                usage.input_bytes > max_compaction_size as u64
                    || usage.input_rows > max_compaction_rows
            }
        }
    }

    /// Update the usage with the fragment.
    fn update_usage(&self, usage: &mut BoundUsage, fragment: &Fragment) {
        usage.input_bytes += fragment
            .files
            .iter()
            .map(|data_file| {
                data_file
                    .file_size_bytes
                    .get()
                    .map(|file_size| file_size.get())
                    .unwrap_or(0)
            })
            .sum::<u64>();
        usage.input_rows += fragment.num_rows().unwrap_or(0);
    }
}

