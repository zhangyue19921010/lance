// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::fragment::write::generate_random_filename;
use crate::dataset::optimize::load_row_id_sequence;
use crate::dataset::WriteParams;
use crate::dataset::DATA_DIR;
use crate::datatypes::Schema;
use crate::Dataset;
use crate::Result;
use lance_arrow::DataTypeExt;
use lance_encoding::decoder::{ColumnInfo, PageEncoding, PageInfo as DecPageInfo};
use lance_encoding::version::LanceFileVersion;
use lance_file::format::pbfile;
use lance_file::reader::FileReader as LFReader;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::object_writer::ObjectWriter;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Writer;
use lance_table::format::{DataFile, Fragment, RowIdMeta};
use lance_table::rowids::{write_row_ids, RowIdSequence};
use prost::Message;
use prost_types::Any;
use std::ops::Range;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Rewrite the files in a single task using binary copy semantics.
///
/// Flow overview (per task):
/// fragments
///   └── data files
///         └── columns
///               └── pages (batched reads) -> aligned writes -> page metadata
///         └── column buffers -> aligned writes -> buffer metadata
///   └── flush when target rows reached -> write footer -> fragment metadata
///   └── final flush for remaining rows
///
/// Behavior highlights:
/// - Assumes all input files share the same Lance file version; version drives column-count
///   calculation (v2.0 includes structural headers, v2.1+ only leaf columns).
/// - Preserves stable row ids by concatenating row-id sequences when enabled.
/// - Enforces 64-byte alignment for page and buffer writes to satisfy downstream readers.
/// - For v2.0, preserves single-page structural headers and normalizes their row counts/priority.
/// - Flushes an output file once `max_rows_per_file` rows are accumulated, then repeats.
///
/// Parameters:
/// - `dataset`: target dataset (for storage/config and schema).
/// - `fragments`: fragments to merge via binary copy (assumed consistent versions).
/// - `params`: write parameters (uses `max_rows_per_file`).
/// - `read_batch_bytes_opt`: optional I/O batch size when coalescing page reads.
pub async fn rewrite_files_binary_copy(
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
        for (col_idx, field) in schema.fields_pre_order().enumerate() {
            // Only mark non-packed Struct fields (lists remain as leaf data carriers)
            let is_non_leaf = field.data_type().is_struct() && !field.is_packed_struct();
            is_non_leaf_column[col_idx] = is_non_leaf;
        }
    }

    let mut out: Vec<Fragment> = Vec::new();
    let mut current_writer: Option<ObjectWriter> = None;
    let mut current_filename: Option<String> = None;
    let mut current_pos: u64 = 0;
    let mut current_page_table: Vec<ColumnInfo> = Vec::new();

    // Column-list<Page-List<DecPageInfo>>
    let mut col_pages: Vec<Vec<DecPageInfo>> = std::iter::repeat_with(Vec::<DecPageInfo>::new)
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
            let src_colum_infos = file_meta.column_infos.clone();
            // Initialize current_page_table
            if current_page_table.is_empty() {
                current_page_table = src_colum_infos
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
            for (col_idx, src_column_info) in src_colum_infos.iter().enumerate() {
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
                    let path = dataset.base.child(DATA_DIR).child(filename.as_str());
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
                            .map(|(_, size)| *size)
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

                    // Append these row IDs to the accumulated sequence for the current output
                    current_row_ids.extend(slice.iter().collect());

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
                        && !pages_vec.is_empty()
                    {
                        pages_vec[0].num_rows = total_rows_in_current;
                        pages_vec[0].priority = 0;
                        pages_vec.truncate(1);
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
                    if field.is_packed_struct() || field.children.is_empty() || !is_structural {
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
    } // Finished writing all fragments; any remaining data in memory will be flushed below

    if total_rows_in_current > 0 {
        // Flush remaining rows as a final output file
        // v2_0 compatibility: same single-page enforcement applies for the final file close
        let mut final_cols: Vec<Arc<ColumnInfo>> = Vec::with_capacity(column_count);
        for (i, ci) in current_page_table.iter().enumerate() {
            // For v2_0 struct headers, force a single page and set num_rows to total
            let mut pages_vec = std::mem::take(&mut col_pages[i]);
            if version == LanceFileVersion::V2_0
                && is_non_leaf_column.get(i).copied().unwrap_or(false)
                && !pages_vec.is_empty()
            {
                pages_vec[0].num_rows = total_rows_in_current;
                pages_vec[0].priority = 0;
                pages_vec.truncate(1);
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
            let path = dataset.base.child(DATA_DIR).child(filename.as_str());
            let writer = dataset.object_store.create(&path).await?;
            current_writer = Some(writer);
            current_filename = Some(filename);
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
            if field.is_packed_struct() || field.children.is_empty() || !is_structural {
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
