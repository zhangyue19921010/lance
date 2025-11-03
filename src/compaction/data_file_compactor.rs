use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use anyhow::{Result, Context};

/// Data file 的元数据
#[derive(Debug, Clone)]
pub struct DataFileMetadata {
    pub file_path: PathBuf,
    pub pages: Vec<PageMetadata>,
}

/// Page 的元数据
#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub offset: u64,
    pub length: u64,
    pub num_rows: u64,
}

/// Data File Compactor
/// 将多个 data file 合并成一个,保持 page 不变
pub struct DataFileCompactor {
    source_files: Vec<DataFileMetadata>,
    output_path: PathBuf,
    buffer_size: usize,
}

impl DataFileCompactor {
    /// 创建新的 compactor
    pub fn new(source_files: Vec<DataFileMetadata>, output_path: PathBuf) -> Self {
        Self {
            source_files,
            output_path,
            buffer_size: 8 * 1024 * 1024, // 8MB buffer
        }
    }

    /// 执行 compaction
    pub fn compact(&self) -> Result<DataFileMetadata> {
        let mut output_file = File::create(&self.output_path)
            .context("Failed to create output file")?;

        let mut new_pages = Vec::new();
        let mut current_offset = 0u64;

        // 遍历所有源文件
        for source_meta in &self.source_files {
            let mut source_file = File::open(&source_meta.file_path)
                .context(format!("Failed to open source file: {:?}", source_meta.file_path))?;

            // 复制每个 page
            for page in &source_meta.pages {
                // 定位到源文件中的 page 位置
                source_file.seek(SeekFrom::Start(page.offset))
                    .context("Failed to seek in source file")?;

                // 复制 page 数据
                let bytes_copied = self.copy_page_data(
                    &mut source_file,
                    &mut output_file,
                    page.length
                )?;

                // 记录新的 page 元数据
                new_pages.push(PageMetadata {
                    offset: current_offset,
                    length: bytes_copied,
                    num_rows: page.num_rows,
                });

                current_offset += bytes_copied;
            }
        }

        output_file.flush().context("Failed to flush output file")?;

        Ok(DataFileMetadata {
            file_path: self.output_path.clone(),
            pages: new_pages,
        })
    }

    /// 复制 page 数据
    fn copy_page_data(
        &self,
        source: &mut File,
        dest: &mut File,
        length: u64,
    ) -> Result<u64> {
        let mut buffer = vec![0u8; self.buffer_size];
        let mut remaining = length;
        let mut total_copied = 0u64;

        while remaining > 0 {
            let to_read = std::cmp::min(remaining, self.buffer_size as u64) as usize;
            let bytes_read = source.read(&mut buffer[..to_read])
                .context("Failed to read from source")?;

            if bytes_read == 0 {
                break;
            }

            dest.write_all(&buffer[..bytes_read])
                .context("Failed to write to destination")?;

            remaining -= bytes_read as u64;
            total_copied += bytes_read as u64;
        }

        Ok(total_copied)
    }

    /// 获取合并后的统计信息
    pub fn get_stats(&self) -> CompactionStats {
        let total_pages: usize = self.source_files.iter()
            .map(|f| f.pages.len())
            .sum();

        let total_rows: u64 = self.source_files.iter()
            .flat_map(|f| &f.pages)
            .map(|p| p.num_rows)
            .sum();

        CompactionStats {
            num_source_files: self.source_files.len(),
            total_pages,
            total_rows,
        }
    }
}

/// Compaction 统计信息
#[derive(Debug)]
pub struct CompactionStats {
    pub num_source_files: usize,
    pub total_pages: usize,
    pub total_rows: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_compact_data_files() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // 创建测试文件 1
        let file1_path = temp_dir.path().join("data1.lance");
        let mut file1 = File::create(&file1_path)?;
        file1.write_all(b"page1_data")?;
        file1.write_all(b"page2_data")?;
        file1.write_all(b"page3_data")?;

        let meta1 = DataFileMetadata {
            file_path: file1_path,
            pages: vec![
                PageMetadata { offset: 0, length: 10, num_rows: 100 },
                PageMetadata { offset: 10, length: 10, num_rows: 100 },
                PageMetadata { offset: 20, length: 10, num_rows: 100 },
            ],
        };

        // 创建测试文件 2
        let file2_path = temp_dir.path().join("data2.lance");
        let mut file2 = File::create(&file2_path)?;
        file2.write_all(b"page4_data")?;
        file2.write_all(b"page5_data")?;
        file2.write_all(b"page6_data")?;

        let meta2 = DataFileMetadata {
            file_path: file2_path,
            pages: vec![
                PageMetadata { offset: 0, length: 10, num_rows: 100 },
                PageMetadata { offset: 10, length: 10, num_rows: 100 },
                PageMetadata { offset: 20, length: 10, num_rows: 100 },
            ],
        };

        // 执行 compaction
        let output_path = temp_dir.path().join("compacted.lance");
        let compactor = DataFileCompactor::new(
            vec![meta1, meta2],
            output_path.clone(),
        );

        let result = compactor.compact()?;

        // 验证结果
        assert_eq!(result.pages.len(), 6);
        assert_eq!(result.pages[0].offset, 0);
        assert_eq!(result.pages[3].offset, 30);

        Ok(())
    }
}

