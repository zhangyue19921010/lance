use lance::compaction::{DataFileCompactor, DataFileMetadata, PageMetadata};
use std::path::PathBuf;
use anyhow::Result;

fn main() -> Result<()> {
    // 准备源文件元数据
    let source_files = vec![
        DataFileMetadata {
            file_path: PathBuf::from("data/file1.lance"),
            pages: vec![
                PageMetadata { offset: 0, length: 1024, num_rows: 100 },
                PageMetadata { offset: 1024, length: 2048, num_rows: 200 },
                PageMetadata { offset: 3072, length: 1024, num_rows: 100 },
            ],
        },
        DataFileMetadata {
            file_path: PathBuf::from("data/file2.lance"),
            pages: vec![
                PageMetadata { offset: 0, length: 2048, num_rows: 200 },
                PageMetadata { offset: 2048, length: 2048, num_rows: 200 },
                PageMetadata { offset: 4096, length: 1024, num_rows: 100 },
            ],
        },
    ];

    // 创建 compactor
    let output_path = PathBuf::from("data/compacted.lance");
    let compactor = DataFileCompactor::new(source_files, output_path);

    // 打印统计信息
    let stats = compactor.get_stats();
    println!("Compaction Stats:");
    println!("  Source files: {}", stats.num_source_files);
    println!("  Total pages: {}", stats.total_pages);
    println!("  Total rows: {}", stats.total_rows);

    // 执行 compaction
    let result = compactor.compact()?;

    println!("\nCompaction completed!");
    println!("Output file: {:?}", result.file_path);
    println!("Pages in output: {}", result.pages.len());

    Ok(())
}
pub mod data_file_compactor;

pub use data_file_compactor::{
    DataFileCompactor,
    DataFileMetadata,
    PageMetadata,
    CompactionStats,
};

