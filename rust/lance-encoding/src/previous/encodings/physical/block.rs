// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::DataType;

use crate::{
    data::{BlockInfo, DataBlock, OpaqueBlock},
    encodings::physical::block::{CompressedBufferEncoder, CompressionConfig, CompressionScheme},
    format::ProtobufUtils,
    previous::encoder::{ArrayEncoder, EncodedArray},
};

use lance_core::Result;

/// 触发实际的数据压缩
/// 
/// 
impl ArrayEncoder for CompressedBufferEncoder {
    fn encode(
        &self,
        data: DataBlock,
        _data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        
        // 获取未压缩的原始数据
        let uncompressed_data = data.as_fixed_width().unwrap();

        // 构建compressed buf，用于压缩数据
        let mut compressed_buf = Vec::with_capacity(uncompressed_data.data.len());
        
        // 基于compressor，压缩数据，这里以zstd为例，展开源码阅读
        // compress 方法会将data数据压缩，并写入到 compressed_buf 数据结构中
        self.compressor
            .compress(&uncompressed_data.data, &mut compressed_buf)?;

        // 构建 compressed_data ，此处为DataBlock::Opaque
        // 该对象，持有压缩后的数据，当前buffer中的条数，以及blockInfo初始对象
        let compressed_data = DataBlock::Opaque(OpaqueBlock {
            buffers: vec![compressed_buf.into()],
            num_values: uncompressed_data.num_values,
            block_info: BlockInfo::new(),
        });

        // 获取 comp_buf_index
        let comp_buf_index = *buffer_index;
        *buffer_index += 1;

        // 构建扁平编码后的 array encoding 元数据信息
        let encoding = ProtobufUtils::flat_encoding(
            uncompressed_data.bits_per_value,
            comp_buf_index,
            Some(CompressionConfig::new(CompressionScheme::Zstd, None)),
        );

        // 组装为 EncodedArray 并返回，该对象持有压缩后的数据 data，以及当前压缩编码的元数据信息 encoding
        Ok(EncodedArray {
            data: compressed_data,
            encoding,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{buffer::LanceBuffer, data::FixedWidthDataBlock};

    use super::*;

    #[test]
    fn test_compressed_buffer_encoder() {
        let encoder = CompressedBufferEncoder::default();
        let data = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            num_values: 8,
            block_info: BlockInfo::new(),
        });

        let mut buffer_index = 0;
        let encoded_array_result = encoder.encode(data, &DataType::Int64, &mut buffer_index);
        assert!(encoded_array_result.is_ok(), "{:?}", encoded_array_result);
        let encoded_array = encoded_array_result.unwrap();
        assert_eq!(encoded_array.data.num_values(), 8);
        let buffers = encoded_array.data.into_buffers();
        assert_eq!(buffers.len(), 1);
        assert!(buffers[0].len() < 64 * 8);
    }
}
