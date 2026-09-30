// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index optimization split into plan, execute and commit steps, so that the
//! work of merging index segments can be distributed the same way compaction
//! is (see [`crate::dataset::optimize`]).

/// Serde adapter for an [`IndexMetadata`](lance_table::format::IndexMetadata)
/// carried inside a task result.
///
/// `IndexMetadata` has no serde implementation; its protobuf form is the
/// serialization the manifest already uses, so a result carries the protobuf
/// bytes as a hex string.
// Used by `IndexOptimizeResult` once the task types land.
#[allow(dead_code)]
mod segment_serde {
    use lance_table::format::{IndexMetadata, pb};
    use prost::Message;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        segment: &Option<IndexMetadata>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match segment {
            Some(segment) => {
                let bytes = pb::IndexMetadata::from(segment).encode_to_vec();
                serializer.serialize_some(&hex::encode(bytes))
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<IndexMetadata>, D::Error> {
        let encoded: Option<String> = Option::deserialize(deserializer)?;
        encoded
            .map(|encoded| {
                let bytes = hex::decode(encoded).map_err(serde::de::Error::custom)?;
                let proto = pb::IndexMetadata::decode(bytes.as_slice())
                    .map_err(serde::de::Error::custom)?;
                IndexMetadata::try_from(proto).map_err(serde::de::Error::custom)
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_table::format::{IndexFile, IndexMetadata};
    use roaring::RoaringBitmap;
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Carrier {
        #[serde(with = "segment_serde")]
        segment: Option<IndexMetadata>,
    }

    #[test]
    fn segment_serde_round_trips_through_protobuf() {
        let segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "vector_idx".to_string(),
            fields: vec![1],
            covering_fields: vec![],
            dataset_version: 7,
            fragment_bitmap: Some(RoaringBitmap::from_iter([1u32, 2, 3])),
            index_details: Some(Arc::new(crate::index::vector_index_details_default())),
            index_version: 3,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(vec![IndexFile {
                path: "index.idx".to_string(),
                size_bytes: 10,
            }]),
        };
        let json = serde_json::to_string(&Carrier {
            segment: Some(segment.clone()),
        })
        .unwrap();
        let decoded: Carrier = serde_json::from_str(&json).unwrap();
        let decoded = decoded.segment.unwrap();
        assert_eq!(decoded.uuid, segment.uuid);
        assert_eq!(decoded.name, segment.name);
        assert_eq!(decoded.fields, segment.fields);
        assert_eq!(decoded.dataset_version, segment.dataset_version);
        assert_eq!(decoded.fragment_bitmap, segment.fragment_bitmap);
        assert_eq!(decoded.index_details, segment.index_details);
        assert_eq!(decoded.index_version, segment.index_version);
        assert_eq!(decoded.files, segment.files);
        // Protobuf keeps millisecond precision.
        assert_eq!(
            decoded.created_at.map(|t| t.timestamp_millis()),
            segment.created_at.map(|t| t.timestamp_millis())
        );

        let json = serde_json::to_string(&Carrier { segment: None }).unwrap();
        let decoded: Carrier = serde_json::from_str(&json).unwrap();
        assert!(decoded.segment.is_none());
    }
}
