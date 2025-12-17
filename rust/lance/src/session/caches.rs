// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance datasets. They are organized in a hierarchical manner to
//! avoid collisions.
//!
//!  GlobalMetadataCache
//!     │
//!     ├─► DSMetadataCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► FileMetadataCache (prefixed by file path)

use std::{borrow::Cow, ops::Deref};

use deepsize::{Context, DeepSizeOf};
use lance_core::{
    cache::{CacheKey, LanceCache},
    utils::{deletion::DeletionVector, mask::RowIdMask},
};
use lance_table::{
    format::{DeletionFile, Manifest},
    rowids::{RowIdIndex, RowIdSequence},
};
use object_store::path::Path;

use crate::dataset::transaction::Transaction;

/// A type-safe wrapper around a LanceCache that enforces namespaces for dataset metadata.
pub struct GlobalMetadataCache(pub(super) LanceCache);

impl GlobalMetadataCache {
    pub fn for_dataset(&self, uri: &str) -> DSMetadataCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSMetadataCache(self.0.with_key_prefix(uri))
    }

    /// Create a file-specific metadata cache with the given prefix.
    /// This is used by file readers and other components that need file-level caching.
    pub(crate) fn file_metadata_cache(&self, path: &Path) -> LanceCache {
        self.0.with_key_prefix(path.as_ref())
    }
}

impl Clone for GlobalMetadataCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl DeepSizeOf for GlobalMetadataCache {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// A type-safe wrapper around a LanceCache that enforces namespaces and keys
/// for dataset metadata.
pub struct DSMetadataCache(pub(crate) LanceCache);

impl Deref for DSMetadataCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Cache key types for type-safe cache access
#[derive(Debug)]
pub struct ManifestKey<'a> {
    pub version: u64,
    pub e_tag: Option<&'a str>,
}

impl CacheKey for ManifestKey<'_> {
    type ValueType = Manifest;

    fn key(&self) -> Cow<'_, str> {
        if let Some(e_tag) = self.e_tag {
            Cow::Owned(format!("manifest/{}/{}", self.version, e_tag))
        } else {
            Cow::Owned(format!("manifest/{}", self.version))
        }
    }
}

#[derive(Debug)]
pub struct TransactionKey {
    pub version: u64,
}

impl CacheKey for TransactionKey {
    type ValueType = Transaction;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("txn/{}", self.version))
    }
}

#[derive(Debug)]
pub struct DeletionFileKey<'a> {
    pub fragment_id: u64,
    pub deletion_file: &'a DeletionFile,
}

impl CacheKey for DeletionFileKey<'_> {
    type ValueType = DeletionVector;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "deletion/{}/{}/{}/{}",
            self.fragment_id,
            self.deletion_file.read_version,
            self.deletion_file.id,
            self.deletion_file.file_type.suffix()
        ))
    }
}

#[derive(Debug)]
pub struct RowIdMaskKey {
    pub version: u64,
}

impl CacheKey for RowIdMaskKey {
    type ValueType = RowIdMask;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_mask/{}", self.version))
    }
}

#[derive(Debug)]
pub struct RowIdIndexKey {
    pub version: u64,
}

impl CacheKey for RowIdIndexKey {
    type ValueType = RowIdIndex;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("row_id_index/{}", self.version))
    }
}

#[derive(Debug)]
pub struct RowIdSequenceKey {
    pub version: u64,
    pub fragment_id: u64,
}

impl CacheKey for RowIdSequenceKey {
    type ValueType = RowIdSequence;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "row_id_sequence/{}/{}",
            self.version, self.fragment_id
        ))
    }
}

impl DSMetadataCache {
    /// Create a file-specific metadata cache with the given prefix.
    /// This is used by file readers and other components that need file-level caching.
    pub(crate) fn file_metadata_cache(&self, prefix: &Path) -> LanceCache {
        self.0.with_key_prefix(prefix.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use std::sync::Arc;

    #[tokio::test]
    async fn rowidsequence_cache_isolated_by_version() {
        // Verify RowIdSequence cache isolation by dataset version.
        // Simulate two versions (v2 and v3) of the same fragment id (42) in the same dataset URI.
        // Ensure lookups do not cross-contaminate between versions.
        let session = Session::default();
        let ds_cache = session.metadata_cache.for_dataset("test://uri");

        // Build two different sequences representing two versions of the same fragment.
        let seq_v2_bytes = {
            let arr: Vec<u64> = vec![1, 2, 3];
            lance_table::rowids::write_row_ids(&RowIdSequence::from(arr.as_slice()))
        };
        let seq_v3_bytes = {
            let arr: Vec<u64> = vec![10, 20];
            lance_table::rowids::write_row_ids(&RowIdSequence::from(arr.as_slice()))
        };
        let seq_v2 = Arc::new(lance_table::rowids::read_row_ids(&seq_v2_bytes).unwrap());
        let seq_v3 = Arc::new(lance_table::rowids::read_row_ids(&seq_v3_bytes).unwrap());

        // Insert V2/frag 42
        let key_v2 = RowIdSequenceKey {
            version: 2,
            fragment_id: 42,
        };
        ds_cache.insert_with_key(&key_v2, seq_v2.clone()).await;

        // Ensure V3/frag 42 does not hit V2 entry
        let key_v3 = RowIdSequenceKey {
            version: 3,
            fragment_id: 42,
        };
        let got = ds_cache.get_with_key(&key_v3).await;
        assert!(
            got.is_none(),
            "V3 lookup should not hit V2 cached entry for same fragment"
        );

        // Ensure V2 lookup hits V2 entry and contents match exactly
        let got_v2 = ds_cache
            .get_with_key(&key_v2)
            .await
            .expect("V2 entry should be present");
        assert_eq!(
            got_v2, seq_v2,
            "V2 cached sequence should match inserted sequence"
        );

        // Insert V3/frag 42 and then ensure hit works
        ds_cache.insert_with_key(&key_v3, seq_v3.clone()).await;
        let got_v3 = ds_cache
            .get_with_key(&key_v3)
            .await
            .expect("V3 entry should be present after insertion");
        assert_eq!(
            got_v3, seq_v3,
            "V3 cached sequence should match inserted sequence"
        );

        assert_ne!(
            got_v2.get(0),
            got_v3.get(0),
            "V2 and V3 sequences should not share identical first element"
        );
    }
}
