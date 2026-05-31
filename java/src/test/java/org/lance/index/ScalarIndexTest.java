/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.index;

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.TestUtils;
import org.lance.WriteParams;
import org.lance.index.scalar.ScalarIndexParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScalarIndexTest {

  @Test
  public void testCreateBTreeIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("btree_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create BTree scalar index parameters
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create BTree index on 'id' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("id"),
                IndexType.BTREE,
                Optional.of("btree_id_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("btree_id_index", index.name());
        assertNotNull(index.uuid());
        assertFalse(index.fields().isEmpty());

        // Verify index was created and is in the list
        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be in the list of indexes: " + dataset.listIndexes());

        // TODO: Verify zone_size parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the zone_size parameter was correctly set to 2048
      }
    }
  }

  @Test
  public void testBtreeMergeIndexMetadataSoftBreak(@TempDir Path tempDir) throws Exception {
    // The historical fragment-based distributed BTree flow (per-fragment build into a
    // shared uuid + mergeIndexMetadata) has been retired in favor of the segmented flow.
    // A fragment-scoped build still succeeds, but mergeIndexMetadata(BTREE) must now fail
    // fast with a migration error.
    String datasetPath = tempDir.resolve("build_index_distributedly").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // 1. write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        UUID uuid = UUID.randomUUID();

        // 2. a distributed worker can still build one fragment-scoped segment (uncommitted).
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("name"), IndexType.BTREE, indexParams)
                .withIndexName("test_index")
                .withIndexUUID(uuid.toString())
                .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                .build());

        // 3. but the shard-then-merge commit step is no longer supported.
        Exception ex =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    dataset.mergeIndexMetadata(uuid.toString(), IndexType.BTREE, Optional.empty()));
        assertTrue(
            ex.getMessage() != null
                && ex.getMessage().contains("no longer supports merge_index_metadata"),
            "expected BTree merge_index_metadata soft-break error, got: " + ex.getMessage());
      }
    }
  }

  @Test
  public void testCreateZonemapIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("zonemap_test").toString();
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("value", new ArrowType.Utf8())),
            null);

    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {

        // Create Zonemap scalar index parameters with rows_per_zone setting
        ScalarIndexParams scalarParams =
            ScalarIndexParams.create("zonemap", "{\"rows_per_zone\": 1024}");

        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // Create Zonemap index on 'value' column
        Index index =
            dataset.createIndex(
                Collections.singletonList("value"),
                IndexType.ZONEMAP,
                Optional.of("zonemap_value_index"),
                indexParams,
                true);

        // Verify the returned Index object
        assertEquals("zonemap_value_index", index.name());
        assertNotNull(index.uuid());

        // Verify index was created
        assertTrue(
            dataset.listIndexes().contains("zonemap_value_index"),
            "Expected 'zonemap_value_index' to be in the list of indexes: "
                + dataset.listIndexes());

        // TODO: Verify rows_per_zone parameter was applied
        // Currently the Java API doesn't expose index configuration details,
        // but we could add a getIndexDetails() method in the future to verify
        // that the rows_per_zone parameter was correctly set to 1024
      }
    }
  }
}
