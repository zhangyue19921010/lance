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
package org.lance.benchmark;

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.WriteParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Shared data generator for BTree index benchmarks. Generates a Lance dataset with 10 schemas
 * (columns) suitable for BTree indexing. Both the range-based and segmented benchmark use the same
 * generated dataset to ensure a controlled comparison.
 */
public class BenchmarkDataGenerator {

  // 10 columns for BTree indexing
  public static final List<String> INDEX_COLUMNS =
      Arrays.asList(
          "col_int_0",
          "col_int_1",
          "col_int_2",
          "col_int_3",
          "col_int_4",
          "col_long_5",
          "col_long_6",
          "col_long_7",
          "col_long_8",
          "col_long_9");

  public static Schema getSchema() {
    List<Field> fields = new ArrayList<>();
    // First 5 columns are Int32
    for (int i = 0; i < 5; i++) {
      fields.add(Field.nullable("col_int_" + i, new ArrowType.Int(32, true)));
    }
    // Next 5 columns are Int64
    for (int i = 5; i < 10; i++) {
      fields.add(Field.nullable("col_long_" + i, new ArrowType.Int(64, true)));
    }
    return new Schema(fields, null);
  }

  /**
   * Generate a dataset with the specified number of total rows, split into fragments.
   *
   * @param datasetPath path to store the dataset
   * @param totalRows total number of rows to generate
   * @param numFragments number of fragments to split data into
   * @param batchSize rows per write batch (to control memory usage)
   * @return the created Dataset (caller must close)
   */
  public static Dataset generateDataset(
      String datasetPath, long totalRows, int numFragments, int batchSize) {
    System.out.printf(
        "[DataGen] Generating dataset: path=%s, totalRows=%d, numFragments=%d, batchSize=%d%n",
        datasetPath, totalRows, numFragments, batchSize);

    long startTime = System.currentTimeMillis();

    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Schema schema = getSchema();

      // Create empty dataset
      Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());
      dataset.close();

      long rowsPerFragment = totalRows / numFragments;
      long remainder = totalRows % numFragments;

      for (int fragIdx = 0; fragIdx < numFragments; fragIdx++) {
        long fragRows = rowsPerFragment + (fragIdx < remainder ? 1 : 0);
        long fragOffset = fragIdx * rowsPerFragment + Math.min(fragIdx, remainder);

        List<FragmentMetadata> fragmentMetas = new ArrayList<>();

        // Write fragment data in batches to avoid OOM
        long written = 0;
        while (written < fragRows) {
          int currentBatch = (int) Math.min(batchSize, fragRows - written);
          long globalOffset = fragOffset + written;

          try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            root.allocateNew();
            fillBatch(root, currentBatch, globalOffset);
            root.setRowCount(currentBatch);

            List<FragmentMetadata> batchMetas =
                Fragment.create(
                    datasetPath,
                    allocator,
                    root,
                    new WriteParams.Builder().withMaxRowsPerFile(Integer.MAX_VALUE).build());
            fragmentMetas.addAll(batchMetas);
          }
          written += currentBatch;
        }

        // Commit this fragment via Append operation
        long currentVersion = fragIdx + 1; // version starts at 1 after create
        FragmentOperation.Append appendOp = new FragmentOperation.Append(fragmentMetas);
        Dataset updated =
            Dataset.commit(allocator, datasetPath, appendOp, Optional.of(currentVersion));
        updated.close();

        if ((fragIdx + 1) % 10 == 0 || fragIdx == numFragments - 1) {
          System.out.printf(
              "[DataGen] Progress: %d/%d fragments written%n", fragIdx + 1, numFragments);
        }
      }

      long elapsed = System.currentTimeMillis() - startTime;
      System.out.printf(
          "[DataGen] Dataset generation complete. Elapsed: %.2fs%n", elapsed / 1000.0);

      // Reopen and return
      return Dataset.open(datasetPath, new RootAllocator(Long.MAX_VALUE));
    }
  }

  private static void fillBatch(VectorSchemaRoot root, int rowCount, long offset) {
    // Fill Int32 columns (col_int_0 to col_int_4)
    for (int col = 0; col < 5; col++) {
      IntVector vec = (IntVector) root.getVector("col_int_" + col);
      for (int i = 0; i < rowCount; i++) {
        // Use a deterministic pattern: value = (offset + i) * prime + col
        int value = (int) ((offset + i) * 31 + col * 7);
        vec.setSafe(i, value);
      }
    }
    // Fill Int64 columns (col_long_5 to col_long_9)
    for (int col = 5; col < 10; col++) {
      BigIntVector vec = (BigIntVector) root.getVector("col_long_" + col);
      for (int i = 0; i < rowCount; i++) {
        long value = (offset + i) * 37L + col * 13L;
        vec.setSafe(i, value);
      }
    }
  }

  /**
   * Main entry point for standalone data generation. Usage: java BenchmarkDataGenerator
   * <datasetPath> <totalRows> <numFragments> [batchSize]
   */
  public static void main(String[] args) {
    if (args.length < 3) {
      System.err.println(
          "Usage: BenchmarkDataGenerator <datasetPath> <totalRows> <numFragments> [batchSize]");
      System.exit(1);
    }

    String datasetPath = args[0];
    long totalRows = Long.parseLong(args[1]);
    int numFragments = Integer.parseInt(args[2]);
    int batchSize = args.length > 3 ? Integer.parseInt(args[3]) : 100000;

    try (Dataset dataset = generateDataset(datasetPath, totalRows, numFragments, batchSize)) {
      System.out.printf(
          "[DataGen] Final dataset: rows=%d, fragments=%d%n",
          dataset.countRows(), dataset.getFragments().size());
    }
  }
}
