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

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.Transaction;
import org.lance.index.Index;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.CreateIndex;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Benchmark for range-based BTree index building (main branch approach).
 *
 * <p>Flow: 1. Scan data from dataset with _rowid 2. Sort data globally per column 3. Divide into
 * ranges (one per thread) 4. Each thread builds a BTree index segment with rangeId +
 * preprocessedData 5. mergeIndexMetadata to merge all ranges 6. Commit index via Transaction
 *
 * <p>Usage: java BTreeRangeBasedBenchmark <datasetPath> <parallelism> Run on main branch.
 */
public class BTreeRangeBasedBenchmark {

  private final String datasetPath;
  private final int parallelism;

  public BTreeRangeBasedBenchmark(String datasetPath, int parallelism) {
    this.datasetPath = datasetPath;
    this.parallelism = parallelism;
  }

  public void run() throws Exception {
    System.out.println("=== BTree Range-Based Index Benchmark (main branch) ===");
    System.out.printf("Dataset: %s%n", datasetPath);
    System.out.printf("Parallelism: %d%n", parallelism);

    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Dataset dataset = Dataset.open(datasetPath, allocator)) {

      long totalRows = dataset.countRows();
      int numFragments = dataset.getFragments().size();
      System.out.printf("Total rows: %d, Fragments: %d%n", totalRows, numFragments);

      long totalBuildTime = 0;

      for (String column : BenchmarkDataGenerator.INDEX_COLUMNS) {
        long columnTime = buildIndexForColumn(dataset, column, allocator);
        totalBuildTime += columnTime;
        System.out.printf("[Column: %s] Index build time: %.2fs%n", column, columnTime / 1000.0);
      }

      System.out.println("\n=== RESULTS (Range-Based) ===");
      System.out.printf(
          "Total index build time for %d columns: %.2fs%n",
          BenchmarkDataGenerator.INDEX_COLUMNS.size(), totalBuildTime / 1000.0);
      System.out.printf(
          "Average per column: %.2fs%n",
          totalBuildTime / 1000.0 / BenchmarkDataGenerator.INDEX_COLUMNS.size());
    }
  }

  private long buildIndexForColumn(Dataset dataset, String column, BufferAllocator allocator)
      throws Exception {
    long startTime = System.currentTimeMillis();

    UUID indexUUID = UUID.randomUUID();
    String indexName = column + "_idx";

    // Step 1: Scan data with _rowid
    System.out.printf("  [%s] Scanning data with _rowid...%n", column);
    List<long[]> data = scanColumnWithRowId(dataset, column, allocator);
    long scanTime = System.currentTimeMillis() - startTime;
    System.out.printf(
        "  [%s] Scan complete: %d rows in %.2fs%n", column, data.size(), scanTime / 1000.0);

    // Step 2: Sort globally
    long sortStart = System.currentTimeMillis();
    data.sort((d1, d2) -> Long.compare(d1[0], d2[0]));
    long sortTime = System.currentTimeMillis() - sortStart;
    System.out.printf("  [%s] Sort complete in %.2fs%n", column, sortTime / 1000.0);

    // Step 3: Divide into ranges and build in parallel
    long buildStart = System.currentTimeMillis();
    int rangeSize = data.size() / parallelism;
    ExecutorService executor = Executors.newFixedThreadPool(parallelism);
    CountDownLatch latch = new CountDownLatch(parallelism);
    AtomicLong errorCount = new AtomicLong(0);

    for (int rangeIdx = 0; rangeIdx < parallelism; rangeIdx++) {
      final int rIdx = rangeIdx;
      final int fromIndex = rIdx * rangeSize;
      final int toIndex = (rIdx == parallelism - 1) ? data.size() : (rIdx + 1) * rangeSize;
      final List<long[]> rangeData = data.subList(fromIndex, toIndex);

      executor.submit(
          () -> {
            try (BufferAllocator threadAllocator = new RootAllocator(Long.MAX_VALUE)) {
              createBtreeIndexForRange(
                  dataset, column, rangeData, rIdx + 1, threadAllocator, indexUUID);
            } catch (Exception e) {
              System.err.printf("  [%s] Error in range %d: %s%n", column, rIdx, e.getMessage());
              errorCount.incrementAndGet();
            } finally {
              latch.countDown();
            }
          });
    }

    latch.await();
    executor.shutdown();
    long buildTime = System.currentTimeMillis() - buildStart;
    System.out.printf(
        "  [%s] Parallel build complete in %.2fs (errors: %d)%n",
        column, buildTime / 1000.0, errorCount.get());

    if (errorCount.get() > 0) {
      throw new RuntimeException("Index build had errors for column: " + column);
    }

    // Step 4: Merge index metadata
    long mergeStart = System.currentTimeMillis();
    dataset.mergeIndexMetadata(indexUUID.toString(), IndexType.BTREE, Optional.empty());
    long mergeTime = System.currentTimeMillis() - mergeStart;
    System.out.printf("  [%s] Merge metadata in %.2fs%n", column, mergeTime / 1000.0);

    // Step 5: Commit the index
    long commitStart = System.currentTimeMillis();
    int fieldId =
        dataset.getLanceSchema().fields().stream()
            .filter(f -> f.getName().equals(column))
            .findAny()
            .orElseThrow(() -> new RuntimeException("Cannot find field: " + column))
            .getId();

    long datasetVersion = dataset.version();
    Index index =
        Index.builder()
            .uuid(indexUUID)
            .name(indexName)
            .fields(Collections.singletonList(fieldId))
            .datasetVersion(datasetVersion)
            .indexVersion(0)
            .fragments(
                dataset.getFragments().stream().map(Fragment::getId).collect(Collectors.toList()))
            .build();

    CreateIndex createIndexOp =
        CreateIndex.builder().withNewIndices(Collections.singletonList(index)).build();

    try (Transaction txn =
        new Transaction.Builder().readVersion(datasetVersion).operation(createIndexOp).build()) {
      Dataset newDataset = new CommitBuilder(dataset).execute(txn);
      newDataset.close();
    }
    long commitTime = System.currentTimeMillis() - commitStart;
    System.out.printf("  [%s] Commit in %.2fs%n", column, commitTime / 1000.0);

    long totalTime = System.currentTimeMillis() - startTime;
    return totalTime;
  }

  private List<long[]> scanColumnWithRowId(
      Dataset dataset, String column, BufferAllocator allocator) throws Exception {
    List<long[]> data = new ArrayList<>();
    try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .withRowId(true)
                    .columns(Collections.singletonList(column))
                    .build());
        ArrowReader arrowReader = scanner.scanBatches()) {
      while (arrowReader.loadNextBatch()) {
        VectorSchemaRoot root = arrowReader.getVectorSchemaRoot();
        UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");

        // Handle both Int32 and Int64 columns
        if (root.getVector(column) instanceof IntVector) {
          IntVector valVec = (IntVector) root.getVector(column);
          for (int i = 0; i < root.getRowCount(); i++) {
            data.add(new long[] {valVec.get(i), rowIdVec.get(i)});
          }
        } else {
          BigIntVector valVec = (BigIntVector) root.getVector(column);
          for (int i = 0; i < root.getRowCount(); i++) {
            data.add(new long[] {valVec.get(i), rowIdVec.get(i)});
          }
        }
      }
    }
    return data;
  }

  private void createBtreeIndexForRange(
      Dataset dataset,
      String column,
      List<long[]> preprocessedData,
      int rangeId,
      BufferAllocator allocator,
      UUID indexUUID) {
    // The preprocessed data schema for BTree: 'value' (the indexed column) + '_rowid'
    boolean isIntColumn = column.startsWith("col_int_");
    Schema schema;
    if (isIntColumn) {
      schema =
          new Schema(
              Arrays.asList(
                  Field.nullable("value", new ArrowType.Int(32, true)),
                  Field.nullable("_rowid", new ArrowType.Int(64, false))),
              null);
    } else {
      schema =
          new Schema(
              Arrays.asList(
                  Field.nullable("value", new ArrowType.Int(64, true)),
                  Field.nullable("_rowid", new ArrowType.Int(64, false))),
              null);
    }

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();

      if (isIntColumn) {
        IntVector valVec = (IntVector) root.getVector("value");
        UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
        for (int i = 0; i < preprocessedData.size(); i++) {
          long[] pair = preprocessedData.get(i);
          valVec.setSafe(i, (int) pair[0]);
          rowIdVec.setSafe(i, pair[1]);
        }
      } else {
        BigIntVector valVec = (BigIntVector) root.getVector("value");
        UInt8Vector rowIdVec = (UInt8Vector) root.getVector("_rowid");
        for (int i = 0; i < preprocessedData.size(); i++) {
          long[] pair = preprocessedData.get(i);
          valVec.setSafe(i, pair[0]);
          rowIdVec.setSafe(i, pair[1]);
        }
      }
      root.setRowCount(preprocessedData.size());

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      } catch (IOException e) {
        throw new RuntimeException("Cannot write arrow stream", e);
      }

      byte[] arrowData = out.toByteArray();
      ByteArrayInputStream in = new ByteArrayInputStream(arrowData);

      try (ArrowStreamReader reader = new ArrowStreamReader(in, allocator);
          ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, stream);

        ScalarIndexParams scalarParams =
            ScalarIndexParams.create("btree", String.format("{\"range_id\": %d}", rangeId));
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList(column), IndexType.BTREE, indexParams)
                .withIndexUUID(indexUUID.toString())
                .withPreprocessedData(stream)
                .build());
      } catch (Exception e) {
        throw new RuntimeException("Cannot create range-based index", e);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: BTreeRangeBasedBenchmark <datasetPath> <parallelism>");
      System.exit(1);
    }

    String datasetPath = args[0];
    int parallelism = Integer.parseInt(args[1]);

    BTreeRangeBasedBenchmark benchmark = new BTreeRangeBasedBenchmark(datasetPath, parallelism);
    benchmark.run();
  }
}
