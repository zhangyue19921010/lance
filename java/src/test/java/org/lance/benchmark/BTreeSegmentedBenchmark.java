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
import org.lance.index.Index;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Benchmark for segmented BTree index building (current branch approach).
 *
 * <p>Flow: 1. Get all fragments from the dataset 2. Distribute fragments across workers
 * (parallelism threads) 3. Each worker builds index segment(s) for its assigned fragments via
 * createIndex(fragmentIds) 4. Collect all uncommitted segments 5. Call buildIndexSegments to build
 * physical segments 6. Call commitExistingIndexSegments to commit
 *
 * <p>Usage: java BTreeSegmentedBenchmark <datasetPath> <parallelism> Run on current branch
 * (btree-distributed-build-segmented-pr1).
 */
public class BTreeSegmentedBenchmark {

  private final String datasetPath;
  private final int parallelism;

  public BTreeSegmentedBenchmark(String datasetPath, int parallelism) {
    this.datasetPath = datasetPath;
    this.parallelism = parallelism;
  }

  public void run() throws Exception {
    System.out.println("=== BTree Segmented Index Benchmark (segmented branch) ===");
    System.out.printf("Dataset: %s%n", datasetPath);
    System.out.printf("Parallelism: %d%n", parallelism);

    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Dataset dataset = Dataset.open(datasetPath, allocator)) {

      long totalRows = dataset.countRows();
      List<Fragment> fragments = dataset.getFragments();
      System.out.printf("Total rows: %d, Fragments: %d%n", totalRows, fragments.size());

      long totalBuildTime = 0;

      for (String column : BenchmarkDataGenerator.INDEX_COLUMNS) {
        long columnTime = buildIndexForColumn(dataset, column, fragments, allocator);
        totalBuildTime += columnTime;
        System.out.printf("[Column: %s] Index build time: %.2fs%n", column, columnTime / 1000.0);
      }

      System.out.println("\n=== RESULTS (Segmented) ===");
      System.out.printf(
          "Total index build time for %d columns: %.2fs%n",
          BenchmarkDataGenerator.INDEX_COLUMNS.size(), totalBuildTime / 1000.0);
      System.out.printf(
          "Average per column: %.2fs%n",
          totalBuildTime / 1000.0 / BenchmarkDataGenerator.INDEX_COLUMNS.size());
    }
  }

  private long buildIndexForColumn(
      Dataset dataset, String column, List<Fragment> fragments, BufferAllocator allocator)
      throws Exception {
    long startTime = System.currentTimeMillis();

    String indexName = column + "_idx";

    // Step 1: Distribute fragments across workers
    List<List<Integer>> fragmentAssignment = distributeFragments(fragments, parallelism);

    // Step 2: Parallel fragment-level index building
    long buildStart = System.currentTimeMillis();
    ExecutorService executor = Executors.newFixedThreadPool(parallelism);
    CountDownLatch latch = new CountDownLatch(parallelism);
    ConcurrentLinkedQueue<Index> segmentResults = new ConcurrentLinkedQueue<>();
    AtomicLong errorCount = new AtomicLong(0);

    ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
    IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

    for (int workerIdx = 0; workerIdx < parallelism; workerIdx++) {
      final List<Integer> assignedFragmentIds = fragmentAssignment.get(workerIdx);
      if (assignedFragmentIds.isEmpty()) {
        latch.countDown();
        continue;
      }

      executor.submit(
          () -> {
            try {
              // Each worker builds index segment for its assigned fragments.
              // Each segment gets its own UUID (no shared UUID for segmented approach).
              Index segment =
                  dataset.createIndex(
                      IndexOptions.builder(
                              Collections.singletonList(column), IndexType.BTREE, indexParams)
                          .withIndexName(indexName)
                          .withFragmentIds(assignedFragmentIds)
                          .build());
              segmentResults.add(segment);
            } catch (Exception e) {
              System.err.printf("  [%s] Worker error: %s%n", column, e.getMessage());
              e.printStackTrace();
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
        "  [%s] Parallel segment build: %.2fs (%d segments, errors: %d)%n",
        column, buildTime / 1000.0, segmentResults.size(), errorCount.get());

    if (errorCount.get() > 0) {
      throw new RuntimeException("Segment build had errors for column: " + column);
    }

    // Step 3: Commit segments directly
    // For BTree, fragments-scoped segments are already complete physical segments.
    // No need for buildIndexSegments (which is only for Vector/Inverted).
    long commitStart = System.currentTimeMillis();
    List<Index> segments = new ArrayList<>(segmentResults);
    List<Index> committed = dataset.commitExistingIndexSegments(indexName, column, segments);
    long commitTime = System.currentTimeMillis() - commitStart;
    System.out.printf(
        "  [%s] Commit: %.2fs (%d committed)%n", column, commitTime / 1000.0, committed.size());

    long totalTime = System.currentTimeMillis() - startTime;
    return totalTime;
  }

  /** Distribute fragments evenly across workers. */
  private List<List<Integer>> distributeFragments(List<Fragment> fragments, int numWorkers) {
    List<List<Integer>> assignment = new ArrayList<>();
    for (int i = 0; i < numWorkers; i++) {
      assignment.add(new ArrayList<>());
    }

    for (int i = 0; i < fragments.size(); i++) {
      int workerIdx = i % numWorkers;
      assignment.get(workerIdx).add(fragments.get(i).getId());
    }

    return assignment;
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: BTreeSegmentedBenchmark <datasetPath> <parallelism>");
      System.exit(1);
    }

    String datasetPath = args[0];
    int parallelism = Integer.parseInt(args[1]);

    BTreeSegmentedBenchmark benchmark = new BTreeSegmentedBenchmark(datasetPath, parallelism);
    benchmark.run();
  }
}
