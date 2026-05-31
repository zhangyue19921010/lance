#!/bin/bash
#
# BTree Index Benchmark Runner
# Compares range-based (main) vs segmented (current branch) BTree index building.
#
# Usage:
#   ./run_benchmark.sh [small|large]
#     small: 100 rows, 20 fragments (for validation)
#     large: 1,000,000,000 rows, 20 fragments (for production benchmark)
#
# Prerequisites:
#   - Java 11+ installed
#   - Maven installed
#   - Lance project compiled on both branches
#
# Run on: ssh zhangyue.1010@10.37.3.153
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
JAVA_DIR="$PROJECT_ROOT/java"

MODE="${1:-small}"
PARALLELISM=20
BENCHMARK_DIR="/tmp/lance_btree_benchmark"
DATASET_PATH="$BENCHMARK_DIR/shared_dataset"
REPORT_FILE="$BENCHMARK_DIR/benchmark_report.txt"

# Configuration based on mode
if [ "$MODE" = "small" ]; then
  TOTAL_ROWS=100
  NUM_FRAGMENTS=20
  BATCH_SIZE=10
  echo "=== Running in SMALL mode (validation: 100 rows) ==="
elif [ "$MODE" = "large" ]; then
  TOTAL_ROWS=1000000000
  NUM_FRAGMENTS=20
  BATCH_SIZE=1000000
  echo "=== Running in LARGE mode (1 billion rows) ==="
else
  echo "Unknown mode: $MODE. Use 'small' or 'large'."
  exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Java dir: $JAVA_DIR"
echo "Dataset path: $DATASET_PATH"
echo "Parallelism: $PARALLELISM"
echo "Total rows: $TOTAL_ROWS"
echo "Num fragments: $NUM_FRAGMENTS"
echo ""

# Clean up previous benchmark data
rm -rf "$BENCHMARK_DIR"
mkdir -p "$BENCHMARK_DIR"

# -----------------------------------------------------------
# Phase 0: Record system info
# -----------------------------------------------------------
{
  echo "=============================================="
  echo " BTree Index Benchmark Report"
  echo "=============================================="
  echo ""
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Mode: $MODE"
  echo "Total Rows: $TOTAL_ROWS"
  echo "Num Fragments: $NUM_FRAGMENTS"
  echo "Parallelism: $PARALLELISM"
  echo "Batch Size: $BATCH_SIZE"
  echo ""
  echo "Java Version:"
  java -version 2>&1
  echo ""
  echo "System Info:"
  uname -a
  echo ""
  echo "CPU Info:"
  if [ -f /proc/cpuinfo ]; then
    grep "model name" /proc/cpuinfo | head -1
    echo "CPU cores: $(nproc)"
  else
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown"
    echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi
  echo ""
  echo "Memory:"
  free -h 2>/dev/null || vm_stat 2>/dev/null || echo "Unknown"
  echo ""
} > "$REPORT_FILE"

# -----------------------------------------------------------
# Phase 1: Build the project (current branch)
# -----------------------------------------------------------
echo "=== Phase 1: Building project on current branch ==="
CURRENT_BRANCH=$(cd "$PROJECT_ROOT" && git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

cd "$JAVA_DIR"
./mvnw -q compile test-compile -DskipTests 2>&1 || {
  echo "Build failed! Trying with mvn..."
  mvn -q compile test-compile -DskipTests 2>&1
}

# Get classpath
CLASSPATH=$(./mvnw -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || \
  mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null)
CLASSPATH="$JAVA_DIR/target/classes:$JAVA_DIR/target/test-classes:$CLASSPATH"

echo "Classpath configured."

# -----------------------------------------------------------
# Phase 2: Generate shared dataset
# -----------------------------------------------------------
echo ""
echo "=== Phase 2: Generating shared dataset ==="
DATA_GEN_START=$(date +%s%N)

java -cp "$CLASSPATH" \
  -Xmx8g \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  -Djava.library.path="$JAVA_DIR/lance-jni/target/release:$JAVA_DIR/lance-jni/target/debug" \
  org.lance.benchmark.BenchmarkDataGenerator \
  "$DATASET_PATH" "$TOTAL_ROWS" "$NUM_FRAGMENTS" "$BATCH_SIZE"

DATA_GEN_END=$(date +%s%N)
DATA_GEN_TIME=$(( (DATA_GEN_END - DATA_GEN_START) / 1000000 ))

echo "Dataset generation: ${DATA_GEN_TIME}ms"
{
  echo "----------------------------------------------"
  echo " Dataset Generation"
  echo "----------------------------------------------"
  echo "Time: ${DATA_GEN_TIME}ms ($(echo "scale=2; $DATA_GEN_TIME / 1000" | bc)s)"
  echo ""
} >> "$REPORT_FILE"

# -----------------------------------------------------------
# Phase 3: Run Segmented Benchmark (current branch)
# -----------------------------------------------------------
echo ""
echo "=== Phase 3: Running Segmented Benchmark (branch: $CURRENT_BRANCH) ==="

# Copy dataset for segmented test
SEGMENTED_DATASET="$BENCHMARK_DIR/dataset_segmented"
cp -r "$DATASET_PATH" "$SEGMENTED_DATASET"

SEGMENTED_START=$(date +%s%N)

java -cp "$CLASSPATH" \
  -Xmx16g \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  -Djava.library.path="$JAVA_DIR/lance-jni/target/release:$JAVA_DIR/lance-jni/target/debug" \
  org.lance.benchmark.BTreeSegmentedBenchmark \
  "$SEGMENTED_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/segmented_output.txt"

SEGMENTED_END=$(date +%s%N)
SEGMENTED_TIME=$(( (SEGMENTED_END - SEGMENTED_START) / 1000000 ))

{
  echo "----------------------------------------------"
  echo " Segmented Index Build (branch: $CURRENT_BRANCH)"
  echo "----------------------------------------------"
  echo "Total wall-clock time: ${SEGMENTED_TIME}ms ($(echo "scale=2; $SEGMENTED_TIME / 1000" | bc)s)"
  echo ""
  echo "Detailed output:"
  cat "$BENCHMARK_DIR/segmented_output.txt"
  echo ""
} >> "$REPORT_FILE"

# -----------------------------------------------------------
# Phase 4: Switch to main, rebuild, and run Range-Based Benchmark
# -----------------------------------------------------------
echo ""
echo "=== Phase 4: Switching to main branch and rebuilding ==="

cd "$PROJECT_ROOT"
# Stash any uncommitted changes
git stash 2>/dev/null || true
git checkout main

cd "$JAVA_DIR"
./mvnw -q compile test-compile -DskipTests 2>&1 || {
  mvn -q compile test-compile -DskipTests 2>&1
}

# Recalculate classpath for main branch
CLASSPATH_MAIN=$(./mvnw -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || \
  mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null)
CLASSPATH_MAIN="$JAVA_DIR/target/classes:$JAVA_DIR/target/test-classes:$CLASSPATH_MAIN"

echo ""
echo "=== Running Range-Based Benchmark (branch: main) ==="

# Copy dataset for range-based test
RANGEBASED_DATASET="$BENCHMARK_DIR/dataset_rangebased"
cp -r "$DATASET_PATH" "$RANGEBASED_DATASET"

RANGEBASED_START=$(date +%s%N)

java -cp "$CLASSPATH_MAIN" \
  -Xmx16g \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  -Djava.library.path="$JAVA_DIR/lance-jni/target/release:$JAVA_DIR/lance-jni/target/debug" \
  org.lance.benchmark.BTreeRangeBasedBenchmark \
  "$RANGEBASED_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/rangebased_output.txt"

RANGEBASED_END=$(date +%s%N)
RANGEBASED_TIME=$(( (RANGEBASED_END - RANGEBASED_START) / 1000000 ))

{
  echo "----------------------------------------------"
  echo " Range-Based Index Build (branch: main)"
  echo "----------------------------------------------"
  echo "Total wall-clock time: ${RANGEBASED_TIME}ms ($(echo "scale=2; $RANGEBASED_TIME / 1000" | bc)s)"
  echo ""
  echo "Detailed output:"
  cat "$BENCHMARK_DIR/rangebased_output.txt"
  echo ""
} >> "$REPORT_FILE"

# -----------------------------------------------------------
# Phase 5: Switch back to PR branch
# -----------------------------------------------------------
echo ""
echo "=== Phase 5: Switching back to PR branch ==="
cd "$PROJECT_ROOT"
git checkout "$CURRENT_BRANCH"
git stash pop 2>/dev/null || true

# -----------------------------------------------------------
# Phase 6: Summary
# -----------------------------------------------------------
{
  echo "=============================================="
  echo " SUMMARY"
  echo "=============================================="
  echo ""
  echo "Configuration:"
  echo "  Total Rows:    $TOTAL_ROWS"
  echo "  Fragments:     $NUM_FRAGMENTS"
  echo "  Parallelism:   $PARALLELISM"
  echo "  Index Columns: 10"
  echo ""
  echo "Results:"
  echo "  Dataset Generation:       $(echo "scale=2; $DATA_GEN_TIME / 1000" | bc)s"
  echo "  Range-Based (main):       $(echo "scale=2; $RANGEBASED_TIME / 1000" | bc)s"
  echo "  Segmented (PR branch):    $(echo "scale=2; $SEGMENTED_TIME / 1000" | bc)s"
  echo ""
  if [ $RANGEBASED_TIME -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $RANGEBASED_TIME / $SEGMENTED_TIME" | bc)
    echo "  Speedup (Segmented vs Range-Based): ${SPEEDUP}x"
  fi
  echo ""
  echo "=============================================="
} >> "$REPORT_FILE"

echo ""
echo "=============================================="
echo " BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Report saved to: $REPORT_FILE"
echo ""
echo "Quick Summary:"
echo "  Range-Based (main):    $(echo "scale=2; $RANGEBASED_TIME / 1000" | bc)s"
echo "  Segmented (PR branch): $(echo "scale=2; $SEGMENTED_TIME / 1000" | bc)s"
if [ $RANGEBASED_TIME -gt 0 ]; then
  echo "  Speedup: ${SPEEDUP}x"
fi
echo ""
cat "$REPORT_FILE"
