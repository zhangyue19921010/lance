#!/bin/bash
#
# Complete benchmark runner for remote server (ssh zhangyue.1010@10.37.3.153)
#
# This script runs both benchmarks sequentially:
# 1. First run segmented benchmark on current PR branch
# 2. Then switch to main branch and run range-based benchmark
# 3. Generate comparison report
#
# Prerequisites:
#   - JDK 11+ 
#   - Maven 3.x (or use included mvnw)
#   - Rust toolchain (for JNI build)
#   - Project already cloned at the expected location
#
# Usage:
#   # Validate with 100 rows first
#   ./run_full_benchmark.sh small
#
#   # Production benchmark with 1 billion rows
#   ./run_full_benchmark.sh large
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
JAVA_DIR="$PROJECT_ROOT/java"

MODE="${1:-small}"
PARALLELISM=20
BENCHMARK_DIR="/tmp/lance_btree_benchmark_$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="$BENCHMARK_DIR/shared_dataset"
REPORT_FILE="$BENCHMARK_DIR/benchmark_report.txt"

# Configuration
if [ "$MODE" = "small" ]; then
  TOTAL_ROWS=100
  NUM_FRAGMENTS=20
  BATCH_SIZE=10
elif [ "$MODE" = "large" ]; then
  TOTAL_ROWS=1000000000
  NUM_FRAGMENTS=20
  BATCH_SIZE=1000000
else
  echo "Usage: $0 [small|large]"
  exit 1
fi

CURRENT_BRANCH=$(cd "$PROJECT_ROOT" && git branch --show-current)
ARROW_OPTS="--add-opens=java.base/java.nio=ALL-UNNAMED"

echo "=============================================="
echo " BTree Index Benchmark"
echo "=============================================="
echo "Mode:       $MODE"
echo "Branch:     $CURRENT_BRANCH"
echo "Rows:       $TOTAL_ROWS"
echo "Fragments:  $NUM_FRAGMENTS"
echo "Parallelism: $PARALLELISM"
echo "Output:     $BENCHMARK_DIR"
echo "=============================================="
echo ""

mkdir -p "$BENCHMARK_DIR"

# Write report header
{
  echo "=============================================="
  echo " BTree Index Benchmark Report"
  echo "=============================================="
  echo "Date:        $(date)"
  echo "Host:        $(hostname)"
  echo "Mode:        $MODE"
  echo "Rows:        $TOTAL_ROWS"
  echo "Fragments:   $NUM_FRAGMENTS"
  echo "Parallelism: $PARALLELISM"
  echo ""
  echo "Java:"
  java -version 2>&1
  echo ""
  echo "System: $(uname -a)"
  if [ -f /proc/cpuinfo ]; then
    echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2)"
    echo "Cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
  fi
  echo ""
} > "$REPORT_FILE"

# Helper function to get classpath
get_classpath() {
  local cp=""
  cd "$JAVA_DIR"
  cp=$(./mvnw -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || \
       mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || echo "")
  if [ -z "$cp" ]; then
    ./mvnw -q dependency:copy-dependencies 2>/dev/null || mvn -q dependency:copy-dependencies 2>/dev/null || true
    cp=$(echo "$JAVA_DIR/target/dependency/"*.jar | tr ' ' ':')
  fi
  echo "$JAVA_DIR/target/classes:$JAVA_DIR/target/test-classes:$cp"
}

# Helper to find JNI path
get_jni_path() {
  if [ -d "$JAVA_DIR/lance-jni/target/release" ]; then
    echo "$JAVA_DIR/lance-jni/target/release"
  elif [ -d "$JAVA_DIR/lance-jni/target/debug" ]; then
    echo "$JAVA_DIR/lance-jni/target/debug"
  else
    echo "$JAVA_DIR/lance-jni/target/release:$JAVA_DIR/lance-jni/target/debug"
  fi
}

# ============================================================
# STEP 1: Build on current branch (segmented)
# ============================================================
echo ">>> Step 1: Building project on branch: $CURRENT_BRANCH"
cd "$JAVA_DIR"
./mvnw -q compile test-compile -DskipTests 2>&1 || mvn -q compile test-compile -DskipTests 2>&1

CLASSPATH=$(get_classpath)
JNI_PATH=$(get_jni_path)

# ============================================================
# STEP 2: Generate shared dataset
# ============================================================
echo ""
echo ">>> Step 2: Generating shared dataset ($TOTAL_ROWS rows, $NUM_FRAGMENTS fragments)"
DATAGEN_START=$(date +%s)

java $ARROW_OPTS -Xmx8g -Djava.library.path="$JNI_PATH" \
  -cp "$CLASSPATH" \
  org.lance.benchmark.BenchmarkDataGenerator \
  "$DATASET_PATH" "$TOTAL_ROWS" "$NUM_FRAGMENTS" "$BATCH_SIZE"

DATAGEN_END=$(date +%s)
DATAGEN_SECS=$((DATAGEN_END - DATAGEN_START))
echo "Dataset generation: ${DATAGEN_SECS}s"
echo "" >> "$REPORT_FILE"
echo "Dataset Generation: ${DATAGEN_SECS}s" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# ============================================================
# STEP 3: Run segmented benchmark (current branch)
# ============================================================
echo ""
echo ">>> Step 3: Running SEGMENTED benchmark (branch: $CURRENT_BRANCH)"

SEGMENTED_DATASET="$BENCHMARK_DIR/dataset_segmented"
cp -r "$DATASET_PATH" "$SEGMENTED_DATASET"

SEGMENTED_START=$(date +%s)

java $ARROW_OPTS -Xmx16g -Djava.library.path="$JNI_PATH" \
  -cp "$CLASSPATH" \
  org.lance.benchmark.BTreeSegmentedBenchmark \
  "$SEGMENTED_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/segmented_output.txt"

SEGMENTED_END=$(date +%s)
SEGMENTED_SECS=$((SEGMENTED_END - SEGMENTED_START))

{
  echo "----------------------------------------------"
  echo " SEGMENTED Index Build (branch: $CURRENT_BRANCH)"
  echo "----------------------------------------------"
  echo "Wall-clock time: ${SEGMENTED_SECS}s"
  echo ""
  cat "$BENCHMARK_DIR/segmented_output.txt"
  echo ""
} >> "$REPORT_FILE"

# ============================================================
# STEP 4: Switch to main, rebuild, run range-based benchmark
# ============================================================
echo ""
echo ">>> Step 4: Switching to main branch"
cd "$PROJECT_ROOT"
git stash 2>/dev/null || true
git checkout main

echo ">>> Building project on branch: main"
cd "$JAVA_DIR"
./mvnw -q compile test-compile -DskipTests 2>&1 || mvn -q compile test-compile -DskipTests 2>&1

CLASSPATH_MAIN=$(get_classpath)
JNI_PATH_MAIN=$(get_jni_path)

echo ""
echo ">>> Running RANGE-BASED benchmark (branch: main)"

RANGEBASED_DATASET="$BENCHMARK_DIR/dataset_rangebased"
cp -r "$DATASET_PATH" "$RANGEBASED_DATASET"

RANGEBASED_START=$(date +%s)

java $ARROW_OPTS -Xmx16g -Djava.library.path="$JNI_PATH_MAIN" \
  -cp "$CLASSPATH_MAIN" \
  org.lance.benchmark.BTreeRangeBasedBenchmark \
  "$RANGEBASED_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/rangebased_output.txt"

RANGEBASED_END=$(date +%s)
RANGEBASED_SECS=$((RANGEBASED_END - RANGEBASED_START))

{
  echo "----------------------------------------------"
  echo " RANGE-BASED Index Build (branch: main)"
  echo "----------------------------------------------"
  echo "Wall-clock time: ${RANGEBASED_SECS}s"
  echo ""
  cat "$BENCHMARK_DIR/rangebased_output.txt"
  echo ""
} >> "$REPORT_FILE"

# ============================================================
# STEP 5: Switch back
# ============================================================
echo ""
echo ">>> Step 5: Switching back to $CURRENT_BRANCH"
cd "$PROJECT_ROOT"
git checkout "$CURRENT_BRANCH"
git stash pop 2>/dev/null || true

# ============================================================
# STEP 6: Summary
# ============================================================
if [ $RANGEBASED_SECS -gt 0 ] && [ $SEGMENTED_SECS -gt 0 ]; then
  SPEEDUP=$(echo "scale=2; $RANGEBASED_SECS / $SEGMENTED_SECS" | bc 2>/dev/null || echo "N/A")
else
  SPEEDUP="N/A"
fi

{
  echo ""
  echo "=============================================="
  echo " SUMMARY"
  echo "=============================================="
  echo ""
  echo "  Total Rows:       $TOTAL_ROWS"
  echo "  Fragments:        $NUM_FRAGMENTS"
  echo "  Parallelism:      $PARALLELISM"
  echo "  Index Columns:    10"
  echo ""
  echo "  Dataset Gen:      ${DATAGEN_SECS}s"
  echo "  Range-Based:      ${RANGEBASED_SECS}s"
  echo "  Segmented:        ${SEGMENTED_SECS}s"
  echo "  Speedup:          ${SPEEDUP}x"
  echo ""
  echo "=============================================="
} | tee -a "$REPORT_FILE"

echo ""
echo "Full report: $REPORT_FILE"
echo "Done."
