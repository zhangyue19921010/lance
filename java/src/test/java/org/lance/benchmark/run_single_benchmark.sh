#!/bin/bash
#
# Simplified benchmark runner - runs on the current branch only.
# Designed to be run twice: once on main (range-based) and once on the PR branch (segmented).
#
# Usage:
#   ./run_single_benchmark.sh <mode> <type> [datasetPath]
#     mode: small | large
#     type: rangebased | segmented | datagen
#     datasetPath: (optional) path to existing dataset. If not provided, generates a new one.
#
# Examples:
#   # Step 1: Generate shared dataset (run once)
#   ./run_single_benchmark.sh small datagen
#
#   # Step 2: On main branch, run range-based benchmark
#   git checkout main && ./mvnw compile test-compile -DskipTests
#   ./run_single_benchmark.sh small rangebased /tmp/lance_btree_benchmark/shared_dataset
#
#   # Step 3: On PR branch, run segmented benchmark
#   git checkout btree-distributed-build-segmented-pr1 && ./mvnw compile test-compile -DskipTests
#   ./run_single_benchmark.sh small segmented /tmp/lance_btree_benchmark/shared_dataset
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
JAVA_DIR="$PROJECT_ROOT/java"

MODE="${1:-small}"
TYPE="${2:-segmented}"
CUSTOM_DATASET="${3:-}"

PARALLELISM=20
BENCHMARK_DIR="/tmp/lance_btree_benchmark"

# Configuration based on mode
if [ "$MODE" = "small" ]; then
  TOTAL_ROWS=100
  NUM_FRAGMENTS=20
  BATCH_SIZE=10
elif [ "$MODE" = "large" ]; then
  TOTAL_ROWS=1000000000
  NUM_FRAGMENTS=20
  BATCH_SIZE=1000000
else
  echo "Unknown mode: $MODE. Use 'small' or 'large'."
  exit 1
fi

mkdir -p "$BENCHMARK_DIR"

# Determine dataset path
if [ -n "$CUSTOM_DATASET" ]; then
  DATASET_PATH="$CUSTOM_DATASET"
else
  DATASET_PATH="$BENCHMARK_DIR/shared_dataset"
fi

echo "=== Lance BTree Index Benchmark ==="
echo "Mode: $MODE"
echo "Type: $TYPE"
echo "Branch: $(cd $PROJECT_ROOT && git branch --show-current)"
echo "Dataset: $DATASET_PATH"
echo "Parallelism: $PARALLELISM"
echo "Total Rows: $TOTAL_ROWS"
echo ""

# Build classpath
cd "$JAVA_DIR"
CLASSPATH=$(./mvnw -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || \
  mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null || \
  echo "")

# Fallback: try to find jars in target/dependency
if [ -z "$CLASSPATH" ]; then
  echo "Warning: Could not get classpath from maven. Trying target/dependency..."
  ./mvnw -q dependency:copy-dependencies 2>/dev/null || mvn -q dependency:copy-dependencies 2>/dev/null
  CLASSPATH=$(echo "$JAVA_DIR/target/dependency/"*.jar | tr ' ' ':')
fi

CLASSPATH="$JAVA_DIR/target/classes:$JAVA_DIR/target/test-classes:$CLASSPATH"

# Find JNI library
JNI_PATH=""
if [ -d "$JAVA_DIR/lance-jni/target/release" ]; then
  JNI_PATH="$JAVA_DIR/lance-jni/target/release"
elif [ -d "$JAVA_DIR/lance-jni/target/debug" ]; then
  JNI_PATH="$JAVA_DIR/lance-jni/target/debug"
else
  echo "Warning: JNI library path not found. JNI may fail."
  JNI_PATH="$JAVA_DIR/lance-jni/target/release:$JAVA_DIR/lance-jni/target/debug"
fi

JAVA_OPTS="-Xmx16g --add-opens=java.base/java.nio=ALL-UNNAMED -Djava.library.path=$JNI_PATH"

case "$TYPE" in
  datagen)
    echo "=== Generating Dataset ==="
    rm -rf "$DATASET_PATH"
    java -cp "$CLASSPATH" $JAVA_OPTS \
      org.lance.benchmark.BenchmarkDataGenerator \
      "$DATASET_PATH" "$TOTAL_ROWS" "$NUM_FRAGMENTS" "$BATCH_SIZE"
    echo ""
    echo "Dataset generated at: $DATASET_PATH"
    ;;

  rangebased)
    echo "=== Running Range-Based Benchmark ==="
    # Copy dataset so original is untouched
    WORK_DATASET="$BENCHMARK_DIR/dataset_rangebased_$(date +%s)"
    cp -r "$DATASET_PATH" "$WORK_DATASET"
    java -cp "$CLASSPATH" $JAVA_OPTS \
      org.lance.benchmark.BTreeRangeBasedBenchmark \
      "$WORK_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/result_rangebased.txt"
    echo ""
    echo "Results saved to: $BENCHMARK_DIR/result_rangebased.txt"
    ;;

  segmented)
    echo "=== Running Segmented Benchmark ==="
    # Copy dataset so original is untouched
    WORK_DATASET="$BENCHMARK_DIR/dataset_segmented_$(date +%s)"
    cp -r "$DATASET_PATH" "$WORK_DATASET"
    java -cp "$CLASSPATH" $JAVA_OPTS \
      org.lance.benchmark.BTreeSegmentedBenchmark \
      "$WORK_DATASET" "$PARALLELISM" 2>&1 | tee "$BENCHMARK_DIR/result_segmented.txt"
    echo ""
    echo "Results saved to: $BENCHMARK_DIR/result_segmented.txt"
    ;;

  *)
    echo "Unknown type: $TYPE. Use 'datagen', 'rangebased', or 'segmented'."
    exit 1
    ;;
esac

echo ""
echo "Done."
