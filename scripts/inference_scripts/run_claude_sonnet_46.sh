#!/bin/bash
set -e

MODEL_NAME="claude-sonnet-4-6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "Running inference with ${MODEL_NAME}..."
echo "Project root: ${PROJECT_ROOT}"

# Default arguments
DEFAULT_ARGS=(
    --model_name "${MODEL_NAME}"
    --benchmark_file data/benchmark/text/benchmark.csv
    --eval_mode both
    # --use_multimodal
    --result_path "results/text/${MODEL_NAME}"
    --size 32k
    --max_items 10
    --parallel 1
)

# Run inference with default args plus any additional args passed to script
python3 inference.py "${DEFAULT_ARGS[@]}" "$@"

echo "Inference completed for ${MODEL_NAME} with multimodal mode"
