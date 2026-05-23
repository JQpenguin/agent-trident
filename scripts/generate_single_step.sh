set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

python scripts/generate_dataset.py \
    --generator_provider "${GENERATOR_PROVIDER:-gemini}" \
    --generator_model "${GENERATOR_MODEL:-gemini-3.1-pro-preview}" \
    --review_provider "${REVIEW_PROVIDER:-openai}" \
    --review_model "${REVIEW_MODEL:-o3}" \
    --prompt_path ./prompt/single_step.txt \
    --output_dir ./data \
    --num_samples "${NUM_SAMPLES:-20}" \
    --batch_size "${BATCH_SIZE:-10}" \
    --subtask single_step \
    --output_name "${OUTPUT_NAME:-single_step_dataset.json}"
