set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

if [ -z "$1" ]; then
    DATA_PATH="$DATA_DIR/trident_benchmark.json"
    if [ ! -f "$DATA_PATH" ]; then
        DATA_PATH="$DATA_DIR/trident_benchmark_full.json"
    fi
    if [ ! -f "$DATA_PATH" ]; then
        DATA_PATH=$(ls -t "$DATA_DIR"/trident_benchmark*.json 2>/dev/null | head -1)
    fi
    if [ -z "$DATA_PATH" ] || [ ! -f "$DATA_PATH" ]; then
        echo "Dataset file not found. Check $DATA_DIR."
        exit 1
    fi
else
    DATA_PATH="$1"
fi

NUM_WORKERS="${2:-10}"
ALL_STRATEGIES="${AGENT_STRATEGIES:-react reflexion plan_execute}"

MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
MODEL_NAME_SAVE="${MODEL_NAME_SAVE:-mistral-7b}"
VLLM_API_URL="${VLLM_API_URL:-http://localhost:8000/v1/chat/completions}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"

if [ "$NUM_WORKERS" -eq 1 ]; then
    PARALLEL_FLAG=""
    PARALLEL_DESC="serial"
else
    PARALLEL_FLAG="--parallel --num_workers $NUM_WORKERS"
    PARALLEL_DESC="parallel ($NUM_WORKERS workers)"
fi

DATASET_NAME=$(basename "$DATA_PATH" .json)
OUTPUT_BASE="$PROJECT_ROOT/results/infer_results(${DATASET_NAME})"

echo "============================================================"
echo "Provider:       vLLM"
echo "============================================================"
echo "Dataset path:   $DATA_PATH"
echo "Dataset name:   $DATASET_NAME"
echo "Model:          $MODEL"
echo "Saved as:       $MODEL_NAME_SAVE"
echo "vLLM endpoint:  $VLLM_API_URL"
echo "Max output tokens: $VLLM_MAX_TOKENS"
echo "Agent strategies: $ALL_STRATEGIES"
echo "Inference mode: $PARALLEL_DESC"
echo "Output directory: $OUTPUT_BASE"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

for LEVEL in 1 2; do
    echo ""
    echo "============================================================"
    echo ">>> Starting Level-$LEVEL inference ($PARALLEL_DESC)..."
    echo "============================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/level${LEVEL}"
    mkdir -p "$OUTPUT_DIR/infer_results"

    LEVEL_ARGS=""
    for L in 1 2 3; do
        if [ $L -eq $LEVEL ]; then
            LEVEL_ARGS="$LEVEL_ARGS --level_$L"
        else
            LEVEL_ARGS="$LEVEL_ARGS --no-level_$L"
        fi
    done

    python main.py \
        --mode infer \
        --data_path "$DATA_PATH" \
        --output_dictory "$OUTPUT_DIR" \
        --model_type vllm \
        --api_url "$VLLM_API_URL" \
        --model "$MODEL" \
        --model_name_save "$MODEL_NAME_SAVE" \
        $LEVEL_ARGS \
        $PARALLEL_FLAG \
        --lang en

    echo ">>> Level-$LEVEL inference complete: $OUTPUT_DIR/infer_results/"
done

REACT_RESULTS_PATH="${OUTPUT_BASE}/level3_react/infer_results/results_${MODEL_NAME_SAVE}.json"

for AGENT_STRATEGY in $ALL_STRATEGIES; do
    echo ""
    echo "============================================================"
    echo ">>> Starting Level-3 inference strategy=$AGENT_STRATEGY ($PARALLEL_DESC)..."
    echo "============================================================"

    OUTPUT_DIR="${OUTPUT_BASE}/level3_${AGENT_STRATEGY}"
    mkdir -p "$OUTPUT_DIR/infer_results"

    REACT_REUSE_FLAG=""
    if [ "$AGENT_STRATEGY" = "reflexion" ] && [ -f "$REACT_RESULTS_PATH" ]; then
        REACT_REUSE_FLAG="--react_results_path $REACT_RESULTS_PATH"
        echo ">>> Reusing ReAct results for Reflexion first trial"
    fi

    python main.py \
        --mode infer \
        --data_path "$DATA_PATH" \
        --output_dictory "$OUTPUT_DIR" \
        --model_type vllm \
        --api_url "$VLLM_API_URL" \
        --model "$MODEL" \
        --model_name_save "$MODEL_NAME_SAVE" \
        --no-level_1 --no-level_2 --level_3 \
        --agent_strategy "$AGENT_STRATEGY" \
        $PARALLEL_FLAG \
        --lang en \
        $REACT_REUSE_FLAG

    echo ">>> Level-3 ($AGENT_STRATEGY) inference complete: $OUTPUT_DIR/infer_results/"
done

echo ""
echo "============================================================"
echo "Three-level inference complete."
echo "============================================================"
echo "Output directory: $OUTPUT_BASE"
echo ""
echo "Structure:"
echo "  $OUTPUT_BASE/"
echo "  |-- level1/infer_results/results_${MODEL_NAME_SAVE}.json"
echo "  |-- level2/infer_results/results_${MODEL_NAME_SAVE}.json"
for s in $ALL_STRATEGIES; do
    echo "  |-- level3_${s}/infer_results/results_${MODEL_NAME_SAVE}.json"
done
echo "============================================================"
echo ""
echo "Next evaluation command:"
echo "  bash scripts/eval_all.sh \"$OUTPUT_BASE\" \"\" \"$MODEL_NAME_SAVE\""
echo "============================================================"
