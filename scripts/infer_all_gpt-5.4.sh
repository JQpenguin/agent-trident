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

MODEL="gpt-5.4"
MODEL_NAME="openai"

if [ -n "$OPENAI_API_KEY" ]; then
    API_KEY="$OPENAI_API_KEY"
fi
API_KEY="${API_KEY:?Please set OPENAI_API_KEY or API_KEY}"

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
echo "Provider:       OpenAI"
echo "============================================================"
echo "Dataset path:   $DATA_PATH"
echo "Dataset name:   $DATASET_NAME"
echo "Model:          $MODEL"
echo "Provider:       OpenAI"
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
        --model_type api \
        --model_name "$MODEL_NAME" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --model_name_save "$MODEL" \
        $LEVEL_ARGS \
        $PARALLEL_FLAG \
        --lang en

    echo ">>> Level-$LEVEL inference complete: $OUTPUT_DIR/infer_results/"
done

REACT_RESULTS_PATH="${OUTPUT_BASE}/level3_react/infer_results/results_${MODEL}.json"

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
        --model_type api \
        --model_name "$MODEL_NAME" \
        --model "$MODEL" \
        --api_key "$API_KEY" \
        --model_name_save "$MODEL" \
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
echo "  |-- level1/infer_results/results_${MODEL}.json"
echo "  |-- level2/infer_results/results_${MODEL}.json"
for s in $ALL_STRATEGIES; do
    echo "  |-- level3_${s}/infer_results/results_${MODEL}.json"
done
echo "============================================================"
echo ""
echo "Next evaluation command:"
echo "  bash scripts/eval_all.sh \"$OUTPUT_BASE\" \"\" \"$MODEL\""
echo "============================================================"
