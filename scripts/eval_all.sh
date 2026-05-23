set -e

CALCULATE_TYPE="soft"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_BASE="$PROJECT_ROOT/results"

if [ -z "$1" ]; then
    INFER_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name 'infer_results(trident_benchmark*)' 2>/dev/null | sort -r | head -1)
    if [ -z "$INFER_DIR" ]; then
        INFER_DIR=$(ls -td "$RESULTS_BASE"/infer_results\(*\) "$RESULTS_BASE"/infer-results-* 2>/dev/null | head -1)
    fi
    if [ -z "$INFER_DIR" ]; then
        echo "Inference results directory not found. Run an inference script first."
        exit 1
    fi
else
    INFER_DIR="$1"
fi

if [ -z "$2" ]; then
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
    DATA_PATH="$2"
fi

MODEL_ARG="${3:-}"

AVAILABLE_MODELS=()
for LEVEL in 1 2 3; do
    for f in "$INFER_DIR/level${LEVEL}/infer_results"/results_*.json "$INFER_DIR"/level3_*/infer_results/results_*.json; do
        [ -f "$f" ] || continue
        m=$(basename "$f" .json | sed 's/results_//')
        if [[ ! " ${AVAILABLE_MODELS[*]} " =~ " ${m} " ]]; then
            AVAILABLE_MODELS+=("$m")
        fi
    done
done

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
    echo "No model results found in the inference results directory."
    exit 1
fi

if [ "$MODEL_ARG" == "all" ]; then
    echo "Evaluating all models: ${AVAILABLE_MODELS[*]}"
    echo ""
    for m in "${AVAILABLE_MODELS[@]}"; do
        echo "============================================================"
        echo ">>> Evaluating model: $m"
        echo "============================================================"
        bash "$0" "$INFER_DIR" "$DATA_PATH" "$m"
        echo ""
    done
    echo "Generating cross-model leaderboard..."
    python utils/generate_summary.py --eval_dir "$INFER_DIR" --leaderboard --calculate_type "$CALCULATE_TYPE"
    echo "All model evaluations complete."
    exit 0
elif [ -n "$MODEL_ARG" ]; then
    MODEL="$MODEL_ARG"
    FOUND=false
    for m in "${AVAILABLE_MODELS[@]}"; do
        if [ "$m" == "$MODEL" ]; then
            FOUND=true
            break
        fi
    done
    if [ "$FOUND" != "true" ]; then
        echo "No inference results found for model '$MODEL'"
        echo "   Available models: ${AVAILABLE_MODELS[*]}"
        exit 1
    fi
elif [ ${#AVAILABLE_MODELS[@]} -eq 1 ]; then
    MODEL="${AVAILABLE_MODELS[0]}"
else
    echo "Multiple model results were found:"
    for i in "${!AVAILABLE_MODELS[@]}"; do
        echo "   [$((i+1))] ${AVAILABLE_MODELS[$i]}"
    done
    echo ""
    echo "Pass the model to evaluate as the third argument, for example:"
    echo "   bash $0 $INFER_DIR \"$DATA_PATH\" ${AVAILABLE_MODELS[0]}"
    echo ""
    echo "Or evaluate all models at once:"
    echo "   bash $0 $INFER_DIR \"$DATA_PATH\" all"
    exit 1
fi

DETECTED_STRATEGIES=""
for strategy in react reflexion plan_execute; do
    if [ -f "$INFER_DIR/level3_${strategy}/infer_results/results_${MODEL}.json" ]; then
        DETECTED_STRATEGIES="$DETECTED_STRATEGIES $strategy"
    fi
done
if [ -z "$DETECTED_STRATEGIES" ] && [ -f "$INFER_DIR/level3/infer_results/results_${MODEL}.json" ]; then
    DETECTED_STRATEGIES="react"
fi
DETECTED_STRATEGIES=$(echo "$DETECTED_STRATEGIES" | xargs)

echo "============================================================"
echo "TRIDENT three-level evaluation (soft mode)"
echo "============================================================"
echo "Inference results directory: $INFER_DIR"
echo "Dataset path:   $DATA_PATH"
echo "Model:          $MODEL"
echo "Detected strategies: ${DETECTED_STRATEGIES:-none}"
echo "Evaluation output:     $INFER_DIR/eval_${MODEL}/"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

EVAL_BASE="$INFER_DIR/eval_${MODEL}"
mkdir -p "$EVAL_BASE"

for LEVEL in 1 2; do
    echo ""
    echo "============================================================"
    echo ">>> Starting Level-$LEVEL evaluation..."
    echo "============================================================"

    RESULTS_DIR="$INFER_DIR/level${LEVEL}/infer_results"
    OUTPUT_DIR="$EVAL_BASE/level${LEVEL}"

    MODEL_RESULT_FILE="$RESULTS_DIR/results_${MODEL}.json"
    if [ ! -f "$MODEL_RESULT_FILE" ]; then
        echo "Level-$LEVEL result file is missing (results_${MODEL}.json); skipping."
        continue
    fi

    mkdir -p "$OUTPUT_DIR/eval_results"
    mkdir -p "$OUTPUT_DIR/table_results"

    TEMP_RESULTS_DIR=$(mktemp -d)
    ln -sf "$(realpath "$MODEL_RESULT_FILE")" "$TEMP_RESULTS_DIR/results_${MODEL}.json"

    LEVEL_ARGS=""
    for L in 1 2 3; do
        if [ $L -eq $LEVEL ]; then
            LEVEL_ARGS="$LEVEL_ARGS --level_$L"
        else
            LEVEL_ARGS="$LEVEL_ARGS --no-level_$L"
        fi
    done

    if [ $LEVEL -eq 2 ]; then
        ANSWER_PATTERN='(?s)(.*)'
    else
        ANSWER_PATTERN='<answer>(.*?)</answer>'
    fi

    python main.py \
        --mode eval \
        --data_path "$DATA_PATH" \
        --results_dictory "$TEMP_RESULTS_DIR" \
        --output_dictory "$OUTPUT_DIR" \
        --calculate_type "$CALCULATE_TYPE" \
        --answer_pattern "$ANSWER_PATTERN" \
        $LEVEL_ARGS \
        --lang en

    rm -rf "$TEMP_RESULTS_DIR"
    echo ">>> Level-$LEVEL evaluation complete: $OUTPUT_DIR/"
done

if [ -z "$DETECTED_STRATEGIES" ]; then
    echo ""
    echo "No Level-3 inference results detected; skipping Level-3 evaluation."
else
    for AGENT_STRATEGY in $DETECTED_STRATEGIES; do
        echo ""
        echo "============================================================"
        echo ">>> Starting Level-3 evaluation - strategy: $AGENT_STRATEGY..."
        echo "============================================================"

        if [ -d "$INFER_DIR/level3_${AGENT_STRATEGY}" ]; then
            L3_RESULTS_DIR="$INFER_DIR/level3_${AGENT_STRATEGY}/infer_results"
        else
            L3_RESULTS_DIR="$INFER_DIR/level3/infer_results"
        fi

        OUTPUT_DIR="$EVAL_BASE/level3_${AGENT_STRATEGY}"
        mkdir -p "$OUTPUT_DIR/eval_results"
        mkdir -p "$OUTPUT_DIR/table_results"

        TEMP_RESULTS_DIR=$(mktemp -d)
        ln -sf "$(realpath "$L3_RESULTS_DIR/results_${MODEL}.json")" "$TEMP_RESULTS_DIR/results_${MODEL}.json"

        python main.py \
            --mode eval \
            --data_path "$DATA_PATH" \
            --results_dictory "$TEMP_RESULTS_DIR" \
            --output_dictory "$OUTPUT_DIR" \
            --calculate_type "$CALCULATE_TYPE" \
            --answer_pattern '<answer>(.*?)</answer>' \
            --no-level_1 --no-level_2 --level_3 \
            --agent_strategy "$AGENT_STRATEGY" \
            --lang en

        rm -rf "$TEMP_RESULTS_DIR"
        echo ">>> Level-3 ($AGENT_STRATEGY) evaluation complete: $OUTPUT_DIR/"
    done
fi

echo ""
echo "============================================================"
echo ">>> Generating summary report and error attribution..."
echo "============================================================"

python utils/generate_summary.py \
    --eval_dir "$EVAL_BASE" \
    --model "$MODEL" \
    --calculate_type "$CALCULATE_TYPE"

echo ""
echo "============================================================"
echo "Three-level evaluation complete."
echo "============================================================"
echo "Evaluation results directory: $EVAL_BASE"
echo ""
echo "Structure:"
echo "  eval_${MODEL}/"
echo "  |-- level1/eval_results/"
echo "  |-- level2/eval_results/"
for s in $DETECTED_STRATEGIES; do
    echo "  |-- level3_${s}/eval_results/"
done
echo "  |-- eval_all/"
echo "      |-- 1_Overall_Performance_*.txt"
echo "      |-- 7_Error_Attribution_*.txt"
echo "      |-- Summary_All_Levels_*.json"
echo "============================================================"
