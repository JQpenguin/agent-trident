
# 🔱 Agent-Trident Reproduction Guide

This repository contains the anonymized code artifact for reproducing TRIDENT inference and evaluation. The artifact is organized around a simple, four-step workflow: 

1. **Install dependencies**
2. **Run model inference**
3. **Run evaluation**
4. **Inspect reports**

TRIDENT evaluates tool-use hallucination across three distinct levels:

| Level | What is Evaluated | Output |
| :--- | :--- | :--- |
| **Level 1** | Solvability judgment | `solvable`, `unsolvable`, or `uncertain` |
| **Level 2** | Static tool planning | Full tool-call sequence |
| **Level 3** | Dynamic tool execution | ReAct, Reflexion, and Plan-and-Solve trajectories in a deterministic simulator |

> **Note on Data:** The bundled artifact includes `data/ares_benchmark.json`, which is a compact anonymous review split containing 280 samples. These samples cover all seven tool-use structures and all 20 domains.

---

## 🛠️ 1. Environment Setup

First, create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

For **Windows PowerShell**, use the following commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

* **Compatibility Note:** The provided scripts are Bash scripts. If you are on Windows, you must run them through Git Bash, WSL, or another Bash-compatible shell.

---

## 🚀 2. Run Inference

The repository includes three primary inference scripts:

| Script | Provider | Model |
| --- | --- | --- |
| `scripts/infer_all_gpt-5.4.sh` | OpenAI official API | `gpt-5.4` |
| `scripts/infer_all_claude-opus-4-6.sh` | Anthropic official API | `claude-opus-4-6` |
| `scripts/infer_all_qwen3.5-35b-vllm.sh` | Local vLLM server | `Qwen/Qwen3.5-35B` |

These scripts automatically run Level 1 solvability judgment, Level 2 static planning, and Level 3 dynamic execution (using `react`, `reflexion`, and `plan_execute` strategies).

### GPT-5.4

```bash
export OPENAI_API_KEY="your_openai_api_key"
bash scripts/infer_all_gpt-5.4.sh data/ares_benchmark.json 10

```

### Claude-Opus-4.6

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
bash scripts/infer_all_claude-opus-4-6.sh data/ares_benchmark.json 10

```

### Qwen3.5-35B with vLLM

First, start a vLLM OpenAI-compatible server in your model-serving environment:

```bash
vllm serve Qwen/Qwen3.5-35B \
  --served-model-name Qwen/Qwen3.5-35B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192

```

Then, run TRIDENT inference against your local endpoint:

```bash
export VLLM_API_URL="http://localhost:8000/v1/chat/completions"
export VLLM_MODEL="Qwen/Qwen3.5-35B"
export MODEL_NAME_SAVE="qwen3.5-35b"
bash scripts/infer_all_qwen3.5-35b-vllm.sh data/ares_benchmark.json 10

```

### Inference Configuration Tips

* **Concurrency:** The second argument in the script dictates the number of workers. Use `1` for serial inference.
* **Targeted Strategies:** To run only selected Level-3 strategies, prepend `AGENT_STRATEGIES` to your command:
```bash
AGENT_STRATEGIES="react" bash scripts/infer_all_gpt-5.4.sh data/ares_benchmark.json 10

```



Inference results are automatically saved to `results/infer_results(ares_benchmark)/`, categorized by level and strategy.

---

## 📊 3. Run Evaluation

You can evaluate the models individually or all at once.

**To evaluate individual models:**

```bash
bash scripts/eval_all.sh "results/infer_results(ares_benchmark)" data/ares_benchmark.json gpt-5.4
bash scripts/eval_all.sh "results/infer_results(ares_benchmark)" data/ares_benchmark.json claude-opus-4-6
bash scripts/eval_all.sh "results/infer_results(ares_benchmark)" data/ares_benchmark.json qwen3.5-35b

```

**To evaluate every model result found in the inference directory:**

```bash
bash scripts/eval_all.sh "results/infer_results(ares_benchmark)" data/ares_benchmark.json all

```

Evaluation outputs are written under `results/infer_results(ares_benchmark)/eval_<model>/`. The `eval_all/` subdirectory contains the final summary tables and error-attribution report.

---

## 🔄 4. Reproduce the Main Artifact Workflow

To run the complete workflow (API models, optional local vLLM model, and full evaluation), execute the following sequence:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export VLLM_API_URL="http://localhost:8000/v1/chat/completions"
export VLLM_MODEL="Qwen/Qwen3.5-35B"
export MODEL_NAME_SAVE="qwen3.5-35b"

bash scripts/infer_all_gpt-5.4.sh data/ares_benchmark.json 10
bash scripts/infer_all_claude-opus-4-6.sh data/ares_benchmark.json 10
bash scripts/infer_all_qwen3.5-35b-vllm.sh data/ares_benchmark.json 10

bash scripts/eval_all.sh "results/infer_results(ares_benchmark)" data/ares_benchmark.json all

```

Afterward, you can inspect the comprehensive summaries and leaderboards located in the respective `eval_all/` and `leaderboard/` directories.

---

