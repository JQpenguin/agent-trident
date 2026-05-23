# 🔱 TRIDENT

<p align="center">
  <b>A benchmark for evaluating tool-use hallucination in agentic systems.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-quick-start-deepseek-v4-pro">Quick Start</a> |
  <a href="#-data">Data</a> |
  <a href="#-run-inference">Inference</a> |
  <a href="#-run-evaluation">Evaluation</a> |
  <a href="#-project-structure">Structure</a>
</p>

TRIDENT evaluates whether an agentic model can correctly judge task solvability, plan valid tool calls, and execute tool-use trajectories in a deterministic simulated environment.

## 🔥 News

- **2026.05**: The anonymized TRIDENT reproduction artifact is released.
- **2026.05**: DeepSeek v4-pro, local vLLM inference, and unified evaluation scripts are supported.

## 📚 Contents

- [Overview](#-overview)
- [Benchmark Figures](#-benchmark-figures)
- [Environment Setup](#-environment-setup)
- [Data](#-data)
- [Quick Start: DeepSeek v4-pro](#-quick-start-deepseek-v4-pro)
- [Run Inference](#-run-inference)
- [Run Evaluation](#-run-evaluation)
- [Local Open-Source Models with vLLM](#-local-open-source-models-with-vllm)
- [Project Structure](#-project-structure)
- [Output Structure](#-output-structure)
- [Optional Data Generation](#-optional-data-generation)
- [Troubleshooting](#-troubleshooting)

## 🧭 Overview

TRIDENT evaluates tool-use behavior across three levels:

| Level | Evaluation Target | Expected Output |
| --- | --- | --- |
| **Level 1** | Solvability judgment | `solvable`, `unsolvable`, or `uncertain` |
| **Level 2** | Static tool planning | Full tool-call sequence |
| **Level 3** | Dynamic tool execution | ReAct, Reflexion, and Plan-Execute trajectories |

The benchmark covers seven tool-use structures and twenty application domains. Each inference script automatically runs Level 1, Level 2, and Level 3.

## 🖼️ Benchmark Figures

### Data Construction Pipeline

![TRIDENT data construction pipeline](./assets/figures/data-construction-pipeline.png)

### Leaderboard

![TRIDENT model leaderboard](./assets/figures/model-leaderboard.png)

## 🛠️ Environment Setup

Create and activate the conda environment:

```bash
conda create -n trident python=3.10
conda activate trident
pip install -r requirements.txt
```

The provided scripts are Bash scripts. On Windows, run them through WSL, Git Bash, or another Bash-compatible shell.

## 📦 Data

The repository includes two benchmark files:

| File | Description |
| --- | --- |
| `data/trident_benchmark.json` | Default 200-sample benchmark file used by the scripts |
| `data/trident_benchmark_full.json` | Full 1120-sample benchmark split |

If no data path is provided, the inference scripts automatically use `data/trident_benchmark.json`.

## 🚀 Quick Start: DeepSeek v4-pro

Run inference and evaluation with DeepSeek v4-pro:

```bash
conda create -n trident python=3.10
conda activate trident
pip install -r requirements.txt

export DEEPSEEK_API_KEY="your_deepseek_api_key"

bash scripts/infer_all_deepseek-v4-pro.sh

bash scripts/eval_all.sh "results/infer_results(trident_benchmark)" data/trident_benchmark.json deepseek-v4-pro
```

The DeepSeek script uses:

- model: `deepseek-v4-pro`
- default dataset: `data/trident_benchmark.json`
- default parallel workers: `10`
- output directory: `results/infer_results(trident_benchmark)`

To run the full benchmark instead:

```bash
bash scripts/infer_all_deepseek-v4-pro.sh data/trident_benchmark_full.json 10
bash scripts/eval_all.sh "results/infer_results(trident_benchmark_full)" data/trident_benchmark_full.json deepseek-v4-pro
```

## 🤖 Run Inference

Available inference scripts:

| Script | Provider | Default Model |
| --- | --- | --- |
| `scripts/infer_all_gpt-5.4.sh` | OpenAI API | `gpt-5.4` |
| `scripts/infer_all_claude-opus-4-6.sh` | Anthropic API | `claude-opus-4-6` |
| `scripts/infer_all_deepseek-v4-pro.sh` | DeepSeek API | `deepseek-v4-pro` |
| `scripts/infer_all_qwen3.5-35b-vllm.sh` | Local vLLM server | `Qwen/Qwen3.5-35B-A3B` |
| `scripts/infer_all_mistral-7b-vllm.sh` | Local vLLM server | `mistralai/Mistral-7B-Instruct-v0.3` |

API model examples:

```bash
export OPENAI_API_KEY="your_openai_api_key"
bash scripts/infer_all_gpt-5.4.sh data/trident_benchmark.json 10

export ANTHROPIC_API_KEY="your_anthropic_api_key"
bash scripts/infer_all_claude-opus-4-6.sh data/trident_benchmark.json 10
```

To use DeepSeek v4-flash instead of v4-pro:

```bash
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_MODEL_NAME_SAVE="deepseek-v4-flash"
bash scripts/infer_all_deepseek-v4-pro.sh data/trident_benchmark.json 10
```

## 📊 Run Evaluation

Evaluate one model:

```bash
bash scripts/eval_all.sh "results/infer_results(trident_benchmark)" data/trident_benchmark.json deepseek-v4-pro
```

Evaluate all detected model results:

```bash
bash scripts/eval_all.sh "results/infer_results(trident_benchmark)" data/trident_benchmark.json all
```

Evaluation outputs are written under:

```text
results/infer_results(trident_benchmark)/eval_<model>/
```

## 🧩 Local Open-Source Models with vLLM

Start a vLLM OpenAI-compatible server first, then run the corresponding TRIDENT inference script.

### Mistral-7B

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --served-model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
```

```bash
export VLLM_API_URL="http://localhost:8000/v1/chat/completions"
export VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
export MODEL_NAME_SAVE="mistral-7b"
bash scripts/infer_all_mistral-7b-vllm.sh data/trident_benchmark.json 10
```

### Qwen3.5-35B-A3B

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --served-model-name Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096
```

```bash
export VLLM_API_URL="http://localhost:8000/v1/chat/completions"
export VLLM_MODEL="Qwen/Qwen3.5-35B-A3B"
export MODEL_NAME_SAVE="qwen3.5-35b-a3b"
bash scripts/infer_all_qwen3.5-35b-vllm.sh data/trident_benchmark.json 10
```

The local vLLM examples use a `4096` token context window. The request output cap is controlled by `VLLM_MAX_TOKENS` and defaults to `1024`, because prompt tokens and output tokens must fit inside the context window together.

## 🗂️ Project Structure

```text
TRIDENT/
|-- assets/
|   `-- figures/                 # README figures
|       |-- data-construction-pipeline.png
|       `-- model-leaderboard.png
|-- data/                        # Benchmark datasets
|   |-- trident_benchmark.json
|   `-- trident_benchmark_full.json
|-- prompt/                      # Dataset generation prompts by task type
|-- scripts/                     # Inference, evaluation, and generation scripts
|-- utils/                       # Generation, processing, evaluation, and reporting utilities
|-- results/                     # Generated inference/evaluation outputs
|-- tools_emb/                   # Generated tool embeddings for evaluation
|-- main.py                      # Main inference/evaluation entrypoint
|-- requirements.txt
`-- README.md
```

`results/`, `tools_emb/`, and Python cache files are generated artifacts and are ignored by `.gitignore`.

## 📁 Output Structure

After inference, results are organized by level and strategy:

```text
results/infer_results(trident_benchmark)/
|-- level1/infer_results/results_<model>.json
|-- level2/infer_results/results_<model>.json
|-- level3_react/infer_results/results_<model>.json
|-- level3_reflexion/infer_results/results_<model>.json
|-- level3_plan_execute/infer_results/results_<model>.json
`-- eval_<model>/
```

The path contains parentheses, so quote it in shell commands:

```bash
bash scripts/eval_all.sh "results/infer_results(trident_benchmark)" data/trident_benchmark.json deepseek-v4-pro
```

## 🧪 Optional Data Generation

The repository keeps a single-step data generation entrypoint:

```bash
export GOOGLE_API_KEY="your_google_api_key"
export OPENAI_API_KEY="your_openai_api_key"
bash scripts/generate_single_step.sh
```

You can override generation settings with environment variables:

```bash
GENERATOR_PROVIDER=gemini \
GENERATOR_MODEL=gemini-3.1-pro-preview \
REVIEW_PROVIDER=openai \
REVIEW_MODEL=o3 \
NUM_SAMPLES=20 \
BATCH_SIZE=10 \
OUTPUT_NAME=single_step_dataset.json \
bash scripts/generate_single_step.sh
```

Generated samples are written to `data/single_step_dataset.json`.

## 🩺 Troubleshooting

**DeepSeek model name error.** Use `deepseek-v4-pro` or `deepseek-v4-flash`. The script `scripts/infer_all_deepseek-v4-pro.sh` defaults to `deepseek-v4-pro`.

**Bash path error near `(`.** Quote paths containing parentheses:

```bash
bash scripts/eval_all.sh "results/infer_results(trident_benchmark)" data/trident_benchmark.json deepseek-v4-pro
```

**CRLF script error.** If Bash reports `$'\r': command not found`, convert shell scripts to LF line endings before running them in WSL or Linux.

**vLLM non-JSON response.** Check that the vLLM server is running and that `VLLM_API_URL` points to `/v1/chat/completions`.
