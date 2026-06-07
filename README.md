# TinyRAG Laptop Assistant

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cxyfer/tinyrag-laptop-assistant/blob/main/notebooks/tinyrag_colab_cuda.ipynb)

TinyRAG is a uv-managed pure Python RAG assistant for answering grounded questions about the GIGABYTE AORUS MASTER 16 AM6H specification page. It uses structured specification ingestion, local vector retrieval, and a llama.cpp-compatible streaming generation layer without LangChain, LlamaIndex, Ollama, hosted LLM APIs, or managed vector databases.

## Features

- Structured AM6H ingestion with variant-aware records for BXH, BYH, and BZH.
- Cached HTML fallback at `data/raw/am6h_spec.html` when the official page is unavailable.
- Field-level and variant-comparison chunks instead of arbitrary full-page windows.
- Local NumPy vector index plus JSON metadata.
- Hybrid retrieval combining dense similarity, keyword matches, field aliases, and variant boosts.
- Traditional Chinese, English, and mixed-language benchmark questions.
- llama.cpp-compatible streaming generation settings for model path, context length, temperature, max tokens, repeat/frequency penalties, and GPU offload layers.
- Benchmark output with retrieved evidence, TTFT, TPS, generated token count, total generation time, answer text, answer-quality checks, and README-ready Markdown summary.

## Setup

For retrieval-only development and mock benchmarks:

```bash
uv sync --extra dev
```

For real GGUF generation, install one llama.cpp extra:

```bash
uv sync --extra llama-cpu      # local CPU
uv sync --extra llama-cu121    # CUDA 12.1 / Colab GPU
```

Run real-model commands with the same extra, for example `uv run --extra llama-cpu ...` or `uv run --extra llama-cu121 ...`. The code loads `llama_cpp.Llama` dynamically so tests and mock benchmarks do not require the binding.

The default tests and benchmark path do not require a real local model. They use a deterministic mock streaming backend.

## Google Colab CUDA Quickstart

Open the notebook directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cxyfer/tinyrag-laptop-assistant/blob/main/notebooks/tinyrag_colab_cuda.ipynb)

Set **Runtime > Change runtime type > GPU**, then run the notebook cells. The notebook clones this repository, installs the CUDA llama.cpp wheel through `uv`, downloads the configured GGUF model, rebuilds the local index from the cached source, and runs a grounded answer with GPU offload.

The notebook defines `MODEL_REPO`, `MODEL_FILE`, `MODEL_PATH`, `N_CTX`, and `N_GPU_LAYERS` near the top. It also keeps the previous Qwen2.5-1.5B baseline and Gemma 4 E2B candidate commented out for quick switching. To switch models, change those values together and keep context length plus GPU layers conservative until VRAM usage is confirmed. Qwen3.5 advertises a much larger training context than this low-VRAM demo uses, so llama.cpp may print `n_ctx_seq (2048) < n_ctx_train (...)`; that is expected and only means the full long-context capacity is not being allocated.

Equivalent Colab shell flow with the default Qwen3.5 2B model:

```bash
nvidia-smi
git clone --branch main https://github.com/cxyfer/tinyrag-laptop-assistant.git
cd tinyrag-laptop-assistant
python -m pip install -q uv
uv sync --frozen --extra llama-cu121
MODEL_REPO=bartowski/Qwen_Qwen3.5-2B-GGUF
MODEL_FILE=Qwen_Qwen3.5-2B-Q4_K_M.gguf
MODEL_PATH=models/$MODEL_FILE
uvx --from huggingface-hub hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir models
uv run --frozen --extra llama-cu121 tinyrag ingest --prefer-cache
uv run --frozen --extra llama-cu121 tinyrag build-index
uv run --frozen --extra llama-cu121 tinyrag ask "BXH 使用哪一張顯示卡？" \
  --model-path "$MODEL_PATH" \
  --n-ctx 2048 \
  --temperature 0.3 \
  --max-tokens 256 \
  --repeat-penalty 1.15 \
  --frequency-penalty 0.3 \
  --n-gpu-layers 35
```

`--n-gpu-layers 35` and `--n-ctx 2048` are Colab starting points for the notebook. `--n-ctx 2048` intentionally keeps KV-cache memory low even when Qwen3.5 reports a much larger `n_ctx_train`; try 4096 or 8192 only after `nvidia-smi` shows enough headroom. The GPU-layer setting fully offloaded the previous Qwen2.5 1.5B Q4_K_M default on a Colab Tesla T4 during validation; reduce it if your runtime reports lower VRAM or the new Qwen3.5 2B default needs more headroom. Unauthenticated Hugging Face downloads work for the demo, but setting `HF_TOKEN` can avoid rate limits.

## Local CPU Development

Run ingestion from the cached source first for deterministic local development:

```bash
uv run tinyrag ingest --prefer-cache
uv run tinyrag build-index
uv run tinyrag benchmark
```

For real CPU generation, install the CPU extra and download the default GGUF model:

```bash
uv sync --extra llama-cpu
uvx --from huggingface-hub hf download \
  bartowski/Qwen_Qwen3.5-2B-GGUF \
  Qwen_Qwen3.5-2B-Q4_K_M.gguf \
  --local-dir models
```

Ask with the downloaded model:

```bash
uv run --extra llama-cpu tinyrag ask "BXH 使用哪一張顯示卡？" \
  --model-path models/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
  --n-ctx 2048 \
  --temperature 0.3 \
  --max-tokens 256 \
  --repeat-penalty 1.15 \
  --frequency-penalty 0.3 \
  --n-gpu-layers 0
```

Use `--n-gpu-layers 0` for CPU-only testing. Increase it during CUDA validation only after confirming VRAM headroom.

## Model Selection and 4GB VRAM Strategy

The primary inference path is llama.cpp through `llama-cpp-python` because GGUF quantization, CPU fallback, and `n_gpu_layers` GPU offload are a better fit for a single-user 4GB VRAM laptop assistant than high-throughput serving stacks.

Recommended model tiers:

| Tier | Candidate | Role | 4GB note |
|---|---|---|---|
| Stable fallback | Qwen2.5-1.5B-Instruct-GGUF or Qwen2.5-3B-Instruct-GGUF | Official Qwen2.5 GGUF baseline for repeatable validation | Safest option; keep for comparison and fallback. |
| Primary 2026 candidate | Qwen3.5-2B Q4_K_M GGUF | Newer Qwen open-weight model via community llama.cpp GGUF quantization | Strong first upgrade candidate; small enough for short-context 4GB testing. |
| Secondary 2026 candidate | Gemma 4 E2B-it 4-bit GGUF | New Gemma 4 edge/laptop candidate with llama.cpp ecosystem support | Viable but tighter; start with conservative context length. |

Practical settings:

- Use conservative anti-repetition decoding, e.g. `--temperature 0.3 --repeat-penalty 1.15 --frequency-penalty 0.3`, because answers must stay exact while avoiding sentence loops.
- Start Qwen2.5 and Qwen3.5-2B with `--n-ctx 2048`; the Qwen3.5 `n_ctx_train` warning is expected at this setting. Increase to 4096 or 8192 only after confirming memory headroom, and reduce if KV cache pressure appears.
- Start Gemma 4 E2B with `--n-ctx 1024` or `1536`, then increase only after confirming memory headroom.
- Start with `--n-gpu-layers 0` for CPU validation, then gradually increase GPU offload layers during CUDA testing.
- Treat Qwen3.5 GGUF files as community quantizations of official Qwen3.5 weights unless the model repository is published by the Qwen organization.

To switch models in Colab or local commands, change the Hugging Face repository, GGUF filename, `--model-path`, `--n-ctx`, and `--n-gpu-layers` together. Example variables:

```bash
MODEL_REPO=bartowski/Qwen_Qwen3.5-2B-GGUF
MODEL_FILE=Qwen_Qwen3.5-2B-Q4_K_M.gguf
MODEL_PATH=models/$MODEL_FILE
```

Then download the selected file and pass `"$MODEL_PATH"` to `tinyrag ask` or `tinyrag benchmark --use-model`.

vLLM is not the default because its strengths are batching and server throughput on larger GPUs. For this assignment, llama.cpp provides lower operational complexity and finer-grained low-VRAM control. Larger models such as Qwen3.5-4B/9B or Gemma 4 E4B/12B can be evaluated later, but they are not recommended as 4GB defaults.

## Data and Artifacts

- `data/raw/am6h_spec.html`: cached source HTML fallback.
- `data/processed/am6h_specs.json`: deterministic structured specification records.
- `data/index/vectors.npy`: local NumPy vector matrix.
- `data/index/metadata.json`: chunk metadata.
- `data/benchmarks/benchmark_results.json`: structured benchmark output.
- `data/benchmarks/benchmark_summary.md`: Markdown-friendly qualitative summary.
- `models/`: local GGUF model placement. GGUF files are ignored by git and should be downloaded locally or in Colab.

## CLI

```bash
uv run tinyrag ingest --help
uv run tinyrag build-index --help
uv run tinyrag ask --help
uv run tinyrag benchmark --help
```

### Ingest

```bash
uv run tinyrag ingest --prefer-cache
```

Fetches the official source URL when available, falls back to `data/raw/am6h_spec.html`, parses structured records, normalizes field aliases, preserves official values, and writes deterministic JSON.

### Build Index

```bash
uv run tinyrag build-index
```

Builds field-level chunks, GPU comparison chunks, hash-based multilingual CPU embeddings, NumPy vectors, and JSON metadata.

### Ask

```bash
uv run --extra llama-cpu tinyrag ask "What GPU does the BXH variant use?" \
  --model-path models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

Retrieves relevant chunks, assembles a grounded prompt, streams a llama.cpp-compatible answer, and prints TTFT/TPS metrics. A real GGUF model and matching `llama-cpu` or `llama-cu121` extra are required for this command unless tests inject a mock backend.

### Benchmark

```bash
uv run tinyrag benchmark
```

Runs fixed questions covering CPU, GPU variants, display, memory, storage, ports, communication, webcam, battery, dimensions, and mixed Traditional Chinese / English phrasing. By default it uses a mock echo backend so benchmark schema, retrieval evidence, and answer-quality checks are testable without a local model. Use `--use-model` to benchmark real llama.cpp generation. Benchmark generation defaults to a concise token budget; pass `--max-tokens` only when intentionally testing longer answers.

## Benchmark Methodology

Each question records:

- Retrieved chunks, scores, dense scores, boost scores, field, variant, aliases, value, and source URL.
- Generation metrics: TTFT, TPS, generated token count, total generation time, and answer text.
- Answer-quality checks for unexpected refusal text, prompt continuation, token-limit saturation, repetition, and unsupported elaboration.
- A summary covering retrieval field hits, retrieval variant hits, bilingual robustness, answer quality, and refusal behavior.

The committed benchmark artifacts preserve a pre-fix Colab real-model baseline and include post-change answer-quality diagnostics. Local mock benchmark runs validate pipeline correctness rather than real model speed. Fresh post-change TTFT/TPS should be measured with the matching llama.cpp extra and a downloaded GGUF model:

```bash
uv run --extra llama-cpu tinyrag benchmark --use-model \
  --model-path models/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
  --n-ctx 2048 \
  --temperature 0.3 \
  --max-tokens 96 \
  --repeat-penalty 1.15 \
  --frequency-penalty 0.3 \
  --n-gpu-layers 0
```

For Colab GPU, use the CUDA extra and the same model variables from the notebook:

```bash
uv run --frozen --extra llama-cu121 tinyrag benchmark --use-model \
  --model-path "$MODEL_PATH" \
  --n-ctx 2048 \
  --temperature 0.3 \
  --max-tokens 96 \
  --repeat-penalty 1.15 \
  --frequency-penalty 0.3 \
  --n-gpu-layers 35
```

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
```

## Sources

- [GIGABYTE U.S.A. official AORUS MASTER 16 AM6H specifications](https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp)
- [AORUS official AORUS MASTER 16 AM6H specifications](https://www.aorus.com/en-us/laptops/aorus-master-16-am6h/Specification)
- [Qwen3 official release blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3.5 official Hugging Face collection](https://huggingface.co/collections/Qwen/qwen35)
- [Qwen2.5 official Hugging Face collection](https://huggingface.co/collections/Qwen/qwen25)
- [Qwen3.5-2B community GGUF quantization](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF)
- [Gemma releases — Google AI for Developers](https://ai.google.dev/gemma/docs/releases)
- [Get started with Gemma models — Google AI for Developers](https://ai.google.dev/gemma/docs/get_started)
- [Gemma 4 E2B-it community GGUF quantization](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
