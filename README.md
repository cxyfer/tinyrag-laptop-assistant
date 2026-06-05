# TinyRAG Laptop Assistant

TinyRAG is a uv-managed pure Python RAG assistant for answering grounded questions about the GIGABYTE AORUS MASTER 16 AM6H specification page. It uses structured specification ingestion, local vector retrieval, and a llama.cpp-compatible streaming generation layer without LangChain, LlamaIndex, Ollama, hosted LLM APIs, or managed vector databases.

## Features

- Structured AM6H ingestion with variant-aware records for BXH, BYH, and BZH.
- Cached HTML fallback at `data/raw/am6h_spec.html` when the official page is unavailable.
- Field-level and variant-comparison chunks instead of arbitrary full-page windows.
- Local NumPy vector index plus JSON metadata.
- Hybrid retrieval combining dense similarity, keyword matches, field aliases, and variant boosts.
- Traditional Chinese, English, and mixed-language benchmark questions.
- llama.cpp-compatible streaming generation settings for model path, context length, temperature, max tokens, and GPU offload layers.
- Benchmark output with retrieved evidence, TTFT, TPS, generated token count, total generation time, answer text, and README-ready Markdown summary.

## Setup

```bash
uv sync --extra dev
```

For real local generation, install an audited llama.cpp runtime in the environment before using `tinyrag ask` with a GGUF model. The code loads `llama_cpp.Llama` dynamically so tests and mock benchmarks do not require the binding.

The default tests and benchmark path do not require a real local model. They use a deterministic mock streaming backend.

## Local CPU Development

Run ingestion from the cached source first for deterministic local development:

```bash
uv run tinyrag ingest --prefer-cache
uv run tinyrag build-index
uv run tinyrag benchmark
```

Ask with a real model after placing a GGUF file under `models/`:

```bash
uv run tinyrag ask "BXH 使用哪一張顯示卡？" \
  --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --n-ctx 2048 \
  --temperature 0.1 \
  --max-tokens 256 \
  --n-gpu-layers 0
```

Use `--n-gpu-layers 0` for CPU-only testing. Increase it during CUDA validation only after confirming VRAM headroom.

## Model Selection and 4GB VRAM Strategy

The primary inference path is llama.cpp through `llama-cpp-python` because GGUF quantization, CPU fallback, and `n_gpu_layers` GPU offload are a better fit for a single-user 4GB VRAM laptop assistant than high-throughput serving stacks.

Recommended baseline:

- Small instruct GGUF model around 1.5B parameters.
- Q4 quantization such as Q4_K_M.
- Context length around 2048 for short specification answers.
- Low temperature, e.g. `0.1`, because values must preserve exact model names, numbers, units, and variants.
- Conservative GPU layer offload; fall back to CPU when VRAM is constrained.

vLLM is not the default because its strengths are batching and server throughput on larger GPUs. For this assignment, llama.cpp provides lower operational complexity and finer-grained low-VRAM control. A 3B GGUF model can be tried later, but it is optional and may require tighter context length or fewer GPU layers.

## Data and Artifacts

- `data/raw/am6h_spec.html`: cached source HTML fallback.
- `data/processed/am6h_specs.json`: deterministic structured specification records.
- `data/index/vectors.npy`: local NumPy vector matrix.
- `data/index/metadata.json`: chunk metadata.
- `data/benchmarks/benchmark_results.json`: structured benchmark output.
- `data/benchmarks/benchmark_summary.md`: Markdown-friendly qualitative summary.
- `models/`: local GGUF model placement.

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
uv run tinyrag ask "What GPU does the BXH variant use?" --model-path models/model.gguf
```

Retrieves relevant chunks, assembles a grounded prompt, streams a llama.cpp-compatible answer, and prints TTFT/TPS metrics. A real GGUF model and `llama-cpp-python` are required for this command unless tests inject a mock backend.

### Benchmark

```bash
uv run tinyrag benchmark
```

Runs fixed questions covering CPU, GPU variants, display, memory, storage, ports, communication, webcam, battery, dimensions, and mixed Traditional Chinese / English phrasing. By default it uses a mock echo backend so benchmark schema and retrieval evidence are testable without a local model. Use `--use-model` to benchmark real llama.cpp generation.

## Benchmark Methodology

Each question records:

- Retrieved chunks, scores, dense scores, boost scores, field, variant, aliases, value, and source URL.
- Generation metrics: TTFT, TPS, generated token count, total generation time, and answer text.
- A qualitative summary covering groundedness, exactness, bilingual robustness, variant awareness, and refusal behavior.

Current local placeholder results are produced by the mock backend and are intended to validate pipeline correctness rather than real model speed. Real TTFT/TPS should be measured after placing a GGUF model under `models/` and running:

```bash
uv run tinyrag benchmark --use-model --model-path models/<model>.gguf
```

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
```

## Sources

- [GIGABYTE U.S.A. official AORUS MASTER 16 AM6H specifications](https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp)
- [AORUS official AORUS MASTER 16 AM6H specifications](https://www.aorus.com/en-us/laptops/aorus-master-16-am6h/Specification)
