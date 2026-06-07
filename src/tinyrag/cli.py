from __future__ import annotations

from pathlib import Path

import typer

from tinyrag.config import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_CACHE_HTML,
    DEFAULT_METADATA_PATH,
    DEFAULT_RECORDS_JSON,
    DEFAULT_SOURCE_URL,
    DEFAULT_SUMMARY_PATH,
    DEFAULT_VECTORS_PATH,
    ensure_data_dirs,
)
from tinyrag.evaluation import BENCHMARK_MODEL_MAX_TOKENS, run_benchmark
from tinyrag.generation import GenerationMetrics, LlamaCppConfig, stream_answer
from tinyrag.indexing import VectorIndex, build_vector_index
from tinyrag.ingestion import ingest_specs
from tinyrag.parsing import read_records_json, write_records_json
from tinyrag.retrieval import search

app = typer.Typer(help="Tiny local RAG assistant for AORUS MASTER 16 AM6H specifications.")


@app.command()
def ingest(
    source_url: str = typer.Option(DEFAULT_SOURCE_URL, help="Official specification page URL."),
    cache_html: Path = typer.Option(DEFAULT_CACHE_HTML, help="Cached HTML fallback path."),
    output: Path = typer.Option(DEFAULT_RECORDS_JSON, help="Processed JSON output path."),
    prefer_cache: bool = typer.Option(False, help="Use cached HTML before remote fetch."),
) -> None:
    """Fetch or load cached source data and write processed specification records."""
    ensure_data_dirs()
    records = ingest_specs(source_url, cache_html, output, prefer_cache=prefer_cache)
    write_records_json(records, output)
    typer.echo(f"Wrote {len(records)} records to {output}")


@app.command("build-index")
def build_index(
    records_path: Path = typer.Option(DEFAULT_RECORDS_JSON, help="Processed records JSON path."),
    vectors_path: Path = typer.Option(DEFAULT_VECTORS_PATH, help="Vector matrix output path."),
    metadata_path: Path = typer.Option(DEFAULT_METADATA_PATH, help="Index metadata output path."),
) -> None:
    """Create chunks, embeddings, and local vector index artifacts."""
    records = read_records_json(records_path)
    index = build_vector_index(records)
    index.save(vectors_path, metadata_path)
    typer.echo(f"Wrote {len(index.chunks)} chunks to {metadata_path} and {vectors_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="User question."),
    vectors_path: Path = typer.Option(DEFAULT_VECTORS_PATH, help="Vector matrix path."),
    metadata_path: Path = typer.Option(DEFAULT_METADATA_PATH, help="Index metadata path."),
    model_path: Path = typer.Option(Path("models/model.gguf"), help="GGUF model path."),
    n_ctx: int = typer.Option(2048, help="llama.cpp context length."),
    temperature: float = typer.Option(0.1, help="Generation temperature."),
    max_tokens: int = typer.Option(256, help="Maximum generated tokens."),
    n_gpu_layers: int = typer.Option(0, help="llama.cpp GPU offload layers."),
    top_k: int = typer.Option(5, help="Retrieved chunks."),
) -> None:
    """Retrieve context and stream a grounded answer."""
    index = VectorIndex.load(vectors_path, metadata_path)
    retrieved = search(index, question, top_k=top_k)
    config = LlamaCppConfig(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=temperature,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers,
    )
    metrics: GenerationMetrics | None = None
    for item in stream_answer(question, retrieved, config):
        if isinstance(item, GenerationMetrics):
            metrics = item
        else:
            typer.echo(item, nl=False)
    typer.echo()
    if metrics:
        typer.echo(
            f"TTFT={metrics.ttft_seconds if metrics.ttft_seconds is not None else 0:.4f}s "
            f"TPS={metrics.tps:.2f} tokens={metrics.generated_tokens}"
        )


@app.command()
def benchmark(
    vectors_path: Path = typer.Option(DEFAULT_VECTORS_PATH, help="Vector matrix path."),
    metadata_path: Path = typer.Option(DEFAULT_METADATA_PATH, help="Index metadata path."),
    output: Path = typer.Option(DEFAULT_BENCHMARK_PATH, help="Benchmark JSON output path."),
    summary: Path = typer.Option(DEFAULT_SUMMARY_PATH, help="Markdown summary output path."),
    model_path: Path = typer.Option(Path("models/model.gguf"), help="GGUF model path."),
    n_ctx: int = typer.Option(2048, help="llama.cpp context length."),
    temperature: float = typer.Option(0.1, help="Generation temperature."),
    max_tokens: int = typer.Option(BENCHMARK_MODEL_MAX_TOKENS, help="Maximum generated tokens."),
    n_gpu_layers: int = typer.Option(0, help="llama.cpp GPU offload layers."),
    use_model: bool = typer.Option(False, help="Use real llama.cpp backend instead of mock echo."),
) -> None:
    """Run the fixed evaluation question set and write structured results."""
    index = VectorIndex.load(vectors_path, metadata_path)
    config = LlamaCppConfig(model_path, n_ctx, temperature, max_tokens, n_gpu_layers)
    payload = run_benchmark(index, config, output, summary, use_model=use_model)
    typer.echo(f"Wrote {len(payload['questions'])} benchmark results to {output}")
    typer.echo(f"Wrote summary to {summary}")


if __name__ == "__main__":
    app()
