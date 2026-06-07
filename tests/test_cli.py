from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tinyrag.cli import app
from tinyrag.indexing import build_vector_index
from tinyrag.models import GenerationMetrics
from tinyrag.parsing import parse_spec_html, write_records_json

FIXTURE_HTML = (Path(__file__).resolve().parents[1] / "data/raw/am6h_spec.html").read_text(encoding="utf-8")


def test_cli_ingest_and_build_index_paths(tmp_path: Path) -> None:
    runner = CliRunner()
    cache = tmp_path / "am6h.html"
    records = tmp_path / "records.json"
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.json"
    cache.write_text(FIXTURE_HTML, encoding="utf-8")

    ingest_result = runner.invoke(
        app,
        ["ingest", "--cache-html", str(cache), "--output", str(records), "--prefer-cache"],
    )
    build_result = runner.invoke(
        app,
        ["build-index", "--records-path", str(records), "--vectors-path", str(vectors), "--metadata-path", str(metadata)],
    )

    assert ingest_result.exit_code == 0, ingest_result.output
    assert build_result.exit_code == 0, build_result.output
    assert records.exists()
    assert vectors.exists()
    assert metadata.exists()


def test_cli_benchmark_writes_outputs(tmp_path: Path) -> None:
    runner = CliRunner()
    records_path = tmp_path / "records.json"
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "benchmark.json"
    summary = tmp_path / "summary.md"
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    write_records_json(records, records_path)
    build_vector_index(records).save(vectors, metadata)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--vectors-path",
            str(vectors),
            "--metadata-path",
            str(metadata),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "TinyRAG Benchmark Summary" in summary.read_text(encoding="utf-8")


def test_cli_ask_forwards_decoding_controls(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.json"
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    build_vector_index(records).save(vectors, metadata)
    captured = {}

    def fake_stream_answer(question, retrieved, config):
        captured["question"] = question
        captured["retrieved"] = retrieved
        captured["config"] = config
        yield GenerationMetrics(answer="ok", ttft_seconds=0.1, tps=10.0, generated_tokens=1, total_seconds=0.1)

    monkeypatch.setattr("tinyrag.cli.stream_answer", fake_stream_answer)

    result = runner.invoke(
        app,
        [
            "ask",
            "BXH GPU?",
            "--vectors-path",
            str(vectors),
            "--metadata-path",
            str(metadata),
            "--model-path",
            str(tmp_path / "model.gguf"),
            "--temperature",
            "0.35",
            "--max-tokens",
            "42",
            "--repeat-penalty",
            "1.2",
            "--frequency-penalty",
            "0.4",
            "--n-gpu-layers",
            "7",
        ],
    )

    config = captured["config"]
    assert result.exit_code == 0, result.output
    assert captured["question"] == "BXH GPU?"
    assert captured["retrieved"]
    assert config.temperature == 0.35
    assert config.max_tokens == 42
    assert config.repeat_penalty == 1.2
    assert config.frequency_penalty == 0.4
    assert config.n_gpu_layers == 7


def test_cli_benchmark_forwards_decoding_controls(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "benchmark.json"
    summary = tmp_path / "summary.md"
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    build_vector_index(records).save(vectors, metadata)
    captured = {}

    def fake_run_benchmark(index, config, output_path, summary_path, use_model=False):
        captured["index"] = index
        captured["config"] = config
        captured["output_path"] = output_path
        captured["summary_path"] = summary_path
        captured["use_model"] = use_model
        return {"questions": []}

    monkeypatch.setattr("tinyrag.cli.run_benchmark", fake_run_benchmark)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--vectors-path",
            str(vectors),
            "--metadata-path",
            str(metadata),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--model-path",
            str(tmp_path / "model.gguf"),
            "--temperature",
            "0.35",
            "--max-tokens",
            "42",
            "--repeat-penalty",
            "1.2",
            "--frequency-penalty",
            "0.4",
            "--n-gpu-layers",
            "7",
            "--use-model",
        ],
    )

    config = captured["config"]
    assert result.exit_code == 0, result.output
    assert captured["index"].chunks
    assert captured["output_path"] == output
    assert captured["summary_path"] == summary
    assert captured["use_model"] is True
    assert config.temperature == 0.35
    assert config.max_tokens == 42
    assert config.repeat_penalty == 1.2
    assert config.frequency_penalty == 0.4
    assert config.n_gpu_layers == 7
