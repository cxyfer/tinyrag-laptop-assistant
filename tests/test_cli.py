from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tinyrag.cli import app
from tinyrag.indexing import build_vector_index
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
