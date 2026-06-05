from __future__ import annotations

from pathlib import Path

from tinyrag.evaluation import format_markdown_summary, run_benchmark
from tinyrag.generation import LlamaCppConfig
from tinyrag.indexing import build_vector_index
from tinyrag.parsing import parse_spec_html

FIXTURE_HTML = Path("/home/usaya/workspace/interview/tinyrag-laptop-assistant/data/raw/am6h_spec.html").read_text(encoding="utf-8")


def test_benchmark_result_schema_and_summary(tmp_path: Path) -> None:
    index = build_vector_index(parse_spec_html(FIXTURE_HTML, "https://example.test/spec"))
    payload = run_benchmark(
        index,
        LlamaCppConfig(model_path=Path("models/mock.gguf")),
        tmp_path / "benchmark.json",
        tmp_path / "summary.md",
    )

    first = payload["questions"][0]
    assert {"id", "question", "retrieved", "generation"}.issubset(first)
    assert {"ttft_seconds", "tps", "generated_tokens", "total_seconds", "answer"}.issubset(first["generation"])
    assert payload["summary"]["exactness_hit_rate"] > 0
    assert "| Question | Top field |" in format_markdown_summary(payload)
