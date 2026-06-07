from __future__ import annotations

from pathlib import Path

from tinyrag.evaluation import assess_answer_quality, format_markdown_summary, run_benchmark
from tinyrag.generation import LlamaCppConfig
from tinyrag.indexing import build_vector_index
from tinyrag.parsing import parse_spec_html
from tinyrag.prompting import INSUFFICIENT_CONTEXT_ANSWER

FIXTURE_HTML = (Path(__file__).resolve().parents[1] / "data/raw/am6h_spec.html").read_text(encoding="utf-8")


def test_benchmark_result_schema_and_summary(tmp_path: Path) -> None:
    index = build_vector_index(parse_spec_html(FIXTURE_HTML, "https://example.test/spec"))
    payload = run_benchmark(
        index,
        LlamaCppConfig(model_path=Path("models/mock.gguf")),
        tmp_path / "benchmark.json",
        tmp_path / "summary.md",
    )

    first = payload["questions"][0]
    assert {"id", "question", "retrieved", "generation", "answer_quality"}.issubset(first)
    assert {"ttft_seconds", "tps", "generated_tokens", "total_seconds", "answer"}.issubset(first["generation"])
    assert {"passes", "issues", "prompt_continuation", "unexpected_refusal"}.issubset(first["answer_quality"])
    assert payload["summary"]["retrieval_field_hit_rate"] > 0

    markdown = format_markdown_summary(payload)
    assert "Retrieval field hit rate" in markdown
    assert "Retrieval variant hit rate" in markdown
    assert "Answer quality pass rate" in markdown
    assert "Exactness hit rate" not in markdown
    assert "| Question | Top field | Top variant | Quality | Issues |" in markdown


def test_answer_quality_flags_generation_issues() -> None:
    result = {
        "id": "display-mixed",
        "expected_fields": ["Display"],
        "expected_variants": [],
        "retrieved": [
            {
                "field": "Display",
                "variant": "BXH",
                "text": "Field: Display; Official value: 16-inch OLED, 240Hz",
                "value": "16-inch OLED, 240Hz",
            }
        ],
        "generation": {
            "ttft_seconds": 0.1,
            "tps": 20.0,
            "generated_tokens": 96,
            "total_seconds": 1.0,
            "answer": "240Hz\n\nQuestion: AM6H 螢幕 display refresh rate 是多少？\nAnswer: 240Hz",
        },
    }

    quality = assess_answer_quality(result, token_limit=96)

    assert not quality["passes"]
    assert "prompt_continuation" in quality["issues"]
    assert "token_limit_saturation" in quality["issues"]


def test_answer_quality_flags_excessive_repetition() -> None:
    result = {
        "id": "communication-en",
        "expected_fields": ["Communication"],
        "expected_variants": [],
        "retrieved": [
            {
                "field": "Communication",
                "variant": "BXH",
                "text": "Field: Communication; Official value: Wi-Fi 7, Bluetooth 5.4",
                "value": "Wi-Fi 7, Bluetooth 5.4",
            }
        ],
        "generation": {
            "ttft_seconds": 0.1,
            "tps": 20.0,
            "generated_tokens": 40,
            "total_seconds": 1.0,
            "answer": (
                "The answer is: Wi-Fi 7 and Bluetooth 5.4. "
                "The answer is: Wi-Fi 7 and Bluetooth 5.4. "
                "The answer is: Wi-Fi 7 and Bluetooth 5.4."
            ),
        },
    }

    quality = assess_answer_quality(result, token_limit=96)

    assert not quality["passes"]
    assert "excessive_repetition" in quality["issues"]


def test_answer_quality_flags_unsupported_elaboration() -> None:
    result = {
        "id": "gpu-bxh-en",
        "expected_fields": ["GPU"],
        "expected_variants": ["BXH"],
        "retrieved": [
            {
                "field": "GPU",
                "variant": "BXH",
                "text": "Field: GPU; Official value: NVIDIA RTX 5090 Laptop GPU 24GB GDDR7",
                "value": "NVIDIA RTX 5090 Laptop GPU 24GB GDDR7",
            }
        ],
        "generation": {
            "ttft_seconds": 0.1,
            "tps": 20.0,
            "generated_tokens": 40,
            "total_seconds": 1.0,
            "answer": (
                "The BXH variant uses NVIDIA RTX 5090 Laptop GPU 24GB GDDR7 "
                "and is ideal for high performance gaming."
            ),
        },
    }

    quality = assess_answer_quality(result, token_limit=96)

    assert not quality["passes"]
    assert "unsupported_elaboration" in quality["issues"]


def test_answer_quality_flags_unexpected_refusal_for_answerable_question() -> None:
    result = {
        "id": "battery-en",
        "expected_fields": ["Battery"],
        "expected_variants": [],
        "retrieved": [
            {
                "field": "Battery",
                "variant": "BXH",
                "text": "Field: Battery; Official value: 99Wh",
                "value": "99Wh",
            }
        ],
        "generation": {
            "ttft_seconds": 0.1,
            "tps": 20.0,
            "generated_tokens": 13,
            "total_seconds": 1.0,
            "answer": f"99Wh {INSUFFICIENT_CONTEXT_ANSWER}",
        },
    }

    quality = assess_answer_quality(result, token_limit=96)

    assert not quality["passes"]
    assert "unexpected_refusal" in quality["issues"]
