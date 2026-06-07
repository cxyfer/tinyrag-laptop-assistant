from __future__ import annotations

from pathlib import Path

from tinyrag.generation import ContextEchoBackend, LlamaCppConfig, fallback_token_count, generate_answer, stream_answer
from tinyrag.indexing import build_vector_index
from tinyrag.models import GenerationMetrics
from tinyrag.parsing import parse_spec_html
from tinyrag.prompting import INSUFFICIENT_CONTEXT_ANSWER, build_prompt
from tinyrag.retrieval import search

FIXTURE_HTML = (Path(__file__).resolve().parents[1] / "data/raw/am6h_spec.html").read_text(encoding="utf-8")


def _retrieved(query: str):
    index = build_vector_index(parse_spec_html(FIXTURE_HTML, "https://example.test/spec"))
    return search(index, query, top_k=3)


def test_prompt_injects_context_and_grounding_instruction() -> None:
    results = _retrieved("BXH GPU")
    prompt = build_prompt("BXH GPU?", results)

    assert "Answer only from the retrieved context" in prompt
    assert "RTX™ 5090" in prompt
    assert "Insufficient specification context" in prompt


def test_streaming_metrics_with_mock_backend() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    chunks = list(stream_answer("BXH GPU?", _retrieved("BXH GPU"), config, ContextEchoBackend()))

    assert any(isinstance(item, str) and item for item in chunks)
    metrics = chunks[-1]
    assert isinstance(metrics, GenerationMetrics)
    assert metrics.ttft_seconds is not None
    assert metrics.generated_tokens > 0
    assert metrics.tps >= 0


def test_insufficient_context_refuses_without_model() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    metrics = generate_answer("Does it include a projector?", [], config, ContextEchoBackend())

    assert metrics.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert metrics.generated_tokens == fallback_token_count(INSUFFICIENT_CONTEXT_ANSWER)


def test_unsupported_query_refuses_with_unrelated_retrieval() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    retrieved = _retrieved("Does it include a built-in projector?")

    metrics = generate_answer("Does it include a built-in projector?", retrieved, config, ContextEchoBackend())

    assert retrieved
    assert metrics.answer == INSUFFICIENT_CONTEXT_ANSWER
