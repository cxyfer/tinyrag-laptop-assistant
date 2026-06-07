from __future__ import annotations

from pathlib import Path

from tinyrag.generation import (
    ContextEchoBackend,
    LlamaCppBackend,
    LlamaCppConfig,
    fallback_token_count,
    generate_answer,
    stream_answer,
)
from tinyrag.indexing import build_vector_index
from tinyrag.models import GenerationMetrics
from tinyrag.parsing import parse_spec_html
from tinyrag.prompting import INSUFFICIENT_CONTEXT_ANSWER, build_prompt
from tinyrag.retrieval import search

FIXTURE_HTML = (Path(__file__).resolve().parents[1] / "data/raw/am6h_spec.html").read_text(encoding="utf-8")


class StaticBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def stream(self, prompt: str, config: LlamaCppConfig):
        del prompt, config
        self.calls += 1
        yield self.text

    def count_tokens(self, text: str) -> int:
        return fallback_token_count(text)


class RaisingBackend:
    def stream(self, prompt: str, config: LlamaCppConfig):
        del prompt, config
        raise AssertionError("model backend should not be called")

    def count_tokens(self, text: str) -> int:
        return fallback_token_count(text)


class FakeLlama:
    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, prompt: str, **kwargs):
        del prompt
        self.kwargs = kwargs
        return [{"choices": [{"text": "ok"}]}]

    def tokenize(self, text: bytes):
        del text
        return [1]


def _retrieved(query: str):
    index = build_vector_index(parse_spec_html(FIXTURE_HTML, "https://example.test/spec"))
    return search(index, query, top_k=3)


def test_prompt_injects_context_and_grounding_instruction() -> None:
    results = _retrieved("BXH GPU")
    prompt = build_prompt("BXH GPU?", results)

    assert "Answer only from the retrieved context" in prompt
    assert "Keep the answer concise" in prompt
    assert "Do not repeat the question" in prompt
    assert "Final answer:" in prompt
    assert "RTX™ 5090" in prompt
    assert "Insufficient specification context" in prompt


def test_llama_backend_passes_stop_sequences() -> None:
    fake_llama = FakeLlama()
    backend = LlamaCppBackend()
    backend._llm = fake_llama
    backend._loaded_path = Path("models/mock.gguf")
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"), stop_sequences=("\nQuestion:", "\nAnswer:"))

    assert list(backend.stream("prompt", config)) == ["ok"]
    assert fake_llama.kwargs is not None
    assert fake_llama.kwargs["stop"] == ["\nQuestion:", "\nAnswer:"]


def test_streaming_metrics_with_mock_backend() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    chunks = list(stream_answer("BXH GPU?", _retrieved("BXH GPU"), config, ContextEchoBackend()))

    assert any(isinstance(item, str) and item for item in chunks)
    metrics = chunks[-1]
    assert isinstance(metrics, GenerationMetrics)
    assert metrics.ttft_seconds is not None
    assert metrics.generated_tokens > 0
    assert metrics.tps >= 0


def test_generated_answer_strips_prompt_continuation() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    backend = StaticBackend("RTX 5090\n\nQuestion: next question\nAnswer: extra")

    metrics = generate_answer("BXH GPU?", _retrieved("BXH GPU"), config, backend)

    assert metrics.answer == "RTX 5090"


def test_answerable_generation_removes_mixed_refusal_text() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    backend = StaticBackend(f"99Wh {INSUFFICIENT_CONTEXT_ANSWER}")

    metrics = generate_answer("What is the battery capacity?", _retrieved("battery capacity"), config, backend)

    assert metrics.answer == "99Wh"
    assert INSUFFICIENT_CONTEXT_ANSWER not in metrics.answer


def test_insufficient_context_refuses_without_model() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    metrics = generate_answer("Does it include a projector?", [], config, RaisingBackend())

    assert metrics.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert metrics.generated_tokens == fallback_token_count(INSUFFICIENT_CONTEXT_ANSWER)


def test_unsupported_query_refuses_with_unrelated_retrieval() -> None:
    config = LlamaCppConfig(model_path=Path("models/mock.gguf"))
    retrieved = _retrieved("Does it include a built-in projector?")

    metrics = generate_answer("Does it include a built-in projector?", retrieved, config, RaisingBackend())

    assert retrieved
    assert metrics.answer == INSUFFICIENT_CONTEXT_ANSWER
