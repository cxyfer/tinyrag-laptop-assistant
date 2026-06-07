from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from tinyrag.models import GenerationMetrics, RetrievedChunk
from tinyrag.prompting import INSUFFICIENT_CONTEXT_ANSWER, build_prompt
from tinyrag.retrieval import has_sufficient_context
from tinyrag.streaming import measure_stream

DEFAULT_STOP_SEQUENCES = (
    "\nQuestion:",
    "\n\nQuestion:",
)
PROMPT_CONTINUATION_RE = re.compile(r"\n\s*(?:Question|Answer|Final answer):", re.IGNORECASE)
LEADING_ANSWER_LABEL_RE = re.compile(r"^\s*(?:Answer|Final answer):\s*", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LlamaCppConfig:
    model_path: Path
    n_ctx: int = 2048
    temperature: float = 0.3
    max_tokens: int = 256
    n_gpu_layers: int = 0
    repeat_penalty: float = 1.15
    frequency_penalty: float = 0.3
    stop_sequences: tuple[str, ...] = field(default_factory=lambda: DEFAULT_STOP_SEQUENCES)


class StreamingBackend(Protocol):
    def stream(self, prompt: str, config: LlamaCppConfig) -> Iterable[str]: ...

    def count_tokens(self, text: str) -> int: ...


class LlamaCppBackend:
    def __init__(self) -> None:
        self._llm = None
        self._loaded_path: Path | None = None

    def _load(self, config: LlamaCppConfig):
        if self._llm is not None and self._loaded_path == config.model_path:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install an audited llama.cpp runtime before real model generation."
            ) from exc
        self._llm = Llama(
            model_path=str(config.model_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )
        self._loaded_path = config.model_path
        return self._llm

    def stream(self, prompt: str, config: LlamaCppConfig) -> Iterable[str]:
        llm = self._load(config)
        stream = llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            repeat_penalty=config.repeat_penalty,
            frequency_penalty=config.frequency_penalty,
            stop=list(config.stop_sequences) or None,
            stream=True,
            echo=False,
        )
        for event in stream:
            choices = event.get("choices", []) if isinstance(event, dict) else []
            if choices:
                yield _choice_text(choices[0])

    def count_tokens(self, text: str) -> int:
        if self._llm is not None:
            return len(self._llm.tokenize(text.encode("utf-8")))
        return fallback_token_count(text)


def fallback_token_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)
    return len(tokens)


def _choice_text(choice: object) -> str:
    if not isinstance(choice, dict):
        return ""

    text = choice.get("text")
    if isinstance(text, str):
        return text

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    content = choice.get("content")
    return content if isinstance(content, str) else ""


def normalize_generated_answer(answer: str) -> str:
    text = THINK_BLOCK_RE.sub("", answer).strip()
    text = LEADING_ANSWER_LABEL_RE.sub("", text, count=1).strip()
    if not text:
        return ""

    continuation = PROMPT_CONTINUATION_RE.search(text)
    if continuation:
        text = text[: continuation.start()].strip()

    if INSUFFICIENT_CONTEXT_ANSWER in text and text != INSUFFICIENT_CONTEXT_ANSWER:
        answer_before_refusal = text.split(INSUFFICIENT_CONTEXT_ANSWER, 1)[0].strip()
        return answer_before_refusal or INSUFFICIENT_CONTEXT_ANSWER

    return text


class ContextEchoBackend:
    def stream(self, prompt: str, config: LlamaCppConfig) -> Iterable[str]:
        del config
        if INSUFFICIENT_CONTEXT_ANSWER in prompt and "Retrieved context:\n\nQuestion:" in prompt:
            yield INSUFFICIENT_CONTEXT_ANSWER
            return
        marker = "Retrieved context:\n"
        context = prompt.split(marker, 1)[1].split("\n\nQuestion:", 1)[0]
        yield "根據檢索到的規格："
        yield context[: max(80, min(600, len(context)))]

    def count_tokens(self, text: str) -> int:
        return fallback_token_count(text)


def stream_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    config: LlamaCppConfig,
    backend: StreamingBackend | None = None,
) -> Iterator[str | GenerationMetrics]:
    selected_backend = backend or LlamaCppBackend()
    context_results = retrieved if has_sufficient_context(retrieved, question) else []
    prompt = build_prompt(question, context_results)
    if not context_results:
        chunks = [INSUFFICIENT_CONTEXT_ANSWER]
    else:
        chunks = selected_backend.stream(prompt, config)
    yield from measure_stream(chunks, selected_backend.count_tokens, answer_transform=normalize_generated_answer)


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    config: LlamaCppConfig,
    backend: StreamingBackend | None = None,
) -> GenerationMetrics:
    metrics: GenerationMetrics | None = None
    for item in stream_answer(question, retrieved, config, backend):
        if isinstance(item, GenerationMetrics):
            metrics = item
    if metrics is None:
        return GenerationMetrics(answer="")
    return metrics
