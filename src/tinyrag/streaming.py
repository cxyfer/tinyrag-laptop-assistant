from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

from tinyrag.models import GenerationMetrics


def measure_stream(
    chunks: Iterable[str],
    token_counter,
    answer_transform: Callable[[str], str] | None = None,
) -> Iterator[str | GenerationMetrics]:
    start = time.perf_counter()
    first_token_time: float | None = None
    answer_parts: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter()
        answer_parts.append(chunk)
        yield chunk
    end = time.perf_counter()
    answer = "".join(answer_parts)
    if answer_transform is not None:
        answer = answer_transform(answer)
    total_seconds = max(end - start, 0.0)
    generated_tokens = token_counter(answer)
    yield GenerationMetrics(
        ttft_seconds=None if first_token_time is None else first_token_time - start,
        tps=generated_tokens / total_seconds if total_seconds > 0 else 0.0,
        generated_tokens=generated_tokens,
        total_seconds=total_seconds,
        answer=answer,
    )
