from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tinyrag.config import DEFAULT_BENCHMARK_PATH, DEFAULT_SUMMARY_PATH
from tinyrag.generation import ContextEchoBackend, LlamaCppConfig, generate_answer
from tinyrag.indexing import VectorIndex
from tinyrag.models import BenchmarkQuestion
from tinyrag.prompting import INSUFFICIENT_CONTEXT_ANSWER
from tinyrag.retrieval import search

BENCHMARK_MODEL_MAX_TOKENS = 96
PROMPT_LABEL_RE = re.compile(r"(?:^|\n)\s*(?:Question|Answer|Final answer):", re.IGNORECASE)
UNSUPPORTED_ELABORATION_MARKERS = (
    "known for",
    "ideal for",
    "designed to deliver",
    "compact form factor",
    "high performance",
    "efficiency",
    "4k uhd",
    "8k uhd",
    "40 gbps",
    "適合需要",
    "效能都相當出色",
)

BENCHMARK_QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion("cpu-zh", "這台 AM6H 的 CPU 規格是什麼？", ("CPU",), ()),
    BenchmarkQuestion("gpu-bxh-en", "What GPU does the BXH variant use?", ("GPU",), ("BXH",)),
    BenchmarkQuestion("gpu-byh-zh", "BYH 使用哪一張顯示卡？", ("GPU",), ("BYH",)),
    BenchmarkQuestion("gpu-bzh-en", "List the BZH graphics card and VRAM.", ("GPU",), ("BZH",)),
    BenchmarkQuestion("display-mixed", "AM6H 螢幕 display refresh rate 是多少？", ("Display",), ()),
    BenchmarkQuestion("memory-en", "What memory does it support?", ("Memory",), ()),
    BenchmarkQuestion("storage-zh", "儲存空間和 M.2 插槽規格？", ("Storage",), ()),
    BenchmarkQuestion("ports-mixed", "有 Thunderbolt 5 和 HDMI 嗎？", ("Ports",), ()),
    BenchmarkQuestion("communication-en", "What Wi-Fi and Bluetooth versions are listed?", ("Communication",), ()),
    BenchmarkQuestion("webcam-zh", "視訊鏡頭支援 Windows Hello 嗎？", ("Webcam",), ()),
    BenchmarkQuestion("battery-en", "What is the battery capacity?", ("Battery",), ()),
    BenchmarkQuestion("dimensions-zh", "機身尺寸和重量是多少？", ("Dimensions", "Weight"), ()),
    BenchmarkQuestion("unknown-refusal", "Does it include a built-in projector?", (), ()),
)


def run_benchmark(
    index: VectorIndex,
    config: LlamaCppConfig,
    output_path: Path = DEFAULT_BENCHMARK_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    use_model: bool = False,
) -> dict[str, Any]:
    backend = None if use_model else ContextEchoBackend()
    results = []
    for question in BENCHMARK_QUESTIONS:
        retrieved = search(index, question.question, top_k=5)
        metrics = generate_answer(question.question, retrieved, config, backend=backend)
        result = {
            "id": question.id,
            "question": question.question,
            "expected_fields": list(question.expected_fields),
            "expected_variants": list(question.expected_variants),
            "retrieved": [item.to_dict() for item in retrieved],
            "generation": metrics.to_dict(),
        }
        result["answer_quality"] = assess_answer_quality(result, token_limit=config.max_tokens)
        results.append(result)
    payload = {"questions": results, "summary": summarize_results(results)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(format_markdown_summary(payload), encoding="utf-8")
    return payload


def retrieval_field_hit(result: dict[str, Any]) -> bool:
    expected_fields = set(result["expected_fields"])
    fields = {item["field"] for item in result["retrieved"][:3]}
    return not expected_fields or bool(expected_fields & fields)


def retrieval_variant_hit(result: dict[str, Any]) -> bool:
    expected_variants = set(result["expected_variants"])
    variants = {item["variant"] for item in result["retrieved"][:3]}
    return not expected_variants or bool(expected_variants & variants) or "ALL" in variants


def _has_excessive_repetition(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer).strip().lower()
    if not normalized:
        return False

    repeated_markers = ("the answer is:", "這些", "根据提供的信息", "根據提供的資訊")
    if any(normalized.count(marker) >= 3 for marker in repeated_markers):
        return True

    segments = [
        segment.strip()
        for segment in re.split(r"[。.!?\n]+", normalized)
        if len(segment.strip()) >= 16
    ]
    return any(segments.count(segment) >= 3 for segment in set(segments))


def _has_unsupported_elaboration(answer: str, retrieved: list[dict[str, Any]]) -> bool:
    normalized_answer = answer.lower()
    context = "\n".join(f"{item.get('text', '')}\n{item.get('value', '')}" for item in retrieved).lower()
    return any(marker in normalized_answer and marker not in context for marker in UNSUPPORTED_ELABORATION_MARKERS)


def assess_answer_quality(result: dict[str, Any], token_limit: int) -> dict[str, Any]:
    answer = result["generation"]["answer"]
    answerable = bool(result["expected_fields"]) and retrieval_field_hit(result)
    checks = {
        "unexpected_refusal": answerable and INSUFFICIENT_CONTEXT_ANSWER in answer,
        "prompt_continuation": bool(PROMPT_LABEL_RE.search(answer)),
        "token_limit_saturation": result["generation"]["generated_tokens"] >= token_limit and answer != INSUFFICIENT_CONTEXT_ANSWER,
        "excessive_repetition": _has_excessive_repetition(answer),
        "unsupported_elaboration": _has_unsupported_elaboration(answer, result["retrieved"]),
    }
    issues = [name for name, failed in checks.items() if failed]
    return {"passes": not issues, "issues": issues, **checks}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    field_hits = 0
    variant_hits = 0
    refusal_hits = 0
    quality_hits = 0
    for result in results:
        if retrieval_field_hit(result):
            field_hits += 1
        if retrieval_variant_hit(result):
            variant_hits += 1
        if result["id"] == "unknown-refusal" and result["generation"]["answer"] == INSUFFICIENT_CONTEXT_ANSWER:
            refusal_hits += 1
        if result.get("answer_quality", {}).get("passes", False):
            quality_hits += 1
    count = len(results) or 1
    return {
        "groundedness": "Retrieved evidence, answer text, and deterministic quality checks are recorded per question.",
        "retrieval_field_hit_rate": field_hits / count,
        "bilingual_robustness": "Question set includes Traditional Chinese, English, and mixed phrasing.",
        "retrieval_variant_hit_rate": variant_hits / count,
        "answer_quality_pass_rate": quality_hits / count,
        "refusal_behavior": "pass" if refusal_hits else "review",
    }


def format_markdown_summary(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TinyRAG Benchmark Summary",
        "",
        f"- Groundedness: {summary['groundedness']}",
        f"- Retrieval field hit rate: {summary['retrieval_field_hit_rate']:.2%}",
        f"- Bilingual robustness: {summary['bilingual_robustness']}",
        f"- Retrieval variant hit rate: {summary['retrieval_variant_hit_rate']:.2%}",
        f"- Answer quality pass rate: {summary['answer_quality_pass_rate']:.2%}",
        f"- Refusal behavior: {summary['refusal_behavior']}",
    ]
    if artifact_context := summary.get("artifact_context"):
        lines.append(f"- Artifact context: {artifact_context}")
    lines.extend([
        "",
        "| Question | Top field | Top variant | Quality | Issues | TTFT | TPS | Tokens |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for result in payload["questions"]:
        top = result["retrieved"][0] if result["retrieved"] else {"field": "-", "variant": "-"}
        generation = result["generation"]
        quality = result.get("answer_quality", {"passes": True, "issues": []})
        ttft = generation["ttft_seconds"]
        issues = ", ".join(quality["issues"]) if quality["issues"] else "-"
        lines.append(
            f"| {result['id']} | {top['field']} | {top['variant']} | "
            f"{'pass' if quality['passes'] else 'review'} | {issues} | "
            f"{ttft if ttft is not None else 0:.4f} | {generation['tps']:.2f} | "
            f"{generation['generated_tokens']} |"
        )
    return "\n".join(lines) + "\n"
