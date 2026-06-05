from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinyrag.config import DEFAULT_BENCHMARK_PATH, DEFAULT_SUMMARY_PATH
from tinyrag.generation import ContextEchoBackend, LlamaCppConfig, generate_answer
from tinyrag.indexing import VectorIndex
from tinyrag.models import BenchmarkQuestion
from tinyrag.retrieval import search

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
        results.append(
            {
                "id": question.id,
                "question": question.question,
                "expected_fields": list(question.expected_fields),
                "expected_variants": list(question.expected_variants),
                "retrieved": [item.to_dict() for item in retrieved],
                "generation": metrics.to_dict(),
            }
        )
    payload = {"questions": results, "summary": summarize_results(results)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(format_markdown_summary(payload), encoding="utf-8")
    return payload


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    exact_hits = 0
    variant_hits = 0
    refusal_hits = 0
    for result in results:
        retrieved = result["retrieved"]
        fields = {item["field"] for item in retrieved[:3]}
        variants = {item["variant"] for item in retrieved[:3]}
        expected_fields = set(result["expected_fields"])
        expected_variants = set(result["expected_variants"])
        if not expected_fields or expected_fields & fields:
            exact_hits += 1
        if not expected_variants or expected_variants & variants or "ALL" in variants:
            variant_hits += 1
        if result["id"] == "unknown-refusal" and "Insufficient" in result["generation"]["answer"]:
            refusal_hits += 1
    count = len(results) or 1
    return {
        "groundedness": "Retrieved evidence and answer text are recorded per question.",
        "exactness_hit_rate": exact_hits / count,
        "bilingual_robustness": "Question set includes Traditional Chinese, English, and mixed phrasing.",
        "variant_awareness_hit_rate": variant_hits / count,
        "refusal_behavior": "pass" if refusal_hits else "review",
    }


def format_markdown_summary(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TinyRAG Benchmark Summary",
        "",
        f"- Groundedness: {summary['groundedness']}",
        f"- Exactness hit rate: {summary['exactness_hit_rate']:.2%}",
        f"- Bilingual robustness: {summary['bilingual_robustness']}",
        f"- Variant awareness hit rate: {summary['variant_awareness_hit_rate']:.2%}",
        f"- Refusal behavior: {summary['refusal_behavior']}",
        "",
        "| Question | Top field | Top variant | TTFT | TPS | Tokens |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in payload["questions"]:
        top = result["retrieved"][0] if result["retrieved"] else {"field": "-", "variant": "-"}
        generation = result["generation"]
        ttft = generation["ttft_seconds"]
        lines.append(
            f"| {result['id']} | {top['field']} | {top['variant']} | "
            f"{ttft if ttft is not None else 0:.4f} | {generation['tps']:.2f} | "
            f"{generation['generated_tokens']} |"
        )
    return "\n".join(lines) + "\n"
