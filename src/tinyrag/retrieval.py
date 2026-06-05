from __future__ import annotations

import re
from collections.abc import Iterable

from tinyrag.indexing import VectorIndex
from tinyrag.models import Chunk, RetrievedChunk

WORD_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]+")
VARIANT_RE = re.compile(r"\b(BXH|BYH|BZH)\b", re.IGNORECASE)

QUERY_ALIASES = {
    "GPU": ("gpu", "graphics", "顯卡", "顯示卡", "顯示晶片", "rtx", "vram"),
    "CPU": ("cpu", "processor", "處理器", "中央處理器"),
    "Display": ("display", "screen", "螢幕", "oled", "顯示器", "refresh", "刷新率"),
    "Battery": ("battery", "capacity", "電池", "99wh"),
    "Ports": ("ports", "連接埠", "thunderbolt", "usb", "hdmi"),
    "Memory": ("memory", "ram", "記憶體", "ddr5"),
    "Storage": ("storage", "ssd", "儲存", "m.2", "m2"),
    "Communication": ("communication", "wireless", "wi-fi", "wifi", "bluetooth", "通訊"),
    "Webcam": ("webcam", "camera", "視訊", "鏡頭", "windows hello"),
    "Dimensions": ("dimensions", "size", "尺寸"),
    "Weight": ("weight", "重量", "kg"),
    "Audio": ("audio", "speaker", "speakers", "喇叭", "音效"),
    "Security": ("security", "tpm", "安全"),
    "Adapter": ("adapter", "charger", "變壓器", "330w"),
    "Operating System": ("os", "operating system", "作業系統", "windows"),
}


def query_terms(query: str) -> set[str]:
    lowered = query.lower()
    terms = {token.lower() for token in WORD_RE.findall(lowered)}
    for aliases in QUERY_ALIASES.values():
        if any(alias.lower() in lowered for alias in aliases):
            terms.update(alias.lower() for alias in aliases)
    return terms


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    lower = haystack.lower()
    return any(needle.lower() in lower for needle in needles)


def keyword_boost(query: str, chunk: Chunk) -> float:
    terms = query_terms(query)
    text = f"{chunk.text} {' '.join(chunk.aliases)} {chunk.field} {chunk.value}".lower()
    boost = 0.0
    for term in terms:
        if term and term in text:
            boost += 0.04
    variant_match = VARIANT_RE.search(query)
    if variant_match and chunk.variant in {variant_match.group(1).upper(), "ALL"}:
        boost += 0.25
    if _contains_any(query, chunk.aliases) or chunk.field.lower() in query.lower():
        boost += 0.20
    if _contains_any(query, ("rtx", "5070", "5080", "5090")) and chunk.field == "GPU":
        boost += 0.15
    if _contains_any(query, ("99wh", "battery", "電池")) and chunk.field == "Battery":
        boost += 0.20
    if _contains_any(query, ("thunderbolt", "usb", "hdmi", "連接埠")) and chunk.field == "Ports":
        boost += 0.20
    return boost


def infer_query_fields(query: str) -> set[str]:
    lowered = query.lower()
    fields = {field for field, aliases in QUERY_ALIASES.items() if any(alias.lower() in lowered for alias in aliases)}
    if not fields and VARIANT_RE.search(query):
        fields.add("GPU")
    return fields


def search(index: VectorIndex, query: str, top_k: int = 5) -> list[RetrievedChunk]:
    if not index.chunks:
        return []
    query_vector = index.embedding_model.embed(query)
    dense_scores = index.vectors @ query_vector
    results: list[RetrievedChunk] = []
    for dense_score, chunk in zip(dense_scores, index.chunks, strict=True):
        dense = float(dense_score)
        boost = keyword_boost(query, chunk)
        results.append(RetrievedChunk(chunk=chunk, score=dense + boost, dense_score=dense, boost_score=boost))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def has_sufficient_context(
    results: list[RetrievedChunk],
    query: str,
    threshold: float = 0.12,
) -> bool:
    if not results or results[0].score < threshold:
        return False
    query_fields = infer_query_fields(query)
    if not query_fields:
        return False
    return any(result.chunk.field in query_fields for result in results[:3])
