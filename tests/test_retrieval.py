from __future__ import annotations

from pathlib import Path

from tinyrag.chunking import build_chunks
from tinyrag.indexing import build_vector_index
from tinyrag.parsing import parse_spec_html
from tinyrag.retrieval import search

FIXTURE_HTML = Path("/home/usaya/workspace/interview/tinyrag-laptop-assistant/data/raw/am6h_spec.html").read_text(encoding="utf-8")


def _index():
    return build_vector_index(parse_spec_html(FIXTURE_HTML, "https://example.test/spec"))


def test_builds_field_and_comparison_chunks() -> None:
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    chunks = build_chunks(records)

    assert any(chunk.kind == "field" and chunk.field == "Battery" for chunk in chunks)
    comparison = [chunk for chunk in chunks if chunk.kind == "comparison" and chunk.field == "GPU"]
    assert comparison
    assert "BXH" in comparison[0].value and "BZH" in comparison[0].value


def test_retrieves_gpu_variants() -> None:
    index = _index()

    bxh = search(index, "BXH GPU", top_k=3)
    byh = search(index, "BYH 顯示卡", top_k=3)
    bzh = search(index, "BZH RTX 5070 Ti", top_k=3)

    assert any(item.chunk.variant == "BXH" and "5090" in item.chunk.value for item in bxh)
    assert any(item.chunk.variant == "BYH" and "5080" in item.chunk.value for item in byh)
    assert any(item.chunk.variant == "BZH" and "5070 Ti" in item.chunk.value for item in bzh)


def test_retrieves_core_specs_in_bilingual_queries() -> None:
    index = _index()
    cases = [
        ("battery capacity 99Wh", "Battery"),
        ("有哪些 Thunderbolt 連接埠", "Ports"),
        ("display OLED 240Hz", "Display"),
        ("CPU 處理器", "CPU"),
        ("storage M.2 SSD", "Storage"),
        ("Wi-Fi Bluetooth 通訊", "Communication"),
    ]

    for query, field in cases:
        results = search(index, query, top_k=3)
        assert any(item.chunk.field == field for item in results), query


def test_index_persistence_roundtrip(tmp_path: Path) -> None:
    index = _index()
    vectors = tmp_path / "vectors.npy"
    metadata = tmp_path / "metadata.json"

    index.save(vectors, metadata)
    from tinyrag.indexing import VectorIndex

    loaded = VectorIndex.load(vectors, metadata)

    assert loaded.vectors.shape == index.vectors.shape
    assert loaded.chunks[0].id == index.chunks[0].id
