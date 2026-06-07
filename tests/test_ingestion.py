from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from tinyrag.ingestion import SourceLoadError, load_source_html
from tinyrag.parsing import aliases_for_field, parse_spec_html, read_records_json, write_records_json

FIXTURE_HTML = (Path(__file__).resolve().parents[1] / "data/raw/am6h_spec.html").read_text(encoding="utf-8")


def test_parse_variant_gpu_records_preserves_values() -> None:
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    gpu = {record.variant: record for record in records if record.field == "GPU"}

    assert set(gpu) == {"BXH", "BYH", "BZH"}
    assert "RTX™ 5090" in gpu["BXH"].value
    assert "24GB GDDR7" in gpu["BXH"].value
    assert "RTX™ 5080" in gpu["BYH"].value
    assert "16GB GDDR7" in gpu["BYH"].value
    assert "RTX™ 5070 Ti" in gpu["BZH"].value
    assert "12GB GDDR7" in gpu["BZH"].value


def test_parse_shared_fields_and_aliases() -> None:
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    batteries = [record for record in records if record.field == "Battery"]
    ports = [record for record in records if record.field == "Ports"]

    assert len(batteries) == 3
    assert {record.value for record in batteries} == {"99Wh"}
    assert all("battery" in {alias.lower() for alias in record.aliases} for record in batteries)
    assert any("Thunderbolt™ 5" in record.value for record in ports)
    assert "GPU" in aliases_for_field("顯示晶片")


def test_cached_fallback_when_remote_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache.html"
    cache.write_text(FIXTURE_HTML, encoding="utf-8")

    def fail_get(*args, **kwargs):
        raise RuntimeError("wrong exception should not be used")

    response_get = Mock()
    response_get.side_effect = __import__("requests").RequestException("offline")
    monkeypatch.setattr("tinyrag.ingestion.requests.get", response_get)

    html, source = load_source_html("https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp", cache, timeout=0.01)

    assert "AORUS MASTER 16 AM6H" in html
    assert source == str(cache)


def test_rejects_unapproved_remote_source_url() -> None:
    with pytest.raises(SourceLoadError):
        load_source_html("https://example.test/spec", timeout=0.01)


def test_prefer_cache_still_rejects_unapproved_source_url(tmp_path: Path) -> None:
    cache = tmp_path / "cache.html"
    cache.write_text(FIXTURE_HTML, encoding="utf-8")

    with pytest.raises(SourceLoadError):
        load_source_html("http://example.test/spec", cache, timeout=0.01, prefer_cache=True)


def test_remote_fetch_disables_redirect_following(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "missing.html"
    response = Mock()
    response.text = FIXTURE_HTML
    response.raise_for_status.return_value = None
    response_get = Mock(return_value=response)
    monkeypatch.setattr("tinyrag.ingestion.requests.get", response_get)

    html, source = load_source_html(
        "https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp",
        cache,
        timeout=0.01,
    )

    assert "AORUS MASTER 16 AM6H" in html
    assert source == "https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp"
    assert response_get.call_args.kwargs["allow_redirects"] is False


def test_deterministic_json_roundtrip(tmp_path: Path) -> None:
    records = parse_spec_html(FIXTURE_HTML, "https://example.test/spec")
    output = tmp_path / "records.json"

    write_records_json(records, output)
    first = output.read_text(encoding="utf-8")
    write_records_json(read_records_json(output), output)
    second = output.read_text(encoding="utf-8")

    assert first == second
