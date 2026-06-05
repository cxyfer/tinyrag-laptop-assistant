from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests

from tinyrag.config import ALLOWED_SOURCE_HOSTS, DEFAULT_CACHE_HTML, DEFAULT_SOURCE_URL, ensure_data_dirs
from tinyrag.parsing import parse_spec_html


class SourceLoadError(RuntimeError):
    pass


def validate_source_url(source_url: str) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
        allowed = ", ".join(sorted(ALLOWED_SOURCE_HOSTS))
        raise SourceLoadError(f"Source URL must use HTTPS and one of these hosts: {allowed}")


def load_source_html(
    source_url: str = DEFAULT_SOURCE_URL,
    cache_path: Path = DEFAULT_CACHE_HTML,
    timeout: float = 10.0,
    prefer_cache: bool = False,
) -> tuple[str, str]:
    ensure_data_dirs()
    validate_source_url(source_url)

    if prefer_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), str(cache_path)

    try:
        response = requests.get(
            source_url,
            timeout=timeout,
            headers={"User-Agent": "TinyRAG/0.1"},
            allow_redirects=False,
        )
        response.raise_for_status()
        html = response.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
        return html, source_url
    except requests.RequestException as exc:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8"), str(cache_path)
        raise SourceLoadError(f"Unable to fetch {source_url} and no cache exists at {cache_path}") from exc


def ingest_specs(
    source_url: str = DEFAULT_SOURCE_URL,
    cache_path: Path = DEFAULT_CACHE_HTML,
    output_path: Path | None = None,
    prefer_cache: bool = False,
):
    html, actual_source = load_source_html(source_url, cache_path, prefer_cache=prefer_cache)
    records = parse_spec_html(html, actual_source if actual_source.startswith("http") else source_url)
    if output_path is not None:
        from tinyrag.parsing import write_records_json

        write_records_json(records, output_path)
    return records
