from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_SOURCE_URL = "https://www.gigabyte.com/us/Laptop/AORUS-MASTER-16-AM6H/sp"
ALLOWED_SOURCE_HOSTS = frozenset({"www.gigabyte.com", "www.aorus.com"})
DEFAULT_CACHE_HTML = RAW_DIR / "am6h_spec.html"
DEFAULT_RECORDS_JSON = PROCESSED_DIR / "am6h_specs.json"
DEFAULT_VECTORS_PATH = INDEX_DIR / "vectors.npy"
DEFAULT_METADATA_PATH = INDEX_DIR / "metadata.json"
DEFAULT_BENCHMARK_PATH = BENCHMARK_DIR / "benchmark_results.json"
DEFAULT_SUMMARY_PATH = BENCHMARK_DIR / "benchmark_summary.md"

VARIANTS = ("BXH", "BYH", "BZH")
PRODUCT_NAME = "AORUS MASTER 16 AM6H"


@dataclass(frozen=True)
class Paths:
    cache_html: Path = DEFAULT_CACHE_HTML
    records_json: Path = DEFAULT_RECORDS_JSON
    vectors_path: Path = DEFAULT_VECTORS_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH


def ensure_data_dirs() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, BENCHMARK_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
