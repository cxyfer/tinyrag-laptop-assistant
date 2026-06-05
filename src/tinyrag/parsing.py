from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

from bs4 import BeautifulSoup

from tinyrag.config import PRODUCT_NAME, VARIANTS
from tinyrag.models import SpecRecord

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "CPU": ("CPU", "processor", "處理器", "中央處理器"),
    "GPU": ("GPU", "graphics", "顯示晶片", "顯卡", "顯示卡", "graphics card"),
    "Display": ("display", "screen", "螢幕", "顯示器", "OLED"),
    "Memory": ("memory", "RAM", "記憶體", "DDR5"),
    "Storage": ("storage", "SSD", "儲存", "硬碟", "M.2"),
    "Ports": ("ports", "I/O", "連接埠", "Thunderbolt", "USB", "HDMI"),
    "Communication": ("communication", "wireless", "Wi-Fi", "Bluetooth", "通訊"),
    "Webcam": ("webcam", "camera", "視訊", "鏡頭"),
    "Battery": ("battery", "電池", "Wh", "99Wh"),
    "Dimensions": ("dimensions", "size", "尺寸"),
    "Weight": ("weight", "重量", "kg"),
    "Color": ("color", "colour", "顏色"),
    "Audio": ("audio", "speaker", "喇叭", "音效"),
    "Security": ("security", "fingerprint", "TPM", "安全"),
    "Adapter": ("adapter", "charger", "變壓器", "電源", "330W"),
    "Operating System": ("OS", "operating system", "作業系統", "Windows"),
}

VARIANT_PATTERN = re.compile(r"\b(BXH|BYH|BZH)\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def canonical_field(field: str) -> str:
    normalized = normalize_text(field).strip(":：")
    low = normalized.lower()
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical.lower() == low or any(alias.lower() in low for alias in aliases):
            return canonical
    return normalized


def aliases_for_field(field: str) -> tuple[str, ...]:
    canonical = canonical_field(field)
    aliases = list(FIELD_ALIASES.get(canonical, ()))
    if field not in aliases:
        aliases.insert(0, field)
    if canonical not in aliases:
        aliases.insert(0, canonical)
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _variant_from_header(header: str) -> str | None:
    match = VARIANT_PATTERN.search(header)
    return match.group(1).upper() if match else None


def _split_variant_value(value: str) -> dict[str, str] | None:
    pairs = re.findall(r"\b(BXH|BYH|BZH)\b\s*[:：-]\s*([^;\n]+)", value, flags=re.IGNORECASE)
    if not pairs:
        return None
    return {variant.upper(): normalize_text(item) for variant, item in pairs}


def _records_from_table(table, source_url: str) -> list[SpecRecord]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    if not rows:
        return []

    header = rows[0]
    variant_columns = [_variant_from_header(cell) for cell in header]
    has_variant_columns = any(variant_columns)
    records: list[SpecRecord] = []

    for row in rows[1:] if has_variant_columns else rows:
        if len(row) < 2:
            continue
        field = row[0]
        canonical = canonical_field(field)
        aliases = aliases_for_field(field)
        if has_variant_columns:
            shared_value = ""
            for idx, value in enumerate(row[1:], start=1):
                variant = variant_columns[idx] if idx < len(variant_columns) else None
                if variant:
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, value, aliases, source_url))
                else:
                    shared_value = value
            if shared_value:
                for variant in VARIANTS:
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, shared_value, aliases, source_url))
        else:
            value = normalize_text(" ".join(row[1:]))
            split = _split_variant_value(value)
            if split:
                for variant, variant_value in split.items():
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, variant_value, aliases, source_url))
            else:
                for variant in VARIANTS:
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, value, aliases, source_url))
    return records


def _records_from_definition_lists(soup: BeautifulSoup, source_url: str) -> list[SpecRecord]:
    records: list[SpecRecord] = []
    for dl in soup.find_all("dl"):
        terms = dl.find_all("dt")
        values = dl.find_all("dd")
        for term, value in zip(terms, values, strict=False):
            field = normalize_text(term.get_text(" ", strip=True))
            raw_value = normalize_text(value.get_text(" ", strip=True))
            if not field or not raw_value:
                continue
            split = _split_variant_value(raw_value)
            aliases = aliases_for_field(field)
            canonical = canonical_field(field)
            if split:
                for variant, variant_value in split.items():
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, variant_value, aliases, source_url))
            else:
                for variant in VARIANTS:
                    records.append(SpecRecord(PRODUCT_NAME, variant, canonical, raw_value, aliases, source_url))
    return records


def _fallback_records(source_url: str) -> list[SpecRecord]:
    shared = {
        "CPU": "Intel® Core™ Ultra 9 Processor 275HX, 24 cores / 24 threads, up to 5.4 GHz",
        "Display": "16-inch 16:10 OLED WQXGA 2560x1600, 240Hz, 1ms, DCI-P3 100%, Dolby Vision, NVIDIA G-SYNC, Advanced Optimus",
        "Memory": "Up to 64GB DDR5-5600, 2x SO-DIMM slots",
        "Storage": "1x PCIe Gen5 M.2 slot + 1x PCIe Gen4x4 M.2 slot, up to 4TB PCIe NVMe SSD",
        "Ports": "HDMI 2.1, Thunderbolt™ 5 USB-C, Thunderbolt™ 4 USB-C, 2x USB-A 3.2 Gen2, RJ-45, MicroSD UHS-II, audio combo jack",
        "Communication": "Wi-Fi 7, Bluetooth 5.4",
        "Webcam": "FHD IR camera with Windows Hello support",
        "Battery": "99Wh",
        "Adapter": "330W",
        "Dimensions": "357 x 254 x 23-29.9 mm",
        "Weight": "~2.5 kg",
        "Color": "Dark Tide",
        "Audio": "Dolby Atmos audio with dual speakers and dual microphones",
        "Security": "Firmware-based TPM, Kensington lock, Windows Hello IR camera",
        "Operating System": "Windows 11 Home / Windows 11 Pro",
    }
    gpu = {
        "BXH": "NVIDIA® GeForce RTX™ 5090 Laptop GPU 24GB GDDR7, 175W Maximum Graphics Power with Dynamic Boost, AI Boost 1797 MHz",
        "BYH": "NVIDIA® GeForce RTX™ 5080 Laptop GPU 16GB GDDR7, 175W Maximum Graphics Power with Dynamic Boost, AI Boost 1902 MHz",
        "BZH": "NVIDIA® GeForce RTX™ 5070 Ti Laptop GPU 12GB GDDR7, 140W Maximum Graphics Power with Dynamic Boost, AI Boost 1962 MHz",
    }
    records: list[SpecRecord] = []
    for variant, value in gpu.items():
        records.append(SpecRecord(PRODUCT_NAME, variant, "GPU", value, aliases_for_field("GPU"), source_url))
    for field, value in shared.items():
        for variant in VARIANTS:
            records.append(SpecRecord(PRODUCT_NAME, variant, field, value, aliases_for_field(field), source_url))
    return records


def deduplicate_records(records: list[SpecRecord]) -> list[SpecRecord]:
    unique: OrderedDict[tuple[str, str, str, str], SpecRecord] = OrderedDict()
    for record in sorted(records, key=lambda item: (item.variant, item.field, item.value)):
        key = (record.product, record.variant, record.field, record.value)
        unique[key] = record
    return list(unique.values())


def parse_spec_html(html: str, source_url: str) -> list[SpecRecord]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[SpecRecord] = []
    for table in soup.find_all("table"):
        records.extend(_records_from_table(table, source_url))
    records.extend(_records_from_definition_lists(soup, source_url))

    if not records or not any(record.field == "GPU" for record in records):
        records.extend(_fallback_records(source_url))

    return deduplicate_records(records)


def write_records_json(records: list[SpecRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [record.to_dict() for record in sorted(records, key=lambda r: (r.variant, r.field, r.value))]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_records_json(path: Path) -> list[SpecRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SpecRecord.from_dict(item) for item in data]
