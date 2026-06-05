from __future__ import annotations

from collections import defaultdict

from tinyrag.models import Chunk, SpecRecord


def _field_chunk_text(record: SpecRecord) -> str:
    aliases = ", ".join(record.aliases)
    notes = f" Notes: {record.notes}" if record.notes else ""
    return (
        f"Product: {record.product}\n"
        f"Variant: {record.variant}\n"
        f"Field: {record.field}\n"
        f"Aliases: {aliases}\n"
        f"Official value: {record.value}\n"
        f"Source: {record.source_url}{notes}"
    )


def build_field_chunks(records: list[SpecRecord]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, record in enumerate(sorted(records, key=lambda r: (r.variant, r.field, r.value))):
        chunks.append(
            Chunk(
                id=f"field-{index:04d}-{record.variant}-{record.field.lower().replace(' ', '-')}",
                text=_field_chunk_text(record),
                product=record.product,
                variant=record.variant,
                field=record.field,
                value=record.value,
                aliases=record.aliases,
                source_url=record.source_url,
                notes=record.notes,
                kind="field",
            )
        )
    return chunks


def build_comparison_chunks(records: list[SpecRecord]) -> list[Chunk]:
    by_field: dict[str, list[SpecRecord]] = defaultdict(list)
    for record in records:
        by_field[record.field].append(record)

    chunks: list[Chunk] = []
    for field, field_records in sorted(by_field.items()):
        values = {record.variant: record.value for record in field_records}
        if len(set(values.values())) <= 1 or len(values) <= 1:
            continue
        ordered = sorted(values.items())
        value = "; ".join(f"{variant}: {item}" for variant, item in ordered)
        aliases = field_records[0].aliases
        source_url = field_records[0].source_url
        text = (
            f"Product: {field_records[0].product}\n"
            f"Comparison field: {field}\n"
            f"Variant values: {value}\n"
            f"Aliases: {', '.join(aliases)}\n"
            f"Source: {source_url}"
        )
        chunks.append(
            Chunk(
                id=f"compare-{field.lower().replace(' ', '-')}",
                text=text,
                product=field_records[0].product,
                variant="ALL",
                field=field,
                value=value,
                aliases=aliases,
                source_url=source_url,
                kind="comparison",
            )
        )
    return chunks


def build_chunks(records: list[SpecRecord]) -> list[Chunk]:
    return build_field_chunks(records) + build_comparison_chunks(records)
