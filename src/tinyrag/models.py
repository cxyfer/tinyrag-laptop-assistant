from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpecRecord:
    product: str
    variant: str
    field: str
    value: str
    aliases: tuple[str, ...]
    source_url: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecRecord:
        return cls(
            product=str(data["product"]),
            variant=str(data["variant"]),
            field=str(data["field"]),
            value=str(data["value"]),
            aliases=tuple(str(alias) for alias in data.get("aliases", ())),
            source_url=str(data["source_url"]),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    product: str
    variant: str
    field: str
    value: str
    aliases: tuple[str, ...]
    source_url: str
    notes: str = ""
    kind: str = "field"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aliases"] = list(self.aliases)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            product=str(data["product"]),
            variant=str(data["variant"]),
            field=str(data["field"]),
            value=str(data["value"]),
            aliases=tuple(str(alias) for alias in data.get("aliases", ())),
            source_url=str(data["source_url"]),
            notes=str(data.get("notes", "")),
            kind=str(data.get("kind", "field")),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float
    boost_score: float

    def to_dict(self) -> dict[str, Any]:
        data = self.chunk.to_dict()
        data.update(
            {
                "score": self.score,
                "dense_score": self.dense_score,
                "boost_score": self.boost_score,
            }
        )
        return data


@dataclass
class GenerationMetrics:
    ttft_seconds: float | None = None
    tps: float = 0.0
    generated_tokens: int = 0
    total_seconds: float = 0.0
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    expected_fields: tuple[str, ...] = field(default_factory=tuple)
    expected_variants: tuple[str, ...] = field(default_factory=tuple)
