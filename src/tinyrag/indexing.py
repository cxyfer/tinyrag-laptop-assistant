from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyrag.chunking import build_chunks
from tinyrag.config import DEFAULT_METADATA_PATH, DEFAULT_VECTORS_PATH
from tinyrag.embeddings import HashEmbeddingModel
from tinyrag.models import Chunk, SpecRecord


@dataclass
class VectorIndex:
    vectors: np.ndarray
    chunks: list[Chunk]
    embedding_model: HashEmbeddingModel

    def save(
        self,
        vectors_path: Path = DEFAULT_VECTORS_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors.astype(np.float32))
        metadata_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        vectors_path: Path = DEFAULT_VECTORS_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        embedding_model: HashEmbeddingModel | None = None,
    ) -> VectorIndex:
        vectors = np.load(vectors_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            vectors=vectors.astype(np.float32),
            chunks=[Chunk.from_dict(item) for item in metadata],
            embedding_model=embedding_model or HashEmbeddingModel(vectors.shape[1]),
        )


def build_vector_index(
    records: list[SpecRecord],
    embedding_model: HashEmbeddingModel | None = None,
) -> VectorIndex:
    model = embedding_model or HashEmbeddingModel()
    chunks = build_chunks(records)
    vectors = model.embed_many([chunk.text for chunk in chunks])
    return VectorIndex(vectors=vectors, chunks=chunks, embedding_model=model)
