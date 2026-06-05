from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]|[぀-ヿ]|[가-힯]")

SYNONYMS = {
    "顯卡": "gpu graphics 顯示卡 顯示晶片",
    "顯示卡": "gpu graphics 顯卡 顯示晶片",
    "顯示晶片": "gpu graphics 顯卡 顯示卡",
    "處理器": "cpu processor 中央處理器",
    "螢幕": "display screen oled 顯示器",
    "電池": "battery 99wh wh",
    "連接埠": "ports usb thunderbolt hdmi",
    "記憶體": "memory ram ddr5",
    "儲存": "storage ssd m2",
    "通訊": "communication wifi bluetooth",
    "視訊": "webcam camera",
    "尺寸": "dimensions size",
    "重量": "weight kg",
}


@dataclass(frozen=True)
class HashEmbeddingModel:
    dimensions: int = 384

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[slot] += sign
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)

    def _tokens(self, text: str) -> list[str]:
        expanded = text.lower()
        for key, value in SYNONYMS.items():
            if key in expanded:
                expanded += " " + value
        tokens = TOKEN_RE.findall(expanded)
        grams = tokens[:]
        grams.extend(" ".join(tokens[index : index + 2]) for index in range(max(0, len(tokens) - 1)))
        grams.extend(" ".join(tokens[index : index + 3]) for index in range(max(0, len(tokens) - 2)))
        return grams
