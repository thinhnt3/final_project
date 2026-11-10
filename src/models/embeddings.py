"""Frozen CLIP adapter with a deterministic hash fallback for smoke tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(t) for t in texts])

    def encode_images(self, paths: list[str | Path]) -> np.ndarray:
        return np.vstack([self._embed(Path(p).read_bytes()) for p in paths])

    def _embed(self, value: str | bytes) -> np.ndarray:
        raw = value.encode() if isinstance(value, str) else value
        out = np.zeros(self.dim, dtype=np.float32)
        for i in range(0, len(raw), 32):
            digest = hashlib.sha256(raw[i:i + 32]).digest()
            out[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = np.linalg.norm(out)
        return out / norm if norm else out


class FrozenCLIPEmbedder:
    """CLIP adapter. The model is loaded once and never put in train mode."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with self.torch.no_grad():
            vectors = self.model.get_text_features(**inputs)
        return self._normalize(vectors.detach().cpu().numpy())

    def encode_images(self, images: list[object]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        with self.torch.no_grad():
            vectors = self.model.get_image_features(**inputs)
        return self._normalize(vectors.detach().cpu().numpy())

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def cosine_search(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = matrix @ query
    order = np.argsort(-scores)[:k]
    return order, scores[order]
