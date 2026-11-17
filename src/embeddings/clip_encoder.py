"""Frozen CLIP encoder for images and text. The model is loaded once and never trained."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.embeddings.device import Device


def resolve_device(device: Device) -> str:
    """Concrete torch device name; `auto` prefers cuda, then mps, then cpu."""
    if device is not Device.AUTO:
        return device.value
    if torch.cuda.is_available():
        return Device.CUDA.value
    if torch.backends.mps.is_available():
        return Device.MPS.value
    return Device.CPU.value


class ClipEncoder:
    """Wraps a Hugging Face CLIP checkpoint; returns raw (not normalized) projection vectors."""

    def __init__(self, model_name: str, device: Device) -> None:
        self.device = resolve_device(device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self._model.requires_grad_(False)

    @property
    def dim(self) -> int:
        """Size of the shared image/text embedding space."""
        return int(self._model.config.projection_dim)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts; long texts are truncated to the model's context length."""
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            vectors = self._model.get_text_features(**inputs.to(self.device))
        return vectors.float().cpu().numpy()

    def encode_image_files(self, paths: Sequence[Path]) -> np.ndarray:
        """Embed image files, converting each to RGB first."""
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            vectors = self._model.get_image_features(**inputs.to(self.device))
        return vectors.float().cpu().numpy()
