"""Modèle d'embedding : celui de ml/stylelens (même prétraitement que le catalogue)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from app.config import Settings
from stylelens.embeddings import load_embedding_model, make_transform, resolve_device


@dataclass(frozen=True)
class ModelInfo:
    model: str
    fine_tuned: bool
    dim: int
    image_size: int
    device: str
    checkpoint: str | None
    load_seconds: float


class EmbeddingModel:
    """CNN chargé une fois au démarrage ; un appel = une image → vecteur normalisé L2.

    Les appels sont sérialisés : sur CPU, PyTorch parallélise déjà chaque
    inférence sur plusieurs cœurs, et des appels simultanés se disputeraient
    les mêmes cœurs.
    """

    def __init__(self, settings: Settings):
        t0 = time.perf_counter()
        if settings.torch_threads:
            torch.set_num_threads(settings.torch_threads)
        if settings.model_checkpoint and not settings.model_checkpoint.is_file():
            raise FileNotFoundError(f"MODEL_CHECKPOINT introuvable : {settings.model_checkpoint}")
        model, info = load_embedding_model(
            None if settings.model_checkpoint else settings.model_name,
            settings.model_checkpoint,
            pretrained=settings.model_pretrained,
        )
        self.device = resolve_device(settings.device)
        self.model = model.eval().to(self.device)
        self.image_size = settings.image_size or info.get("image_size") or 224
        self.transform = make_transform(self.image_size)
        self._lock = threading.Lock()

        dim = len(self.embed(Image.new("RGB", (self.image_size, self.image_size), "white")))  # échauffement
        self.info = ModelInfo(
            model=info["model"],
            fine_tuned=bool(info["fine_tuned"]),
            dim=dim,
            image_size=self.image_size,
            device=str(self.device),
            checkpoint=info.get("checkpoint"),
            load_seconds=round(time.perf_counter() - t0, 2),
        )

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> np.ndarray:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._lock:
            out = self.model(x)
        return F.normalize(out.flatten(1).float(), dim=1)[0].cpu().numpy()
