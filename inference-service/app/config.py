"""Configuration du service, lue dans les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    # Modèle fine-tuné (stylelens.train) ; sinon backbone ImageNet ``model_name``.
    model_checkpoint: Path | None = None
    model_name: str = "resnet50"
    model_pretrained: bool = True
    image_size: int | None = None  # défaut : taille d'entraînement du checkpoint, ou 224
    device: str | None = None  # cuda, mps, cpu (défaut : automatique)
    torch_threads: int | None = None
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        checkpoint = env.get("MODEL_CHECKPOINT") or None
        return cls(
            model_checkpoint=Path(checkpoint) if checkpoint else None,
            model_name=env.get("MODEL_NAME") or "resnet50",
            model_pretrained=_bool(env.get("MODEL_PRETRAINED"), True),
            image_size=_int(env.get("IMAGE_SIZE")),
            device=env.get("DEVICE") or None,
            torch_threads=_int(env.get("TORCH_THREADS")),
            max_upload_bytes=_int(env.get("MAX_UPLOAD_BYTES")) or cls.max_upload_bytes,
            max_image_pixels=_int(env.get("MAX_IMAGE_PIXELS")) or cls.max_image_pixels,
        )
