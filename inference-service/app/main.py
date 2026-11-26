"""Service IA StyleLens : image → embedding (P5).

Lancement (depuis inference-service/) :
    MODEL_CHECKPOINT=../ml/models/resnet50-ft/model.pt uvicorn --factory app.main:create_app --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.config import Settings
from app.images import InvalidImageError, decode_image
from app.model import EmbeddingModel

log = logging.getLogger("uvicorn.error")
ACCEPTED_TYPES = ("image/", "application/octet-stream")


class Health(BaseModel):
    status: str
    model: str
    fine_tuned: bool
    dim: int
    image_size: int
    device: str


class Embedding(BaseModel):
    embedding: list[float]
    dim: int
    model: str
    fine_tuned: bool
    elapsed_ms: float


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = EmbeddingModel(settings)
        info = app.state.model.info
        log.info("Modèle %s (fine-tuné : %s, %d-d, %s) chargé en %.1f s",
                 info.model, info.fine_tuned, info.dim, info.device, info.load_seconds)
        yield

    app = FastAPI(
        title="StyleLens — service IA",
        version="0.1.0",
        description="Calcule l'embedding visuel (CNN, normalisé L2) d'une image d'article.",
        lifespan=lifespan,
    )

    def model_of(request: Request) -> EmbeddingModel:
        return request.app.state.model

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        info = model_of(request).info
        return Health(status="UP", model=info.model, fine_tuned=info.fine_tuned, dim=info.dim,
                      image_size=info.image_size, device=info.device)

    @app.post("/embed", response_model=Embedding,
              responses={400: {"description": "Image illisible"}, 413: {"description": "Fichier trop lourd"},
                         415: {"description": "Type de fichier non accepté"}})
    async def embed(request: Request, file: UploadFile = File(..., description="image JPEG, PNG ou WebP")) -> Embedding:
        if file.content_type and not file.content_type.startswith(ACCEPTED_TYPES):
            raise HTTPException(415, f"Type de fichier non accepté : {file.content_type}")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"Fichier trop lourd (maximum {settings.max_upload_bytes // (1024 * 1024)} Mo)")

        model = model_of(request)
        t0 = time.perf_counter()
        try:
            image = await run_in_threadpool(decode_image, data, settings.max_image_pixels)
        except InvalidImageError as e:
            raise HTTPException(400, str(e)) from e
        vector = await run_in_threadpool(model.embed, image)
        info = model.info
        return Embedding(embedding=vector.tolist(), dim=len(vector), model=info.model,
                         fine_tuned=info.fine_tuned, elapsed_ms=round(1000 * (time.perf_counter() - t0), 2))

    return app

