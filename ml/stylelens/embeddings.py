"""P2 — Extraction des embeddings avec un CNN pré-entraîné sur ImageNet.

La couche de classification est retirée : l'embedding est la sortie de
l'avant-dernière couche (après le pooling global), normalisée L2.

Entrée : ``catalog.csv`` produit par P1 (``stylelens.data``).
Sortie (``--out``, un dossier par modèle) :
    embeddings.npy  matrice float32 (N, D), lignes normalisées L2
    index.csv       une ligne par embedding, même ordre (id, split, étiquettes…)
    meta.json       modèle, dimension, taille d'image, durée d'extraction

Usage (depuis ml/) :
    python -m stylelens.embeddings --catalog data/processed/catalog.csv \\
        --data-dir data/fashion-product-images-small --model resnet50 --out data/embeddings/resnet50
    # modèle fine-tuné (P3)
    python -m stylelens.embeddings --checkpoint models/resnet50-ft/model.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from stylelens.embeddings_io import save_index

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INDEX_COLUMNS = [
    "id", "image_path", "split", "group", "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "productDisplayName",
]


@dataclass(frozen=True)
class Backbone:
    builder: Callable[..., nn.Module]
    weights: models.WeightsEnum
    head: str  # attribut remplacé par nn.Identity
    dim: int


BACKBONES: dict[str, Backbone] = {
    "resnet50": Backbone(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc", 2048),
    "efficientnet_b0": Backbone(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier", 1280),
    "mobilenet_v3_large": Backbone(models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier", 960),
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """CNN sans couche de classification, en mode évaluation."""
    if name not in BACKBONES:
        raise ValueError(f"Modèle inconnu {name!r}, choix : {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    model = spec.builder(weights=spec.weights if pretrained else None)
    setattr(model, spec.head, nn.Identity())
    return model.eval()


class EmbeddingNet(nn.Module):
    """Backbone CNN + projection optionnelle, sortie normalisée L2.

    Sans projection (``embed_dim=None``), l'embedding est la sortie du pooling
    global du backbone : c'est le modèle de base P2. Avec projection, une couche
    linéaire suivie d'une normalisation par lot réduit la dimension (P3).
    """

    def __init__(self, backbone: str, embed_dim: int | None = None, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = build_model(backbone, pretrained=pretrained)
        in_dim = BACKBONES[backbone].dim
        self.embed_dim = embed_dim or in_dim
        self.head = (
            nn.Sequential(nn.Linear(in_dim, embed_dim), nn.BatchNorm1d(embed_dim))
            if embed_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x).flatten(1)), dim=1)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[EmbeddingNet, dict]:
    """Recharge un modèle fine-tuné enregistré par ``stylelens.train``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = EmbeddingNet(ckpt["backbone"], ckpt["embed_dim"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def load_embedding_model(
    model_name: str | None = None,
    checkpoint: Path | None = None,
    pretrained: bool = True,
) -> tuple[nn.Module, dict]:
    """Modèle d'extraction : fine-tuné si ``checkpoint`` est donné, sinon pré-entraîné ImageNet."""
    if checkpoint:
        model, ckpt = load_checkpoint(checkpoint)
        if model_name and model_name != ckpt["backbone"]:
            raise ValueError(f"--model {model_name} différent du backbone du checkpoint ({ckpt['backbone']})")
        info = {
            "model": ckpt["backbone"],
            "fine_tuned": True,
            "checkpoint": str(checkpoint),
            "image_size": ckpt["image_size"],
            "weights": f"fine-tuné (époque {ckpt['epoch']})",
        }
        return model, info
    model_name = model_name or "resnet50"
    info = {
        "model": model_name,
        "fine_tuned": False,
        "checkpoint": None,
        "weights": str(BACKBONES[model_name].weights) if pretrained else None,
    }
    return build_model(model_name, pretrained=pretrained), info


class PadToSquare:
    """Complète l'image en carré avec un fond blanc, sans la recadrer.

    Les photos du catalogue sont en portrait (60×80) sur fond blanc : le
    ``CenterCrop`` des transformations ImageNet couperait le haut et le bas de
    l'article (col, semelle). On garde l'article entier.
    """

    def __init__(self, fill: tuple[int, int, int] = (255, 255, 255)):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        if w == h:
            return img
        return ImageOps.pad(img, (side, side), color=self.fill, centering=(0.5, 0.5))


def make_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageListDataset(Dataset):
    def __init__(self, paths: list[Path], transform: Callable):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def extract(
    model: nn.Module,
    paths: list[Path],
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Embeddings normalisés L2, une ligne par image, dans l'ordre de ``paths``."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.eval().to(device)
    loader = DataLoader(
        ImageListDataset(paths, make_transform(image_size)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    n_batches = len(loader)
    for b, batch in enumerate(loader, 1):
        out = model(batch.to(device, non_blocking=True))
        chunks.append(F.normalize(out.flatten(1).float(), dim=1).cpu().numpy())
        if progress and (b % 20 == 0 or b == n_batches):
            print(f"  lot {b}/{n_batches}", flush=True)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def embed_images(
    paths: list[Path],
    model_name: str,
    device: str | None = None,
    image_size: int = 224,
    checkpoint: Path | None = None,
) -> np.ndarray:
    """Raccourci pour quelques images (requête de recherche)."""
    model, _ = load_embedding_model(model_name, checkpoint)
    return extract(model, paths, image_size=image_size, device=device)

# --------------------------------------------------------------------------- pipeline

def build_index(
    catalog_csv: Path,
    data_dir: Path,
    out_dir: Path,
    model_name: str | None = "resnet50",
    image_size: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | None = None,
    limit: int | None = None,
    pretrained: bool = True,
    progress: bool = True,
    checkpoint: Path | None = None,
) -> dict:
    """Calcule les embeddings de tout le catalogue P1 et les enregistre."""
    catalog = pd.read_csv(catalog_csv, dtype={"group": "string"})
    if limit and limit < len(catalog):
        catalog = (
            catalog.groupby("split").sample(frac=limit / len(catalog), random_state=0)
            .sort_values("id").reset_index(drop=True)
        )
    paths = [Path(data_dir) / p for p in catalog["image_path"]]

    dev = resolve_device(device)
    model, info = load_embedding_model(model_name, checkpoint, pretrained)
    image_size = image_size or info.pop("image_size", None) or 224
    info.pop("image_size", None)
    t0 = time.perf_counter()
    embeddings = extract(model, paths, image_size, batch_size, num_workers, dev, progress)
    elapsed = time.perf_counter() - t0

    meta = {
        **info,
        "pretrained": pretrained,
        "dim": int(embeddings.shape[1]),
        "n_images": int(len(embeddings)),
        "image_size": image_size,
        "preprocessing": "pad-to-square (blanc) + resize + normalisation ImageNet",
        "normalized": "l2",
        "device": str(dev),
        "extraction_seconds": round(elapsed, 2),
        "ms_per_image": round(1000 * elapsed / max(len(embeddings), 1), 3),
        "catalog": str(catalog_csv),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_index(out_dir, embeddings, catalog[[c for c in INDEX_COLUMNS if c in catalog.columns]], meta)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/fashion-product-images-small"))
    parser.add_argument("--model", choices=sorted(BACKBONES), help="défaut : resnet50, ou le backbone du checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="modèle fine-tuné (stylelens.train), ex. models/resnet50-ft/model.pt")
    parser.add_argument("--out", type=Path, help="défaut : data/embeddings/<model>, ou <dossier du checkpoint>")
    parser.add_argument("--image-size", type=int, help="défaut : 224, ou la taille d'entraînement du checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, mps ou cpu (défaut : automatique)")
    parser.add_argument("--limit", type=int, help="sous-échantillon stratifié par split (essai rapide)")
    args = parser.parse_args(argv)

    if args.out:
        out = args.out
    elif args.checkpoint:
        out = Path("data/embeddings") / args.checkpoint.parent.name
    else:
        out = Path("data/embeddings") / (args.model or "resnet50")
    meta = build_index(
        args.catalog, args.data_dir, out, args.model if args.checkpoint else (args.model or "resnet50"),
        args.image_size, args.batch_size, args.num_workers, args.device, args.limit,
        checkpoint=args.checkpoint,
    )
    kind = "fine-tuné" if meta["fine_tuned"] else "ImageNet"
    print(f"{meta['n_images']} embeddings de dimension {meta['dim']} ({meta['model']} {kind}, {meta['device']})")
    print(f"Durée : {meta['extraction_seconds']} s, soit {meta['ms_per_image']} ms/image")
    print(f"Écrit dans {out}")


if __name__ == "__main__":
    main()
