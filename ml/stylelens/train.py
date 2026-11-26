"""P3 — Transfert d'apprentissage : fine-tuning du CNN pour la recherche par similarité.

Le backbone pré-entraîné sur ImageNet est spécialisé sur le catalogue mode avec
une perte de classification **cosinus à marge** (CosFace) sur ``articleType`` :
les embeddings d'une même classe se resserrent autour d'un centre, ceux de
classes différentes s'écartent d'au moins la marge. C'est la géométrie que la
recherche kNN cosinus exploite ensuite.

Déroulement :
1. ``--freeze-epochs`` époques backbone gelé : seules la projection et la tête
   de classification apprennent (les poids ImageNet ne sont pas abîmés par des
   gradients aléatoires du début d'entraînement) ;
2. puis tout le réseau est entraîné, le backbone avec un taux d'apprentissage
   plus faible (``--backbone-lr-mult``) ;
3. à chaque époque, mAP@10 de recherche sur le split ``val`` (requêtes et
   galerie = val) ; le meilleur modèle est conservé, arrêt anticipé après
   ``--patience`` époques sans progrès.

Le split ``test`` n'est jamais utilisé ici : il reste réservé à
``stylelens.evaluate`` pour comparer modèle de base et modèle fine-tuné.

Usage (depuis ml/, GPU conseillé) :
    python -m stylelens.train --backbone resnet50 --out models/resnet50-ft
    python -m stylelens.embeddings --checkpoint models/resnet50-ft/model.pt
    python -m stylelens.evaluate --index data/embeddings/resnet50-ft
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from torchvision import transforms

from stylelens.embeddings import (
    BACKBONES, IMAGENET_MEAN, IMAGENET_STD, EmbeddingNet, PadToSquare, extract, resolve_device,
)
from stylelens.evaluate import evaluate_retrieval


@dataclass
class TrainConfig:
    backbone: str = "resnet50"
    embed_dim: int | None = 512
    image_size: int = 224
    epochs: int = 15
    freeze_epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    scale: float = 30.0
    margin: float = 0.2
    balance: str = "sqrt"  # none | sqrt | inverse
    label: str = "articleType"
    patience: int = 4
    seed: int = 42
    num_workers: int = 0
    steps_per_epoch: int | None = None
    device: str | None = None
    pretrained: bool = True
    amp: bool = True
    freeze_bn: bool = True


# --------------------------------------------------------------------------- modèle

class CosineMarginHead(nn.Module):
    """Classifieur cosinus avec marge additive (CosFace).

    logit_c = s · (cos θ_c − m · [c = y]) : la classe correcte doit l'emporter
    avec une marge m en similarité cosinus, ce qui sépare nettement les classes
    sur l'hypersphère. Sans étiquette (inférence), pas de marge.
    """

    def __init__(self, in_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.n_classes, self.scale, self.margin = n_classes, scale, margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.linear(F.normalize(emb, dim=1), F.normalize(self.weight, dim=1))
        if labels is not None and self.margin:
            cos = cos - self.margin * F.one_hot(labels, self.n_classes).to(cos.dtype)
        return self.scale * cos


def set_backbone_trainable(model: EmbeddingNet, trainable: bool, freeze_bn: bool = True) -> None:
    """Gèle ou dégèle le backbone (gelé : pas de gradient et BatchNorm figées).

    Avec ``freeze_bn``, les couches BatchNorm du backbone gardent leurs
    statistiques ImageNet même une fois dégelé (leurs poids affines restent
    entraînables). Sinon, au dégel, elles basculent brutalement sur les
    statistiques de petits lots d'images sur fond blanc, très différentes
    d'ImageNet : la perte remonte et une partie de l'acquis est perdue.
    """
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)
    model.backbone.train(trainable)
    if trainable and freeze_bn:
        for m in model.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


# --------------------------------------------------------------------------- données

def make_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmentations légères qui préservent la couleur (teinte inchangée).

    La couleur est un critère de ressemblance pour l'utilisateur : pas de
    variation de teinte, saturation à peine modifiée.
    """
    return transforms.Compose([
        PadToSquare(),
        transforms.RandomResizedCrop(
            image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LabeledImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform):
        self.paths, self.labels, self.transform = [Path(p) for p in paths], labels, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB")), int(self.labels[i])


def sample_weights(labels: np.ndarray, mode: str) -> np.ndarray | None:
    """Poids d'échantillonnage par image pour atténuer le déséquilibre des classes.

    ``sqrt`` : poids ∝ 1/√effectif (compromis : les classes rares sont vues plus
    souvent sans être sur-apprises) ; ``inverse`` : classes équiprobables.
    """
    if mode == "none":
        return None
    counts = np.bincount(labels)[labels].astype(np.float64)
    if mode == "sqrt":
        return 1.0 / np.sqrt(counts)
    if mode == "inverse":
        return 1.0 / counts
    raise ValueError(f"balance inconnu : {mode!r} (none, sqrt, inverse)")


def lr_lambdas(steps_per_epoch: int, epochs: int, freeze_epochs: int):
    """Multiplicateurs de LR par pas : (backbone, tête).

    Les deux suivent une décroissance cosinus jusqu'à 1 % du LR initial. Le
    backbone reste à 0 tant qu'il est gelé, puis remonte linéairement pendant
    une époque après le dégel : on évite le choc d'un LR plein sur des poids
    ImageNet alors que la tête vient à peine de converger.
    """
    total = max(1, steps_per_epoch * epochs)
    unfreeze = steps_per_epoch * freeze_epochs
    warmup = max(1, steps_per_epoch)

    def cosine(step: int) -> float:
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))

    def backbone(step: int) -> float:
        if step < unfreeze:
            return 0.0
        return cosine(step) * min(1.0, (step - unfreeze + 1) / warmup)

    return [backbone, cosine]


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- entraînement

def _validate(model, val_df, data_dir, cfg, device) -> dict:
    paths = [Path(data_dir) / p for p in val_df["image_path"]]
    emb = extract(model, paths, cfg.image_size, cfg.batch_size, cfg.num_workers, device)
    report = evaluate_retrieval(
        emb, val_df.reset_index(drop=True), query_splits=("val",), gallery_splits=("val",),
        label=cfg.label, ks=(1, 5, 10), per_class=False, latency_queries=0,
    )
    return report["overall"]


def train(
    catalog_csv: Path,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    model: EmbeddingNet | None = None,
    progress: bool = True,
) -> dict:
    """Fine-tune ``model`` (ou un nouveau backbone) et enregistre le meilleur checkpoint."""
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    log = print if progress else (lambda *a, **k: None)

    catalog = pd.read_csv(catalog_csv, dtype={"group": "string"})
    train_df = catalog[catalog["split"] == "train"].reset_index(drop=True)
    classes = sorted(train_df[cfg.label].unique())
    class_index = {c: i for i, c in enumerate(classes)}
    val_df = catalog[(catalog["split"] == "val") & catalog[cfg.label].isin(class_index)].reset_index(drop=True)
    if len(train_df) < 2:
        raise ValueError("Split train trop petit pour entraîner")

    labels = train_df[cfg.label].map(class_index).to_numpy()
    batch_size = min(cfg.batch_size, len(train_df))
    n_samples = cfg.steps_per_epoch * batch_size if cfg.steps_per_epoch else len(train_df)
    weights = sample_weights(labels, cfg.balance)
    sampler = (
        WeightedRandomSampler(weights, n_samples, replacement=True, generator=generator)
        if weights is not None else RandomSampler(range(len(train_df)), num_samples=n_samples, generator=generator)
    )
    loader = DataLoader(
        LabeledImageDataset([Path(data_dir) / p for p in train_df["image_path"]], labels,
                            make_train_transform(cfg.image_size)),
        batch_size=batch_size, sampler=sampler, num_workers=cfg.num_workers,
        drop_last=True, pin_memory=device.type == "cuda",
    )

    model = (model or EmbeddingNet(cfg.backbone, cfg.embed_dim, pretrained=cfg.pretrained)).to(device)
    head = CosineMarginHead(model.embed_dim, len(classes), cfg.scale, cfg.margin).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": [*model.head.parameters(), *head.parameters()], "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas(len(loader), cfg.epochs, cfg.freeze_epochs))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log(f"Fine-tuning {cfg.backbone} -> {model.embed_dim}-d sur {device} : "
        f"{len(train_df)} images train, {len(val_df)} val, {len(classes)} classes")

    history, best, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        frozen = epoch <= cfg.freeze_epochs
        if epoch == cfg.freeze_epochs + 1:
            stale = 0  # patience remise à zéro au dégel du backbone
        set_backbone_trainable(model, not frozen, cfg.freeze_bn)
        model.head.train()
        head.train()
        t0 = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                if frozen:
                    with torch.no_grad():
                        feats = model.backbone(x).flatten(1)
                    emb = F.normalize(model.head(feats.float()), dim=1)
                else:
                    emb = model(x)
                loss = F.cross_entropy(head(emb, y), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                correct += int((head(emb.detach().float()).argmax(1) == y).sum())
            loss_sum += loss.item() * len(y)
            seen += len(y)

        row = {
            "epoch": epoch,
            "phase": "backbone gelé" if frozen else "réseau complet",
            "loss": round(loss_sum / max(seen, 1), 4),
            "train_acc": round(correct / max(seen, 1), 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        if len(val_df):
            val = _validate(model, val_df, data_dir, cfg, device)
            row.update({f"val_{k}": round(v, 4) for k, v in val.items() if "@" in k})
            score = val.get("map@10", -1.0)
        else:
            score = -row["loss"]  # pas de validation : on garde l'époque de plus faible perte
        history.append(row)

        improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            _save_checkpoint(out_dir / "model.pt", model, cfg, classes, epoch, row)
        else:
            stale += 1
        log(f"  époque {epoch:>2}/{cfg.epochs} [{row['phase']}] perte {row['loss']:.4f} "
            f"acc {row['train_acc']:.3f}"
            + (f"  val mAP@10 {row['val_map@10']:.4f} P@1 {row['val_precision@1']:.4f}" if "val_map@10" in row else "")
            + ("  *" if improved else "") + f"  ({row['seconds']} s)")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        if not frozen and stale >= cfg.patience:
            log(f"  arrêt anticipé : pas de progrès depuis {cfg.patience} époques")
            break

    summary = {
        "checkpoint": str(out_dir / "model.pt"),
        "best_epoch": best_epoch,
        "best_val_map@10": round(best, 4) if len(val_df) else None,
        "epochs_run": len(history),
        "n_classes": len(classes),
        "config": _config_dict(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**summary, "history": history}


def _config_dict(cfg: TrainConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def _save_checkpoint(path: Path, model: EmbeddingNet, cfg: TrainConfig, classes, epoch: int, metrics: dict) -> None:
    torch.save({
        "backbone": model.backbone_name,
        "embed_dim": cfg.embed_dim,
        "image_size": cfg.image_size,
        "epoch": epoch,
        "classes": list(classes),
        "label": cfg.label,
        "metrics": metrics,
        "config": _config_dict(cfg),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainConfig()
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/fashion-product-images-small"))
    parser.add_argument("--out", type=Path, help="défaut : models/<backbone>-ft")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default=d.backbone)
    parser.add_argument("--embed-dim", type=int, default=d.embed_dim, help="0 = pas de projection")
    parser.add_argument("--image-size", type=int, default=d.image_size)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--freeze-epochs", type=int, default=d.freeze_epochs)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--backbone-lr-mult", type=float, default=d.backbone_lr_mult)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--scale", type=float, default=d.scale, help="facteur s de CosFace")
    parser.add_argument("--margin", type=float, default=d.margin, help="marge m de CosFace")
    parser.add_argument("--balance", choices=["none", "sqrt", "inverse"], default=d.balance)
    parser.add_argument("--label", default=d.label)
    parser.add_argument("--patience", type=int, default=d.patience)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--num-workers", type=int, default=d.num_workers)
    parser.add_argument("--steps-per-epoch", type=int, help="limite le nombre de lots par époque (essai rapide)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true", help="désactive la précision mixte sur GPU")
    parser.add_argument("--train-bn", action="store_true",
                        help="met à jour les statistiques BatchNorm du backbone (gelées par défaut)")
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        backbone=args.backbone, embed_dim=args.embed_dim or None, image_size=args.image_size,
        epochs=args.epochs, freeze_epochs=args.freeze_epochs, batch_size=args.batch_size, lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay, scale=args.scale,
        margin=args.margin, balance=args.balance, label=args.label, patience=args.patience,
        seed=args.seed, num_workers=args.num_workers, steps_per_epoch=args.steps_per_epoch,
        device=args.device, amp=not args.no_amp, freeze_bn=not args.train_bn,
    )
    out = args.out or Path("models") / f"{args.backbone}-ft"
    summary = train(args.catalog, args.data_dir, out, cfg)
    print(f"Meilleure époque : {summary['best_epoch']} (val mAP@10 = {summary['best_val_map@10']})")
    print(f"Checkpoint : {summary['checkpoint']}")
    print(f"Suite : python -m stylelens.embeddings --checkpoint {summary['checkpoint']}")


if __name__ == "__main__":
    main()
