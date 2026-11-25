"""Lecture / écriture d'un index d'embeddings (sans dépendance à PyTorch).

Un index est un dossier contenant :
    embeddings.npy  matrice float32 (N, D), lignes normalisées L2
    index.csv       une ligne par embedding, même ordre (id, split, étiquettes…)
    meta.json       modèle, dimension, taille d'image, durée d'extraction
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_index(out_dir: Path, embeddings: np.ndarray, index: pd.DataFrame, meta: dict) -> None:
    if len(embeddings) != len(index):
        raise ValueError(f"{len(embeddings)} embeddings pour {len(index)} lignes d'index")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    index.to_csv(out_dir / "index.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(index_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "embeddings.npy")
    index = pd.read_csv(index_dir / "index.csv", dtype={"group": "string"})
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    if len(embeddings) != len(index):
        raise ValueError(f"Index incohérent dans {index_dir} : {len(embeddings)} embeddings, {len(index)} lignes")
    return embeddings, index, meta
