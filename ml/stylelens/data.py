"""P1 — Préparation des données : lecture, nettoyage et séparation par produit.

Entrée : le jeu *Fashion Product Images (Small)* (styles.csv + images/<id>.jpg).
Sortie : un catalogue propre (catalog.csv) avec une colonne ``split``
(train / val / test) et un rapport chiffré (report.json).

Usage (depuis ml/) :
    python -m stylelens.data --data-dir data/fashion-product-images-small --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COLUMNS = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]
KEPT_MASTER_CATEGORIES = ("Apparel", "Footwear", "Accessories")
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


# --------------------------------------------------------------------------- lecture

def load_styles(csv_path: Path) -> pd.DataFrame:
    """Lit styles.csv sans perdre de lignes.

    Certains noms de produits contiennent des virgules non échappées :
    ``pd.read_csv(on_bad_lines='skip')`` les jette. Le nom étant la dernière
    colonne, on découpe chaque ligne sur les 9 premières virgules seulement.
    """
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().lstrip("﻿").split(",")
        if header != COLUMNS:
            raise ValueError(f"En-tête inattendu dans {csv_path} : {header}")
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(",", len(COLUMNS) - 1)
            parts += [""] * (len(COLUMNS) - len(parts))
            rows.append(parts)

    df = pd.DataFrame(rows, columns=COLUMNS, dtype="string")
    df = df.apply(lambda s: s.str.strip().replace("", pd.NA))
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------- images

def inspect_image(path: Path, min_side: int = 32) -> dict:
    """Vérifie qu'une image existe, se décode entièrement et n'est pas minuscule."""
    if not path.is_file():
        return {"status": "missing"}
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()  # décodage complet : détecte les fichiers tronqués
            width, height = img.size
    except Exception:
        return {"status": "corrupt"}
    if min(width, height) < min_side:
        return {"status": "too_small"}
    return {
        "status": "ok",
        "width": width,
        "height": height,
        "md5": hashlib.md5(data).hexdigest(),
    }


# --------------------------------------------------------------------------- nettoyage

def product_group(name, fallback_id) -> str:
    """Clé de regroupement : nom de produit normalisé.

    Les déclinaisons d'un même produit (même nom, couleur ou taille différente)
    partagent la même clé et iront donc dans le même split : aucune fuite
    d'un produit quasi identique entre train et test.
    """
    if name is None or pd.isna(name):
        return f"id:{fallback_id}"
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s or f"id:{fallback_id}"


def clean(
    df: pd.DataFrame,
    images_dir: Path,
    min_class_count: int = 20,
    min_side: int = 32,
) -> tuple[pd.DataFrame, dict]:
    """Applique les règles de nettoyage et renvoie (catalogue, compte rendu)."""
    steps: dict[str, int] = {"raw": len(df)}

    df = df.dropna(subset=["id", "masterCategory", "subCategory", "articleType"])
    df = df.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    steps["valid_metadata"] = len(df)

    df = df[df["masterCategory"].isin(KEPT_MASTER_CATEGORIES)].reset_index(drop=True)
    steps["kept_master_categories"] = len(df)

    checks = pd.DataFrame(
        [inspect_image(images_dir / f"{i}.jpg", min_side) for i in df["id"]],
        index=df.index,
    )
    image_issues = checks["status"].value_counts().drop("ok", errors="ignore")
    df = df.join(checks)
    df = df[df["status"] == "ok"].drop(columns="status").reset_index(drop=True)
    df[["width", "height"]] = df[["width", "height"]].astype(int)
    steps["valid_images"] = len(df)

    # Doublons exacts (mêmes octets sous deux id) : on garde le plus petit id.
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
    steps["unique_images"] = len(df)

    counts = df["articleType"].value_counts()
    rare = counts[counts < min_class_count]
    df = df[~df["articleType"].isin(rare.index)].reset_index(drop=True)
    steps["frequent_classes"] = len(df)

    df["image_path"] = "images/" + df["id"].astype(str) + ".jpg"
    df["group"] = [product_group(n, i) for n, i in zip(df["productDisplayName"], df["id"])]

    report = {
        "rows_after_each_step": steps,
        "image_issues": {k: int(v) for k, v in image_issues.items()},
        "dropped_rare_classes": {k: int(v) for k, v in rare.items()},
        "min_class_count": min_class_count,
    }
    return df, report


# --------------------------------------------------------------------------- séparation

def split_by_group(
    df: pd.DataFrame,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 42,
    stratify: str = "articleType",
    group: str = "group",
) -> pd.Series:
    """Séparation train/val/test stratifiée par classe et groupée par produit.

    Chaque groupe reçoit l'étiquette majoritaire de ses images. Pour chaque
    étiquette, les groupes (mélangés, puis du plus gros au plus petit) sont
    affectés au split le plus en retard sur sa proportion cible. Un groupe
    n'apparaît que dans un seul split.
    """
    if len(ratios) != len(SPLITS):
        raise ValueError(f"{len(SPLITS)} proportions attendues, reçu {ratios}")
    target = np.asarray(ratios, dtype=float) / sum(ratios)

    groups = df.groupby(group)[stratify].agg(size="size", label=lambda s: s.mode().iloc[0])
    groups = groups.iloc[np.random.default_rng(seed).permutation(len(groups))]

    assignment: dict[str, str] = {}
    for _, g in groups.groupby("label", sort=True):
        filled = np.zeros(len(SPLITS))
        total = g["size"].sum()
        for key, size in g["size"].sort_values(ascending=False, kind="stable").items():
            k = int(np.argmax(target * total - filled))
            assignment[key] = SPLITS[k]
            filled[k] += size
    return df[group].map(assignment).rename("split")


def split_summary(df: pd.DataFrame) -> dict:
    """Effectifs par split et vérification de l'absence de fuite entre splits."""
    groups_per_split = df.groupby("group")["split"].nunique()
    per_class = pd.crosstab(df["articleType"], df["split"]).reindex(columns=SPLITS, fill_value=0)
    return {
        "images": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "groups": {s: int(df.loc[df["split"] == s, "group"].nunique()) for s in SPLITS},
        "leaking_groups": int((groups_per_split > 1).sum()),
        "n_classes": int(df["articleType"].nunique()),
        "classes_missing_from": {
            s: sorted(per_class.index[per_class[s] == 0].tolist()) for s in SPLITS
        },
        "per_class": {c: {s: int(row[s]) for s in SPLITS} for c, row in per_class.iterrows()},
    }


# --------------------------------------------------------------------------- pipeline

OUTPUT_COLUMNS = [
    "id", "image_path", "split", "group", "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "year", "usage", "productDisplayName",
    "width", "height", "md5",
]


def prepare(
    data_dir: Path,
    out_dir: Path,
    min_class_count: int = 20,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    df = load_styles(data_dir / "styles.csv")
    df, report = clean(df, data_dir / "images", min_class_count=min_class_count)
    df["split"] = split_by_group(df, ratios=ratios, seed=seed)
    report["split"] = split_summary(df)
    report["seed"] = seed
    report["ratios"] = dict(zip(SPLITS, ratios))
    report["source"] = str(data_dir)

    if report["split"]["leaking_groups"]:
        raise RuntimeError("Un même produit se retrouve dans plusieurs splits")

    out_dir.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(out_dir / "catalog.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return df, report


def _print_summary(report: dict) -> None:
    print("Lignes conservées à chaque étape :")
    for step, n in report["rows_after_each_step"].items():
        print(f"  {step:<24} {n:>7}")
    if report["image_issues"]:
        print("Images écartées :", report["image_issues"])
    print(f"Classes rares écartées : {len(report['dropped_rare_classes'])}")
    split = report["split"]
    print(f"Classes conservées : {split['n_classes']}")
    for s in SPLITS:
        print(f"  {s:<5} {split['images'][s]:>7} images  {split['groups'][s]:>6} produits")
    print("Produits présents dans plusieurs splits :", split["leaking_groups"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fashion-product-images-small"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    _, report = prepare(args.data_dir, args.out, args.min_class_count, tuple(args.ratios), args.seed)
    _print_summary(report)
    print(f"Écrit : {args.out / 'catalog.csv'} et {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
