from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from stylelens.data import (
    COLUMNS, SPLITS, load_styles, prepare, product_group, split_by_group,
)

HEADER = ",".join(COLUMNS)


def write_image(path: Path, color, size=(60, 80)):
    Image.new("RGB", size, color).save(path, "JPEG")


@pytest.fixture
def dataset(tmp_path):
    """Mini jeu de données reprenant les pièges du vrai styles.csv."""
    root = tmp_path / "fashion"
    images = root / "images"
    images.mkdir(parents=True)

    lines = [HEADER]
    next_id = 1
    # 3 classes fréquentes, 10 produits chacune, 2 déclinaisons par produit
    for k, article in enumerate(["Tshirts", "Watches", "Casual Shoes"]):
        master = {"Tshirts": "Apparel", "Watches": "Accessories", "Casual Shoes": "Footwear"}[article]
        for p in range(10):
            for v in range(2):
                i = next_id
                next_id += 1
                lines.append(f"{i},Men,{master},Sub,{article},Blue,Summer,2012,Casual,Brand {article} Model {p}")
                write_image(images / f"{i}.jpg", (k * 80, p * 20, v * 100))

    # Nom contenant des virgules (perdu avec on_bad_lines='skip')
    lines.append(f"{next_id},Women,Apparel,Sub,Tshirts,Red,Summer,2012,Casual,Tee, red, slim fit")
    write_image(images / f"{next_id}.jpg", (250, 10, 10))
    comma_id = next_id
    next_id += 1
    # Catégorie hors périmètre
    lines.append(f"{next_id},Men,Personal Care,Sub,Deodorant,Black,Summer,2012,Casual,Deo")
    write_image(images / f"{next_id}.jpg", (1, 2, 3))
    next_id += 1
    # Image manquante
    lines.append(f"{next_id},Men,Apparel,Sub,Tshirts,Blue,Summer,2012,Casual,Missing tee")
    next_id += 1
    # Image corrompue
    lines.append(f"{next_id},Men,Apparel,Sub,Tshirts,Blue,Summer,2012,Casual,Broken tee")
    (images / f"{next_id}.jpg").write_bytes(b"not a jpeg")
    next_id += 1
    # Doublon exact de l'image 1 sous un autre id
    lines.append(f"{next_id},Men,Apparel,Sub,Tshirts,Blue,Summer,2012,Casual,Copy tee")
    (images / f"{next_id}.jpg").write_bytes((images / "1.jpg").read_bytes())
    dup_id = next_id
    next_id += 1
    # Classe rare
    lines.append(f"{next_id},Men,Accessories,Sub,Cufflinks,Gold,Summer,2012,Formal,Rare cufflinks")
    write_image(images / f"{next_id}.jpg", (200, 200, 0))
    next_id += 1
    # Ligne sans articleType
    lines.append(f"{next_id},Men,Apparel,Sub,,Blue,Summer,2012,Casual,No type")

    (root / "styles.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, {"comma_id": comma_id, "dup_id": dup_id}


def test_load_styles_keeps_names_with_commas(dataset):
    root, ids = dataset
    df = load_styles(root / "styles.csv")
    row = df[df["id"] == ids["comma_id"]].iloc[0]
    assert row["productDisplayName"] == "Tee, red, slim fit"
    assert row["articleType"] == "Tshirts"


def test_product_group_normalises_names():
    assert product_group("Nike Men's  T-Shirt", 1) == product_group("nike men s t shirt", 2)
    assert product_group(pd.NA, 7) == "id:7"
    assert product_group("---", 8) == "id:8"


def test_prepare_cleans_and_splits(dataset, tmp_path):
    root, ids = dataset
    out = tmp_path / "processed"
    df, report = prepare(root, out, min_class_count=5, ratios=(0.6, 0.2, 0.2))

    assert (out / "catalog.csv").exists() and (out / "report.json").exists()
    assert report["image_issues"] == {"missing": 1, "corrupt": 1}
    assert report["dropped_rare_classes"] == {"Cufflinks": 1}
    assert ids["dup_id"] not in set(df["id"])
    assert set(df["masterCategory"]) <= {"Apparel", "Footwear", "Accessories"}
    assert len(df) == 61  # 60 images fréquentes + le nom à virgules
    assert set(df["split"]) == set(SPLITS)

    # Pas de fuite : chaque produit (groupe) n'est que dans un seul split
    assert report["split"]["leaking_groups"] == 0
    assert (df.groupby("group")["split"].nunique() == 1).all()
    # Les deux déclinaisons d'un même produit restent ensemble
    assert df.groupby("group").size().max() == 2

    # Chaque classe est représentée dans chaque split
    assert all(not missing for missing in report["split"]["classes_missing_from"].values())


def test_split_is_deterministic_and_follows_ratios():
    df = pd.DataFrame({
        "group": [f"g{i}" for i in range(1000)],
        "articleType": ["A"] * 600 + ["B"] * 400,
    })
    a = split_by_group(df, ratios=(0.7, 0.15, 0.15), seed=0)
    b = split_by_group(df, ratios=(0.7, 0.15, 0.15), seed=0)
    assert a.equals(b)
    shares = a.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.01)
    assert shares["val"] == pytest.approx(0.15, abs=0.01)
    assert shares["test"] == pytest.approx(0.15, abs=0.01)
    assert not split_by_group(df, seed=1).equals(a)
