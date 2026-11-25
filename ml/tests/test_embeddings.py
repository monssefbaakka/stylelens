import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from stylelens.embeddings import BACKBONES, PadToSquare, build_index, build_model, extract, make_transform  # noqa: E402
from stylelens.embeddings_io import load_index  # noqa: E402


def test_pad_to_square_keeps_whole_image():
    img = Image.new("RGB", (60, 80), (10, 20, 30))
    out = PadToSquare()(img)
    assert out.size == (80, 80)
    assert out.getpixel((40, 40)) == (10, 20, 30)  # centre : l'article
    assert out.getpixel((0, 40)) == (255, 255, 255)  # bords : fond blanc
    assert PadToSquare()(Image.new("RGB", (50, 50))).size == (50, 50)


def test_transform_output_shape():
    x = make_transform(96)(Image.new("RGB", (60, 80), "red"))
    assert tuple(x.shape) == (3, 96, 96)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbones_output_declared_dim(name):
    model = build_model(name, pretrained=False)
    assert not model.training
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 64, 64))
    assert tuple(out.shape) == (2, BACKBONES[name].dim)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("vgg16")


@pytest.fixture
def tiny_catalog(tmp_path):
    root = tmp_path / "fashion"
    (root / "images").mkdir(parents=True)
    rows = []
    for i, (split, article) in enumerate(
        [("train", "Tshirts"), ("train", "Watches"), ("val", "Tshirts"), ("test", "Watches"), ("test", "Tshirts")], 1
    ):
        Image.new("RGB", (60, 80), (i * 40, 255 - i * 40, 100)).save(root / "images" / f"{i}.jpg")
        rows.append({"id": i, "image_path": f"images/{i}.jpg", "split": split, "group": f"g{i}",
                     "masterCategory": "Apparel", "articleType": article, "productDisplayName": f"P {i}"})
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(rows).to_csv(catalog, index=False)
    return root, catalog


def test_extract_returns_normalized_embeddings(tiny_catalog):
    root, _ = tiny_catalog
    paths = sorted((root / "images").glob("*.jpg"))
    emb = extract(build_model("mobilenet_v3_large", pretrained=False), paths, image_size=64, batch_size=2, device="cpu")
    assert emb.shape == (5, BACKBONES["mobilenet_v3_large"].dim)
    assert emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)


def test_build_index_writes_aligned_files(tiny_catalog, tmp_path):
    root, catalog = tiny_catalog
    out = tmp_path / "emb"
    meta = build_index(catalog, root, out, "mobilenet_v3_large", image_size=64, batch_size=4,
                       device="cpu", pretrained=False, progress=False)
    emb, index, saved_meta = load_index(out)
    assert saved_meta == meta
    assert meta["n_images"] == len(index) == len(emb) == 5
    assert index["id"].tolist() == [1, 2, 3, 4, 5]
    assert {"split", "articleType", "masterCategory", "group"} <= set(index.columns)

    small = build_index(catalog, root, tmp_path / "emb_small", "mobilenet_v3_large", image_size=64,
                        device="cpu", limit=3, pretrained=False, progress=False)
    assert small["n_images"] < 5
