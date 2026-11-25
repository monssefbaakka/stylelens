import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from stylelens.compare import compare, main as compare_main, to_markdown  # noqa: E402
from stylelens.embeddings import EmbeddingNet, build_index, load_checkpoint  # noqa: E402
from stylelens.embeddings_io import load_index, save_index  # noqa: E402
from stylelens.train import (  # noqa: E402
    CosineMarginHead, TrainConfig, lr_lambdas, make_train_transform, sample_weights, set_backbone_trainable, train,
)

TINY = dict(backbone="mobilenet_v3_large", embed_dim=32, image_size=64, batch_size=8,
            pretrained=False, device="cpu", num_workers=0)


def test_cosine_margin_head_applies_margin_to_target_only():
    head = CosineMarginHead(4, 3, scale=10.0, margin=0.25)
    emb = torch.randn(5, 4)
    y = torch.tensor([0, 1, 2, 0, 1])
    plain, with_margin = head(emb), head(emb, y)
    diff = plain - with_margin
    expected = 10.0 * 0.25 * torch.nn.functional.one_hot(y, 3).float()
    torch.testing.assert_close(diff, expected)
    assert plain.abs().max() <= 10.0 + 1e-5  # s · cos θ


def test_embedding_net_projection_and_identity():
    with torch.inference_mode():
        x = torch.zeros(2, 3, 64, 64)
        proj = EmbeddingNet("mobilenet_v3_large", 32, pretrained=False).eval()
        out = proj(x)
        assert out.shape == (2, 32) and proj.embed_dim == 32
        torch.testing.assert_close(out.norm(dim=1), torch.ones(2))
        plain = EmbeddingNet("mobilenet_v3_large", None, pretrained=False).eval()
        assert plain(x).shape == (2, 960) and plain.embed_dim == 960


def test_set_backbone_trainable_freezes_grads_and_batchnorm():
    model = EmbeddingNet("mobilenet_v3_large", 16, pretrained=False)
    set_backbone_trainable(model, False)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not model.backbone.training
    assert all(p.requires_grad for p in model.head.parameters())
    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.backbone.parameters()) and model.backbone.training
    bns = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)  # stats ImageNet conservées par défaut
    set_backbone_trainable(model, True, freeze_bn=False)
    assert all(m.training for m in bns)


def test_lr_lambdas_freeze_warmup_and_cosine():
    backbone, head = lr_lambdas(steps_per_epoch=10, epochs=5, freeze_epochs=2)
    assert head(0) == pytest.approx(1.0) and head(50) == pytest.approx(0.01)
    assert all(head(s) >= head(s + 1) for s in range(50))
    assert backbone(0) == backbone(19) == 0.0  # gelé pendant 2 époques
    assert backbone(20) == pytest.approx(head(20) * 0.1)  # début de la remontée
    assert backbone(29) == pytest.approx(head(29))  # remontée terminée après une époque
    assert backbone(40) == pytest.approx(head(40))


def test_sample_weights_modes():
    labels = np.array([0, 0, 0, 0, 1])
    np.testing.assert_allclose(sample_weights(labels, "inverse"), [0.25] * 4 + [1.0])
    np.testing.assert_allclose(sample_weights(labels, "sqrt"), [0.5] * 4 + [1.0])
    assert sample_weights(labels, "none") is None
    with pytest.raises(ValueError):
        sample_weights(labels, "bogus")


def test_train_transform_shape():
    x = make_train_transform(48)(Image.new("RGB", (60, 80), "blue"))
    assert tuple(x.shape) == (3, 48, 48)


@pytest.fixture
def shapes_catalog(tmp_path):
    """3 classes de formes distinctes, plusieurs produits par split."""
    root = tmp_path / "fashion"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, i = [], 1
    for article, master in [("Tshirts", "Apparel"), ("Watches", "Accessories"), ("Casual Shoes", "Footwear")]:
        for p in range(8):
            split = "train" if p < 5 else "val" if p < 7 else "test"
            color = tuple(int(c) for c in rng.integers(0, 200, 3))
            img = Image.new("RGB", (60, 80), "white")
            d = ImageDraw.Draw(img)
            if article == "Tshirts":
                d.rectangle([10, 15, 50, 70], fill=color)
            elif article == "Watches":
                d.ellipse([12, 22, 48, 58], fill=color)
            else:
                d.polygon([(5, 60), (55, 60), (55, 50), (25, 35), (5, 45)], fill=color)
            img.save(root / "images" / f"{i}.jpg")
            rows.append({"id": i, "image_path": f"images/{i}.jpg", "split": split, "group": f"{article}-{p}",
                         "masterCategory": master, "articleType": article, "productDisplayName": f"{article} {p}"})
            i += 1
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(rows).to_csv(catalog, index=False)
    return root, catalog


def test_train_writes_checkpoint_and_history(shapes_catalog, tmp_path):
    root, catalog = shapes_catalog
    out = tmp_path / "ft"
    cfg = TrainConfig(**TINY, epochs=2, freeze_epochs=1, steps_per_epoch=2, patience=5)
    summary = train(catalog, root, out, cfg, progress=False)

    assert (out / "model.pt").exists() and (out / "summary.json").exists()
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert [h["phase"] for h in history] == ["backbone gelé", "réseau complet"]
    assert all(np.isfinite(h["loss"]) for h in history)
    assert all("val_map@10" in h for h in history)
    assert 1 <= summary["best_epoch"] <= 2 and summary["n_classes"] == 3

    model, ckpt = load_checkpoint(out / "model.pt")
    assert ckpt["backbone"] == "mobilenet_v3_large" and ckpt["embed_dim"] == 32
    assert ckpt["classes"] == ["Casual Shoes", "Tshirts", "Watches"]
    assert model.embed_dim == 32


def test_frozen_phase_leaves_backbone_untouched(shapes_catalog, tmp_path):
    root, catalog = shapes_catalog
    model = EmbeddingNet("mobilenet_v3_large", 32, pretrained=False)
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
    cfg = TrainConfig(**TINY, epochs=1, freeze_epochs=1, steps_per_epoch=2)
    train(catalog, root, tmp_path / "ft", cfg, model=model, progress=False)

    after = model.backbone.state_dict()
    assert all(torch.equal(before[k], after[k].cpu()) for k in before)  # poids et stats BatchNorm
    assert any(not torch.equal(head_before[k], v.cpu()) for k, v in model.head.state_dict().items())


def test_index_from_checkpoint_and_compare(shapes_catalog, tmp_path, capsys):
    root, catalog = shapes_catalog
    cfg = TrainConfig(**TINY, epochs=1, freeze_epochs=0, steps_per_epoch=2)
    train(catalog, root, tmp_path / "ft", cfg, progress=False)

    meta = build_index(catalog, root, tmp_path / "emb-ft", model_name=None, device="cpu",
                       checkpoint=tmp_path / "ft" / "model.pt", progress=False)
    emb, index, saved = load_index(tmp_path / "emb-ft")
    assert meta["fine_tuned"] and saved["checkpoint"].endswith("model.pt")
    assert meta["image_size"] == 64 and emb.shape == (len(index), 32)

    base = build_index(catalog, root, tmp_path / "emb-base", "mobilenet_v3_large", image_size=64,
                       device="cpu", pretrained=False, progress=False)
    assert base["fine_tuned"] is False and base["dim"] == 960

    rows = compare([tmp_path / "emb-base", tmp_path / "emb-ft"], latency_queries=2)
    assert [r["fine_tuned"] for r in rows] == [False, True]
    assert rows[0]["delta_map@10"] == 0.0
    assert rows[1]["delta_map@10"] == pytest.approx(rows[1]["map@10"] - rows[0]["map@10"])
    table = to_markdown(rows)
    assert table.count("\n") == 3 and "| emb-ft | mobilenet_v3_large | oui | 32 |" in table

    compare_main([str(tmp_path / "emb-base"), str(tmp_path / "emb-ft"), "--latency-queries", "0",
                  "--out", str(tmp_path / "cmp.md")])
    assert "mAP@10" in (tmp_path / "cmp.md").read_text(encoding="utf-8")


def test_compare_on_fake_indexes(tmp_path):
    rng = np.random.default_rng(0)
    index = pd.DataFrame({"id": range(40), "split": ["train"] * 30 + ["test"] * 10,
                          "articleType": ["A", "B"] * 20, "masterCategory": ["Apparel"] * 40})
    good = np.where(index["articleType"].to_numpy()[:, None] == "A", 1.0, -1.0) * np.ones((40, 4))
    good = good + 0.01 * rng.normal(size=(40, 4))
    noise = rng.normal(size=(40, 4))
    for name, emb, ft in [("base", noise, False), ("ft", good, True)]:
        save_index(tmp_path / name, emb / np.linalg.norm(emb, axis=1, keepdims=True), index,
                   {"model": "fake", "fine_tuned": ft, "ms_per_image": 1.0})
    rows = compare([tmp_path / "base", tmp_path / "ft"], latency_queries=0)
    assert rows[1]["map@10"] == 1.0 and rows[1]["delta_map@10"] > 0
    assert rows[0]["latency_p95"] is None
