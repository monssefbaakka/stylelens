import json

import numpy as np
import pandas as pd
import pytest

from stylelens.embeddings_io import load_index, save_index
from stylelens.evaluate import (
    count_relevant, evaluate_retrieval, main as evaluate_main, relevance_matrix, retrieval_metrics,
)
from stylelens.search import knn_search, l2_normalize


def brute_force(queries, gallery, k):
    sims = queries @ gallery.T
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def test_knn_matches_brute_force_across_chunks():
    rng = np.random.default_rng(0)
    q = l2_normalize(rng.normal(size=(37, 16)))
    g = l2_normalize(rng.normal(size=(200, 16)))
    idx, scores = knn_search(q, g, k=7, chunk_size=10)
    np.testing.assert_array_equal(idx, brute_force(q, g, 7))
    assert (np.diff(scores, axis=1) <= 1e-6).all()  # scores décroissants
    np.testing.assert_allclose(scores[:, 0], (q @ g.T).max(axis=1), rtol=1e-5)


def test_knn_excludes_self_and_applies_filter():
    g = l2_normalize(np.array([[1, 0], [0.9, 0.1], [0, 1], [0.8, 0.2]]))
    ids = np.array([10, 11, 12, 13])
    cats = np.array(["A", "B", "A", "A"])

    idx, _ = knn_search(g[:1], g, k=2, query_ids=ids[:1], gallery_ids=ids)
    assert 0 not in idx[0]  # la requête ne se retrouve pas elle-même
    assert idx[0].tolist() == [1, 3]

    idx, _ = knn_search(g[:1], g, k=2, query_ids=ids[:1], gallery_ids=ids,
                        query_filter=cats[:1], gallery_filter=cats)
    assert idx[0].tolist() == [3, 2]  # 1 est écarté (catégorie B)


def test_knn_pads_when_not_enough_candidates():
    g = l2_normalize(np.eye(3))
    idx, scores = knn_search(g[:1], g, k=5, query_filter=np.array(["x"]), gallery_filter=np.array(["x", "y", "x"]))
    assert idx[0].tolist() == [0, 2, -1, -1, -1]
    assert np.isneginf(scores[0, 2:]).all()


def test_knn_rejects_bad_shapes():
    with pytest.raises(ValueError):
        knn_search(np.zeros((2, 3)), np.zeros((4, 5)), k=1)
    with pytest.raises(ValueError):
        knn_search(np.zeros((2, 3)), np.zeros((4, 3)), k=1, query_ids=np.arange(2))


def test_retrieval_metrics_hand_computed():
    # requête 1 : pertinents aux rangs 1 et 3, 4 pertinents dans la galerie
    # requête 2 : aucun pertinent dans le top 3, 2 pertinents dans la galerie
    rel = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.float32)
    m = retrieval_metrics(rel, np.array([4, 2]), ks=(1, 3))
    assert m["n_queries"] == 2
    assert m["precision@1"] == pytest.approx(0.5)
    assert m["precision@3"] == pytest.approx((2 / 3 + 0) / 2)
    assert m["recall@3"] == pytest.approx((2 / 4 + 0) / 2)
    assert m["hit_rate@3"] == pytest.approx(0.5)
    # AP@3 requête 1 = (1/1 + 2/3) / min(3, 4)
    assert m["map@3"] == pytest.approx(((1 + 2 / 3) / 3 + 0) / 2)


def test_retrieval_metrics_skips_queries_without_relevant_items():
    rel = np.array([[1, 1], [0, 0]], dtype=np.float32)
    m = retrieval_metrics(rel, np.array([2, 0]), ks=(2,))
    assert m["n_queries"] == 1
    assert m["precision@2"] == m["recall@2"] == m["map@2"] == 1.0
    with pytest.raises(ValueError):
        retrieval_metrics(rel, np.array([2, 0]), ks=(3,))


def test_relevance_and_count_relevant():
    neighbors = np.array([[0, 2, -1]])
    rel = relevance_matrix(neighbors, np.array(["a"]), np.array(["a", "b", "a"]))
    assert rel.tolist() == [[1, 1, 0]]

    n = count_relevant(np.array(["a", "b"]), np.array(["a", "b", "a"]))
    assert n.tolist() == [2, 1]
    # la requête 7 est dans la galerie : elle ne compte pas comme pertinente pour elle-même
    n = count_relevant(np.array(["a"]), np.array(["a", "b", "a"]), np.array([7]), np.array([7, 8, 9]))
    assert n.tolist() == [1]
    n = count_relevant(np.array(["a"]), np.array(["a", "b", "a"]),
                       query_filter=np.array(["F"]), gallery_filter=np.array(["F", "F", "S"]))
    assert n.tolist() == [1]


@pytest.fixture
def clustered_index():
    """3 classes bien séparées dans l'espace des embeddings, splits par produit."""
    rng = np.random.default_rng(1)
    classes = {"Tshirts": "Apparel", "Watches": "Accessories", "Casual Shoes": "Footwear"}
    centers = l2_normalize(rng.normal(size=(len(classes), 32)))
    rows, vecs = [], []
    next_id = 1
    for c, (article, master) in enumerate(classes.items()):
        for p in range(20):
            split = "train" if p < 12 else "val" if p < 16 else "test"
            for _ in range(2):
                rows.append({"id": next_id, "split": split, "group": f"{article}-{p}",
                             "articleType": article, "masterCategory": master})
                vecs.append(centers[c] + 0.05 * rng.normal(size=32))
                next_id += 1
    return l2_normalize(np.array(vecs)), pd.DataFrame(rows)


def test_evaluate_retrieval_on_separable_clusters(clustered_index):
    emb, index = clustered_index
    report = evaluate_retrieval(emb, index, ks=(1, 5), latency_queries=5)
    p = report["protocol"]
    assert (p["n_queries"], p["n_gallery"]) == (24, 96)
    o = report["overall"]
    assert o["precision@1"] == o["precision@5"] == o["map@5"] == 1.0
    assert o["recall@5"] == pytest.approx(5 / 32)  # 32 images par classe dans train+val
    assert set(report["by_master_category"]) == {"Apparel", "Accessories", "Footwear"}
    assert set(report["per_class"]) == {"Tshirts", "Watches", "Casual Shoes"}
    assert report["latency_ms"]["n"] == 5 and report["latency_ms"]["p95"] >= 0


def test_evaluate_retrieval_same_split_excludes_self(clustered_index):
    emb, index = clustered_index
    report = evaluate_retrieval(emb, index, query_splits=("test",), gallery_splits=("test",),
                                ks=(1,), filter_by="masterCategory", latency_queries=0)
    # 8 images de test par classe : 7 pertinentes pour chaque requête (elle-même exclue)
    assert report["overall"]["precision@1"] == 1.0
    assert report["overall"]["recall@1"] == pytest.approx(1 / 7)
    assert report["latency_ms"] == {}


def test_evaluate_cli_writes_metrics(clustered_index, tmp_path, capsys):
    emb, index = clustered_index
    save_index(tmp_path / "idx", emb, index, {"model": "fake", "dim": 32})
    loaded, loaded_index, meta = load_index(tmp_path / "idx")
    np.testing.assert_array_equal(loaded, emb.astype(np.float32))
    assert meta["model"] == "fake" and len(loaded_index) == len(index)

    evaluate_main(["--index", str(tmp_path / "idx"), "--ks", "1", "5", "--latency-queries", "3"])
    metrics = json.loads((tmp_path / "idx" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["precision@5"] == 1.0
    assert "P@K" in capsys.readouterr().out


def test_save_index_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        save_index(tmp_path, np.zeros((3, 4)), pd.DataFrame({"id": [1, 2]}), {})
