"""Tests P4. Les tests PostgreSQL ne tournent que si STYLELENS_TEST_DSN est défini, ex. :
    STYLELENS_TEST_DSN=postgresql://stylelens:stylelens@localhost:5432/stylelens python -m pytest
Ils travaillent dans un schéma temporaire, supprimé à la fin.
"""

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stylelens.vectordb import (
    HNSW_INDEX, PRODUCT_COLUMNS, SchemaError, default_dsn, evaluate_hnsw, load_catalog, product_rows,
)

MIGRATION = Path(__file__).resolve().parents[2] / "catalog-api/src/main/resources/db/migration/V1__create_products.sql"
DIM = 8


def test_default_dsn_prefers_explicit_url_then_postgres_vars():
    assert default_dsn({"STYLELENS_DSN": "postgresql://a@b/c"}) == "postgresql://a@b/c"
    assert default_dsn({}) == "postgresql://stylelens:stylelens@localhost:5432/stylelens"
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_PORT": "6543", "POSTGRES_DB": "d"}
    assert default_dsn(env) == "postgresql://u:p@localhost:6543/d"


def test_product_rows_maps_columns_and_missing_values():
    index = pd.DataFrame({
        "id": [7, 8], "image_path": ["images/7.jpg", "images/8.jpg"], "split": ["train", "test"],
        "masterCategory": ["Apparel", "Footwear"], "articleType": ["Tshirts", "Casual Shoes"],
        "productDisplayName": ["Red Tee", np.nan], "baseColour": [np.nan, "Black"],
    })
    emb = np.eye(2, 3, dtype=np.float64)
    rows = list(product_rows(emb, index))
    assert len(rows) == 2 and all(len(r) == len(PRODUCT_COLUMNS) for r in rows)
    first = dict(zip(PRODUCT_COLUMNS, rows[0]))
    second = dict(zip(PRODUCT_COLUMNS, rows[1]))
    assert first["id"] == 7 and isinstance(first["id"], int)
    assert first["name"] == "Red Tee" and first["base_colour"] is None and first["gender"] is None
    assert second["name"] == "Casual Shoes"  # pas de nom : type d'article
    assert first["embedding"].dtype == np.float32


def test_product_rows_rejects_incomplete_index():
    with pytest.raises(ValueError, match="manquantes"):
        list(product_rows(np.zeros((1, 2)), pd.DataFrame({"id": [1]})))
    index = pd.DataFrame({"id": [1], "image_path": ["x"], "masterCategory": ["A"], "articleType": ["T"]})
    with pytest.raises(ValueError):
        list(product_rows(np.zeros((2, 2)), index))


# --------------------------------------------------------------------------- PostgreSQL

@pytest.fixture
def pg():
    dsn = os.environ.get("STYLELENS_TEST_DSN")
    if not dsn:
        pytest.skip("STYLELENS_TEST_DSN non défini")
    psycopg = pytest.importorskip("psycopg")
    from pgvector.psycopg import register_vector

    schema = f"stylelens_test_{uuid.uuid4().hex[:8]}"
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path = {schema}, public")
    conn.execute(MIGRATION.read_text(encoding="utf-8").replace("${embedding_dim}", str(DIM)))
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


def synthetic_catalog(n_per_class: int = 30, seed: int = 0):
    """Trois classes bien séparées sur l'hypersphère, 15 % de requêtes test."""
    rng = np.random.default_rng(seed)
    centers = np.eye(3, DIM)
    rows, emb = [], []
    for c, (master, article) in enumerate([("Apparel", "Tshirts"), ("Footwear", "Casual Shoes"),
                                           ("Accessories", "Watches")]):
        for j in range(n_per_class):
            v = centers[c] + 0.1 * rng.standard_normal(DIM)
            emb.append(v / np.linalg.norm(v))
            pid = 100 * (c + 1) + j
            rows.append({"id": pid, "image_path": f"images/{pid}.jpg", "masterCategory": master,
                         "articleType": article, "productDisplayName": f"{article} {j}",
                         "split": "test" if j % 7 == 0 else "train"})
    return np.asarray(emb, dtype=np.float32), pd.DataFrame(rows)


def test_load_catalog_replaces_rows_and_rebuilds_index(pg):
    emb, index = synthetic_catalog()
    meta = {"model": "resnet50", "fine_tuned": True, "image_size": 224}
    info = load_catalog(pg, emb, index, meta, source="synthetic")
    assert info["index_rebuilt"] and info["n_products"] == len(index)

    info = load_catalog(pg, emb[:10], index.iloc[:10], meta)  # second import : remplace
    assert pg.execute("SELECT count(*) FROM products").fetchone()[0] == 10
    assert pg.execute(
        "SELECT count(*) FROM pg_indexes WHERE schemaname = current_schema() AND indexname = %s", (HNSW_INDEX,)
    ).fetchone()[0] == 1
    stored = dict(pg.execute("SELECT key, value FROM catalog_meta").fetchall())
    assert stored["model"] == "resnet50" and stored["fine_tuned"] == "true"
    assert stored["dim"] == str(DIM) and stored["n_products"] == "10"

    row = pg.execute("SELECT name, master_category, embedding FROM products WHERE id = %s",
                     (int(index.at[0, "id"]),)).fetchone()
    assert row[0] == index.at[0, "productDisplayName"] and row[1] == "Apparel"
    np.testing.assert_allclose(row[2].to_numpy(), emb[0], rtol=1e-6)


def test_load_catalog_rejects_dimension_mismatch_without_touching_data(pg):
    emb, index = synthetic_catalog()
    load_catalog(pg, emb, index, {"model": "m"})
    with pytest.raises(SchemaError, match=f"vector\\({DIM}\\)"):
        load_catalog(pg, np.ones((len(index), DIM + 1), dtype=np.float32), index, {"model": "m"})
    assert pg.execute("SELECT count(*) FROM products").fetchone()[0] == len(index)


def test_evaluate_hnsw_matches_exact_search_on_small_catalog(pg):
    emb, index = synthetic_catalog()
    load_catalog(pg, emb, index, {"model": "m"})
    report = evaluate_hnsw(pg, emb, index, ef_values=(40,), k=5)
    assert report["protocol"]["n_products_in_db"] == len(index)
    exact, hnsw = report["results"]
    assert exact["config"] == "exact" and not exact["uses_hnsw_index"]
    assert exact["recall@5_vs_exact"] == 1.0
    assert hnsw["recall@5_vs_exact"] >= 0.9
    assert hnsw["precision@5"] == exact["precision@5"] == pytest.approx(1.0)  # classes séparées

    filtered = evaluate_hnsw(pg, emb, index, ef_values=(40,), k=5, filter_by="masterCategory")
    assert filtered["results"][0]["recall@5_vs_exact"] == 1.0
