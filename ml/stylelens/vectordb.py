"""P4 — Index vectoriel pgvector (HNSW) : import du catalogue et mesure du rappel.

Le schéma (table ``products``, index HNSW cosinus, ``catalog_meta``) appartient
aux migrations Flyway de catalog-api (``V1__create_products.sql``) : démarrer
l'API une fois (ou ``docker compose up -d catalog-api``) avant le premier import.

Sous-commandes (depuis ml/) :
    python -m stylelens.vectordb load --index data/embeddings/resnet50-ft
    python -m stylelens.vectordb evaluate --index data/embeddings/resnet50-ft --ef-search 20 40 100 200
    python -m stylelens.vectordb evaluate --index data/embeddings/resnet50-ft --filter masterCategory

load
    Remplace le contenu de ``products`` par un index d'embeddings (COPY binaire),
    reconstruit l'index HNSW après l'import (plus rapide que d'insérer les lignes
    une à une dans le graphe) et enregistre le modèle dans ``catalog_meta``.
    Tout se fait dans une transaction : l'API continue de servir l'ancien
    catalogue jusqu'au commit.
evaluate
    Pour un échantillon de requêtes du split ``test``, compare les voisins
    renvoyés par PostgreSQL (galerie = ``train`` + ``val``, comme P2) au kNN
    exact NumPy : rappel@K, Precision@K, mAP@K et latence SQL, pour plusieurs
    valeurs de ``hnsw.ef_search``. La ligne « exact » est le même kNN sans index
    (parcours séquentiel), référence de latence.

Connexion : ``--dsn``, sinon ``$STYLELENS_DSN``, sinon variables ``POSTGRES_*``
(mêmes valeurs par défaut que ``.env.example``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from stylelens.embeddings_io import load_index
from stylelens.evaluate import count_relevant, relevance_matrix, retrieval_metrics
from stylelens.search import knn_search

HNSW_INDEX = "products_embedding_hnsw"
HNSW_MAX_DIM = 2000  # limite pgvector pour un index HNSW sur le type vector

# colonne SQL → (colonne de index.csv, type pour COPY binaire)
PRODUCT_COLUMNS: dict[str, tuple[str | None, str]] = {
    "id": ("id", "bigint"),
    "name": ("productDisplayName", "text"),
    "gender": ("gender", "text"),
    "master_category": ("masterCategory", "text"),
    "sub_category": ("subCategory", "text"),
    "article_type": ("articleType", "text"),
    "base_colour": ("baseColour", "text"),
    "image_path": ("image_path", "text"),
    "split": ("split", "text"),
    "embedding": (None, "vector"),
}
REQUIRED_INDEX_COLUMNS = {"id", "image_path", "masterCategory", "articleType"}


class SchemaError(RuntimeError):
    """Schéma absent ou incompatible avec l'index d'embeddings."""


# --------------------------------------------------------------------------- connexion

def default_dsn(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    if env.get("STYLELENS_DSN"):
        return env["STYLELENS_DSN"]
    user = env.get("POSTGRES_USER", "stylelens")
    password = env.get("POSTGRES_PASSWORD", "stylelens")
    host = env.get("POSTGRES_HOST", "localhost")
    port = env.get("POSTGRES_PORT", "5432")
    db = env.get("POSTGRES_DB", "stylelens")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def connect(dsn: str):
    """Connexion psycopg en autocommit, type ``vector`` enregistré (tableaux NumPy)."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as e:
        conn.close()
        raise SchemaError("Extension pgvector absente : démarrer PostgreSQL via docker compose") from e
    return conn


def column_dim(cur) -> int:
    """Dimension déclarée de ``products.embedding`` (typmod de ``vector(n)``)."""
    cur.execute(
        "SELECT a.atttypmod FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass('products') AND a.attname = 'embedding' AND NOT a.attisdropped"
    )
    row = cur.fetchone()
    if row is None:
        raise SchemaError(
            "Table products absente : démarrer catalog-api une fois pour appliquer les migrations Flyway"
        )
    return int(row[0])


# --------------------------------------------------------------------------- import

def _text(value) -> str | None:
    return None if pd.isna(value) else str(value)


def product_rows(embeddings: np.ndarray, index: pd.DataFrame) -> Iterator[tuple]:
    """Lignes de ``products`` dans l'ordre de ``PRODUCT_COLUMNS``."""
    missing = REQUIRED_INDEX_COLUMNS - set(index.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans index.csv : {sorted(missing)}")
    if len(embeddings) != len(index):
        raise ValueError(f"{len(embeddings)} embeddings pour {len(index)} lignes d'index")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    for i, rec in enumerate(index.to_dict("records")):
        row = []
        for column, (source, _) in PRODUCT_COLUMNS.items():
            if column == "id":
                row.append(int(rec["id"]))
            elif column == "embedding":
                row.append(embeddings[i])
            elif column == "name":
                row.append(_text(rec.get(source)) or str(rec["articleType"]))
            else:
                row.append(_text(rec.get(source)))
        yield tuple(row)


def load_catalog(
    conn,
    embeddings: np.ndarray,
    index: pd.DataFrame,
    meta: dict,
    source: str | None = None,
    rebuild_index: bool = True,
    maintenance_work_mem: str = "256MB",
) -> dict:
    """Remplace le catalogue par ``embeddings`` / ``index`` dans une seule transaction."""
    dim = int(embeddings.shape[1])
    if dim > HNSW_MAX_DIM:
        raise SchemaError(
            f"{dim} dimensions : pgvector n'indexe en HNSW que {HNSW_MAX_DIM} dimensions au plus "
            "(utiliser le modèle fine-tuné 512-d, ou mobilenet_v3_large / efficientnet_b0)"
        )
    columns = ", ".join(PRODUCT_COLUMNS)
    types = [t for _, t in PRODUCT_COLUMNS.values()]
    timings = {}
    with conn.transaction(), conn.cursor() as cur:
        declared = column_dim(cur)
        if declared != dim:
            raise SchemaError(
                f"products.embedding est en vector({declared}) mais l'index fait {dim} dimensions : "
                f"recréer la base avec STYLELENS_EMBEDDING_DIM={dim}"
            )

        indexdef = None
        if rebuild_index:
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = %s",
                (HNSW_INDEX,),
            )
            found = cur.fetchone()
            if found:
                indexdef = found[0]
                cur.execute(f"DROP INDEX {HNSW_INDEX}")

        cur.execute("TRUNCATE products")
        t0 = time.perf_counter()
        with cur.copy(f"COPY products ({columns}) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(types)
            for row in product_rows(embeddings, index):
                copy.write_row(row)
        timings["copy_seconds"] = round(time.perf_counter() - t0, 2)

        if indexdef:  # même définition que la migration Flyway
            cur.execute("SELECT set_config('maintenance_work_mem', %s, true)", (maintenance_work_mem,))
            t0 = time.perf_counter()
            cur.execute(indexdef)
            timings["index_seconds"] = round(time.perf_counter() - t0, 2)

        info = {
            "model": meta.get("model"),
            "fine_tuned": str(bool(meta.get("fine_tuned"))).lower(),
            "dim": dim,
            "image_size": meta.get("image_size"),
            "checkpoint": meta.get("checkpoint"),
            "n_products": len(index),
            "source": source,
            "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cur.execute("DELETE FROM catalog_meta")
        cur.executemany(
            "INSERT INTO catalog_meta (key, value) VALUES (%s, %s)",
            [(k, str(v)) for k, v in info.items() if v is not None],
        )

    conn.execute("ANALYZE products")
    if not conn.autocommit:
        conn.commit()
    return {**info, **timings, "index_rebuilt": indexdef is not None}


# --------------------------------------------------------------------------- évaluation

def knn_sql(category: bool) -> str:
    """kNN cosinus servi par l'index HNSW (même forme que la requête de catalog-api)."""
    where = "split <> %s" + (" AND master_category = %s" if category else "")
    return (
        f"SELECT id, embedding <=> %s AS distance FROM products WHERE {where} "
        "ORDER BY embedding <=> %s LIMIT %s"
    )


def _configure(cur, ef_search: int | None) -> None:
    if ef_search is None:  # référence : parcours séquentiel exact
        cur.execute("SET enable_indexscan = off")
    else:
        cur.execute("SET enable_indexscan = on")
        cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")


def _enable_iterative_scan(cur) -> bool:
    """pgvector >= 0.8 : poursuit le parcours du graphe tant que le filtre écarte des candidats."""
    import psycopg

    try:
        cur.execute("SET hnsw.iterative_scan = relaxed_order")
        return True
    except psycopg.Error:
        return False


def evaluate_hnsw(
    conn,
    embeddings: np.ndarray,
    index: pd.DataFrame,
    ef_values=(20, 40, 100, 200),
    k: int = 10,
    query_split: str = "test",
    filter_by: str | None = None,
    label: str = "articleType",
    max_queries: int = 1000,
    seed: int = 0,
) -> dict:
    """Rappel et latence de l'index HNSW par rapport au kNN exact, pour chaque ``ef_search``."""
    if filter_by not in (None, "masterCategory"):
        raise ValueError("Seul le filtre masterCategory existe côté SQL")
    split = index["split"].to_numpy()
    q_pos = np.flatnonzero(split == query_split)
    g_mask = split != query_split
    if len(q_pos) == 0 or not g_mask.any():
        raise ValueError(f"Requêtes ({query_split}) ou galerie vides")
    rng = np.random.default_rng(seed)
    q_pos = np.sort(rng.choice(q_pos, size=min(max_queries, len(q_pos)), replace=False))
    q, q_emb = index.iloc[q_pos].reset_index(drop=True), np.asarray(embeddings[q_pos], dtype=np.float32)
    g, g_emb = index[g_mask].reset_index(drop=True), embeddings[g_mask]
    filt = (q[filter_by].to_numpy(), g[filter_by].to_numpy()) if filter_by else (None, None)

    exact, _ = knn_search(q_emb, g_emb, k, None, None, *filt)
    g_ids = g["id"].to_numpy()
    exact_sets = [set(g_ids[row[row >= 0]].tolist()) for row in exact]
    position = {int(i): j for j, i in enumerate(g_ids)}
    n_rel = count_relevant(q[label].to_numpy(), g[label].to_numpy(), None, None, *filt)
    ks = tuple(sorted({1, k}))

    sql = knn_sql(filter_by is not None)
    configs = [("exact", None)] + [(f"hnsw ef_search={ef}", int(ef)) for ef in ef_values]
    rows = []
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM products")
        n_db = cur.fetchone()[0]
        iterative = _enable_iterative_scan(cur) if filter_by else False

        def params(i: int) -> tuple:
            vec = q_emb[i]
            return (vec, query_split, *([q.at[i, filter_by]] if filter_by else []), vec, k)

        for name, ef in configs:
            _configure(cur, ef)
            cur.execute("EXPLAIN " + sql, params(0))
            uses_index = any(HNSW_INDEX in line for (line,) in cur.fetchall())
            for i in range(min(5, len(q))):  # échauffement (cache, requête préparée)
                cur.execute(sql, params(i)).fetchall()

            neighbors = np.full((len(q), k), -1, dtype=np.int64)
            recalls, times = [], []
            for i in range(len(q)):
                t0 = time.perf_counter()
                found = cur.execute(sql, params(i)).fetchall()
                times.append(1000 * (time.perf_counter() - t0))
                found.sort(key=lambda r: r[1])  # relaxed_order : ordre approximatif
                ids = [int(r[0]) for r in found]
                neighbors[i, :len(ids)] = [position.get(x, -1) for x in ids]
                if exact_sets[i]:
                    recalls.append(len(exact_sets[i].intersection(ids)) / len(exact_sets[i]))

            rel = relevance_matrix(neighbors, q[label].to_numpy(), g[label].to_numpy())
            metrics = retrieval_metrics(rel, n_rel, ks)
            times = np.asarray(times)
            rows.append({
                "config": name,
                "ef_search": ef,
                "uses_hnsw_index": uses_index,
                f"recall@{k}_vs_exact": round(float(np.mean(recalls)), 4) if recalls else None,
                f"precision@{k}": round(metrics.get(f"precision@{k}", float("nan")), 4),
                f"map@{k}": round(metrics.get(f"map@{k}", float("nan")), 4),
                "latency_ms": {
                    "mean": round(float(times.mean()), 3),
                    "p50": round(float(np.percentile(times, 50)), 3),
                    "p95": round(float(np.percentile(times, 95)), 3),
                },
            })
        cur.execute("RESET enable_indexscan")

    return {
        "protocol": {
            "query_split": query_split,
            "gallery": f"split <> '{query_split}'",
            "label": label,
            "filter": filter_by,
            "k": k,
            "n_queries": int(len(q)),
            "n_gallery": int(len(g)),
            "n_index": int(len(index)),
            "n_products_in_db": int(n_db),
            "dim": int(embeddings.shape[1]),
            "iterative_scan": iterative,
        },
        "results": rows,
    }


# --------------------------------------------------------------------------- CLI

def _print_evaluation(report: dict) -> None:
    p, k = report["protocol"], report["protocol"]["k"]
    print(f"Requêtes {p['query_split']} : {p['n_queries']}  |  galerie : {p['n_gallery']}  |  "
          f"{p['dim']}-d" + (f"  |  filtre {p['filter']} (iterative_scan : {p['iterative_scan']})"
                            if p["filter"] else ""))
    if p["n_products_in_db"] != p["n_index"]:
        print(f"Attention : {p['n_products_in_db']} produits en base pour {p['n_index']} dans l'index : "
              "relancer « load » avec cet index")
    print(f"{'Configuration':<22} {'Rappel@' + str(k):>10} {'P@' + str(k):>8} {'mAP@' + str(k):>8} "
          f"{'moy. ms':>9} {'p95 ms':>9}  index")
    for r in report["results"]:
        lat = r["latency_ms"]
        print(f"{r['config']:<22} {r[f'recall@{k}_vs_exact']:>10.4f} {r[f'precision@{k}']:>8.4f} "
              f"{r[f'map@{k}']:>8.4f} {lat['mean']:>9.2f} {lat['p95']:>9.2f}  "
              f"{'HNSW' if r['uses_hnsw_index'] else 'séquentiel'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=None, help="défaut : $STYLELENS_DSN ou variables POSTGRES_*")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="importer un index d'embeddings dans products")
    p_load.add_argument("--index", type=Path, required=True, help="dossier produit par stylelens.embeddings")
    p_load.add_argument("--no-rebuild-index", action="store_true",
                        help="garder l'index HNSW pendant l'import (plus lent)")
    p_load.add_argument("--maintenance-work-mem", default="256MB",
                        help="mémoire pour construire l'index HNSW")

    p_eval = sub.add_parser("evaluate", help="rappel et latence HNSW contre le kNN exact")
    p_eval.add_argument("--index", type=Path, required=True)
    p_eval.add_argument("--ef-search", type=int, nargs="+", default=[20, 40, 100, 200])
    p_eval.add_argument("-k", type=int, default=10)
    p_eval.add_argument("--filter", dest="filter_by", choices=["masterCategory"])
    p_eval.add_argument("--max-queries", type=int, default=1000)
    p_eval.add_argument("--out", type=Path, help="défaut : <index>/hnsw[_<filtre>].json")
    args = parser.parse_args(argv)

    embeddings, index, meta = load_index(args.index)
    conn = connect(args.dsn or default_dsn())
    try:
        if args.command == "load":
            info = load_catalog(conn, embeddings, index, meta, str(args.index),
                                not args.no_rebuild_index, args.maintenance_work_mem)
            print(f"{info['n_products']} produits importés ({info['model']}, {info['dim']}-d) "
                  f"en {info['copy_seconds']} s"
                  + (f", index HNSW reconstruit en {info['index_seconds']} s" if info["index_rebuilt"] else ""))
        else:
            report = evaluate_hnsw(conn, embeddings, index, tuple(args.ef_search), args.k,
                                   filter_by=args.filter_by, max_queries=args.max_queries)
            report["model"] = meta
            out = args.out or args.index / (f"hnsw_{args.filter_by}.json" if args.filter_by else "hnsw.json")
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            _print_evaluation(report)
            print(f"Écrit : {out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
