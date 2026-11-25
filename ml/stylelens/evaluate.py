"""P2 — Évaluation de la recherche par similarité : Precision@K, Recall@K, mAP@K, latence.

Protocole par défaut :
- requêtes : images du split ``test`` ;
- galerie : images des splits ``train`` et ``val`` (le catalogue « connu ») ;
- pertinent : même ``articleType`` que la requête (``--label``).

Les splits P1 étant séparés par produit, aucune requête ne retrouve une
déclinaison de son propre produit dans la galerie : le score mesure la
ressemblance visuelle, pas la détection de quasi-doublons.

Si requêtes et galerie partagent des images (ex. ``--gallery-splits test``),
une requête n'est jamais comparée à elle-même.

Usage (depuis ml/) :
    python -m stylelens.evaluate --index data/embeddings/resnet50
    python -m stylelens.evaluate --index data/embeddings/resnet50 --filter masterCategory
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stylelens.embeddings_io import load_index
from stylelens.search import knn_search

DEFAULT_KS = (1, 5, 10, 20)


# --------------------------------------------------------------------------- métriques

def relevance_matrix(
    neighbors: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray
) -> np.ndarray:
    """rel[q, j] = 1 si le j-ième voisin de q a la même étiquette que q (-1 = vide → 0)."""
    gallery_labels = np.asarray(gallery_labels)
    valid = neighbors >= 0
    labels = gallery_labels[np.where(valid, neighbors, 0)]
    return (valid & (labels == np.asarray(query_labels)[:, None])).astype(np.float32)


def retrieval_metrics(rel: np.ndarray, n_relevant: np.ndarray, ks=DEFAULT_KS) -> dict[str, float]:
    """Métriques moyennées sur les requêtes ayant au moins un élément pertinent.

    - Precision@K = pertinents dans le top K / K
    - Recall@K    = pertinents dans le top K / pertinents dans la galerie
    - HitRate@K   = part des requêtes avec au moins un pertinent dans le top K
    - mAP@K       = moyenne de AP@K = Σ_k P@k·rel_k / min(K, pertinents)
    """
    rel = np.asarray(rel, dtype=np.float32)
    n_relevant = np.asarray(n_relevant, dtype=np.float64)
    keep = n_relevant > 0
    rel, n_relevant = rel[keep], n_relevant[keep]
    out: dict[str, float] = {"n_queries": int(keep.sum())}
    if not keep.any():
        return out

    max_k = rel.shape[1]
    cum = np.cumsum(rel, axis=1)
    precision_at = cum / np.arange(1, max_k + 1)
    for k in ks:
        if k > max_k:
            raise ValueError(f"K={k} > nombre de voisins calculés ({max_k})")
        hits = cum[:, k - 1]
        ap = (precision_at[:, :k] * rel[:, :k]).sum(axis=1) / np.minimum(k, n_relevant)
        out[f"precision@{k}"] = float((hits / k).mean())
        out[f"recall@{k}"] = float((hits / n_relevant).mean())
        out[f"hit_rate@{k}"] = float((hits > 0).mean())
        out[f"map@{k}"] = float(ap.mean())
    return out


def count_relevant(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_ids: np.ndarray | None = None,
    gallery_ids: np.ndarray | None = None,
    query_filter: np.ndarray | None = None,
    gallery_filter: np.ndarray | None = None,
) -> np.ndarray:
    """Nombre d'éléments pertinents candidats pour chaque requête (dénominateur du rappel)."""
    query_labels = np.asarray(query_labels)
    key_g = pd.Series(np.asarray(gallery_labels), dtype="object")
    if query_filter is not None:
        key_g = key_g.astype(str) + "\x1f" + pd.Series(np.asarray(gallery_filter)).astype(str)
        key_q = pd.Series(query_labels).astype(str) + "\x1f" + pd.Series(np.asarray(query_filter)).astype(str)
    else:
        key_q = pd.Series(query_labels, dtype="object")
    n = key_q.map(key_g.value_counts()).fillna(0).to_numpy(dtype=np.int64)

    if query_ids is not None:  # une requête présente dans la galerie ne compte pas pour elle-même
        n = n - np.isin(np.asarray(query_ids), np.asarray(gallery_ids)).astype(np.int64)
    return n


def evaluate_retrieval(
    embeddings: np.ndarray,
    index: pd.DataFrame,
    query_splits=("test",),
    gallery_splits=("train", "val"),
    label: str = "articleType",
    filter_by: str | None = None,
    ks=DEFAULT_KS,
    per_class: bool = True,
    latency_queries: int = 200,
) -> dict:
    """Évalue la recherche kNN exhaustive sur un index d'embeddings."""
    q_mask = index["split"].isin(query_splits).to_numpy()
    g_mask = index["split"].isin(gallery_splits).to_numpy()
    if not q_mask.any() or not g_mask.any():
        raise ValueError(f"Requêtes ({query_splits}) ou galerie ({gallery_splits}) vides")
    q, g = index[q_mask].reset_index(drop=True), index[g_mask].reset_index(drop=True)
    q_emb, g_emb = embeddings[q_mask], embeddings[g_mask]

    overlap = bool(set(query_splits) & set(gallery_splits))
    ids = (q["id"].to_numpy(), g["id"].to_numpy()) if overlap else (None, None)
    filt = (q[filter_by].to_numpy(), g[filter_by].to_numpy()) if filter_by else (None, None)

    max_k = max(ks)
    neighbors, _ = knn_search(q_emb, g_emb, max_k, *ids, *filt)
    rel = relevance_matrix(neighbors, q[label].to_numpy(), g[label].to_numpy())
    n_rel = count_relevant(q[label].to_numpy(), g[label].to_numpy(), *ids, *filt)

    report = {
        "protocol": {
            "query_splits": list(query_splits),
            "gallery_splits": list(gallery_splits),
            "label": label,
            "filter": filter_by,
            "n_queries": int(len(q)),
            "n_gallery": int(len(g)),
            "dim": int(embeddings.shape[1]),
        },
        "overall": retrieval_metrics(rel, n_rel, ks),
        "latency_ms": search_latency(q_emb, g_emb, max_k, latency_queries),
    }

    by_master = {}
    for cat, rows in q.groupby("masterCategory").groups.items():
        rows = np.asarray(rows)
        by_master[cat] = retrieval_metrics(rel[rows], n_rel[rows], ks)
    report["by_master_category"] = by_master

    if per_class:
        k = min(10, max_k)
        rows_out = {}
        for cls, rows in q.groupby(label).groups.items():
            rows = np.asarray(rows)
            m = retrieval_metrics(rel[rows], n_rel[rows], (k,))
            if m["n_queries"]:
                rows_out[cls] = {"n_queries": m["n_queries"], f"precision@{k}": m[f"precision@{k}"], f"map@{k}": m[f"map@{k}"]}
        report["per_class"] = dict(sorted(rows_out.items(), key=lambda kv: kv[1][f"precision@{k}"]))
    return report


def search_latency(q_emb: np.ndarray, g_emb: np.ndarray, k: int, n: int = 200, seed: int = 0) -> dict:
    """Latence d'une requête isolée (kNN exhaustif CPU), moyenne et p95 en ms."""
    if n <= 0 or len(q_emb) == 0:
        return {}
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(q_emb), size=min(n, len(q_emb)), replace=False)
    knn_search(q_emb[picks[:1]], g_emb, k)  # échauffement
    times = []
    for i in picks:
        t0 = time.perf_counter()
        knn_search(q_emb[i:i + 1], g_emb, k)
        times.append(1000 * (time.perf_counter() - t0))
    times = np.asarray(times)
    return {
        "n": int(len(times)),
        "mean": round(float(times.mean()), 3),
        "p50": round(float(np.percentile(times, 50)), 3),
        "p95": round(float(np.percentile(times, 95)), 3),
    }


# --------------------------------------------------------------------------- CLI

def _print_report(report: dict, meta: dict | None) -> None:
    p = report["protocol"]
    name = f"{meta['model']} ({p['dim']}-d)" if meta else f"{p['dim']}-d"
    print(f"Modèle : {name}")
    print(f"Requêtes {'+'.join(p['query_splits'])} : {p['n_queries']}  |  "
          f"galerie {'+'.join(p['gallery_splits'])} : {p['n_gallery']}  |  "
          f"pertinent = même {p['label']}" + (f"  |  filtre {p['filter']}" if p["filter"] else ""))
    o = report["overall"]
    ks = sorted({int(key.split("@")[1]) for key in o if "@" in key})
    print(f"{'K':>4} {'P@K':>8} {'R@K':>8} {'Hit@K':>8} {'mAP@K':>8}")
    for k in ks:
        print(f"{k:>4} {o[f'precision@{k}']:>8.4f} {o[f'recall@{k}']:>8.4f} "
              f"{o[f'hit_rate@{k}']:>8.4f} {o[f'map@{k}']:>8.4f}")
    lat = report["latency_ms"]
    if lat:
        print(f"Latence kNN exhaustif : moyenne {lat['mean']} ms, p95 {lat['p95']} ms")
    if meta and "ms_per_image" in meta:
        print(f"Extraction embedding : {meta['ms_per_image']} ms/image ({meta.get('device')})")
    if report.get("per_class"):
        worst = list(report["per_class"].items())[:5]
        key = next(k for k in worst[0][1] if k.startswith("precision@"))
        print("Classes les plus difficiles :", ", ".join(f"{c} ({v[key]:.2f})" for c, v in worst))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="dossier produit par stylelens.embeddings")
    parser.add_argument("--query-splits", nargs="+", default=["test"])
    parser.add_argument("--gallery-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--label", default="articleType", help="colonne définissant la pertinence")
    parser.add_argument("--filter", dest="filter_by", help="colonne de filtre (ex. masterCategory)")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--out", type=Path, help="défaut : <index>/metrics[_<filtre>].json")
    args = parser.parse_args(argv)

    embeddings, index, meta = load_index(args.index)
    report = evaluate_retrieval(
        embeddings, index, args.query_splits, args.gallery_splits, args.label,
        args.filter_by, tuple(sorted(args.ks)), latency_queries=args.latency_queries,
    )
    report["model"] = meta
    out = args.out or args.index / (f"metrics_{args.filter_by}.json" if args.filter_by else "metrics.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report, meta)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
