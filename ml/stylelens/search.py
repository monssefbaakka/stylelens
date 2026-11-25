"""P2 — Recherche des plus proches voisins par similarité cosinus.

Les embeddings étant normalisés L2, la similarité cosinus est un simple produit
scalaire. La recherche est exhaustive (force brute, par blocs de requêtes) :
c'est la référence exacte à laquelle l'index HNSW de pgvector (P4) sera comparé.

Usage (depuis ml/) :
    python -m stylelens.search --index data/embeddings/resnet50 --image chemin/vers/photo.jpg -k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def knn_search(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int,
    query_ids: np.ndarray | None = None,
    gallery_ids: np.ndarray | None = None,
    query_filter: np.ndarray | None = None,
    gallery_filter: np.ndarray | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (indices, scores) des ``k`` voisins les plus proches de chaque requête.

    - ``queries`` (Q, D) et ``gallery`` (N, D) doivent être normalisés L2.
    - ``query_ids`` / ``gallery_ids`` : une requête ne se retrouve jamais elle-même.
    - ``query_filter`` / ``gallery_filter`` : seuls les éléments de la galerie de même
      valeur que la requête sont candidats (ex. ``masterCategory``), comme le filtre
      SQL appliqué en même temps que le kNN dans pgvector.

    Les voisins sont triés par score décroissant. Si moins de ``k`` candidats
    existent, les places vides ont l'indice -1 et le score ``-inf``.
    """
    queries = np.asarray(queries, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError(f"Dimensions incompatibles : {queries.shape} et {gallery.shape}")
    if (query_ids is None) != (gallery_ids is None):
        raise ValueError("query_ids et gallery_ids vont ensemble")
    if (query_filter is None) != (gallery_filter is None):
        raise ValueError("query_filter et gallery_filter vont ensemble")

    n_gallery = len(gallery)
    k_eff = min(k, n_gallery)
    indices = np.full((len(queries), k), -1, dtype=np.int64)
    scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
    if k_eff == 0:
        return indices, scores

    for start in range(0, len(queries), chunk_size):
        stop = start + chunk_size
        sims = queries[start:stop] @ gallery.T  # (q, N)
        if query_ids is not None:
            sims[np.asarray(query_ids)[start:stop, None] == np.asarray(gallery_ids)[None, :]] = -np.inf
        if query_filter is not None:
            sims[np.asarray(query_filter)[start:stop, None] != np.asarray(gallery_filter)[None, :]] = -np.inf

        # argpartition (O(N)) puis tri des k candidats seulement
        top = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top[~np.isfinite(top_scores)] = -1

        indices[start:stop, :k_eff] = top
        scores[start:stop, :k_eff] = top_scores
    return indices, scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True, help="dossier produit par stylelens.embeddings")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--category", help="filtre masterCategory (Apparel, Footwear, Accessories)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from stylelens.embeddings import embed_images  # import tardif : torch
    from stylelens.embeddings_io import load_index

    gallery, index, meta = load_index(args.index)
    query = embed_images([args.image], meta["model"], device=args.device, image_size=meta["image_size"])

    if args.category:
        keep = (index["masterCategory"] == args.category).to_numpy()
        if not keep.any():
            raise SystemExit(f"Aucun produit dans la catégorie {args.category!r}")
        gallery, index = gallery[keep], index[keep].reset_index(drop=True)

    idx, sims = knn_search(query, gallery, args.k)
    hits = index.iloc[idx[0][idx[0] >= 0]].assign(score=sims[0][idx[0] >= 0])
    cols = ["score", "id", "articleType", "baseColour", "productDisplayName", "image_path"]
    print(hits[[c for c in cols if c in hits.columns]].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
