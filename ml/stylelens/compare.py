"""P3 — Comparaison de plusieurs index d'embeddings (avec et sans fine-tuning).

Chaque index est évalué avec exactement le même protocole (``stylelens.evaluate``),
puis les résultats sont réunis dans un tableau Markdown, prêt pour le rapport.

Usage (depuis ml/) :
    python -m stylelens.compare data/embeddings/resnet50 data/embeddings/resnet50-ft
    python -m stylelens.compare data/embeddings/* --filter masterCategory --out comparaison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stylelens.embeddings_io import load_index
from stylelens.evaluate import evaluate_retrieval

KS = (1, 5, 10, 20)
COLUMNS = [
    ("Index", "name"), ("Backbone", "model"), ("Fine-tuné", "fine_tuned"), ("Dim.", "dim"),
    ("P@1", "precision@1"), ("P@10", "precision@10"), ("mAP@10", "map@10"), ("Hit@5", "hit_rate@5"),
    ("R@20", "recall@20"), ("Gain mAP@10", "delta_map@10"), ("Extraction (ms/img)", "ms_per_image"),
    ("kNN p95 (ms)", "latency_p95"),
]


def compare(
    index_dirs: list[Path],
    query_splits=("test",),
    gallery_splits=("train", "val"),
    label: str = "articleType",
    filter_by: str | None = None,
    latency_queries: int = 100,
) -> list[dict]:
    """Une ligne de résultats par index ; ``delta_map@10`` est relatif au premier."""
    rows = []
    for d in index_dirs:
        emb, index, meta = load_index(d)
        report = evaluate_retrieval(
            emb, index, query_splits, gallery_splits, label, filter_by, KS,
            per_class=False, latency_queries=latency_queries,
        )
        rows.append({
            "name": Path(d).name,
            "model": meta.get("model", "?"),
            "fine_tuned": bool(meta.get("fine_tuned", False)),
            "dim": int(emb.shape[1]),
            **{k: v for k, v in report["overall"].items() if "@" in k},
            "ms_per_image": meta.get("ms_per_image"),
            "latency_p95": report["latency_ms"].get("p95"),
            "n_queries": report["protocol"]["n_queries"],
            "n_gallery": report["protocol"]["n_gallery"],
        })
    if rows:
        ref = rows[0]["map@10"]
        for r in rows:
            r["delta_map@10"] = r["map@10"] - ref
    return rows


def to_markdown(rows: list[dict]) -> str:
    def fmt(key, value):
        if value is None:
            return "–"
        if key == "fine_tuned":
            return "oui" if value else "non"
        if key == "delta_map@10":
            return f"{value:+.4f}"
        if isinstance(value, float) and "@" in key:
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(title for title, _ in COLUMNS) + " |",
        "|" + "|".join("---" if key in ("name", "model", "fine_tuned") else "---:" for _, key in COLUMNS) + "|",
    ]
    lines += ["| " + " | ".join(fmt(key, r.get(key)) for _, key in COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indexes", type=Path, nargs="+", help="dossiers produits par stylelens.embeddings")
    parser.add_argument("--query-splits", nargs="+", default=["test"])
    parser.add_argument("--gallery-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--label", default="articleType")
    parser.add_argument("--filter", dest="filter_by")
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--out", type=Path, help="fichier Markdown de sortie")
    args = parser.parse_args(argv)

    rows = compare(args.indexes, args.query_splits, args.gallery_splits, args.label,
                   args.filter_by, args.latency_queries)
    table = to_markdown(rows)
    r = rows[0]
    header = (f"Requêtes {'+'.join(args.query_splits)} : {r['n_queries']} · galerie "
              f"{'+'.join(args.gallery_splits)} : {r['n_gallery']} · pertinent = même {args.label}"
              + (f" · filtre {args.filter_by}" if args.filter_by else ""))
    print(header)
    print(table)
    if args.out:
        args.out.write_text(f"{header}\n\n{table}\n", encoding="utf-8")
        print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
