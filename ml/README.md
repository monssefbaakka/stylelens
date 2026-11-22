# ml/ — Pipeline d'intelligence artificielle

Expérimentations : données, embeddings CNN, fine-tuning, évaluation.

```
ml/
├── stylelens/   # code réutilisable (data.py : préparation des données, P1)
├── tests/       # tests pytest
├── notebooks/   # 01_poc_embeddings.ipynb (preuve de concept)
├── data/        # jeu de données (ignoré par git)
└── models/      # poids entraînés (ignorés par git)
```

## Données

Jeu conseillé : [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), environ 44 000 images avec `masterCategory`, `subCategory` et `articleType`.

```bash
# nécessite un token Kaggle (~/.kaggle/kaggle.json)
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p ml/data --unzip
```

Attendu : `ml/data/fashion-product-images-small/styles.csv` et `ml/data/fashion-product-images-small/images/`.
Vérifier la licence du jeu de données avant de diffuser des images.

## Installation

```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_poc_embeddings.ipynb
```

## P1 — Préparation des données

```bash
cd ml
python -m stylelens.data --data-dir data/fashion-product-images-small --out data/processed
python -m pytest            # tests du pipeline (données synthétiques, pas besoin du jeu réel)
```

Sorties dans `data/processed/` :

- `catalog.csv` : une ligne par image conservée, avec `image_path` (relatif au dossier du jeu), les étiquettes, `group` (produit) et `split` (`train` / `val` / `test`)
- `report.json` : nombre de lignes après chaque étape, images écartées, classes rares, effectifs par split et par classe

Règles de nettoyage, dans l'ordre :

1. Lecture tolérante de `styles.csv` : les noms contenant des virgules sont conservés, alors que `read_csv(on_bad_lines='skip')` les perd.
2. Suppression des lignes sans `id`, `masterCategory`, `subCategory` ou `articleType`, et des `id` en double.
3. Conservation de `Apparel`, `Footwear` et `Accessories` uniquement.
4. Suppression des images manquantes, corrompues (décodage complet) ou dont un côté fait moins de 32 px.
5. Suppression des doublons exacts (même MD5) : on garde le plus petit `id`.
6. Suppression des `articleType` de moins de 20 images (`--min-class-count`).

Séparation **par produit** (70 / 15 / 15, `--ratios`, graine `--seed 42`) : les images sont regroupées par nom de produit normalisé. Toutes les déclinaisons d'un même produit vont dans le même split, donc le test ne contient aucun quasi-doublon d'une image d'entraînement. La séparation est stratifiée par `articleType`. Le script s'arrête si un produit apparaît dans plusieurs splits.

## Preuve de concept (semaine 1)

`01_poc_embeddings.ipynb` : ResNet50 pré-entraîné (ImageNet) sans couche de classification, embeddings 2048-d normalisés L2 pour 1 000 images, recherche des plus proches voisins par similarité cosinus, Precision@5 indicative.
