# ml/ — Pipeline d'intelligence artificielle

Expérimentations : données, embeddings CNN, fine-tuning, évaluation.

```
ml/
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

## Preuve de concept (semaine 1)

`01_poc_embeddings.ipynb` : ResNet50 pré-entraîné (ImageNet) sans couche de classification, embeddings 2048-d normalisés L2 pour 1 000 images, recherche des plus proches voisins par similarité cosinus, Precision@5 indicative.
