# ml/ — Pipeline d'intelligence artificielle

Expérimentations : données, embeddings CNN, fine-tuning, évaluation.

```
ml/
├── stylelens/   # code réutilisable : data.py (P1), embeddings.py, search.py, evaluate.py (P2),
│                #   train.py, compare.py (P3), vectordb.py (P4)
├── tests/       # tests pytest
├── notebooks/   # 01_poc_embeddings.ipynb (preuve de concept), 02_fine_tuning.ipynb (P1 → P3 sur GPU)
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

## P2 — Modèle de base : embeddings CNN + recherche kNN

```bash
cd ml
# 1. Embeddings de tout le catalogue P1 (GPU conseillé ; --limit 2000 pour un essai rapide)
python -m stylelens.embeddings --model resnet50 --out data/embeddings/resnet50

# 2. Évaluation : requêtes = test, galerie = train + val, pertinent = même articleType
python -m stylelens.evaluate --index data/embeddings/resnet50
python -m stylelens.evaluate --index data/embeddings/resnet50 --filter masterCategory

# 3. Recherche à partir d'une photo
python -m stylelens.search --index data/embeddings/resnet50 --image photo.jpg -k 10 --category Footwear
```

Modèles disponibles (`--model`) : `resnet50` (2048-d), `efficientnet_b0` (1280-d), `mobilenet_v3_large` (960-d), pré-entraînés sur ImageNet, couche de classification retirée.

Choix :

- **Prétraitement** : l'image est complétée en carré sur fond blanc puis redimensionnée en 224×224, au lieu du `CenterCrop` ImageNet qui couperait le haut et le bas des photos en portrait (col, semelle).
- **Embedding** : sortie du pooling global, normalisée L2. La similarité cosinus devient un produit scalaire.
- **Recherche** : kNN exhaustif en NumPy, par blocs de requêtes. C'est la référence exacte pour mesurer plus tard le rappel de l'index HNSW (P4).
- **Filtre** : `--filter masterCategory` ne garde que les candidats de la même catégorie que la requête, comme le filtre SQL appliqué en même temps que le kNN dans pgvector.

Sorties dans `data/embeddings/<modèle>/` :

- `embeddings.npy` (float32, N × D), `index.csv` (même ordre : `id`, `split`, étiquettes), `meta.json` (modèle, dimension, durée d'extraction en ms/image)
- `metrics.json` : Precision@K, Recall@K, HitRate@K, mAP@K (K = 1, 5, 10, 20), latence de recherche moyenne et p95, détail par `masterCategory` et par classe

Définitions (moyennes sur les requêtes) :

| Métrique | Définition |
|----------|------------|
| Precision@K | pertinents dans le top K / K |
| Recall@K | pertinents dans le top K / pertinents dans la galerie |
| HitRate@K | au moins un pertinent dans le top K |
| mAP@K | moyenne de AP@K = Σₖ P@k · relₖ / min(K, pertinents) |

Les splits P1 séparent les produits : une requête ne retrouve jamais une déclinaison de son propre produit, le score mesure donc la ressemblance visuelle et non la détection de quasi-doublons. Avec une étiquette de classe comme critère, Recall@K reste faible par construction (des centaines de pertinents par classe) : Precision@K et mAP@K sont les métriques principales.

## P3 — Transfert d'apprentissage et fine-tuning

```bash
cd ml
# 1. Fine-tuning (GPU conseillé : notebooks/02_fine_tuning.ipynb sur Colab ou Kaggle)
python -m stylelens.train --backbone resnet50 --out models/resnet50-ft

# 2. Embeddings du catalogue avec le modèle fine-tuné
python -m stylelens.embeddings --checkpoint models/resnet50-ft/model.pt     # → data/embeddings/resnet50-ft

# 3. Comparaison avec et sans fine-tuning, même protocole que P2
python -m stylelens.compare data/embeddings/resnet50 data/embeddings/resnet50-ft --out data/comparaison.md
```

Essai rapide sur CPU : `--backbone mobilenet_v3_large --image-size 128 --steps-per-epoch 20 --epochs 3`.

Méthode :

| Élément | Choix | Raison |
|---------|-------|--------|
| Modèle | backbone ImageNet + projection linéaire 512-d + BatchNorm, sortie normalisée L2 (`--embed-dim`, 0 = sans projection) | embedding plus compact (512 au lieu de 2048) : index pgvector 4× plus petit |
| Perte | classification cosinus à marge **CosFace** sur `articleType` (s = 30, m = 0,2) | optimise directement la similarité cosinus utilisée par la recherche ; stable, sans minage de triplets |
| Calendrier | 2 époques backbone gelé, puis réseau complet ; LR backbone = 0,1 × LR tête, remontée linéaire sur une époque après le dégel ; décroissance cosinus ; AdamW | la tête converge avant de toucher aux poids ImageNet |
| BatchNorm | statistiques du backbone figées par défaut (`--train-bn` pour les mettre à jour) | au dégel, des petits lots d'images sur fond blanc faussent les statistiques ImageNet : sur le jeu de test synthétique, la perte remonte de 1,5 à 4,0 sans ce gel, contre 0,0002 avec |
| Déséquilibre | échantillonnage pondéré ∝ 1/√effectif de la classe (`--balance`) | les classes rares sont vues plus souvent sans être sur-apprises |
| Augmentations | pad-to-square, recadrage aléatoire léger (75–100 %), miroir horizontal, luminosité et contraste ; **teinte inchangée** | la couleur est un critère de ressemblance pour l'utilisateur |
| Sélection | mAP@10 de recherche sur `val` à chaque époque, meilleur checkpoint conservé, arrêt anticipé (`--patience 4`) | le split `test` reste réservé à la comparaison finale |
| GPU | précision mixte automatique (`--no-amp` pour la désactiver) | |

Sorties dans `models/<backbone>-ft/` (ignoré par git) : `model.pt` (poids, configuration, classes, métriques de l'époque retenue), `history.json` (perte, précision, métriques val par époque), `summary.json`.

`stylelens.compare` évalue chaque index avec le même protocole et produit un tableau Markdown pour le rapport : P@1, P@10, mAP@10, Hit@5, R@20, gain de mAP@10 par rapport au premier index, temps d'extraction et latence kNN p95.

## P4 — Index vectoriel pgvector (HNSW)

```bash
# 0. PostgreSQL + schéma (migrations Flyway de catalog-api, appliquées à son démarrage)
docker compose up -d postgres catalog-api        # depuis la racine du dépôt

# 1. Import du catalogue (depuis ml/)
python -m stylelens.vectordb load --index data/embeddings/resnet50-ft

# 2. Rappel et latence de l'index HNSW par rapport au kNN exact
python -m stylelens.vectordb evaluate --index data/embeddings/resnet50-ft --ef-search 20 40 100 200
python -m stylelens.vectordb evaluate --index data/embeddings/resnet50-ft --filter masterCategory
```

Connexion : `--dsn postgresql://…`, sinon `$STYLELENS_DSN`, sinon les variables `POSTGRES_*` de `.env`.

Schéma ([`V1__create_products.sql`](../catalog-api/src/main/resources/db/migration/V1__create_products.sql)) :

| Élément | Choix | Raison |
|---------|-------|--------|
| `products` | une ligne par image : étiquettes, `image_path`, `split`, `embedding vector(n)` | `n` = `STYLELENS_EMBEDDING_DIM` (512 par défaut, modèle fine-tuné P3) |
| Index | HNSW `vector_cosine_ops`, `m = 16`, `ef_construction = 64` | embeddings normalisés L2 : distance cosinus `<=>`, score = 1 − distance |
| Filtre | index B-tree sur `master_category`, condition dans la même requête que le kNN ; `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) | sans parcours itératif, un filtre sélectif renverrait moins de K résultats |
| `catalog_meta` | modèle, dimension, nombre de produits, date d'import | l'API vérifie que le service IA utilise le même modèle |

Limite : pgvector n'indexe en HNSW que 2 000 dimensions au plus. ResNet50 brut (2048-d) ne passe pas ; le modèle fine-tuné (512-d), `mobilenet_v3_large` (960-d) et `efficientnet_b0` (1280-d) passent. La base est créée avec la dimension `STYLELENS_EMBEDDING_DIM` ; pour en changer, supprimer le volume (`docker compose down -v`).

`load` remplace tout le catalogue dans une transaction : suppression de l'index HNSW, `TRUNCATE`, `COPY` binaire, reconstruction de l'index avec la définition de la migration (plus rapide qu'une insertion dans le graphe ligne à ligne), `catalog_meta`, puis `ANALYZE`. L'API sert l'ancien catalogue jusqu'au commit.

`evaluate` reprend le protocole P2 (requêtes = `test`, galerie = le reste, pertinent = même `articleType`) sur 1 000 requêtes (`--max-queries`) et écrit `<index>/hnsw.json` : pour chaque `ef_search`, rappel@K par rapport au kNN exact NumPy, Precision@K, mAP@K, latence SQL moyenne et p95, et si le plan utilise bien l'index HNSW. La ligne `exact` est la même requête sans index (parcours séquentiel).

Tests PostgreSQL (schéma temporaire, supprimé à la fin) : `STYLELENS_TEST_DSN=postgresql://stylelens:stylelens@localhost:5432/stylelens python -m pytest`.

## Preuve de concept (semaine 1)

`01_poc_embeddings.ipynb` : ResNet50 pré-entraîné (ImageNet) sans couche de classification, embeddings 2048-d normalisés L2 pour 1 000 images, recherche des plus proches voisins par similarité cosinus, Precision@5 indicative.
