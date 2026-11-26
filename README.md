# StyleLens

**Système de recommandation visuelle par IA pour le commerce électronique.**

L'utilisateur dépose la photo d'un article (vêtement, chaussure, accessoire) et StyleLens renvoie les produits du catalogue qui lui ressemblent le plus. On ne sait pas toujours nommer ce que l'on cherche, mais on sait le montrer.

## Idée centrale

Un réseau de neurones convolutif (CNN) transforme chaque image en un **embedding**, un vecteur de quelques centaines de nombres qui résume son apparence (forme, couleur, texture, coupe). Deux articles qui se ressemblent ont des vecteurs proches. Chercher des articles similaires revient donc à chercher les plus proches voisins du vecteur de l'image requête.

## Architecture cible

```
Interface React ──image──▶ API Spring Boot ──image──▶ Service IA FastAPI (PyTorch : CNN → embedding)
      ▲                      │      │
      └──── résultats ───────┘      ├── kNN + filtre ──▶ PostgreSQL + pgvector (HNSW)
                                    └── cache ─────────▶ Redis
```

Détails : [docs/architecture.md](docs/architecture.md).

## Structure du dépôt

```
stylelens/
├── ml/                 # pipeline IA : données (P1), embeddings CNN + kNN (P2), fine-tuning (P3), import pgvector (P4)
├── inference-service/  # service IA FastAPI : image → embedding (P5)
├── catalog-api/        # API Spring Boot : schéma pgvector HNSW (P4), recherche, catalogue (P5)
├── frontend/           # interface React + TypeScript : recherche par image, filtres (P6)
├── db/init/            # scripts SQL exécutés au premier démarrage de PostgreSQL
├── docs/               # architecture et documentation
└── docker-compose.yml  # PostgreSQL + pgvector, service IA, API, interface
```

## Démarrage

### 1. Données, modèle et embeddings

Voir [ml/README.md](ml/README.md) : téléchargement de *Fashion Product Images*, préparation (P1), fine-tuning sur GPU (P3, `ml/models/resnet50-ft/model.pt`) et embeddings du catalogue (`data/embeddings/resnet50-ft`).

### 2. Application complète

```bash
cp .env.example .env
docker compose up -d --build                               # PostgreSQL, service IA, API, interface
cd ml && python -m stylelens.vectordb load --index data/embeddings/resnet50-ft   # import du catalogue (P4)
```

| Service | URL |
|---------|-----|
| Interface | http://localhost:3000 |
| API catalogue | http://localhost:8080/api/info |
| Service IA | http://localhost:8000/docs |

Le service IA, la dimension de la base (`STYLELENS_EMBEDDING_DIM`) et les embeddings importés doivent venir du même modèle. Sans modèle fine-tuné, un backbone ImageNet indexable convient : `MODEL_CHECKPOINT=` (vide), `MODEL_NAME=mobilenet_v3_large`, `STYLELENS_EMBEDDING_DIM=960`, embeddings `python -m stylelens.embeddings --model mobilenet_v3_large`.

### Tests

```bash
cd ml && python -m pytest                      # + STYLELENS_TEST_DSN=postgresql://… pour les tests pgvector
cd inference-service && python -m pytest
cd catalog-api && ./mvnw test                  # tests PostgreSQL via Testcontainers si Docker est disponible
cd frontend && npm test
```

## Feuille de route

| Phase | Contenu | Statut |
|-------|---------|--------|
| P0 | Cadrage, environnement, dépôt Git | En cours |
| P1 | Données : collecte, nettoyage, séparation par produit | En cours (pipeline prêt, à exécuter sur le jeu complet) |
| P2 | Modèle de base : embeddings CNN + recherche kNN simple | En cours (extraction, recherche et évaluation prêtes, à exécuter sur le jeu complet) |
| P3 | Transfert d'apprentissage, fine-tuning, évaluation (Precision@K, Recall@K, mAP) | En cours (entraînement et comparaison prêts, à exécuter sur GPU) |
| P4 | Index vectoriel pgvector (HNSW) | En cours (schéma, import et mesure du rappel prêts, à exécuter sur le jeu complet) |
| P5 | Service IA FastAPI + API catalogue Spring Boot | En cours (services prêts et testés, à brancher sur le modèle fine-tuné) |
| P6 | Interface React : recherche par image, filtres par catégorie | En cours (interface prête et testée) |
| P7 | Cache Redis, indexation Kafka (optionnel), tests | À faire |
| P8 | Docker Compose complet, observabilité, CI | À faire |
| P9 | Documentation, rapport, démo vidéo | À faire |

## Technologies

PyTorch · torchvision · FastAPI · Java 21 / Spring Boot · PostgreSQL + pgvector · Redis · React + TypeScript · Docker Compose · GitHub Actions
