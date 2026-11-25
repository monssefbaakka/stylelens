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
├── ml/                 # pipeline IA : données (P1), embeddings CNN + kNN + évaluation (P2)
├── inference-service/  # service IA FastAPI (à venir, P5)
├── catalog-api/        # API métier Spring Boot (à venir, P4–P5)
├── frontend/           # interface React + TypeScript (à venir, P6)
├── db/init/            # scripts SQL exécutés au premier démarrage de PostgreSQL
├── docs/               # architecture et documentation
└── docker-compose.yml  # pour l'instant : PostgreSQL + pgvector
```

## Démarrage (semaine 1)

### 1. Base de données PostgreSQL + pgvector

```bash
cp .env.example .env
docker compose up -d
docker compose exec postgres psql -U stylelens -d stylelens -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

### 2. Données et preuve de concept

Voir [ml/README.md](ml/README.md) : téléchargement de *Fashion Product Images*, puis notebook `ml/notebooks/01_poc_embeddings.ipynb` (ResNet50 → embeddings de 1 000 images → recherche par similarité cosinus).

## Feuille de route

| Phase | Contenu | Statut |
|-------|---------|--------|
| P0 | Cadrage, environnement, dépôt Git | En cours |
| P1 | Données : collecte, nettoyage, séparation par produit | En cours (pipeline prêt, à exécuter sur le jeu complet) |
| P2 | Modèle de base : embeddings CNN + recherche kNN simple | En cours (extraction, recherche et évaluation prêtes, à exécuter sur le jeu complet) |
| P3 | Transfert d'apprentissage, fine-tuning, évaluation (Precision@K, Recall@K, mAP) | À faire |
| P4 | Index vectoriel pgvector (HNSW) | À faire |
| P5 | Service IA FastAPI + API catalogue Spring Boot | À faire |
| P6 | Interface React : recherche par image, filtres par catégorie | À faire |
| P7 | Cache Redis, indexation Kafka (optionnel), tests | À faire |
| P8 | Docker Compose complet, observabilité, CI | À faire |
| P9 | Documentation, rapport, démo vidéo | À faire |

## Technologies

PyTorch · torchvision · FastAPI · Java 21 / Spring Boot · PostgreSQL + pgvector · Redis · React + TypeScript · Docker Compose · GitHub Actions
