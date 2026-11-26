# catalog-api — API métier (Spring Boot)

**Statut : prêt (phases P4 et P5).**

Catalogue produits, validation des images et orchestration de la recherche : l'image est envoyée au service IA, qui renvoie son embedding, puis PostgreSQL/pgvector renvoie les k produits les plus proches (index HNSW, distance cosinus), filtre de catégorie compris dans la même requête SQL.

Java 21 · Spring Boot 4.1 (Web MVC, JDBC, Flyway, Actuator, Validation) · PostgreSQL + pgvector · Testcontainers.

## Endpoints

| Méthode | Chemin | Rôle |
|---------|--------|------|
| `POST` | `/api/search` | multipart `image` (JPEG, PNG, WebP ≤ 10 Mo) + `category` (masterCategory, facultatif) + `k` (1–60, défaut 12) |
| `GET` | `/api/products/{id}` | fiche produit |
| `GET` | `/api/products/{id}/similar` | articles similaires à un produit du catalogue (`category`, `k`) ; son embedding est déjà en base, pas d'appel au service IA |
| `GET` | `/api/categories` | catégories et nombre de produits, pour les filtres |
| `GET` | `/api/info` | taille du catalogue, modèle des embeddings (`catalog_meta`), modèle servi par le service IA |
| `GET` | `/images/{id}.jpg` | photos du catalogue (dossier du jeu de données) |
| `GET` | `/actuator/health` | santé (base de données comprise) |

Réponse de recherche :

```json
{
  "results": [
    {"id": 1855, "name": "Inkfruit Mens Chain Reaction T-shirt", "gender": "Men", "masterCategory": "Apparel",
     "subCategory": "Topwear", "articleType": "Tshirts", "baseColour": "Grey", "imageUrl": "/images/1855.jpg", "score": 0.8731}
  ],
  "query": {"category": "Apparel", "k": 12, "productId": null},
  "timings": {"embeddingMs": 41.2, "searchMs": 3.8, "totalMs": 45.3}
}
```

`score` = similarité cosinus (1 − distance pgvector `<=>`). Erreurs au format `application/problem+json` (RFC 9457) : `400` paramètre ou image invalide, `404` produit inconnu, `413` fichier trop lourd, `415` format refusé, `502` modèle du service IA incompatible avec le catalogue (dimension), `503` service IA injoignable.

## Base de données

Migrations Flyway dans [`src/main/resources/db/migration`](src/main/resources/db/migration) : table `products` (colonne `embedding vector(n)`), index HNSW `vector_cosine_ops`, index sur `master_category`, table `catalog_meta`. La dimension `n` vient de `STYLELENS_EMBEDDING_DIM`. Le contenu est importé par `python -m stylelens.vectordb load` (voir [ml/README.md](../ml/README.md#p4--index-vectoriel-pgvector-hnsw)).

Chaque recherche fixe dans sa transaction `hnsw.ef_search` (`STYLELENS_EF_SEARCH`, 100 par défaut) et, avec un filtre, `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) pour toujours obtenir k résultats ; les résultats sont retriés par score.

Cohérence avec le service IA : une dimension différente de la colonne renvoie `502` avec un message explicite ; un modèle différent de celui de `catalog_meta` (même dimension) est signalé dans les journaux et visible dans `/api/info`.

## Configuration

| Variable | Défaut |
|----------|--------|
| `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | `localhost` / `5432` / `stylelens` / `stylelens` / `stylelens` |
| `STYLELENS_EMBEDDING_DIM` | `512` |
| `INFERENCE_URL` | `http://localhost:8000` |
| `STYLELENS_EF_SEARCH` | `100` |
| `STYLELENS_IMAGES_DIR` | `../ml/data/fashion-product-images-small` |
| `STYLELENS_IMAGES_BASE_URL` | `/` (préfixe des URL d'images, ex. un CDN) |

## Lancement

```bash
cd catalog-api
./mvnw spring-boot:run          # PostgreSQL (docker compose up -d postgres) et service IA démarrés
./mvnw test                     # tests unitaires + tests PostgreSQL/pgvector via Testcontainers (ignorés sans Docker)
```
