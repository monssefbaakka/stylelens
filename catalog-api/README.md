# catalog-api — API métier (Spring Boot)

**Statut : à venir (phases P4 et P5).**

Rôle : catalogue produits, validation des images, orchestration de la recherche (appel au service IA puis requête kNN pgvector avec filtre de catégorie).

Technologies prévues : Java 21, Spring Boot, PostgreSQL + pgvector (index HNSW), Flyway, JUnit/MockMvc.

Endpoints prévus :

- `POST /api/search` : image + `category` (optionnel) + `k`, renvoie les produits les plus similaires
- `GET /api/categories` : liste des catégories pour les filtres
