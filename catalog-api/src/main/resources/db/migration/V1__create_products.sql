-- P4 — Catalogue produits et index vectoriel HNSW (pgvector).
--
-- embedding_dim : placeholder Flyway (propriété stylelens.embedding-dim, 512 par défaut,
-- soit la projection du modèle fine-tuné P3). pgvector n'indexe en HNSW que les vecteurs
-- de 2 000 dimensions au plus : ResNet50 brut (2048-d) n'est pas indexable tel quel.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE products (
    id              BIGINT PRIMARY KEY,
    name            TEXT   NOT NULL,
    gender          TEXT,
    master_category TEXT   NOT NULL,
    sub_category    TEXT,
    article_type    TEXT   NOT NULL,
    base_colour     TEXT,
    image_path      TEXT   NOT NULL,
    split           TEXT,
    embedding       vector(${embedding_dim}) NOT NULL
);

-- Embeddings normalisés L2 : distance cosinus (<=>), score = 1 - distance.
-- m = 16, ef_construction = 64 : valeurs par défaut de pgvector. Le rappel par rapport
-- au kNN exact est mesuré par python -m stylelens.vectordb evaluate (ml/).
CREATE INDEX products_embedding_hnsw ON products
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Filtre de catégorie appliqué dans la même requête que le kNN.
CREATE INDEX products_master_category_idx ON products (master_category);

-- Modèle ayant produit les embeddings, écrit par l'import (stylelens.vectordb load).
CREATE TABLE catalog_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
