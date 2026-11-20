-- Exécuté au premier démarrage du conteneur PostgreSQL.
-- Le schéma (table products, index HNSW) sera géré par des migrations dans catalog-api.
CREATE EXTENSION IF NOT EXISTS vector;
