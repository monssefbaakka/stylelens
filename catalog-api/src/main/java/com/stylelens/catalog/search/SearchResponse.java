package com.stylelens.catalog.search;

import com.stylelens.catalog.product.Product;
import java.util.List;

/** Résultat d'une recherche : produits triés par similarité décroissante. */
public record SearchResponse(List<Item> results, Query query, Timings timings) {

    public record Item(
            long id,
            String name,
            String gender,
            String masterCategory,
            String subCategory,
            String articleType,
            String baseColour,
            String imageUrl,
            double score) {

        static Item of(Product p, String imageUrl, double score) {
            return new Item(p.id(), p.name(), p.gender(), p.masterCategory(), p.subCategory(), p.articleType(),
                    p.baseColour(), imageUrl, Math.round(score * 10_000) / 10_000.0);
        }
    }

    /** {@code productId} : produit de départ d'une recherche « articles similaires ». */
    public record Query(String category, int k, Long productId) {
    }

    /** Durées en millisecondes : embedding (service IA, absent pour « similaires »), kNN, total. */
    public record Timings(Double embeddingMs, double searchMs, double totalMs) {
    }
}
