package com.stylelens.catalog.search;

import com.stylelens.catalog.product.Product;

/** Produit exposé par l'API (URL d'image au lieu du chemin dans le jeu de données). */
public record ProductView(
        long id,
        String name,
        String gender,
        String masterCategory,
        String subCategory,
        String articleType,
        String baseColour,
        String imageUrl) {

    static ProductView of(Product p, String imageUrl) {
        return new ProductView(p.id(), p.name(), p.gender(), p.masterCategory(), p.subCategory(), p.articleType(),
                p.baseColour(), imageUrl);
    }
}
