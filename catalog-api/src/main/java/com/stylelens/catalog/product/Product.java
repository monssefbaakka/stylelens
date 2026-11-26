package com.stylelens.catalog.product;

/** Produit du catalogue (une ligne de {@code products}, sans l'embedding). */
public record Product(
        long id,
        String name,
        String gender,
        String masterCategory,
        String subCategory,
        String articleType,
        String baseColour,
        String imagePath) {
}
