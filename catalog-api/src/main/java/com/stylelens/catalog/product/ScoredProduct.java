package com.stylelens.catalog.product;

/** Produit et similarité cosinus avec la requête (1 = identique). */
public record ScoredProduct(Product product, double score) {
}
