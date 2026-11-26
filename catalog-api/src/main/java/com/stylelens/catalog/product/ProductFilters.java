package com.stylelens.catalog.product;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Filtres d'une recherche, combinés par ET. Chaque filtre vaut {@code null} s'il n'est pas utilisé.
 *
 * @param category    {@code master_category} : Apparel, Footwear, Accessories
 * @param subCategory {@code sub_category} : Topwear, Shoes, Watches…
 * @param articleType {@code article_type} : Tshirts, Casual Shoes…
 * @param gender      {@code gender} : Men, Women, Boys, Girls, Unisex
 * @param colour      {@code base_colour} : Black, Navy Blue…
 */
public record ProductFilters(String category, String subCategory, String articleType, String gender, String colour) {

    public static final ProductFilters NONE = new ProductFilters(null, null, null, null, null);
    static final int MAX_LENGTH = 100;

    /** Dimensions filtrables, dans l'ordre d'affichage : nom dans l'API → colonne SQL. */
    public enum Facet {
        CATEGORY("category", "master_category"),
        SUB_CATEGORY("subCategory", "sub_category"),
        ARTICLE_TYPE("articleType", "article_type"),
        GENDER("gender", "gender"),
        COLOUR("colour", "base_colour");

        private final String param;
        private final String column;

        Facet(String param, String column) {
            this.param = param;
            this.column = column;
        }

        public String param() {
            return param;
        }

        String column() {
            return column;
        }
    }

    /** Valeurs vides ignorées, espaces retirés ; refuse les valeurs anormalement longues. */
    public ProductFilters {
        category = clean(category);
        subCategory = clean(subCategory);
        articleType = clean(articleType);
        gender = clean(gender);
        colour = clean(colour);
    }

    public String value(Facet facet) {
        return switch (facet) {
            case CATEGORY -> category;
            case SUB_CATEGORY -> subCategory;
            case ARTICLE_TYPE -> articleType;
            case GENDER -> gender;
            case COLOUR -> colour;
        };
    }

    @JsonIgnore
    public boolean isEmpty() {
        return values(null).isEmpty();
    }

    /** Filtres actifs (colonne → valeur), sauf {@code excluded} (comptage des facettes). */
    Map<String, String> values(Facet excluded) {
        Map<String, String> active = new LinkedHashMap<>();
        for (Facet f : Facet.values()) {
            if (f != excluded && value(f) != null) {
                active.put(f.column(), value(f));
            }
        }
        return active;
    }

    /** Clause SQL ({@code col = ? AND …}, ou chaîne vide) et ses paramètres. */
    SqlCondition toSql(Facet excluded) {
        List<String> clauses = new ArrayList<>();
        List<Object> params = new ArrayList<>();
        values(excluded).forEach((column, value) -> {
            clauses.add(column + " = ?");
            params.add(value);
        });
        return new SqlCondition(String.join(" AND ", clauses), params);
    }

    record SqlCondition(String sql, List<Object> params) {
    }

    private static String clean(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String trimmed = value.trim();
        if (trimmed.length() > MAX_LENGTH) {
            throw new IllegalArgumentException("Valeur de filtre trop longue (" + MAX_LENGTH + " caractères maximum)");
        }
        return trimmed;
    }
}
