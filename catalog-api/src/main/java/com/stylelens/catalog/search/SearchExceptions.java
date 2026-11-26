package com.stylelens.catalog.search;

/** Erreurs métier de la recherche, traduites en réponses HTTP par {@code ApiExceptionHandler}. */
public final class SearchExceptions {

    private SearchExceptions() {
    }

    /** Paramètre invalide (400). */
    public static class BadRequestException extends RuntimeException {
        public BadRequestException(String message) {
            super(message);
        }
    }

    /** Format d'image non accepté (415). */
    public static class UnsupportedImageTypeException extends RuntimeException {
        public UnsupportedImageTypeException(String message) {
            super(message);
        }
    }

    /** Produit absent du catalogue (404). */
    public static class ProductNotFoundException extends RuntimeException {
        public ProductNotFoundException(long id) {
            super("Produit introuvable : " + id);
        }
    }

    /** Le service IA ne produit pas des vecteurs comparables à ceux du catalogue (502). */
    public static class IncompatibleModelException extends RuntimeException {
        public IncompatibleModelException(String message) {
            super(message);
        }
    }
}
