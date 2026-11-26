package com.stylelens.catalog.inference;

/** Image refusée (illisible, trop lourde, type non accepté) : erreur du client, renvoyée telle quelle. */
public class ImageRejectedException extends RuntimeException {

    private final int status;

    public ImageRejectedException(int status, String message) {
        super(message);
        this.status = status;
    }

    public int status() {
        return status;
    }
}
