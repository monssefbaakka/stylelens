package com.stylelens.catalog.inference;

/** Service IA injoignable, trop lent ou en erreur. */
public class InferenceUnavailableException extends RuntimeException {

    public InferenceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
