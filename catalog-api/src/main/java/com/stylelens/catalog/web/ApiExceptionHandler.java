package com.stylelens.catalog.web;

import com.stylelens.catalog.inference.ImageRejectedException;
import com.stylelens.catalog.inference.InferenceUnavailableException;
import com.stylelens.catalog.search.SearchExceptions.BadRequestException;
import com.stylelens.catalog.search.SearchExceptions.IncompatibleModelException;
import com.stylelens.catalog.search.SearchExceptions.ProductNotFoundException;
import com.stylelens.catalog.search.SearchExceptions.UnsupportedImageTypeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

/**
 * Erreurs au format RFC 9457 ({@code application/problem+json}). Les erreurs Spring MVC standard
 * (partie multipart manquante, fichier trop lourd, paramètre mal typé…) sont gérées par la classe mère.
 */
@RestControllerAdvice
public class ApiExceptionHandler extends ResponseEntityExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    @ExceptionHandler(BadRequestException.class)
    ProblemDetail badRequest(BadRequestException e) {
        return problem(HttpStatus.BAD_REQUEST, "Requête invalide", e.getMessage());
    }

    @ExceptionHandler(UnsupportedImageTypeException.class)
    ProblemDetail unsupportedType(UnsupportedImageTypeException e) {
        return problem(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "Format d'image non accepté", e.getMessage());
    }

    @ExceptionHandler(ImageRejectedException.class)
    ProblemDetail imageRejected(ImageRejectedException e) {
        HttpStatusCode status = switch (e.status()) {
            case 413, 415 -> HttpStatusCode.valueOf(e.status());
            default -> HttpStatus.BAD_REQUEST;
        };
        return problem(status, "Image refusée", e.getMessage());
    }

    @ExceptionHandler(ProductNotFoundException.class)
    ProblemDetail notFound(ProductNotFoundException e) {
        return problem(HttpStatus.NOT_FOUND, "Produit introuvable", e.getMessage());
    }

    @ExceptionHandler(InferenceUnavailableException.class)
    ProblemDetail inferenceUnavailable(InferenceUnavailableException e) {
        log.warn("Service IA indisponible : {}", e.getMessage(), e.getCause());
        return problem(HttpStatus.SERVICE_UNAVAILABLE, "Service IA indisponible",
                e.getMessage() + ". Réessayer dans quelques instants.");
    }

    @ExceptionHandler(IncompatibleModelException.class)
    ProblemDetail incompatibleModel(IncompatibleModelException e) {
        log.error(e.getMessage());
        return problem(HttpStatus.BAD_GATEWAY, "Modèle incompatible avec le catalogue", e.getMessage());
    }

    private static ProblemDetail problem(HttpStatusCode status, String title, String detail) {
        ProblemDetail pd = ProblemDetail.forStatusAndDetail(status, detail);
        pd.setTitle(title);
        return pd;
    }
}
