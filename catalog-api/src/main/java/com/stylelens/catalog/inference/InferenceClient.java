package com.stylelens.catalog.inference;

import java.util.Map;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClientException;

/** Client du service IA (inference-service) : image → embedding. */
public class InferenceClient {

    private final RestClient rest;

    public InferenceClient(RestClient rest) {
        this.rest = rest;
    }

    /**
     * Embedding normalisé L2 de l'image.
     *
     * @throws ImageRejectedException si le service IA refuse l'image (illisible, trop lourde…)
     * @throws InferenceUnavailableException si le service IA est injoignable ou en erreur
     */
    public Embedding embed(byte[] image, String filename, String contentType) {
        String name = filename == null || filename.isBlank() ? "image" : filename;
        HttpHeaders partHeaders = new HttpHeaders();
        partHeaders.setContentType(contentType == null ? MediaType.APPLICATION_OCTET_STREAM : MediaType.parseMediaType(contentType));
        partHeaders.setContentDispositionFormData("file", name);
        MultiValueMap<String, Object> parts = new LinkedMultiValueMap<>();
        parts.add("file", new HttpEntity<>(new NamedResource(image, name), partHeaders));
        try {
            return rest.post()
                    .uri("/embed")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(parts)
                    .exchange((request, response) -> {
                        HttpStatusCode status = response.getStatusCode();
                        if (status.is2xxSuccessful()) {
                            Embedding body = response.bodyTo(Embedding.class);
                            if (body == null || body.embedding() == null) {
                                throw new InferenceUnavailableException("Réponse vide du service IA", null);
                            }
                            return body;
                        }
                        if (status.is4xxClientError()) {
                            throw new ImageRejectedException(status.value(), detail(response.bodyTo(Map.class)));
                        }
                        throw new InferenceUnavailableException("Service IA en erreur (HTTP " + status.value() + ")", null);
                    });
        } catch (ResourceAccessException e) {
            throw new InferenceUnavailableException("Service IA injoignable", e);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("Réponse invalide du service IA", e);
        }
    }

    /** État et modèle du service IA ({@code GET /health}). */
    public ModelInfo health() {
        try {
            return rest.get().uri("/health").retrieve().body(ModelInfo.class);
        } catch (RestClientException e) {
            throw new InferenceUnavailableException("Service IA injoignable", e);
        }
    }

    private static String detail(Map<?, ?> body) {
        Object detail = body == null ? null : body.get("detail");
        return detail instanceof String s && !s.isBlank() ? s : "Image refusée par le service IA";
    }

    /** Ressource multipart avec nom de fichier (exigé pour un champ fichier). */
    private static final class NamedResource extends ByteArrayResource {
        private final String filename;

        NamedResource(byte[] bytes, String filename) {
            super(bytes);
            this.filename = filename;
        }

        @Override
        public String getFilename() {
            return filename;
        }
    }
}
