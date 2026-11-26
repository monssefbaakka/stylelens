package com.stylelens.catalog.inference;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Réponse de {@code GET /health}. */
public record ModelInfo(
        String status,
        String model,
        @JsonProperty("fine_tuned") boolean fineTuned,
        int dim,
        @JsonProperty("image_size") int imageSize,
        String device) {
}
