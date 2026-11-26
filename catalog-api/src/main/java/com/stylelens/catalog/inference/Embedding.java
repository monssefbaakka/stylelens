package com.stylelens.catalog.inference;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Réponse de {@code POST /embed}. */
public record Embedding(
        float[] embedding,
        int dim,
        String model,
        @JsonProperty("fine_tuned") boolean fineTuned,
        @JsonProperty("elapsed_ms") double elapsedMs) {
}
