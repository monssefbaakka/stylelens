package com.stylelens.catalog.config;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.nio.file.Path;
import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

/** Configuration {@code stylelens.*} (voir application.yml). */
@Validated
@ConfigurationProperties("stylelens")
public record StylelensProperties(
        @Min(1) @Max(2000) int embeddingDim,
        @Valid @NotNull Inference inference,
        @Valid @NotNull Search search,
        @Valid @NotNull Images images) {

    public record Inference(@NotBlank String baseUrl, @NotNull Duration connectTimeout, @NotNull Duration readTimeout) {
    }

    public record Search(@Min(1) int defaultK, @Min(1) int maxK, @Min(1) @Max(1000) int efSearch) {
    }

    /** {@code dir} : dossier du jeu de données ; URL d'une image = {@code baseUrl + image_path}. */
    public record Images(@NotNull Path dir, @NotNull String baseUrl) {
    }
}
