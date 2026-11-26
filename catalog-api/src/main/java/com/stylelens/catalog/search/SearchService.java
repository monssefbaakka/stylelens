package com.stylelens.catalog.search;

import com.stylelens.catalog.config.StylelensProperties;
import com.stylelens.catalog.inference.Embedding;
import com.stylelens.catalog.inference.InferenceClient;
import com.stylelens.catalog.product.CategoryCount;
import com.stylelens.catalog.product.Product;
import com.stylelens.catalog.product.ProductRepository;
import com.stylelens.catalog.product.ScoredProduct;
import com.stylelens.catalog.product.Vectors;
import com.stylelens.catalog.search.SearchExceptions.BadRequestException;
import com.stylelens.catalog.search.SearchExceptions.IncompatibleModelException;
import com.stylelens.catalog.search.SearchExceptions.ProductNotFoundException;
import com.stylelens.catalog.search.SearchExceptions.UnsupportedImageTypeException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/** Orchestration de la recherche : image → service IA (embedding) → kNN pgvector avec filtre. */
@Service
public class SearchService {

    static final Set<String> ACCEPTED_TYPES = Set.of("image/jpeg", "image/png", "image/webp");
    private static final Logger log = LoggerFactory.getLogger(SearchService.class);

    private final InferenceClient inference;
    private final ProductRepository products;
    private final StylelensProperties properties;
    private final AtomicBoolean modelChecked = new AtomicBoolean();

    public SearchService(InferenceClient inference, ProductRepository products, StylelensProperties properties) {
        this.inference = inference;
        this.products = products;
        this.properties = properties;
    }

    public SearchResponse searchByImage(MultipartFile image, String category, Integer k) {
        int limit = resolveK(k);
        String cat = normalize(category);
        if (image == null || image.isEmpty()) {
            throw new BadRequestException("Image manquante (champ multipart « image »)");
        }
        String type = image.getContentType() == null ? "" : image.getContentType().toLowerCase(Locale.ROOT);
        if (!ACCEPTED_TYPES.contains(type)) {
            throw new UnsupportedImageTypeException(
                    "Formats acceptés : JPEG, PNG, WebP (reçu : " + (type.isEmpty() ? "inconnu" : type) + ")");
        }

        long t0 = System.nanoTime();
        Embedding embedding = inference.embed(bytes(image), image.getOriginalFilename(), type);
        checkModel(embedding);
        double embeddingMs = millis(t0);

        long t1 = System.nanoTime();
        List<ScoredProduct> hits = products.findSimilar(Vectors.toLiteral(embedding.embedding()), cat, null, limit);
        double searchMs = millis(t1);
        return new SearchResponse(items(hits), new SearchResponse.Query(cat, limit, null),
                new SearchResponse.Timings(round(embeddingMs), round(searchMs), round(millis(t0))));
    }

    /** Articles ressemblant à un produit du catalogue : son embedding est déjà en base, pas d'appel au service IA. */
    public SearchResponse similarTo(long productId, String category, Integer k) {
        int limit = resolveK(k);
        String cat = normalize(category);
        long t0 = System.nanoTime();
        String vector = products.findEmbedding(productId).orElseThrow(() -> new ProductNotFoundException(productId));
        List<ScoredProduct> hits = products.findSimilar(vector, cat, productId, limit);
        double ms = round(millis(t0));
        return new SearchResponse(items(hits), new SearchResponse.Query(cat, limit, productId),
                new SearchResponse.Timings(null, ms, ms));
    }

    public ProductView product(long id) {
        Product p = products.findById(id).orElseThrow(() -> new ProductNotFoundException(id));
        return ProductView.of(p, imageUrl(p));
    }

    public List<CategoryCount> categories() {
        return products.categories();
    }

    String imageUrl(Product p) {
        String path = p.imagePath().startsWith("/") ? p.imagePath().substring(1) : p.imagePath();
        return properties.images().baseUrl() + path;
    }

    private List<SearchResponse.Item> items(List<ScoredProduct> hits) {
        return hits.stream().map(h -> SearchResponse.Item.of(h.product(), imageUrl(h.product()), h.score())).toList();
    }

    private int resolveK(Integer k) {
        int max = properties.search().maxK();
        int value = k == null ? properties.search().defaultK() : k;
        if (value < 1 || value > max) {
            throw new BadRequestException("k doit être compris entre 1 et " + max);
        }
        return value;
    }

    /**
     * La dimension doit correspondre à la colonne {@code vector(n)}, sinon PostgreSQL refuse la requête.
     * Un autre modèle de même dimension donnerait des résultats sans rapport avec l'image : l'écart avec
     * {@code catalog_meta} est signalé une fois dans les journaux.
     */
    private void checkModel(Embedding e) {
        int expected = properties.embeddingDim();
        if (e.embedding().length != expected) {
            throw new IncompatibleModelException("Le service IA renvoie des vecteurs de dimension "
                    + e.embedding().length + ", le catalogue attend " + expected
                    + " : servir le modèle qui a produit les embeddings du catalogue");
        }
        if (modelChecked.compareAndSet(false, true)) {
            Map<String, String> meta = products.catalogMeta();
            String model = meta.get("model");
            String fineTuned = meta.get("fine_tuned");
            if (model != null && (!Objects.equals(model, e.model())
                    || !Objects.equals(fineTuned, Boolean.toString(e.fineTuned())))) {
                log.warn("Modèle du service IA ({}, fine-tuné : {}) différent de celui du catalogue ({}, fine-tuné : {})",
                        e.model(), e.fineTuned(), model, fineTuned);
            }
        }
    }

    private static String normalize(String category) {
        return category == null || category.isBlank() ? null : category.trim();
    }

    private static byte[] bytes(MultipartFile file) {
        try {
            return file.getBytes();
        } catch (IOException e) {
            throw new UncheckedIOException("Lecture de l'image impossible", e);
        }
    }

    private static double millis(long since) {
        return (System.nanoTime() - since) / 1_000_000.0;
    }

    private static double round(double ms) {
        return Math.round(ms * 100) / 100.0;
    }
}
