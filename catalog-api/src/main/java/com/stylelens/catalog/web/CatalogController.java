package com.stylelens.catalog.web;

import com.stylelens.catalog.inference.InferenceClient;
import com.stylelens.catalog.inference.InferenceUnavailableException;
import com.stylelens.catalog.product.CategoryCount;
import com.stylelens.catalog.product.ProductFilters;
import com.stylelens.catalog.product.ProductRepository;
import com.stylelens.catalog.search.ProductView;
import com.stylelens.catalog.search.SearchResponse;
import com.stylelens.catalog.search.SearchService;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class CatalogController {

    private final SearchService search;
    private final ProductRepository products;
    private final InferenceClient inference;

    public CatalogController(SearchService search, ProductRepository products, InferenceClient inference) {
        this.search = search;
        this.products = products;
        this.inference = inference;
    }

    /**
     * Recherche par image : multipart {@code image} ; filtres facultatifs combinés par ET :
     * {@code category}, {@code subCategory}, {@code articleType}, {@code gender}, {@code colour}.
     */
    @PostMapping(path = "/search", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public SearchResponse search(
            @RequestPart("image") MultipartFile image,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String subCategory,
            @RequestParam(required = false) String articleType,
            @RequestParam(required = false) String gender,
            @RequestParam(required = false) String colour,
            @RequestParam(required = false) Integer k) {
        return search.searchByImage(image, new ProductFilters(category, subCategory, articleType, gender, colour), k);
    }

    @GetMapping("/products/{id}")
    public ProductView product(@PathVariable long id) {
        return search.product(id);
    }

    /** Articles similaires à un produit du catalogue (son embedding est déjà indexé). */
    @GetMapping("/products/{id}/similar")
    public SearchResponse similar(
            @PathVariable long id,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String subCategory,
            @RequestParam(required = false) String articleType,
            @RequestParam(required = false) String gender,
            @RequestParam(required = false) String colour,
            @RequestParam(required = false) Integer k) {
        return search.similarTo(id, new ProductFilters(category, subCategory, articleType, gender, colour), k);
    }

    @GetMapping("/categories")
    public List<CategoryCount> categories() {
        return search.categories();
    }

    /**
     * Valeurs possibles de chaque filtre et nombre de produits correspondants, compte tenu des
     * autres filtres déjà choisis (mêmes paramètres que la recherche).
     */
    @GetMapping("/filters")
    public Map<String, List<CategoryCount>> filters(
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String subCategory,
            @RequestParam(required = false) String articleType,
            @RequestParam(required = false) String gender,
            @RequestParam(required = false) String colour) {
        return search.facets(new ProductFilters(category, subCategory, articleType, gender, colour));
    }

    /** Catalogue chargé (taille, modèle des embeddings) et modèle servi par le service IA. */
    @GetMapping("/info")
    public Map<String, Object> info() {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("products", products.count());
        catalog.put("embeddings", products.catalogMeta());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("catalog", catalog);
        try {
            body.put("inference", inference.health());
        } catch (InferenceUnavailableException e) {
            body.put("inference", Map.of("status", "DOWN", "error", e.getMessage()));
        }
        return body;
    }
}
