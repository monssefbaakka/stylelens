package com.stylelens.catalog.web;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.stylelens.catalog.inference.Embedding;
import com.stylelens.catalog.inference.ImageRejectedException;
import com.stylelens.catalog.inference.InferenceClient;
import com.stylelens.catalog.inference.InferenceUnavailableException;
import com.stylelens.catalog.product.CategoryCount;
import com.stylelens.catalog.product.Product;
import com.stylelens.catalog.product.ProductFilters;
import com.stylelens.catalog.product.ProductRepository;
import com.stylelens.catalog.product.ScoredProduct;
import com.stylelens.catalog.search.SearchService;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(CatalogController.class)
@Import(SearchService.class)
@TestPropertySource(properties = "stylelens.embedding-dim=2")
class CatalogControllerTest {

    private static final Product SNEAKER = new Product(42, "White Sneaker", "Men", "Footwear", "Shoes",
            "Casual Shoes", "White", "images/42.jpg");
    private static final MockMultipartFile PHOTO =
            new MockMultipartFile("image", "photo.jpg", "image/jpeg", new byte[] {1, 2, 3});

    @Autowired
    private MockMvc mvc;

    @MockitoBean
    private InferenceClient inference;

    @MockitoBean
    private ProductRepository products;

    @Test
    void searchEmbedsImageThenQueriesCatalogWithFilter() throws Exception {
        when(inference.embed(any(), eq("photo.jpg"), eq("image/jpeg")))
                .thenReturn(new Embedding(new float[] {0.6f, 0.8f}, 2, "resnet50", true, 10));
        when(products.findSimilar("[0.6,0.8]", new ProductFilters("Footwear", null, null, null, null), null, 5))
                .thenReturn(List.of(new ScoredProduct(SNEAKER, 0.912345)));
        when(products.catalogMeta()).thenReturn(Map.of("model", "resnet50", "fine_tuned", "true"));

        mvc.perform(multipart("/api/search").file(PHOTO).param("category", " Footwear ").param("k", "5"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].id").value(42))
                .andExpect(jsonPath("$.results[0].name").value("White Sneaker"))
                .andExpect(jsonPath("$.results[0].imageUrl").value("/images/42.jpg"))
                .andExpect(jsonPath("$.results[0].score").value(0.9123))
                .andExpect(jsonPath("$.query.filters.category").value("Footwear"))
                .andExpect(jsonPath("$.query.filters.empty").doesNotExist())
                .andExpect(jsonPath("$.query.k").value(5))
                .andExpect(jsonPath("$.timings.embeddingMs").isNumber());
    }

    @Test
    void searchUsesDefaultKWithoutCategory() throws Exception {
        when(inference.embed(any(), any(), any())).thenReturn(new Embedding(new float[] {1f, 0f}, 2, "m", false, 1));
        when(products.findSimilar(any(), eq(ProductFilters.NONE), isNull(), eq(12))).thenReturn(List.of());

        mvc.perform(multipart("/api/search").file(PHOTO).param("category", ""))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results").isEmpty())
                .andExpect(jsonPath("$.query.k").value(12));
    }

    @Test
    void searchCombinesAllFilters() throws Exception {
        when(inference.embed(any(), any(), any())).thenReturn(new Embedding(new float[] {1f, 0f}, 2, "m", false, 1));
        var filters = new ProductFilters("Footwear", "Shoes", "Casual Shoes", "Men", "White");
        when(products.findSimilar("[1.0,0.0]", filters, null, 12)).thenReturn(List.of(new ScoredProduct(SNEAKER, 0.8)));

        mvc.perform(multipart("/api/search").file(PHOTO)
                        .param("category", "Footwear").param("subCategory", "Shoes")
                        .param("articleType", "Casual Shoes").param("gender", "Men").param("colour", "White"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].id").value(42))
                .andExpect(jsonPath("$.query.filters.articleType").value("Casual Shoes"))
                .andExpect(jsonPath("$.query.filters.colour").value("White"));
    }

    @Test
    void similarAcceptsFilters() throws Exception {
        when(products.findEmbedding(42)).thenReturn(Optional.of("[0.6,0.8]"));
        when(products.findSimilar("[0.6,0.8]", new ProductFilters(null, null, null, "Women", "Red"), 42L, 12))
                .thenReturn(List.of());
        mvc.perform(get("/api/products/42/similar").param("gender", "Women").param("colour", "Red"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.query.filters.gender").value("Women"));
    }

    @Test
    void filtersEndpointReturnsFacetsForCurrentSelection() throws Exception {
        var selection = new ProductFilters("Footwear", null, null, null, null);
        when(products.facets(selection)).thenReturn(Map.of(
                "category", List.of(new CategoryCount("Apparel", 10), new CategoryCount("Footwear", 4)),
                "articleType", List.of(new CategoryCount("Casual Shoes", 3))));

        mvc.perform(get("/api/filters").param("category", "Footwear"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.category[1].name").value("Footwear"))
                .andExpect(jsonPath("$.articleType[0].count").value(3));
    }

    @Test
    void rejectsOverlongFilterValue() throws Exception {
        mvc.perform(get("/api/filters").param("colour", "x".repeat(101)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.detail").value(org.hamcrest.Matchers.containsString("trop longue")));
    }

    @Test
    void rejectsInvalidK() throws Exception {
        mvc.perform(multipart("/api/search").file(PHOTO).param("k", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_PROBLEM_JSON))
                .andExpect(jsonPath("$.detail").value("k doit être compris entre 1 et 60"));
        verify(inference, never()).embed(any(), any(), any());
    }

    @Test
    void rejectsMissingImagePart() throws Exception {
        mvc.perform(multipart("/api/search")).andExpect(status().isBadRequest());
    }

    @Test
    void rejectsUnsupportedImageType() throws Exception {
        var gif = new MockMultipartFile("image", "a.gif", "image/gif", new byte[] {1});
        mvc.perform(multipart("/api/search").file(gif))
                .andExpect(status().isUnsupportedMediaType())
                .andExpect(jsonPath("$.title").value("Format d'image non accepté"));
    }

    @Test
    void forwardsImageRejectionFromInferenceService() throws Exception {
        when(inference.embed(any(), any(), any())).thenThrow(new ImageRejectedException(400, "Fichier corrompu"));
        mvc.perform(multipart("/api/search").file(PHOTO))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.detail").value("Fichier corrompu"));
    }

    @Test
    void reportsUnavailableInferenceService() throws Exception {
        when(inference.embed(any(), any(), any()))
                .thenThrow(new InferenceUnavailableException("Service IA injoignable", null));
        mvc.perform(multipart("/api/search").file(PHOTO))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.title").value("Service IA indisponible"));
    }

    @Test
    void rejectsEmbeddingOfWrongDimension() throws Exception {
        when(inference.embed(any(), any(), any()))
                .thenReturn(new Embedding(new float[] {1f, 0f, 0f}, 3, "resnet50", false, 1));
        mvc.perform(multipart("/api/search").file(PHOTO))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.detail").value(org.hamcrest.Matchers.containsString("dimension 3")));
        verify(products, never()).findSimilar(any(), any(), any(), anyInt());
    }

    @Test
    void similarUsesStoredEmbeddingAndExcludesProduct() throws Exception {
        when(products.findEmbedding(42)).thenReturn(Optional.of("[0.6,0.8]"));
        when(products.findSimilar("[0.6,0.8]", ProductFilters.NONE, 42L, 3)).thenReturn(List.of(new ScoredProduct(SNEAKER, 0.5)));

        mvc.perform(get("/api/products/42/similar").param("k", "3"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.query.productId").value(42))
                .andExpect(jsonPath("$.timings.embeddingMs").doesNotExist());
        verify(inference, never()).embed(any(), any(), any());
    }

    @Test
    void unknownProductIs404() throws Exception {
        when(products.findEmbedding(7)).thenReturn(Optional.empty());
        when(products.findById(7)).thenReturn(Optional.empty());
        mvc.perform(get("/api/products/7/similar")).andExpect(status().isNotFound());
        mvc.perform(get("/api/products/7")).andExpect(status().isNotFound());
    }

    @Test
    void productAndCategories() throws Exception {
        when(products.findById(42)).thenReturn(Optional.of(SNEAKER));
        when(products.categories()).thenReturn(List.of(new CategoryCount("Apparel", 10), new CategoryCount("Footwear", 4)));

        mvc.perform(get("/api/products/42"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.articleType").value("Casual Shoes"))
                .andExpect(jsonPath("$.imageUrl").value("/images/42.jpg"))
                .andExpect(jsonPath("$.imagePath").doesNotExist());
        mvc.perform(get("/api/categories"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].name").value("Apparel"))
                .andExpect(jsonPath("$[1].count").value(4));
    }

    @Test
    void infoReportsInferenceDown() throws Exception {
        when(products.count()).thenReturn(3L);
        when(products.catalogMeta()).thenReturn(Map.of("model", "resnet50"));
        when(inference.health()).thenThrow(new InferenceUnavailableException("Service IA injoignable", null));

        mvc.perform(get("/api/info"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.catalog.products").value(3))
                .andExpect(jsonPath("$.catalog.embeddings.model").value("resnet50"))
                .andExpect(jsonPath("$.inference.status").value("DOWN"));
    }
}
