package com.stylelens.catalog;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.stylelens.catalog.inference.Embedding;
import com.stylelens.catalog.inference.InferenceClient;
import com.stylelens.catalog.product.ProductFilters;
import com.stylelens.catalog.product.ProductRepository;
import com.stylelens.catalog.product.ScoredProduct;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Import;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Bout en bout sur un vrai PostgreSQL + pgvector (Testcontainers) : migration Flyway, index HNSW,
 * kNN cosinus, filtre de catégorie, images. Le service IA est simulé. Ignoré sans Docker.
 */
@SpringBootTest(properties = "stylelens.embedding-dim=3")
@AutoConfigureMockMvc
@Import(TestcontainersConfiguration.class)
@Testcontainers(disabledWithoutDocker = true)
class CatalogApiIntegrationTests {

    @TempDir
    static Path images;

    @DynamicPropertySource
    static void imagesDir(DynamicPropertyRegistry registry) {
        registry.add("stylelens.images.dir", () -> images.toString());
    }

    @Autowired
    private MockMvc mvc;

    @Autowired
    private JdbcClient jdbc;

    @Autowired
    private ProductRepository products;

    @MockitoBean
    private InferenceClient inference;

    @BeforeEach
    void seed() throws Exception {
        jdbc.sql("TRUNCATE products").update();
        jdbc.sql("DELETE FROM catalog_meta").update();
        insert(1, "Red Tee", "Apparel", "Topwear", "Tshirts", "Men", "Red", "[1,0,0]");
        insert(2, "Pink Tee", "Apparel", "Topwear", "Tshirts", "Women", "Pink", "[0.9,0.1,0]");
        insert(3, "Red Sneaker", "Footwear", "Shoes", "Casual Shoes", "Men", "Red", "[0.8,0,0.2]");
        insert(4, "Blue Watch", "Accessories", "Watches", "Watches", "Women", "Blue", "[0,0,1]");
        insert(5, "Red Dress", "Apparel", "Dress", "Dresses", "Women", "Red", "[0.7,0.3,0]");
        jdbc.sql("INSERT INTO catalog_meta (key, value) VALUES ('model', 'resnet50'), ('fine_tuned', 'true')").update();

        Files.createDirectories(images.resolve("images"));
        Files.write(images.resolve("images/1.jpg"), new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
    }

    private void insert(long id, String name, String master, String sub, String article, String gender,
            String colour, String vector) {
        jdbc.sql("""
                INSERT INTO products (id, name, master_category, sub_category, article_type, gender, base_colour,
                                      image_path, split, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'train', CAST(? AS vector))
                """).params(id, name, master, sub, article, gender, colour, "images/" + id + ".jpg", vector).update();
    }

    @Test
    void migrationCreatesHnswCosineIndex() {
        String indexdef = jdbc.sql("SELECT indexdef FROM pg_indexes WHERE indexname = 'products_embedding_hnsw'")
                .query(String.class).single();
        assertThat(indexdef).contains("USING hnsw").contains("vector_cosine_ops");
    }

    @Test
    void knnReturnsNearestFirstWithCosineScore() {
        List<ScoredProduct> hits = products.findSimilar("[1,0,0]", ProductFilters.NONE, null, 3);
        assertThat(hits).extracting(h -> h.product().id()).containsExactly(1L, 2L, 3L);
        assertThat(hits.get(0).score()).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(hits.get(1).score()).isLessThan(hits.get(0).score());
    }

    @Test
    void searchByImageAppliesCategoryFilter() throws Exception {
        when(inference.embed(any(), any(), any()))
                .thenReturn(new Embedding(new float[] {1f, 0f, 0f}, 3, "resnet50", true, 5));
        var photo = new MockMultipartFile("image", "q.png", "image/png", new byte[] {1});

        mvc.perform(multipart("/api/search").file(photo).param("category", "Footwear").param("k", "5"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results.length()").value(1))
                .andExpect(jsonPath("$.results[0].id").value(3))
                .andExpect(jsonPath("$.results[0].masterCategory").value("Footwear"));

        mvc.perform(multipart("/api/search").file(photo).param("k", "2"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].id").value(1))
                .andExpect(jsonPath("$.results[1].id").value(2));
    }

    @Test
    void similarExcludesTheProductItself() throws Exception {
        mvc.perform(get("/api/products/1/similar").param("k", "2"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results[0].id").value(2))
                .andExpect(jsonPath("$.results[1].id").value(3));
    }

    @Test
    void categoriesAreCounted() throws Exception {
        mvc.perform(get("/api/categories"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].name").value("Apparel"))
                .andExpect(jsonPath("$[0].count").value(3))
                .andExpect(jsonPath("$.length()").value(3));
    }

    @Test
    void combinedFiltersRestrictKnnResults() {
        List<ScoredProduct> hits = products.findSimilar("[1,0,0]", new ProductFilters(null, null, null, "Women", "Red"), null, 5);
        assertThat(hits).extracting(h -> h.product().id()).containsExactly(5L);

        hits = products.findSimilar("[1,0,0]", new ProductFilters("Apparel", null, null, "Men", null), null, 5);
        assertThat(hits).extracting(h -> h.product().id()).containsExactly(1L);

        hits = products.findSimilar("[1,0,0]", new ProductFilters(null, null, "Tshirts", null, null), 1L, 5);
        assertThat(hits).extracting(h -> h.product().id()).containsExactly(2L);
    }

    @Test
    void facetsCountEachDimensionWithTheOtherFilters() throws Exception {
        // colour=Red : chaque dimension est comptée parmi les produits rouges, sauf la couleur elle-même
        mvc.perform(get("/api/filters").param("colour", "Red"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.category[0].name").value("Apparel"))
                .andExpect(jsonPath("$.category[0].count").value(2))
                .andExpect(jsonPath("$.category[1].name").value("Footwear"))
                .andExpect(jsonPath("$.category.length()").value(2))
                .andExpect(jsonPath("$.gender[0].name").value("Men"))
                .andExpect(jsonPath("$.gender[0].count").value(2))
                .andExpect(jsonPath("$.colour.length()").value(3))
                .andExpect(jsonPath("$.colour[0].name").value("Red"))
                .andExpect(jsonPath("$.colour[0].count").value(3))
                .andExpect(jsonPath("$.articleType.length()").value(3))
                .andExpect(jsonPath("$.subCategory.length()").value(3));
    }

    @Test
    void searchByImageCombinesFilters() throws Exception {
        when(inference.embed(any(), any(), any()))
                .thenReturn(new Embedding(new float[] {1f, 0f, 0f}, 3, "resnet50", true, 5));
        var photo = new MockMultipartFile("image", "q.png", "image/png", new byte[] {1});

        mvc.perform(multipart("/api/search").file(photo).param("gender", "Women").param("category", "Apparel"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results.length()").value(2))
                .andExpect(jsonPath("$.results[0].id").value(2))
                .andExpect(jsonPath("$.results[1].id").value(5));
    }

    @Test
    void servesCatalogImages() throws Exception {
        mvc.perform(get("/images/1.jpg")).andExpect(status().isOk());
        mvc.perform(get("/images/999.jpg")).andExpect(status().isNotFound());
    }
}
