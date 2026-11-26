package com.stylelens.catalog.product;

import com.stylelens.catalog.config.StylelensProperties;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

/**
 * Accès à {@code products} : kNN cosinus servi par l'index HNSW de pgvector (P4).
 *
 * <p>Le filtre de catégorie est dans la même requête que le kNN. Avec un filtre, le parcours
 * itératif de pgvector (≥ 0.8, {@code hnsw.iterative_scan = relaxed_order}) continue d'explorer le
 * graphe tant qu'il manque des résultats ; l'ordre peut alors être légèrement approximatif, d'où
 * le tri final par score.
 */
@Repository
public class ProductRepository {

    private static final String COLUMNS =
            "id, name, gender, master_category, sub_category, article_type, base_colour, image_path";

    private final JdbcClient jdbc;
    private final int efSearch;

    public ProductRepository(JdbcClient jdbc, StylelensProperties properties) {
        this.jdbc = jdbc;
        this.efSearch = properties.search().efSearch();
    }

    /**
     * Les {@code k} produits les plus proches de {@code vector} (littéral pgvector).
     *
     * @param category  {@code master_category} imposée, ou {@code null}
     * @param excludeId produit à exclure (recherche « articles similaires »), ou {@code null}
     */
    @Transactional(readOnly = true)
    public List<ScoredProduct> findSimilar(String vector, String category, Long excludeId, int k) {
        setLocal("hnsw.ef_search", Integer.toString(efSearch));
        if (category != null) {
            setLocal("hnsw.iterative_scan", "relaxed_order");
        }

        List<String> where = new ArrayList<>();
        List<Object> params = new ArrayList<>();
        params.add(vector);
        if (category != null) {
            where.add("master_category = ?");
            params.add(category);
        }
        if (excludeId != null) {
            where.add("id <> ?");
            params.add(excludeId);
        }
        params.add(vector);
        params.add(k);

        String sql = "SELECT " + COLUMNS + ", embedding <=> CAST(? AS vector) AS distance FROM products"
                + (where.isEmpty() ? "" : " WHERE " + String.join(" AND ", where))
                + " ORDER BY embedding <=> CAST(? AS vector) LIMIT ?";
        List<ScoredProduct> results = new ArrayList<>(jdbc.sql(sql)
                .params(params)
                .query((rs, n) -> new ScoredProduct(product(rs), 1.0 - rs.getDouble("distance")))
                .list());
        results.sort(Comparator.comparingDouble(ScoredProduct::score).reversed());
        return results;
    }

    public Optional<Product> findById(long id) {
        return jdbc.sql("SELECT " + COLUMNS + " FROM products WHERE id = ?")
                .param(id)
                .query((rs, n) -> product(rs))
                .optional();
    }

    /** Embedding d'un produit, au format texte pgvector (réutilisable tel quel dans une requête). */
    public Optional<String> findEmbedding(long id) {
        return jdbc.sql("SELECT embedding::text FROM products WHERE id = ?").param(id).query(String.class).optional();
    }

    public List<CategoryCount> categories() {
        return jdbc.sql("SELECT master_category, count(*) AS n FROM products GROUP BY master_category ORDER BY n DESC, master_category")
                .query((rs, n) -> new CategoryCount(rs.getString(1), rs.getLong(2)))
                .list();
    }

    public long count() {
        return jdbc.sql("SELECT count(*) FROM products").query(Long.class).single();
    }

    /** Modèle ayant produit les embeddings ({@code catalog_meta}, écrit par l'import P4). */
    public Map<String, String> catalogMeta() {
        Map<String, String> meta = new LinkedHashMap<>();
        jdbc.sql("SELECT key, value FROM catalog_meta ORDER BY key")
                .query((ResultSet rs) -> {
                    meta.put(rs.getString(1), rs.getString(2));
                });
        return meta;
    }

    private void setLocal(String name, String value) {
        jdbc.sql("SELECT set_config(?, ?, true)").param(name).param(value).query(String.class).single();
    }

    private static Product product(ResultSet rs) throws SQLException {
        return new Product(
                rs.getLong("id"),
                rs.getString("name"),
                rs.getString("gender"),
                rs.getString("master_category"),
                rs.getString("sub_category"),
                rs.getString("article_type"),
                rs.getString("base_colour"),
                rs.getString("image_path"));
    }
}
