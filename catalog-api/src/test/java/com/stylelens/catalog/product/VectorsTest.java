package com.stylelens.catalog.product;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class VectorsTest {

    @Test
    void formatsPgvectorLiteral() {
        assertThat(Vectors.toLiteral(new float[] {0.5f, -1f, 1.25e-3f})).isEqualTo("[0.5,-1.0,0.00125]");
    }

    @Test
    void keepsFloatPrecision() {
        float v = 0.12345679f;
        String literal = Vectors.toLiteral(new float[] {v});
        assertThat(Float.parseFloat(literal.substring(1, literal.length() - 1))).isEqualTo(v);
    }

    @Test
    void rejectsEmptyOrNonFiniteVectors() {
        assertThatThrownBy(() -> Vectors.toLiteral(new float[0])).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> Vectors.toLiteral(new float[] {1f, Float.NaN}))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
