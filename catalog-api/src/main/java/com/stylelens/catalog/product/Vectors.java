package com.stylelens.catalog.product;

/** Conversion vers le format texte de pgvector : {@code [0.1,0.2,…]}. */
public final class Vectors {

    private Vectors() {
    }

    public static String toLiteral(float[] vector) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException("Vecteur vide");
        }
        StringBuilder sb = new StringBuilder(vector.length * 12).append('[');
        for (int i = 0; i < vector.length; i++) {
            float v = vector[i];
            if (!Float.isFinite(v)) {
                throw new IllegalArgumentException("Composante non finie à l'indice " + i);
            }
            if (i > 0) {
                sb.append(',');
            }
            sb.append(v);
        }
        return sb.append(']').toString();
    }
}
