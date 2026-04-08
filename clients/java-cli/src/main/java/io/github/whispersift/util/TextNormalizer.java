package io.github.whispersift.util;

public final class TextNormalizer {

    private TextNormalizer() {
    }

    public static String normalizeWhitespace(String value) {
        String normalized = value.replace("\r\n", "\n").replace("\r", "\n");
        normalized = normalized.replaceAll("[ \\t]+", " ");
        normalized = normalized.replaceAll(" ?\\n ?", "\n");
        return normalized.trim();
    }
}
