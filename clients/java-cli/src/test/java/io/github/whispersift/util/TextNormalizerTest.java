package io.github.whispersift.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class TextNormalizerTest {

    @Test
    void normalizeWhitespaceCollapsesSpacesAndNormalizesNewlines() {
        String value = "  one   two\r\n\r\n three \t four \r five  ";

        String normalized = TextNormalizer.normalizeWhitespace(value);

        assertEquals("one two\n\nthree four\nfive", normalized);
    }
}
