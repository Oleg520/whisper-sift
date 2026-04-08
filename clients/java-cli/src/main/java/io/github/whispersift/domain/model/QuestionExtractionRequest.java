package io.github.whispersift.domain.model;

import java.nio.file.Path;
import java.util.List;

public record QuestionExtractionRequest(
        List<Path> files,
        Path outputDir,
        String suffix,
        boolean deduplicate,
        int minLength,
        int maxLength
) {
}
