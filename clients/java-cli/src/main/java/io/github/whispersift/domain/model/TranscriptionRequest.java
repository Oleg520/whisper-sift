package io.github.whispersift.domain.model;

import java.nio.file.Path;
import java.util.List;

public record TranscriptionRequest(
        List<Path> files,
        Path outputDir,
        String model,
        String language,
        String device,
        List<String> formats
) {
}
