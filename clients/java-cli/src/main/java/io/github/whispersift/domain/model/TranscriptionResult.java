package io.github.whispersift.domain.model;

import java.nio.file.Path;
import java.util.List;

public record TranscriptionResult(List<Path> generatedFiles) {
}
