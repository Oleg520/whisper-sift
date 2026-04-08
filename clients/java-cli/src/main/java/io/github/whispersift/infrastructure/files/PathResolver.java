package io.github.whispersift.infrastructure.files;

import java.nio.file.Path;

public final class PathResolver {

    public Path buildQuestionOutputPath(Path source, Path outputDir, String suffix) {
        Path targetDir = outputDir != null ? outputDir.toAbsolutePath().normalize() : source.getParent();
        String fileName = source.getFileName().toString();
        int dotIndex = fileName.lastIndexOf('.');
        String stem = dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
        return targetDir.resolve(stem + suffix);
    }
}
