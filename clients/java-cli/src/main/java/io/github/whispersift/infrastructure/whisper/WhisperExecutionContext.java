package io.github.whispersift.infrastructure.whisper;

import java.nio.file.Path;

public record WhisperExecutionContext(
        Path projectRoot,
        Path launcherScript,
        String pythonExecutable
) {
}
