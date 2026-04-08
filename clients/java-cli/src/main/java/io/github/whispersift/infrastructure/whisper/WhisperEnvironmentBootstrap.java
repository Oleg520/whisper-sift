package io.github.whispersift.infrastructure.whisper;

import java.nio.file.Files;
import java.nio.file.Path;

public final class WhisperEnvironmentBootstrap {

    public WhisperExecutionContext bootstrap(Path explicitProjectRoot, String pythonExecutable) {
        Path projectRoot = explicitProjectRoot != null
                ? explicitProjectRoot.toAbsolutePath().normalize()
                : findProjectRoot();

        Path launcherScript = projectRoot.resolve("transcribe_whisper.py");
        if (!Files.exists(launcherScript)) {
            throw new IllegalStateException("Could not find transcribe_whisper.py at " + launcherScript);
        }

        return new WhisperExecutionContext(projectRoot, launcherScript, pythonExecutable);
    }

    private Path findProjectRoot() {
        Path current = Path.of("").toAbsolutePath().normalize();
        for (Path cursor = current; cursor != null; cursor = cursor.getParent()) {
            if (Files.exists(cursor.resolve("transcribe_whisper.py"))
                    && Files.exists(cursor.resolve("pyproject.toml"))) {
                return cursor;
            }
        }

        throw new IllegalStateException(
                "Could not auto-detect project root. Use --project-root to specify it explicitly."
        );
    }
}
