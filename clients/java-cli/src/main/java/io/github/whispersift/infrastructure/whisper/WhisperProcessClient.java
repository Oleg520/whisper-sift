package io.github.whispersift.infrastructure.whisper;

import java.nio.file.Path;
import java.util.List;

public final class WhisperProcessClient {

    public void execute(List<String> command, Path workingDirectory) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(workingDirectory.toFile());
        processBuilder.inheritIO();

        Process process = processBuilder.start();
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IllegalStateException("Whisper process failed with exit code " + exitCode);
        }
    }
}
