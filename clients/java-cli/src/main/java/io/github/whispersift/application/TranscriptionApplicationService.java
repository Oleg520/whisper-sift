package io.github.whispersift.application;

import io.github.whispersift.domain.model.TranscriptionRequest;
import io.github.whispersift.domain.model.TranscriptionResult;
import io.github.whispersift.infrastructure.whisper.WhisperCommandBuilder;
import io.github.whispersift.infrastructure.whisper.WhisperEnvironmentBootstrap;
import io.github.whispersift.infrastructure.whisper.WhisperExecutionContext;
import io.github.whispersift.infrastructure.whisper.WhisperProcessClient;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class TranscriptionApplicationService {

    private final WhisperEnvironmentBootstrap environmentBootstrap;
    private final WhisperProcessClient processClient;

    public TranscriptionApplicationService(
            WhisperEnvironmentBootstrap environmentBootstrap,
            WhisperProcessClient processClient
    ) {
        this.environmentBootstrap = environmentBootstrap;
        this.processClient = processClient;
    }

    public TranscriptionResult transcribe(
            TranscriptionRequest request,
            String pythonExecutable,
            Path explicitProjectRoot
    ) throws Exception {
        validateRequest(request);

        WhisperExecutionContext context = environmentBootstrap.bootstrap(
                explicitProjectRoot,
                pythonExecutable
        );
        List<String> command = WhisperCommandBuilder.buildTranscribeCommand(request, context);
        processClient.execute(command, context.projectRoot());

        List<Path> generatedFiles = new ArrayList<>();
        for (Path source : request.files()) {
            String stem = stripExtension(source.getFileName().toString());
            for (String format : request.formats()) {
                generatedFiles.add(request.outputDir().resolve(stem + "." + format));
            }
        }
        return new TranscriptionResult(generatedFiles);
    }

    private void validateRequest(TranscriptionRequest request) {
        if (request.files().isEmpty()) {
            throw new IllegalArgumentException("At least one file is required for transcription.");
        }
        for (Path file : request.files()) {
            if (!file.toFile().exists()) {
                throw new IllegalArgumentException("Input file not found: " + file.toAbsolutePath());
            }
        }
    }

    private String stripExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(0, dotIndex) : fileName;
    }
}
