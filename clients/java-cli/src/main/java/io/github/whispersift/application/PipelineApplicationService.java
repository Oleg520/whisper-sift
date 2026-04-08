package io.github.whispersift.application;

import io.github.whispersift.domain.model.QuestionExtractionRequest;
import io.github.whispersift.domain.model.TranscriptionRequest;
import java.nio.file.Path;
import java.util.List;

public final class PipelineApplicationService {

    private final TranscriptionApplicationService transcriptionService;
    private final QuestionExtractionApplicationService questionExtractionService;

    public PipelineApplicationService(
            TranscriptionApplicationService transcriptionService,
            QuestionExtractionApplicationService questionExtractionService
    ) {
        this.transcriptionService = transcriptionService;
        this.questionExtractionService = questionExtractionService;
    }

    public void run(
            TranscriptionRequest transcriptionRequest,
            QuestionExtractionRequest extractionDefaults,
            String pythonExecutable,
            Path explicitProjectRoot
    ) throws Exception {
        if (transcriptionRequest.formats().stream().noneMatch("txt"::equalsIgnoreCase)) {
            throw new IllegalArgumentException(
                    "Pipeline requires txt output because question extraction reads generated transcript files."
            );
        }

        var transcriptionResult = transcriptionService.transcribe(
                transcriptionRequest,
                pythonExecutable,
                explicitProjectRoot
        );

        List<Path> transcriptFiles = transcriptionResult.generatedFiles().stream()
                .filter(path -> path.toString().toLowerCase().endsWith(".txt"))
                .toList();

        questionExtractionService.extract(
                new QuestionExtractionRequest(
                        transcriptFiles,
                        extractionDefaults.outputDir() != null
                                ? extractionDefaults.outputDir()
                                : transcriptionRequest.outputDir(),
                        extractionDefaults.suffix(),
                        extractionDefaults.deduplicate(),
                        extractionDefaults.minLength(),
                        extractionDefaults.maxLength()
                )
        );
    }
}
