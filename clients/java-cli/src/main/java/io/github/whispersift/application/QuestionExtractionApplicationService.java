package io.github.whispersift.application;

import io.github.whispersift.domain.model.QuestionExtractionRequest;
import io.github.whispersift.domain.model.QuestionExtractionResult;
import io.github.whispersift.domain.service.QuestionExtractionService;
import io.github.whispersift.infrastructure.files.FileStorageService;
import io.github.whispersift.infrastructure.files.PathResolver;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class QuestionExtractionApplicationService {

    private final FileStorageService fileStorageService;
    private final PathResolver pathResolver;
    private final QuestionExtractionService questionExtractionService;

    public QuestionExtractionApplicationService(
            FileStorageService fileStorageService,
            PathResolver pathResolver,
            QuestionExtractionService questionExtractionService
    ) {
        this.fileStorageService = fileStorageService;
        this.pathResolver = pathResolver;
        this.questionExtractionService = questionExtractionService;
    }

    public QuestionExtractionResult extract(QuestionExtractionRequest request) throws Exception {
        List<Path> generatedFiles = new ArrayList<>();

        for (Path file : request.files()) {
            Path resolvedSource = file.toAbsolutePath().normalize();
            if (!resolvedSource.toFile().exists()) {
                throw new IllegalArgumentException("Transcript file not found: " + resolvedSource);
            }

            String content = fileStorageService.readUtf8(resolvedSource);
            List<String> questions = questionExtractionService.extractQuestionCandidates(
                    content,
                    request.deduplicate(),
                    request.minLength(),
                    request.maxLength()
            );
            Path outputPath = pathResolver.buildQuestionOutputPath(
                    resolvedSource,
                    request.outputDir(),
                    request.suffix()
            );
            fileStorageService.writeUtf8(outputPath, String.join(System.lineSeparator(), questions));
            System.out.printf(
                    "[questions] %s -> %s (%d items)%n",
                    resolvedSource.getFileName(),
                    outputPath.getFileName(),
                    questions.size()
            );
            generatedFiles.add(outputPath);
        }

        return new QuestionExtractionResult(generatedFiles);
    }
}
