package io.github.whispersift.application;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.whispersift.domain.model.QuestionExtractionRequest;
import io.github.whispersift.domain.service.QuestionExtractionService;
import io.github.whispersift.infrastructure.files.FileStorageService;
import io.github.whispersift.infrastructure.files.PathResolver;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class QuestionExtractionApplicationServiceTest {

    @TempDir
    Path tempDir;

    @Test
    void writesQuestionFileToRequestedOutputDirectory() throws Exception {
        Path transcript = tempDir.resolve("interview.txt");
        Files.writeString(
                transcript,
                """
                Расскажите о вашем опыте работы?
                Конечно.
                Какие технологии вы использовали?
                """
        );
        Path outputDir = tempDir.resolve("questions");

        var service = new QuestionExtractionApplicationService(
                new FileStorageService(),
                new PathResolver(),
                new QuestionExtractionService()
        );

        var result = service.extract(
                new QuestionExtractionRequest(
                        java.util.List.of(transcript),
                        outputDir,
                        "_questions.txt",
                        true,
                        10,
                        240
                )
        );

        Path outputFile = outputDir.resolve("interview_questions.txt");
        assertEquals(java.util.List.of(outputFile), result.generatedFiles());
        assertTrue(Files.exists(outputFile));
        assertEquals(
                """
                Расскажите о вашем опыте работы?
                Какие технологии вы использовали?
                """.strip(),
                Files.readString(outputFile)
                        .replace("\r\n", "\n")
                        .strip()
        );
    }
}
