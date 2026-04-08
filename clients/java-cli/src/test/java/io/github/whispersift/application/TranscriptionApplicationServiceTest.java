package io.github.whispersift.application;

import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.whispersift.domain.model.TranscriptionRequest;
import io.github.whispersift.infrastructure.whisper.WhisperEnvironmentBootstrap;
import io.github.whispersift.infrastructure.whisper.WhisperProcessClient;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class TranscriptionApplicationServiceTest {

    @Test
    void rejectsMissingInputFilesBeforeTryingToRunWhisper() {
        var service = new TranscriptionApplicationService(
                new WhisperEnvironmentBootstrap(),
                new WhisperProcessClient()
        );

        var request = new TranscriptionRequest(
                List.of(Path.of("missing-interview.mkv")),
                Path.of("."),
                "small",
                "ru",
                "auto",
                List.of("txt", "srt")
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> service.transcribe(request, "python", null)
        );
    }
}
