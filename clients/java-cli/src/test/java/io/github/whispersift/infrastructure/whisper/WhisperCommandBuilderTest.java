package io.github.whispersift.infrastructure.whisper;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.whispersift.domain.model.TranscriptionRequest;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class WhisperCommandBuilderTest {

    @Test
    void buildsExpectedCommandLineForTranscription() {
        var request = new TranscriptionRequest(
                List.of(Path.of("part1.mkv"), Path.of("part2.mkv")),
                Path.of("results"),
                "medium",
                "ru",
                "cuda",
                List.of("txt", "srt")
        );
        var context = new WhisperExecutionContext(
                Path.of("C:/projects/whisper-sift"),
                Path.of("C:/projects/whisper-sift/transcribe_whisper.py"),
                "python"
        );

        List<String> command = WhisperCommandBuilder.buildTranscribeCommand(request, context);

        assertEquals(
                List.of(
                        "python",
                        context.launcherScript().toString(),
                        "transcribe",
                        "part1.mkv",
                        "part2.mkv",
                        "--model",
                        "medium",
                        "--language",
                        "ru",
                        "--device",
                        "cuda",
                        "--output-dir",
                        "results",
                        "--formats",
                        "txt",
                        "srt"
                ),
                command
        );
    }
}
