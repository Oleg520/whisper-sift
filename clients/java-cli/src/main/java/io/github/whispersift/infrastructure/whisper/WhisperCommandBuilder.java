package io.github.whispersift.infrastructure.whisper;

import io.github.whispersift.domain.model.TranscriptionRequest;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class WhisperCommandBuilder {

    private WhisperCommandBuilder() {
    }

    public static List<String> buildTranscribeCommand(
            TranscriptionRequest request,
            WhisperExecutionContext context
    ) {
        List<String> command = new ArrayList<>();
        command.add(context.pythonExecutable());
        command.add(context.launcherScript().toString());
        command.add("transcribe");

        for (Path file : request.files()) {
            command.add(file.toString());
        }

        command.add("--model");
        command.add(request.model());
        command.add("--language");
        command.add(request.language());
        command.add("--device");
        command.add(request.device());
        command.add("--output-dir");
        command.add(request.outputDir().toString());
        command.add("--formats");
        command.addAll(request.formats());

        return command;
    }
}
