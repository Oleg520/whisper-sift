package io.github.whispersift.cli;

import io.github.whispersift.application.TranscriptionApplicationService;
import io.github.whispersift.domain.model.TranscriptionRequest;
import io.github.whispersift.infrastructure.whisper.WhisperEnvironmentBootstrap;
import io.github.whispersift.infrastructure.whisper.WhisperProcessClient;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public final class TranscribeCommand {

    public int execute(String[] args) throws Exception {
        List<Path> files = new ArrayList<>();
        String model = "small";
        String language = "ru";
        String device = "auto";
        Path outputDir = Path.of(".");
        List<String> formats = new ArrayList<>(List.of("txt", "srt"));
        String pythonExecutable = "python";
        Path projectRoot = null;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--help", "-h" -> {
                    printHelp();
                    return 0;
                }
                case "--model" -> model = requireValue(args, ++i, "--model");
                case "--language" -> language = requireValue(args, ++i, "--language");
                case "--device" -> device = requireValue(args, ++i, "--device");
                case "--output-dir" -> outputDir = Path.of(requireValue(args, ++i, "--output-dir"));
                case "--python" -> pythonExecutable = requireValue(args, ++i, "--python");
                case "--project-root" -> projectRoot = Path.of(requireValue(args, ++i, "--project-root"));
                case "--formats" -> {
                    formats.clear();
                    i = collectValues(args, i + 1, formats, "--formats");
                }
                default -> {
                    if (arg.startsWith("--")) {
                        throw new IllegalArgumentException("Unknown option for transcribe: " + arg);
                    }
                    files.add(Path.of(arg));
                }
            }
        }

        if (files.isEmpty()) {
            throw new IllegalArgumentException("transcribe requires at least one media file.");
        }

        var service = new TranscriptionApplicationService(
                new WhisperEnvironmentBootstrap(),
                new WhisperProcessClient()
        );
        service.transcribe(
                new TranscriptionRequest(files, outputDir, model, language, device, formats),
                pythonExecutable,
                projectRoot
        );
        return 0;
    }

    private String requireValue(String[] args, int index, String optionName) {
        if (index >= args.length || args[index].startsWith("--")) {
            throw new IllegalArgumentException("Option " + optionName + " requires a value.");
        }
        return args[index];
    }

    private int collectValues(String[] args, int startIndex, List<String> target, String optionName) {
        int index = startIndex;
        while (index < args.length && !args[index].startsWith("--")) {
            target.addAll(Arrays.stream(args[index].split(","))
                    .map(String::trim)
                    .filter(value -> !value.isBlank())
                    .toList());
            index++;
        }
        if (target.isEmpty()) {
            throw new IllegalArgumentException("Option " + optionName + " requires at least one value.");
        }
        return index - 1;
    }

    private void printHelp() {
        System.out.println("""
                Usage: whisper-sift-java transcribe <files...> [options]

                Options:
                  --model <name>         Whisper model name, default: small
                  --language <code>      Language code or auto, default: ru
                  --device <name>        auto/cpu/cuda/mps, default: auto
                  --output-dir <path>    Output directory, default: .
                  --formats <list>       Output formats, default: txt srt
                  --python <exe>         Python executable, default: python
                  --project-root <path>  Explicit project root
                """);
    }
}
