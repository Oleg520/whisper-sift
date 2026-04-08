package io.github.whispersift.cli;

import io.github.whispersift.application.PipelineApplicationService;
import io.github.whispersift.application.QuestionExtractionApplicationService;
import io.github.whispersift.application.TranscriptionApplicationService;
import io.github.whispersift.domain.model.QuestionExtractionRequest;
import io.github.whispersift.domain.model.TranscriptionRequest;
import io.github.whispersift.domain.service.QuestionExtractionService;
import io.github.whispersift.infrastructure.files.FileStorageService;
import io.github.whispersift.infrastructure.files.PathResolver;
import io.github.whispersift.infrastructure.whisper.WhisperEnvironmentBootstrap;
import io.github.whispersift.infrastructure.whisper.WhisperProcessClient;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class PipelineCommand {

    public int execute(String[] args) throws Exception {
        List<Path> files = new ArrayList<>();
        String model = "small";
        String language = "ru";
        String device = "auto";
        Path outputDir = Path.of(".");
        List<String> formats = new ArrayList<>(List.of("txt", "srt"));
        Path questionsDir = null;
        String suffix = "_questions.txt";
        boolean noDeduplicate = false;
        int minLength = 10;
        int maxLength = 240;
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
                case "--formats" -> {
                    formats.clear();
                    i = collectValues(args, i + 1, formats, "--formats");
                }
                case "--questions-dir" -> questionsDir = Path.of(requireValue(args, ++i, "--questions-dir"));
                case "--suffix" -> suffix = requireValue(args, ++i, "--suffix");
                case "--no-deduplicate" -> noDeduplicate = true;
                case "--min-length" -> minLength = Integer.parseInt(requireValue(args, ++i, "--min-length"));
                case "--max-length" -> maxLength = Integer.parseInt(requireValue(args, ++i, "--max-length"));
                case "--python" -> pythonExecutable = requireValue(args, ++i, "--python");
                case "--project-root" -> projectRoot = Path.of(requireValue(args, ++i, "--project-root"));
                default -> {
                    if (arg.startsWith("--")) {
                        throw new IllegalArgumentException("Unknown option for pipeline: " + arg);
                    }
                    files.add(Path.of(arg));
                }
            }
        }

        if (files.isEmpty()) {
            throw new IllegalArgumentException("pipeline requires at least one media file.");
        }

        var transcriptionService = new TranscriptionApplicationService(
                new WhisperEnvironmentBootstrap(),
                new WhisperProcessClient()
        );
        var questionService = new QuestionExtractionApplicationService(
                new FileStorageService(),
                new PathResolver(),
                new QuestionExtractionService()
        );
        var pipeline = new PipelineApplicationService(transcriptionService, questionService);

        pipeline.run(
                new TranscriptionRequest(files, outputDir, model, language, device, formats),
                new QuestionExtractionRequest(
                        List.of(),
                        questionsDir,
                        suffix,
                        !noDeduplicate,
                        minLength,
                        maxLength
                ),
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
            for (String value : args[index].split(",")) {
                String trimmed = value.trim();
                if (!trimmed.isBlank()) {
                    target.add(trimmed);
                }
            }
            index++;
        }
        if (target.isEmpty()) {
            throw new IllegalArgumentException("Option " + optionName + " requires at least one value.");
        }
        return index - 1;
    }

    private void printHelp() {
        System.out.println("""
                Usage: whisper-sift-java pipeline <files...> [options]

                Options:
                  --model <name>         Whisper model name, default: small
                  --language <code>      Language code or auto, default: ru
                  --device <name>        auto/cpu/cuda/mps, default: auto
                  --output-dir <path>    Transcript output directory, default: .
                  --formats <list>       Output formats, default: txt srt
                  --questions-dir <path> Question output directory
                  --suffix <value>       Question file suffix, default: _questions.txt
                  --no-deduplicate       Disable question deduplication
                  --min-length <int>     Minimum question length, default: 10
                  --max-length <int>     Maximum question length, default: 240
                  --python <exe>         Python executable, default: python
                  --project-root <path>  Explicit project root
                """);
    }
}
