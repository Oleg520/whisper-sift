package io.github.whispersift.cli;

import io.github.whispersift.application.QuestionExtractionApplicationService;
import io.github.whispersift.domain.model.QuestionExtractionRequest;
import io.github.whispersift.domain.service.QuestionExtractionService;
import io.github.whispersift.infrastructure.files.FileStorageService;
import io.github.whispersift.infrastructure.files.PathResolver;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class ExtractQuestionsCommand {

    public int execute(String[] args) throws Exception {
        List<Path> files = new ArrayList<>();
        Path outputDir = null;
        String suffix = "_questions.txt";
        boolean noDeduplicate = false;
        int minLength = 10;
        int maxLength = 240;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--help", "-h" -> {
                    printHelp();
                    return 0;
                }
                case "--output-dir" -> outputDir = Path.of(requireValue(args, ++i, "--output-dir"));
                case "--suffix" -> suffix = requireValue(args, ++i, "--suffix");
                case "--no-deduplicate" -> noDeduplicate = true;
                case "--min-length" -> minLength = Integer.parseInt(requireValue(args, ++i, "--min-length"));
                case "--max-length" -> maxLength = Integer.parseInt(requireValue(args, ++i, "--max-length"));
                default -> {
                    if (arg.startsWith("--")) {
                        throw new IllegalArgumentException("Unknown option for extract-questions: " + arg);
                    }
                    files.add(Path.of(arg));
                }
            }
        }

        if (files.isEmpty()) {
            throw new IllegalArgumentException("extract-questions requires at least one transcript file.");
        }

        var service = new QuestionExtractionApplicationService(
                new FileStorageService(),
                new PathResolver(),
                new QuestionExtractionService()
        );
        service.extract(
                new QuestionExtractionRequest(
                        files,
                        outputDir,
                        suffix,
                        !noDeduplicate,
                        minLength,
                        maxLength
                )
        );
        return 0;
    }

    private String requireValue(String[] args, int index, String optionName) {
        if (index >= args.length || args[index].startsWith("--")) {
            throw new IllegalArgumentException("Option " + optionName + " requires a value.");
        }
        return args[index];
    }

    private void printHelp() {
        System.out.println("""
                Usage: whisper-sift-java extract-questions <files...> [options]

                Options:
                  --output-dir <path>    Output directory for question files
                  --suffix <value>       Output file suffix, default: _questions.txt
                  --no-deduplicate       Disable question deduplication
                  --min-length <int>     Minimum question length, default: 10
                  --max-length <int>     Maximum question length, default: 240
                """);
    }
}
