package io.github.whispersift.cli;

public final class Main {

    public static void main(String[] args) {
        int exitCode = new Main().run(args);
        System.exit(exitCode);
    }

    private int run(String[] args) {
        if (args.length == 0 || isHelp(args[0])) {
            printHelp();
            return 0;
        }

        String command = args[0];
        String[] commandArgs = java.util.Arrays.copyOfRange(args, 1, args.length);

        try {
            return switch (command) {
                case "transcribe" -> new TranscribeCommand().execute(commandArgs);
                case "extract-questions" -> new ExtractQuestionsCommand().execute(commandArgs);
                case "pipeline" -> new PipelineCommand().execute(commandArgs);
                default -> {
                    System.err.println("Unknown command: " + command);
                    printHelp();
                    yield 1;
                }
            };
        } catch (IllegalArgumentException ex) {
            System.err.println(ex.getMessage());
            return 2;
        } catch (Exception ex) {
            ex.printStackTrace(System.err);
            return 1;
        }
    }

    private boolean isHelp(String value) {
        return "--help".equals(value) || "-h".equals(value) || "help".equals(value);
    }

    private void printHelp() {
        System.out.println("""
                whisper-sift-java

                Commands:
                  transcribe         Transcribe media files using the existing Python app
                  extract-questions  Extract interviewer questions from transcript txt files
                  pipeline           Run transcription and then extract questions

                Examples:
                  java -jar target/whisper-sift-java-cli.jar transcribe interview.mkv
                  java -jar target/whisper-sift-java-cli.jar extract-questions interview.txt
                  java -jar target/whisper-sift-java-cli.jar pipeline interview.mkv --output-dir results
                """);
    }
}
