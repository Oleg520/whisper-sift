package io.github.whispersift.domain.service;

import io.github.whispersift.util.TextNormalizer;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class QuestionExtractionService {

    private static final Pattern QUESTION_SPLIT = Pattern.compile("(?<=[?.!])\\s+|\\n+");
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[A-Za-zА-Яа-яЁё0-9-]+");
    private static final Pattern LETTER_PATTERN = Pattern.compile("[A-Za-zА-Яа-яЁё]");

    public List<String> extractQuestionCandidates(
            String text,
            boolean deduplicate,
            int minLength,
            int maxLength
    ) {
        String preparedText = TextNormalizer.normalizeWhitespace(text);
        String[] chunks = QUESTION_SPLIT.split(preparedText);

        List<String> questions = new ArrayList<>();
        for (String chunk : chunks) {
            if (!chunk.contains("?")) {
                continue;
            }

            for (String part : chunk.split("\\?")) {
                String cleaned = cleanupQuestion(part);
                if (cleaned.isEmpty()) {
                    continue;
                }
                cleaned = cleaned + "?";
                if (cleaned.length() < minLength || cleaned.length() > maxLength) {
                    continue;
                }
                if (looksLikeNoise(cleaned)) {
                    continue;
                }
                questions.add(cleaned);
            }
        }

        return deduplicate ? deduplicatePreservingOrder(questions) : questions;
    }

    private String cleanupQuestion(String value) {
        String cleaned = TextNormalizer.normalizeWhitespace(value).strip();
        cleaned = cleaned.replaceAll("^[\\s\\.,!;:-]+", "");
        cleaned = cleaned.replaceAll("[\\s\\.,!;:-]+$", "");
        return cleaned.replace(" ?", "?");
    }

    private boolean looksLikeNoise(String value) {
        Matcher tokenMatcher = TOKEN_PATTERN.matcher(value.toLowerCase(Locale.ROOT));
        List<String> tokens = new ArrayList<>();
        while (tokenMatcher.find()) {
            tokens.add(tokenMatcher.group());
        }
        if (tokens.isEmpty()) {
            return true;
        }

        if (tokens.size() >= 8) {
            Set<String> uniqueTokens = new LinkedHashSet<>(tokens);
            double uniqueRatio = (double) uniqueTokens.size() / tokens.size();
            if (uniqueRatio < 0.35) {
                return true;
            }
        }

        Matcher letterMatcher = LETTER_PATTERN.matcher(value);
        int letters = 0;
        while (letterMatcher.find()) {
            letters++;
        }
        return letters < 5;
    }

    private List<String> deduplicatePreservingOrder(List<String> values) {
        return new ArrayList<>(new LinkedHashSet<>(values));
    }
}
