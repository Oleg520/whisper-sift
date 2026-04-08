package io.github.whispersift.domain.service;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class QuestionExtractionServiceTest {

    private final QuestionExtractionService service = new QuestionExtractionService();

    @Test
    void extractsQuestionsAndRemovesDuplicatesWhenEnabled() {
        String transcript = """
                Расскажите, пожалуйста, о вашем опыте работы?
                Да, конечно.
                Какие технологии вы использовали на последнем проекте?
                Java и Spring.
                Какие технологии вы использовали на последнем проекте?
                """;

        List<String> questions = service.extractQuestionCandidates(transcript, true, 10, 240);

        assertEquals(
                List.of(
                        "Расскажите, пожалуйста, о вашем опыте работы?",
                        "Какие технологии вы использовали на последнем проекте?"
                ),
                questions
        );
    }

    @Test
    void keepsDuplicatesWhenDeduplicationDisabled() {
        String transcript = """
                Какие технологии вы использовали на последнем проекте?
                Какие технологии вы использовали на последнем проекте?
                """;

        List<String> questions = service.extractQuestionCandidates(transcript, false, 10, 240);

        assertEquals(
                List.of(
                        "Какие технологии вы использовали на последнем проекте?",
                        "Какие технологии вы использовали на последнем проекте?"
                ),
                questions
        );
    }

    @Test
    void filtersOutNoisyQuestions() {
        String transcript = """
                test test test test test test test test?
                Чем вы занимались на последнем проекте?
                """;

        List<String> questions = service.extractQuestionCandidates(transcript, true, 10, 240);

        assertEquals(List.of("Чем вы занимались на последнем проекте?"), questions);
    }
}
