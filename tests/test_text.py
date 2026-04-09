from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.utils.text import extract_question_candidates, normalize_whitespace


class TextUtilityTests(unittest.TestCase):
    def test_normalize_whitespace_collapses_spaces_and_newlines(self) -> None:
        value = "  one   two\r\n\r\n three \t four \r five  "

        normalized = normalize_whitespace(value)

        self.assertEqual("one two\n\nthree four\nfive", normalized)

    def test_extract_question_candidates_deduplicates_by_default(self) -> None:
        transcript = (
            "Расскажите, пожалуйста, о вашем опыте работы?\n"
            "Да, конечно.\n"
            "Какие технологии вы использовали на последнем проекте?\n"
            "Java и Spring.\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Расскажите, пожалуйста, о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_keeps_duplicates_when_requested(self) -> None:
        transcript = (
            "Какие технологии вы использовали на последнем проекте?\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript, deduplicate=False)

        self.assertEqual(
            [
                "Какие технологии вы использовали на последнем проекте?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_noise(self) -> None:
        transcript = (
            "test test test test test test test test?\n"
            "Чем вы занимались на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Чем вы занимались на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_accepts_question_like_phrase_without_mark(self) -> None:
        transcript = (
            "Расскажите, пожалуйста, про ваш последний проект\n"
            "Я работал над платежным модулем.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Расскажите, пожалуйста, про ваш последний проект?"],
            questions,
        )

    def test_extract_question_candidates_accepts_est_li_question_without_mark(self) -> None:
        transcript = (
            "Есть ли у вас опыт работы с Kafka\n"
            "Да, у меня был такой опыт.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Есть ли у вас опыт работы с Kafka?"],
            questions,
        )

    def test_extract_question_candidates_accepts_kakoy_question_without_mark(self) -> None:
        transcript = (
            "Какой размер сообщения в Kafka по умолчанию\n"
            "Кажется, около одного мегабайта.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какой размер сообщения в Kafka по умолчанию?"],
            questions,
        )

    def test_extract_question_candidates_rejects_answer_like_sentences_without_question_mark(self) -> None:
        transcript = (
            "Можно, соответственно, поднимать инстанции по отдельности, если нагрузка у нас неравномерная по разным сервисам\n"
            "Когда у нас намного больше чтений, чем записи, можно отдельно масштабировать чтение\n"
            "Как сказать, абстракция создает какую-то базу, которую мы используем\n"
            "Можешь рассказать, пожалуйста, про ваш последний проект\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Можешь рассказать, пожалуйста, про ваш последний проект?"],
            questions,
        )

    def test_extract_question_candidates_filters_answer_like_phrase_with_question_mark(self) -> None:
        transcript = (
            "Я работал над платежным модулем?\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие технологии вы использовали на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_keeps_question_like_phrase_starting_with_ya(self) -> None:
        transcript = (
            "Я правильно понимаю, что у вас был опыт работы с Kafka\n"
            "Да, был.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Я правильно понимаю, что у вас был опыт работы с Kafka?"],
            questions,
        )

    def test_extract_question_candidates_does_not_treat_kakoy_to_as_question(self) -> None:
        transcript = (
            "какой-то дополнительный на диске.\n"
            "А вспомнишь, как кластеризованы и не кластеризованы индексы?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["А вспомнишь, как кластеризованы и не кластеризованы индексы?"],
            questions,
        )

    def test_extract_question_candidates_filters_filler_heavy_fragment(self) -> None:
        transcript = (
            "А, ContekMap, да, наверное, то есть что?\n"
            "И вот знаешь, что такое проба в куберныце?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["И вот знаешь, что такое проба в куберныце?"],
            questions,
        )

    def test_extract_question_candidates_filters_answer_like_guess_with_fillers(self) -> None:
        transcript = (
            "А, так, если честно не помню, но могу предположить, что может быть в районе там 2 мегабайт?\n"
            "Так, а может, ты знаешь, какой размер сообщения в Кавке по умолчанию?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["А может, ты знаешь, какой размер сообщения в Кавке по умолчанию?"],
            questions,
        )

    def test_extract_question_candidates_filters_answer_like_bilo_phrase(self) -> None:
        transcript = (
            "Было тоже самописным, да-да?\n"
            "А Inbox был?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["А Inbox был?"],
            questions,
        )

    def test_extract_question_candidates_filters_low_signal_followups(self) -> None:
        transcript = (
            "Или знаешь, что это такое?\n"
            "Ну, а что у нас?\n"
            "Что у нас еще там есть?\n"
            "А какие еще эти?\n"
            "Не трогаем пока паттерные микросервисы?\n"
            "Так, видишь, да?\n"
            "Можешь, пожалуйста?\n"
            "Что-то еще добавить нужно?\n"
            "Для чего еще?\n"
            "Для чего использовать?\n"
            "Что хотелось сказать?\n"
            "Какие группы паттерных проектирований ты знаешь?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие группы паттерных проектирований ты знаешь?"],
            questions,
        )

    def test_extract_question_candidates_filters_answer_like_explanation_with_question_mark(self) -> None:
        transcript = (
            "То есть адаптер, он как бы берет один интерфейс и адаптирует его к другому, условно, да?\n"
            "Обычно делается вот в тех случаях, когда запросов много, да?\n"
            "А чем декоратор отличается от адаптера?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["А чем декоратор отличается от адаптера?"],
            questions,
        )

    def test_extract_question_candidates_keeps_contextual_followup_with_previous_subject(self) -> None:
        transcript = (
            "Можешь рассказать про устройство памяти Java?\n"
            "Что содержится в ней?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Можешь рассказать про устройство памяти Java?",
                "Что содержится в ней?",
            ],
            questions,
        )

    def test_extract_question_candidates_stops_after_invitation_for_candidate_questions(self) -> None:
        transcript = (
            "Какие технологии вы использовали на последнем проекте?\n"
            "Тогда, наверное, может, у вас какие-то вопросы, мы будем ответить.\n"
            "А в чем разница с CICD?\n"
            "У вас много поточка вообще в целом как используется?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие технологии вы использовали на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_filters_organizational_and_interface_prompts(self) -> None:
        transcript = (
            "Мы на ты или на вы общаться будем?\n"
            "Когда ты указываешь, что она в Питерском офисе, а не в Минске?\n"
            "Можешь посмотреть в чатик?\n"
            "Можно вкладку закрыть?\n"
            "Расскажите, пожалуйста, про ваш последний проект\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Расскажите, пожалуйста, про ваш последний проект?"],
            questions,
        )

    def test_extract_question_candidates_filters_explanatory_answer_like_prompts(self) -> None:
        transcript = (
            "Обычно используется hashCode для определения индекса бакетов, да?\n"
            "Если говорить про память, можно разделить ее на хип и стек, да?\n"
            "Чем hashCode отличается от equals?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Чем hashCode отличается от equals?"],
            questions,
        )

    def test_extract_question_candidates_filters_meta_prompts(self) -> None:
        transcript = (
            "А у вас, ты задаёшь вопрос, потому что у вас второй используете?\n"
            "Это так, просто к слову?\n"
            "Кстати, под Java, вы в основном в 17-й работали?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["По Java вы в основном с 17-й версией работали?"],
            questions,
        )

    def test_extract_question_candidates_filters_truncated_technical_fragments(self) -> None:
        transcript = (
            "А как сделать так, чтобы вот сообщения, которые, ну, не успешно\n"
            "где не стоит, но, если часто вставка редко запись, индекса, они же, по-моему, каждый раз пере\n"
            "Может ли таблица быть без первичного ключа?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Может ли таблица быть без первичного ключа?"],
            questions,
        )

    def test_extract_question_candidates_normalizes_dirty_asr_questions(self) -> None:
        transcript = (
            "У вас получается несколько дай микросервисов у вашей команды в разработке?\n"
            "Кстати, поджаве, вы-то получается, в основном, в 17-е, да, работал?\n"
            "Кстати, вот какой, например, нравится подход, больше функциональное, например, программирование, в каком вот стиле обычно пишешь?\n"
            "Вот, ну, а про Metaspace знаешь, который раньше пермгеном был?\n"
            "Можешь вообще рассказать про Spring, что, зачем нужен, вот, и чем отличается, там, от Spring Boot?\n"
            "Ну, всё-таки, можно ли обрабатывать error?\n"
            "Можешь рассказать вообще про у жизни цикл бина, как он создается, как его найти, как использовать?\n"
            "Вот ты, кстати, упомянул, да, прототайп, может, получается, побольше сказать, про сколпу бинов, вот ты один из них назвал, а с другими знаком?\n"
            "Можешь рассказать чем, например, Unic отличается от Праймер K?\n"
            "А использовали это, может быть, на предыдущем проекте, или в целом, опыт с аконными функциями, было?\n"
            "Вот, касательно спрингана, вот, и а ОП работал с АОП?\n"
            "Вот, зачем они нужны?\n"
            "К осадьему, вот, бас данных, так понимаю, получается, с пасгрессом работали, ну, на предыдущем проекте, а с какими-то другими басами данными был, например, опыт?\n"
            "А, можешь рассказать, для чего индексы вообще нужны в таблице БОЗ-Данных?\n"
            "Так, кстати, вот я вспомнил, по индексам, вот еще все-таки спрашиваю, какие вот типы индексов приходилось использовать?\n"
            "селективности, да, насколько полезен будет индекс, как именно будет сканироваться таблица?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Сколько микросервисов у вашей команды в разработке?",
                "По Java вы в основном с 17-й версией работали?",
                "Какой подход вам больше нравится: функциональное программирование или что-то другое, и в каком стиле обычно пишете?",
                "А про Metaspace знаешь? Это то, что раньше называлось PermGen?",
                "Можешь рассказать про Spring: зачем он нужен и чем отличается от Spring Boot?",
                "Можно ли обрабатывать Error?",
                "Можешь рассказать про жизненный цикл бина: как он создаётся, как его найти и как использовать?",
                "Ты упомянул prototype. Можешь рассказать про scope бинов и какие ещё scope тебе знакомы?",
                "Чем UNIQUE отличается от Primary Key?",
                "Был ли опыт с оконными функциями?",
                "С AOP в Spring работал?",
                "Зачем нужен AOP?",
                "Кроме Postgres, был ли опыт работы с другими базами данных?",
                "Для чего индексы нужны в базе данных?",
                "Какие типы индексов приходилось использовать?",
                "Насколько полезен будет индекс с точки зрения селективности, и как именно будет сканироваться таблица?",
            ],
            questions,
        )

    def test_extract_question_candidates_fuzzy_deduplicates_similar_questions(self) -> None:
        transcript = (
            "Какие технологии вы использовали на последнем проекте?\n"
            "Какие технологии использовали на последнем проекте?\n"
            "Какие технологии вы использовали на последнем проекте ?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие технологии вы использовали на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_filters_topic_bridges(self) -> None:
        transcript = (
            "Так, что я еще не упомянул?\n"
            "А можно пойти дальше тогда?\n"
            "Так, вот как раз пора к исключениям вернуться?\n"
            "Можешь рассказать про иерархию исключений?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Можешь рассказать про иерархию исключений?"],
            questions,
        )

    def test_extract_question_candidates_uses_detected_interviewer_label(self) -> None:
        transcript = (
            "Интервьюер: Расскажите о вашем опыте работы\n"
            "Кандидат: Я работаю в backend уже пять лет.\n"
            "Интервьюер: Какие технологии вы использовали на последнем проекте?\n"
            "Кандидат: Java и Spring.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_supports_explicit_speaker_label(self) -> None:
        transcript = (
            "SPEAKER_00: Расскажите о вашем опыте работы\n"
            "SPEAKER_01: Я работаю в backend уже пять лет.\n"
            "SPEAKER_00: Какие технологии вы использовали на последнем проекте?\n"
            "SPEAKER_01: Java и Spring.\n"
        )

        questions = extract_question_candidates(
            transcript,
            interviewer_labels=("SPEAKER_00",),
        )

        self.assertEqual(
            [
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_uses_remaining_speaker_when_candidate_is_detected(self) -> None:
        transcript = (
            "Team Lead: Расскажите о вашем опыте работы\n"
            "Candidate: Я работаю в backend уже пять лет.\n"
            "Team Lead: Какие технологии вы использовали на последнем проекте?\n"
            "Candidate: Python и FastAPI.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_fails_for_unknown_explicit_speaker_label(self) -> None:
        transcript = (
            "SPEAKER_00: Расскажите о вашем опыте работы\n"
            "SPEAKER_01: Я работаю в backend уже пять лет.\n"
        )

        with self.assertRaisesRegex(RuntimeError, "None of the provided interviewer labels"):
            extract_question_candidates(
                transcript,
                interviewer_labels=("SPEAKER_42",),
            )

    def test_extract_question_candidates_extracts_from_srt_segments(self) -> None:
        transcript = (
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Можешь рассказать\n\n"
            "2\n"
            "00:00:02,000 --> 00:00:05,000\n"
            "про Spring и Spring Boot\n\n"
            "3\n"
            "00:00:05,000 --> 00:00:07,000\n"
            "Да, конечно.\n"
        )

        questions = extract_question_candidates(
            transcript,
            source_name="interview.srt",
        )

        self.assertEqual(
            ["Можешь рассказать про Spring и Spring Boot?"],
            questions,
        )

    def test_extract_question_candidates_supports_srt_with_explicit_speaker_label(self) -> None:
        transcript = (
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "SPEAKER_00: Расскажите про ваш последний проект\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:05,000\n"
            "SPEAKER_01: Я работал над платежным сервисом.\n\n"
            "3\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "SPEAKER_00: Какие технологии вы использовали?\n"
        )

        questions = extract_question_candidates(
            transcript,
            source_name="interview.srt",
            interviewer_labels=("SPEAKER_00",),
        )

        self.assertEqual(
            [
                "Расскажите про ваш последний проект?",
                "Какие технологии вы использовали?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_smalltalk_wrapup_and_answer_leaks(self) -> None:
        transcript = (
            "Ну что ты такой, что ты от себя чиста?\n"
            "Вот как пример, это подключение к BDS-ке?\n"
            "Как бы, в принципе, нельзя сказать конкретно, какую проблему они решают?\n"
            "У меня в целом такая часть, она закончена, дальше вопросы такие, а остатки?\n"
            "Почему современные framework, в основном, выбирают непроверяемые исключения?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Почему современные фреймворки и библиотеки чаще выбирают непроверяемые исключения?"],
            questions,
        )

    def test_extract_question_candidates_normalizes_noisy_collection_and_map_questions(self) -> None:
        transcript = (
            "Что вообще такого коллекция на P?\n"
            "А вообще лист сет Q-мэп, что это такое?\n"
            "А в хэш-мапу мы можем положить элемент с ключом ну?\n"
            "Так что там unique check?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Что такое Collection API?",
                "List, Set, Queue и Map — что это такое?",
                "Можно ли положить в HashMap элемент с ключом null?",
                "Что такое UNIQUE constraint?",
            ],
            questions,
        )

    def test_extract_question_candidates_normalizes_noisy_spring_and_sql_questions(self) -> None:
        transcript = (
            "Что такое сервизация, десервизация, для чего нужен serial version of uit?\n"
            "А что такое dispatcher-serivallet в архитектуре Springer?\n"
            "Что такое ORM, GP and HyperNate?\n"
            "А почему рекомендуют сбегать в Force Push?\n"
            "Если у тебя есть очень медленный скель запрос, как его анализировать на ревнирной акцентрести?\n"
            "Приходился ли когда-нибудь смотреть тжава дамп и трет дамп?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Что такое сериализация и десериализация, и для чего нужен serialVersionUID?",
                "Что такое DispatcherServlet в архитектуре Spring?",
                "Что такое ORM, JPA и Hibernate?",
                "Почему не рекомендуют делать force push?",
                "Если у тебя есть очень медленный SQL-запрос, как ты будешь его анализировать?",
                "Приходилось ли когда-нибудь смотреть Java dump и thread dump?",
            ],
            questions,
        )

    def test_extract_question_candidates_normalizes_contextual_noisy_followups(self) -> None:
        transcript = (
            "Что такое сервизация, десервизация, для чего нужен serial version of uit?\n"
            "А для чего нужен транси?\n"
            "Можешь ли ты просто немножко теории, что такое вообще спринк, для чего его создали, что там у него под комботом?\n"
            "А что у него под комботом, за технологией?\n"
            "Может ли Кавка обеспечить порядок сообщений?\n"
            "Как бы творили комит по этому сообщению?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Что такое сериализация и десериализация, и для чего нужен serialVersionUID?",
                "Для чего нужен transient?",
                "Что такое Spring, для чего его создали и что у него под капотом?",
                "А что у Spring под капотом, какая там технология?",
                "Может ли Кавка обеспечить порядок сообщений?",
                "Когда бы ты коммитил offset по сообщению в Kafka?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_remaining_garbled_followups(self) -> None:
        transcript = (
            "Может быть, тут еще у них должна быть?\n"
            "Как сейчас эту еллю в вару использовала ли на работе?\n"
            "Можешь рассказать про уровни изоляции транзакций?\n"
            "Может, про какой-нибудь один любой над головой?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Можешь рассказать про уровни изоляции транзакций?"],
            questions,
        )

    def test_extract_question_candidates_filters_short_contextual_noise_from_tutko(self) -> None:
        transcript = (
            "А по стеку что?\n"
            "Бинарное или не бинарное?\n"
            "Что это у меня в ступор, кто вгоняет?\n"
            "Ты знаком с проблемой n плюс 1?\n"
            "Какие есть операции над стримом по типу?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Ты знаком с проблемой n плюс 1?",
                "Какие есть операции над стримом по типу?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_short_contextual_noise_from_golubev(self) -> None:
        transcript = (
            "А какой стек у тебя, вот, сейчас?\n"
            "Как функциональной интерфейсе, да?\n"
            "Что несколько инстинкцев есть у одного сервиса, да?\n"
            "Можешь попробовать, например, объяснить вот, допустим, про дедлок?\n"
            "А про Spring, то вот сейчас теперь давай, что такое dependency injection, и как он связан с спрингом?\n"
            "А что такое DLQ?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Можешь попробовать, например, объяснить вот, допустим, про дедлок?",
                "А про Spring, то вот сейчас теперь давай, что такое dependency injection, и как он связан с спрингом?",
                "А что такое DLQ?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_short_contextual_noise_from_safargulov(self) -> None:
        transcript = (
            "В сервисе, да?\n"
            "Или это был первый опыт?\n"
            "Плюс длинной транзакцией?\n"
            "Скажите, каким феррборками вы пользовались, когда делали отправку в кавку?\n"
            "Для чего индексы нужны в базе данных?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Скажите, каким феррборками вы пользовались, когда делали отправку в кавку?",
                "Для чего индексы нужны в базе данных?",
            ],
            questions,
        )


if __name__ == "__main__":
    unittest.main()
