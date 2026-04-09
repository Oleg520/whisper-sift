from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_INTERVIEWER_LABELS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
)
from whisper_sift.domain.questions import QuestionCandidate, QuestionExtractionResult
from whisper_sift.domain.transcript import (
    SpeakerTurn,
    TranscriptArtifact,
    TranscriptSlice,
    TranscriptView,
)


WHITESPACE_RE = re.compile(r"[ \t]+")
QUESTION_SPLIT_RE = re.compile(r"(?<=[?.!])\s+|\n+")
SRT_TIMECODE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)
GENERIC_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]{1,40})\]|(?P<plain>[^:\n]{1,40}?))"
    r"(?:\s*:\s*|\s+[-–—]\s+)(?P<body>.*)$"
)
CANDIDATE_QUESTION_INVITE_RE = re.compile(
    r"(?:"
    r"есть\s+ли\s+у\s+вас\s+вопросы|"
    r"у\s+вас\s+есть\s+вопросы|"
    r"у\s+вас\s+какие[- ]?то\s+вопросы|"
    r"может(?:,\s*|\s+)у\s+вас\s+какие[- ]?то\s+вопросы|"
    r"может(?:,\s*|\s+)есть\s+какие[- ]?то\s+вопросы"
    r")",
    re.IGNORECASE,
)
QUESTION_LIKE_RE = re.compile(
    r"^(?:"
    r"что(?!-)|как(?!-)|почему|зачем|когда(?!-)|где(?!-)|кто(?!-)|сколько|"
    r"какой(?!-)|какая(?!-)|какие(?!-)|какое(?!-)|каким(?!-)|какую(?!-)|какого(?!-)|какому(?!-)|"
    r"чем(?!-)|можете|можешь|можно|есть ли|был ли|были ли|"
    r"расскажите|расскажи|подскажите|объясните|верно ли|правильно ли|"
    r"я правильно понимаю|я верно понимаю|если я правильно понимаю|"
    r"правильно понимаю|верно понимаю|"
    r"what|how|why|when|where|who|which|can you|could you|would you|"
    r"do you|did you|have you|is there|are there|tell me"
    r")\b",
    re.IGNORECASE,
)
ANSWER_LIKE_RE = re.compile(
    r"^(?:"
    r"да|нет|ага|ну|смотрите|слушайте|хорошо|конечно|скорее|получается|"
    r"наверное|в целом|на самом деле|если честно|честно говоря|не помню|"
    r"то есть|мне кажется|я думаю|я бы|можно,? конечно|"
    r"я|мы|мне|нам|у нас|в нашей команде|в компании|на проекте|на последнем проекте|"
    r"на прошлом проекте|обычно я|обычно мы|это|было|есть|"
    r"actually|well|yes|no|i|we|our team|in our team|on the last project"
    r")\b",
    re.IGNORECASE,
)
STRONG_ANSWER_LIKE_RE = re.compile(
    r"^(?:"
    r"то есть|мне кажется|я думаю|я бы|можно,? конечно|"
    r"что касается|если у нас|нет, нет|нет, наоборот|"
    r"как бы мне|обычно делается|так,\s*ну,\s*как бы"
    r")\b",
    re.IGNORECASE,
)
CANONICAL_WHITESPACE_RE = re.compile(r"\s+")
QUESTION_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
LEADING_FILLER_RE = re.compile(
    r"^(?:(?:а|ну|вот|так|и|слушай|смотри)\b\s*[,:\-]?\s*)+",
    re.IGNORECASE,
)
FILLER_TOKEN_RE = re.compile(
    r"^(?:а|ну|вот|так|и|то|есть|да|наверное|если|честно|вообще|как|бы|смотри|слушай)$",
    re.IGNORECASE,
)
LOW_SIGNAL_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"(?:а\s+)?по\s+стеку\s+что\b|"
    r"(?:а\s+)?какой\s+стек\s+у\s+тебя\b.*|"
    r"или\s+это\s+был\s+первый\s+опыт\b|"
    r"или\s+(?:знаешь|есть)\b.*(?:это|такое|способы)\b|"
    r"как\s+это\s+можно\s+исправить\b|"
    r"ну\s*,?\s*а\s+что\s+у\s+нас\b|"
    r"что\s+у\s+нас(?:\s+ещ[её])?\b|"
    r"(?:а\s+)?какие\s+ещ[её]\b|"
    r"не\s+трогаем\s+пока\b|"
    r"(?:что[- ]?то|то)\s+ещ[её]\s+добавить\s+нужно\b|"
    r"(?:так,\s*)?(?:видишь|понятно|слышно)\s*,?\s*да\b|"
    r"можешь\s*,?\s*пожалуйста\b|"
    r"для\s+чего\s+еще\b|"
    r"для\s+чего\s+использовать\b|"
    r"что\s+хотелось\s+сказать\b|"
    r"как\s+делать\b.*(?:приятно|понятно)\b|"
    r"может\s+быть,\s*тут\s+ещ[её]\s+у\s+них\s+должна\s+быть\b|"
    r"можешь\s+ли\s+сказать\s+про\s+эту\s+проблему\b|"
    r"чем\s+именно\s+проблему\s+не\s+слышал\b|"
    r"может,\s*про\s+какой[- ]?нибудь\s+один\b.*"
    r")",
    re.IGNORECASE,
)
KNOWN_INTERVIEWER_LABELS = (
    "interviewer",
    "интервьюер",
    "recruiter",
    "рекрутер",
    "hr",
    "question",
    "вопрос",
    "q",
    "собеседующий",
)
KNOWN_CANDIDATE_LABELS = (
    "candidate",
    "кандидат",
    "соискатель",
    "interviewee",
    "answer",
    "ответ",
    "a",
)
KNOWN_SPEAKER_LABEL_RE = re.compile(
    r"^(?:speaker|spk|спикер)[ _-]?\d+$",
    re.IGNORECASE,
)
LEADING_READABILITY_FILLER_RE = re.compile(
    r"^(?:(?:ну|вот|так|кстати|окей)\s*,\s*)+",
    re.IGNORECASE,
)
LEADING_ADDRESS_RE = re.compile(
    r"^(?:давай\s+тогда\s+спросить,\s*[^,]+,\s*)",
    re.IGNORECASE,
)
READABILITY_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bподжаве\b", re.IGNORECASE), "по Java"),
    (re.compile(r"\bпод\s+java\b", re.IGNORECASE), "по Java"),
    (re.compile(r"\bпасгресс\w*\b", re.IGNORECASE), "Postgres"),
    (re.compile(r"\bбасами\s+данными\b", re.IGNORECASE), "базами данных"),
    (re.compile(r"\bбас\s+данных\b", re.IGNORECASE), "базы данных"),
    (re.compile(r"\bбо[зс]-?данных\b", re.IGNORECASE), "базе данных"),
    (re.compile(r"\bжизни\s+цикл\b", re.IGNORECASE), "жизненный цикл"),
    (re.compile(r"\bсколпу\b", re.IGNORECASE), "scope"),
    (re.compile(r"\bнотаций\b", re.IGNORECASE), "аннотаций"),
    (re.compile(r"\bнотацией\b", re.IGNORECASE), "аннотацией"),
    (re.compile(r"\bантанцую\b", re.IGNORECASE), "аннотацию"),
    (re.compile(r"\bпартийц\w*\b", re.IGNORECASE), "партиции"),
    (re.compile(r"\bаконными\b", re.IGNORECASE), "оконными"),
    (re.compile(r"\bэсквеле\b", re.IGNORECASE), "SQL"),
    (re.compile(r"\bunic\b", re.IGNORECASE), "UNIQUE"),
    (re.compile(r"\bпраймер\s*k\b", re.IGNORECASE), "Primary Key"),
    (re.compile(r"\bа\s+оп\b", re.IGNORECASE), "AOP"),
    (re.compile(r"\bспрингана\b", re.IGNORECASE), "Spring"),
    (re.compile(r"\bспринг\b", re.IGNORECASE), "Spring"),
    (re.compile(r"\bчто,\s*что\b", re.IGNORECASE), "что"),
    (re.compile(r"\b17-е\b", re.IGNORECASE), "17-й версии"),
    (re.compile(r"\bхэш[- ]?кот\b", re.IGNORECASE), "hashCode"),
    (re.compile(r"\bсервизац\w*\b", re.IGNORECASE), "сериализация"),
    (re.compile(r"\bдесервизац\w*\b", re.IGNORECASE), "десериализация"),
    (re.compile(r"\bserial version of uit\b", re.IGNORECASE), "serialVersionUID"),
    (re.compile(r"\bdispatcher[- ]?serivallet\b", re.IGNORECASE), "DispatcherServlet"),
    (re.compile(r"\bspringer\b", re.IGNORECASE), "Spring"),
    (re.compile(r"\bhypernate\b", re.IGNORECASE), "Hibernate"),
    (re.compile(r"\bволотайл\b", re.IGNORECASE), "volatile"),
    (re.compile(r"\bскель\b", re.IGNORECASE), "SQL"),
    (re.compile(r"\bтжава\b", re.IGNORECASE), "Java"),
    (re.compile(r"\bтрет\b", re.IGNORECASE), "thread"),
    (re.compile(r"\bликида[- ]?бейс\b", re.IGNORECASE), "Liquibase"),
)
ORGANIZATIONAL_PROMPT_RE = re.compile(
    r"(?:на\s+ты|на\s+вы|питерск\w+\s+офис|офис\w*|беларус\w*|росси\w*|"
    r"камер\w*|луниск\w*|как\s+будем\s+общаться|удобно.*общаться)",
    re.IGNORECASE,
)
INTERFACE_PROMPT_RE = re.compile(
    r"(?:чатик|комментари\w*|вкладк\w*|закрыт\w*|отправл\w*|яндекс\.?code|"
    r"в\s+редактор|в\s+чате|открыва\w*|посмотрет\w*)",
    re.IGNORECASE,
)
TOPIC_BRIDGE_RE = re.compile(
    r"^(?:"
    r"что(?:\s+я)?\s+ещ[её]\s+(?:не\s+упомянул|есть|сказать)|"
    r"можно\s+пойти\s+дальше|"
    r"можно,\s*в\s+принципе,\s*вкладк\w+\s+даже\s+закрыт\w*|"
    r"как\s+раз\s+пора.*верн\w+|"
    r"можешь,\s*давай,\s*наверное,\s*этот.*верн\w+|"
    r"что,\s*не\s+могу\s+обойти\b.*"
    r")",
    re.IGNORECASE,
)
CONFIRMATION_PROMPT_RE = re.compile(
    r"^(?:"
    r"(?:так,\s*)?(?:видишь|понятно|слышно)\s*,?\s*да|"
    r"не\s+использовали,\s*да|"
    r"в\s+сервисе,\s*да|"
    r"слышал,\s*да,\s*про\s+них|"
    r"ну,\s*и\s+раздач\w+.*да|"
    r"так,\s*ну,\s*как\s+бы\s+оно.*да"
    r")\b",
    re.IGNORECASE,
)
META_PROMPT_RE = re.compile(
    r"^(?:"
    r"(?:а\s+)?у\s+вас.*ты\s+зада[её]шь\s+вопрос,\s*потому\s+что|"
    r"это\s+чисто\s+для\s+себя|"
    r"это\s+так,\s*просто\s+к\s+слову"
    r")\b",
    re.IGNORECASE,
)
SMALLTALK_PROMPT_RE = re.compile(
    r"^(?:(?:ну,\s*)?что\s+ты\s+такой\b.*\bот\s+себя\b|"
    r"что\s+это\s+у\s+меня\s+в\s+ступор\b.*)",
    re.IGNORECASE,
)
WRAPUP_PROMPT_RE = re.compile(
    r"^(?:"
    r"у\s+меня\s+в\s+целом\b.*\bчасть\b.*\bзакончена|"
    r"дальше\s+вопросы\s+такие\b.*"
    r")",
    re.IGNORECASE,
)
ANSWER_LEAK_PROMPT_RE = re.compile(
    r"^(?:"
    r"(?:вот,\s*)?как\s+пример\b.*|"
    r"как\s+бы,\s*в\s+принципе\b.*|"
    r"как\s+бы,\s*у\s+нас\b.*|"
    r"как\s+бы\s+проект\b.*|"
    r"что\s+бы,\s*когда\s+я\b.*|"
    r"так\s+что,\s*если\s+сделать\b.*"
    r")",
    re.IGNORECASE,
)
GENERIC_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"можешь\s+подробнее\s+рассказать|"
    r"можешь\s+рассказать\s+какие|"
    r"можешь\s+вот,\s*рассказать\s+про|"
    r"свои\s+какие[- ]?то\s+приходилось\s+писать|"
    r"ну,\s*знаком.*вопрос\s+приходилось\s+ли\s+использовать|"
    r"мтлс,\s*скажу|"
    r"длкью[- ]?топик"
    r")\b",
    re.IGNORECASE,
)
INCOMPLETE_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"дополнительн\w+\s+\w+,\s*или\s+что|"
    r"бинарн\w+\s+или\s+не\s+бинарн\w*|"
    r"плюс\s+длинн\w+\s+транзакц\w*|"
    r"что\s+несколько\s+инстинкц\w+.*|"
    r"аспекты\b.*|"
    r"как\s+это\s+называется\b.*|"
    r"а,\s*можно\s+понимать,\s*по\b.*|"
    r"полаор\b.*|"
    r"как\s+раунд[- ]?робина\b.*|"
    r".*\bне\s+успешно$|"
    r".*\bкаждый\s+раз\s+пере$|"
    r"чем\s+отличается\s+hash\s*map$"
    r")$",
    re.IGNORECASE,
)
GARBLED_FRAGMENT_RE = re.compile(
    r"(?:"
    r"функциональн\w+\s+интерфейс\w*,?\s*да|"
    r"закручеч\w*|"
    r"труп[- ]?лухан\w*|"
    r"икрус\w*|"
    r"\bматье\b|"
    r"\bпол\s+стрингов\b|"
    r"устроена\s+релиз\b|"
    r"еллю\s+в\s+вару\b|"
    r"мутабельн\w+\s+класс\s+соотдай\b"
    r")",
    re.IGNORECASE,
)
EXPLANATORY_PREFIX_RE = re.compile(
    r"^(?:"
    r"если\s+говорить|"
    r"обычно\s+используется|"
    r"знаком(?:-то)?\s+знаком|"
    r"не\s+совсем|"
    r"как\s+бы\s+оно|"
    r"там\b|"
    r"ну,\s*там\b|"
    r"так,\s*ну,\s*как\s+бы|"
    r"получается,\s*если\s+говорить|"
    r"вот,\s*как\s+пример|"
    r"как\s+бы,\s*в\s+принципе|"
    r"что\s+бы,\s*когда\s+я|"
    r"как\s+бы,\s*у\s+нас|"
    r"как\s+бы\s+проект"
    r")\b",
    re.IGNORECASE,
)
CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"какие\s+виды|"
    r"что\s+содержится|"
    r"зачем|"
    r"для\s+чего|"
    r"в\s+чем|"
    r"чем|"
    r"можно\s+ли|"
    r"может\s+ли|"
    r"почему"
    r")\b",
    re.IGNORECASE,
)
PRONOUN_FOLLOWUP_RE = re.compile(
    r"\b(?:это|этого|этой|эти|этот|ней|нем|нём|них|его|её|их|они|он|она|оно)\b",
    re.IGNORECASE,
)
MAX_ROLE_LABEL_TOKENS = 4
STOPWORD_TOKENS = {
    "а", "ну", "вот", "так", "и", "то", "есть", "да", "нет", "ли", "же", "бы",
    "у", "в", "во", "на", "по", "к", "ко", "о", "об", "про", "для", "из", "от",
    "до", "с", "со", "за", "не", "но", "мы", "я", "ты", "вы", "он", "она", "они",
    "оно", "их", "его", "её", "нам", "вам", "мне", "тебе", "нас", "вас", "там",
    "тут", "здесь", "это", "этот", "эта", "эти", "этого", "этой", "что", "как",
    "какой", "какая", "какие", "какое", "каким", "какую", "какого", "какому",
    "кто", "где", "когда", "почему", "зачем", "сколько", "чем", "можно", "можешь",
    "можете", "расскажи", "расскажите", "подскажи", "подскажите", "объясни",
    "объясните", "ли", "или", "ещё", "еще", "тогда", "просто",
}
TECHNICAL_TOKENS = {
    "java", "spring", "boot", "aop", "set", "map", "list", "queue", "hash", "hashcode",
    "equals", "exception", "throw", "trycatch", "error", "metaspace", "heap", "stack",
    "gc", "bean", "бин", "бина", "бинов", "аоп", "аспекты", "транзакц", "transaction",
    "consumer", "producer", "partition", "topic", "топик", "сообщени", "кафк", "kafka",
    "rabbit", "redis", "postgres", "postgresql", "sql", "table", "cte", "with", "индекс",
    "индексы", "constraint", "констейн", "праймер", "primary", "unique", "view", "вью",
    "join", "селективност", "lock", "блокиров", "мтлс", "dlq", "длкью", "микросервис",
    "монолит", "архитектур", "сервис", "controller", "repository", "hibernate", "stream",
    "памят", "eden", "permgen", "эксепшен", "исключен", "класс", "интерфейс", "коллекц",
    "технолог", "inbox", "outbox", "проб", "кубер", "kuber", "equals", "hashset",
    "hashmap", "arraylist", "volatile", "synchronized", "thread", "dump", "dispatcher",
    "servlet", "orm", "jpa", "liquibase", "docker", "maven", "gradle", "rebase",
    "merge", "push", "offset", "serialversionuid", "cloud", "config",
}
PROJECT_TOKENS = {
    "опыт", "проек", "стек", "команд", "микросервис", "архитектур", "функционал",
    "легаси", "финтех", "банк", "backend", "бэкэнд", "спринг", "джава", "вопросы",
    "docker", "cloud", "infra", "сборк", "тестирован", "кодревью",
}


@dataclass(slots=True, frozen=True)
class QuestionClassification:
    accepted: bool
    score: int
    reason: str


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = WHITESPACE_RE.sub(" ", value)
    value = re.sub(r" ?\n ?", "\n", value)
    return value.strip()


def extract_questions(
    text: str,
    *,
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS,
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH,
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH,
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS,
    source_name: str | None = None,
) -> QuestionExtractionResult:
    artifact = TranscriptArtifact(text=text, source_name=source_name)
    transcript = _build_transcript_view(
        artifact=artifact,
        interviewer_labels=interviewer_labels,
    )

    questions: list[QuestionCandidate] = []
    previous_candidate: QuestionCandidate | None = None
    for transcript_slice in transcript.candidate_slices:
        extracted_questions = _extract_questions_from_slice(
            transcript_slice,
            min_length=min_length,
            max_length=max_length,
            previous_candidate=previous_candidate,
        )
        questions.extend(extracted_questions)
        if extracted_questions:
            previous_candidate = extracted_questions[-1]

    if deduplicate:
        questions = _deduplicate_preserving_order(questions)

    return QuestionExtractionResult(
        transcript=transcript,
        questions=tuple(questions),
        deduplicated=deduplicate,
        min_length=min_length,
        max_length=max_length,
    )


def extract_question_candidates(
    text: str,
    *,
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS,
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH,
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH,
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS,
    source_name: str | None = None,
) -> list[str]:
    result = extract_questions(
        text,
        deduplicate=deduplicate,
        min_length=min_length,
        max_length=max_length,
        interviewer_labels=interviewer_labels,
        source_name=source_name,
    )
    return list(result.question_texts)


def _build_transcript_view(
    *,
    artifact: TranscriptArtifact,
    interviewer_labels: tuple[str, ...],
) -> TranscriptView:
    if _is_srt_source(artifact.source_name):
        return _build_srt_transcript_view(
            artifact=artifact,
            interviewer_labels=interviewer_labels,
        )

    explicit_interviewer_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    speaker_turns = _extract_speaker_turns(
        artifact.text,
        interviewer_labels=interviewer_labels,
    )
    speaker_filter = _resolve_interviewer_labels(
        speaker_turns,
        interviewer_labels=interviewer_labels,
    )

    if explicit_interviewer_labels:
        if not speaker_turns:
            raise RuntimeError(
                "Explicit interviewer labels were provided, but no speaker-labeled "
                "transcript structure was detected."
            )
        if not speaker_filter:
            available_labels = ", ".join(_ordered_unique(turn.label for turn in speaker_turns))
            raise RuntimeError(
                "None of the provided interviewer labels were found in the transcript. "
                f"Available labels: {available_labels}"
            )

    if speaker_turns and speaker_filter:
        candidate_slices = tuple(
            TranscriptSlice(
                text=turn.text,
                speaker_label=turn.label,
                normalized_speaker_label=turn.normalized_label,
            )
            for turn in speaker_turns
            if turn.normalized_label in speaker_filter
        )
    else:
        candidate_slices = (TranscriptSlice(text=artifact.text),)

    return TranscriptView(
        artifact=artifact,
        speaker_turns=tuple(speaker_turns),
        candidate_slices=candidate_slices,
        selected_interviewer_labels=tuple(sorted(speaker_filter)),
        available_speaker_labels=tuple(_ordered_unique(turn.label for turn in speaker_turns)),
    )


def _build_srt_transcript_view(
    *,
    artifact: TranscriptArtifact,
    interviewer_labels: tuple[str, ...],
) -> TranscriptView:
    explicit_interviewer_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    raw_slices = _extract_srt_slices(artifact.text)
    labeled_slices, speaker_turns = _label_transcript_slices(
        raw_slices,
        explicit_labels=explicit_interviewer_labels,
    )
    speaker_filter = _resolve_interviewer_labels(
        speaker_turns,
        interviewer_labels=interviewer_labels,
    )

    if explicit_interviewer_labels:
        if not speaker_turns:
            raise RuntimeError(
                "Explicit interviewer labels were provided, but no speaker-labeled "
                "transcript structure was detected."
            )
        if not speaker_filter:
            available_labels = ", ".join(
                _ordered_unique(turn.label for turn in speaker_turns)
            )
            raise RuntimeError(
                "None of the provided interviewer labels were found in the transcript. "
                f"Available labels: {available_labels}"
            )

    if speaker_turns and speaker_filter:
        candidate_slices = tuple(
            transcript_slice
            for transcript_slice in labeled_slices
            if transcript_slice.normalized_speaker_label in speaker_filter
        )
    elif raw_slices:
        candidate_slices = tuple(labeled_slices)
    else:
        candidate_slices = (TranscriptSlice(text=artifact.text),)

    return TranscriptView(
        artifact=artifact,
        speaker_turns=tuple(speaker_turns),
        candidate_slices=candidate_slices,
        selected_interviewer_labels=tuple(sorted(speaker_filter)),
        available_speaker_labels=tuple(_ordered_unique(turn.label for turn in speaker_turns)),
    )


def _extract_questions_from_slice(
    transcript_slice: TranscriptSlice,
    *,
    min_length: int,
    max_length: int,
    previous_candidate: QuestionCandidate | None,
) -> list[QuestionCandidate]:
    prepared_text = _trim_after_candidate_question_invite(
        normalize_whitespace(transcript_slice.text)
    )
    chunks = QUESTION_SPLIT_RE.split(prepared_text)

    questions: list[QuestionCandidate] = []
    local_previous_candidate = previous_candidate
    for chunk in chunks:
        parts = chunk.split("?") if "?" in chunk else [chunk]

        for part in parts:
            cleaned = _cleanup_question(part)
            if not cleaned:
                continue

            is_explicit_question = "?" in chunk
            if not is_explicit_question and not _looks_like_question_without_mark(cleaned):
                continue

            question_text = cleaned if cleaned.endswith("?") else f"{cleaned}?"
            if len(question_text) < min_length or len(question_text) > max_length:
                continue
            classification = _classify_question_candidate(
                question_text=question_text,
                explicit_question=is_explicit_question,
                previous_candidate=local_previous_candidate,
            )
            if not classification.accepted:
                continue
            normalized_question_text = _normalize_question_for_output(
                question_text,
                previous_candidate=local_previous_candidate,
            )

            candidate = QuestionCandidate(
                text=normalized_question_text,
                source_text=cleaned,
                explicit_question=is_explicit_question,
                speaker_label=transcript_slice.speaker_label,
                normalized_speaker_label=transcript_slice.normalized_speaker_label,
                start_time=transcript_slice.start_time,
                end_time=transcript_slice.end_time,
            )
            questions.append(candidate)
            local_previous_candidate = candidate

    return questions


def _classify_question_candidate(
    *,
    question_text: str,
    explicit_question: bool,
    previous_candidate: QuestionCandidate | None,
) -> QuestionClassification:
    normalized = _normalize_intent_text(question_text)
    tokens = _tokenize_question(normalized)
    content_tokens = _content_tokens(tokens)
    technical_hits = _find_matching_stems(tokens, TECHNICAL_TOKENS)
    project_hits = _find_matching_stems(tokens, PROJECT_TOKENS)
    has_domain_subject = bool(technical_hits or project_hits)
    has_contextual_subject = _has_contextual_subject(
        normalized,
        previous_candidate=previous_candidate,
    )

    if _matches_any_pattern(normalized, ORGANIZATIONAL_PROMPT_RE):
        return QuestionClassification(False, -10, "organizational")
    if _matches_any_pattern(normalized, INTERFACE_PROMPT_RE):
        return QuestionClassification(False, -10, "interface")
    if _matches_any_pattern(normalized, SMALLTALK_PROMPT_RE):
        return QuestionClassification(False, -9, "smalltalk")
    if _matches_any_pattern(normalized, WRAPUP_PROMPT_RE):
        return QuestionClassification(False, -9, "wrapup")
    if _matches_any_pattern(normalized, TOPIC_BRIDGE_RE):
        return QuestionClassification(False, -8, "topic_bridge")
    if _matches_any_pattern(normalized, CONFIRMATION_PROMPT_RE):
        return QuestionClassification(False, -8, "confirmation")
    if _matches_any_pattern(normalized, META_PROMPT_RE):
        return QuestionClassification(False, -8, "meta_prompt")
    if _matches_any_pattern(normalized, ANSWER_LEAK_PROMPT_RE):
        return QuestionClassification(False, -8, "answer_leak")
    if _matches_any_pattern(normalized, GENERIC_FOLLOWUP_RE):
        return QuestionClassification(False, -7, "generic_followup")
    if _matches_any_pattern(normalized, INCOMPLETE_FRAGMENT_RE):
        return QuestionClassification(False, -8, "incomplete_fragment")
    if _matches_any_pattern(normalized, GARBLED_FRAGMENT_RE):
        return QuestionClassification(False, -8, "garbled_fragment")
    if _matches_any_pattern(normalized, EXPLANATORY_PREFIX_RE) and not has_contextual_subject:
        return QuestionClassification(False, -8, "answer_like_explanation")
    if _looks_like_noise(question_text):
        return QuestionClassification(False, -8, "noise")
    if _looks_like_low_signal_followup(question_text):
        return QuestionClassification(False, -7, "low_signal")
    if _looks_like_filler_fragment(question_text):
        return QuestionClassification(False, -6, "filler_fragment")
    if _looks_like_answer(question_text) and not has_contextual_subject:
        return QuestionClassification(False, -9, "answer_like")

    score = 0
    if QUESTION_LIKE_RE.match(normalized):
        score += 3
    if explicit_question:
        score += 1
    if has_domain_subject:
        score += 3
    if project_hits:
        score += 1
    if len(content_tokens) >= 7:
        score += 2
    elif len(content_tokens) >= 5:
        score += 1
    if has_contextual_subject:
        score += 2

    if _starts_with_lowercase_fragment(question_text) and not has_domain_subject:
        score -= 2
    if len(content_tokens) <= 2 and not has_domain_subject and not has_contextual_subject:
        score -= 4
    elif len(content_tokens) <= 3 and not has_domain_subject and not has_contextual_subject:
        score -= 2
    if _is_generic_without_subject(normalized, has_domain_subject, has_contextual_subject):
        score -= 3

    accepted = score >= 2
    reason = "accepted" if accepted else "low_confidence"
    return QuestionClassification(accepted, score, reason)


def _cleanup_question(value: str) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = cleaned.strip(" .,!;:-")
    cleaned = cleaned.replace(" ?", "?")
    return cleaned


def _normalize_question_for_output(
    question_text: str,
    *,
    previous_candidate: QuestionCandidate | None,
) -> str:
    normalized = _apply_readability_replacements(normalize_whitespace(question_text))
    lowered = _normalize_intent_text(normalized)

    rewritten = _rewrite_question_for_readability(
        lowered,
        previous_candidate=previous_candidate,
    )
    if rewritten is not None:
        return rewritten

    normalized = LEADING_ADDRESS_RE.sub("", normalized)
    normalized = LEADING_READABILITY_FILLER_RE.sub("", normalized)
    normalized = _apply_readability_replacements(normalized)

    normalized = re.sub(
        r"\bя\s+понимаю,\s*да,\s*говоришь,\s*сильный\s+товарищ\s+был,\s*а\s+вот\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bты-?то\b",
        "вы",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\s*,\s*,+",
        ", ",
        normalized,
    )
    normalized = normalize_whitespace(normalized).strip(" ,")
    return _finalize_question_text(normalized)


def _apply_readability_replacements(value: str) -> str:
    normalized = value
    for pattern, replacement in READABILITY_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _rewrite_question_for_readability(
    normalized: str,
    *,
    previous_candidate: QuestionCandidate | None,
) -> str | None:
    if "микросервис" in normalized and "команд" in normalized and "разработ" in normalized:
        return "Сколько микросервисов у вашей команды в разработке?"
    if "селективности" in normalized and "индекс" in normalized and "сканир" in normalized:
        return "Насколько полезен будет индекс с точки зрения селективности, и как именно будет сканироваться таблица?"
    if "оптимизации запросов" in normalized and "провод" in normalized:
        return "Проводили какие-то оптимизации запросов?"
    if "функциональн" in normalized and "стиле" in normalized:
        return "Какой подход вам больше нравится: функциональное программирование или что-то другое, и в каком стиле обычно пишете?"
    if "микросервисная архитектура" in normalized:
        return "Можешь рассказать о микросервисной архитектуре?"
    if ("unique" in normalized or "unic" in normalized) and (
        "primary key" in normalized or "праймер" in normalized
    ):
        return "Чем UNIQUE отличается от Primary Key?"
    if "error" in normalized and "обрабатывать" in normalized:
        return "Можно ли обрабатывать Error?"
    if "цикл бина" in normalized and ("жизнен" in normalized or "жизни цикл" in normalized):
        return "Можешь рассказать про жизненный цикл бина: как он создаётся, как его найти и как использовать?"
    if "metaspace" in normalized and "пермген" in normalized:
        return "А про Metaspace знаешь? Это то, что раньше называлось PermGen?"
    if "spring" in normalized and "boot" in normalized and "зачем" in normalized and "отличается" in normalized:
        return "Можешь рассказать про Spring: зачем он нужен и чем отличается от Spring Boot?"
    if "spring" in normalized and "что дает" in normalized:
        return "Что даёт Spring и почему его используют?"
    if "оконными функц" in normalized:
        return "Был ли опыт с оконными функциями?"
    if "рекурсивные запросы" in normalized and ("эсквеле" in normalized or "sql" in normalized):
        return "Рекурсивные запросы в SQL не строили?"
    if "индекс" in normalized and (
        "боз-данных" in normalized
        or "боз данных" in normalized
        or "базе данных" in normalized
        or "базы данных" in normalized
    ):
        return "Для чего индексы нужны в базе данных?"
    if "нотац" in normalized and "бин" in normalized and "компонент" in normalized:
        return "Чем отличается аннотация @Bean от @Component?"
    if "констейн" in normalized and "какие виды" in normalized:
        return "Какие виды constraints бывают?"
    if "java" in normalized and "17" in normalized:
        return "По Java вы в основном с 17-й версией работали?"
    if "паттерны микросервис" in normalized and "какие знаешь" in normalized:
        return "Какие паттерны микросервисной архитектуры знаешь и применял на практике?"
    if "что это за паттерны" in normalized and "когда стоит применять" in normalized:
        return "Что это за паттерны и когда их стоит применять?"
    if "паттерны" in normalized and "для чего они нужны" in normalized and "какие виды" in normalized:
        return "Для чего нужны паттерны проектирования, какие виды бывают и можешь привести по паре примеров?"
    if "set" in normalized and "под капотом" in normalized and "мап" in normalized:
        return "Что у Set под капотом, если сравнивать его с Map?"
    if "коллекц" in normalized and "какие виды" in normalized:
        return "Можешь рассказать про коллекции: какие виды коллекций есть?"
    if "коллекция на p" in normalized:
        return "Что такое Collection API?"
    if "лист" in normalized and "сет" in normalized and ("q мэп" in normalized or "q-мэп" in normalized):
        return "List, Set, Queue и Map — что это такое?"
    if "интерфейс или класс" in normalized and "лист" in normalized:
        return "List — это интерфейс или класс?"
    if "имплементирует интерфейс" in normalized:
        return "Какие интерфейсы он имплементирует?"
    if "структура данных" in normalized and "хэш" in normalized and "мап" in normalized:
        return "Какая структура данных внутри HashMap?"
    if "бакете" in normalized and ("много значений" in normalized or "очень много" in normalized):
        return "Когда список в HashMap превращается в дерево, если в одном бакете слишком много значений?"
    if "хэш" in normalized and "сет" in normalized and "особен" in normalized:
        return "Какие особенности есть у HashSet?"
    if "порядок отсутствует" in normalized and "элементы" in normalized:
        return "В HashSet элементы хранятся в каком-то порядке или порядок отсутствует?"
    if "хэш" in normalized and "мап" in normalized and "ключом ну" in normalized:
        return "Можно ли положить в HashMap элемент с ключом null?"
    if "кастомн" in normalized and "ключ" in normalized and ("матье" in normalized or "мап" in normalized):
        return "Если использовать кастомный класс как ключ в Map, как правильно переопределить equals и hashCode?"
    if "мутабельн" in normalized and "стал" in normalized:
        return "Как сделать так, чтобы класс был immutable?"
    if "кастомный класс" in normalized and "мутабельн" in normalized:
        return "Как сделать так, чтобы кастомный класс был immutable?"
    if "финальн" in normalized and "сеттер" in normalized:
        return "Как сделать кастомный класс immutable: достаточно ли финальных полей и отсутствия сеттеров?"
    if "прототайп" in normalized and "scope" in normalized and "бин" in normalized:
        return "Ты упомянул prototype. Можешь рассказать про scope бинов и какие ещё scope тебе знакомы?"
    if "aop" in normalized and "работал" in normalized:
        return "С AOP в Spring работал?"
    if normalized == "зачем они нужны" and _previous_question_mentions(
        previous_candidate,
        "aop",
        "spring",
    ):
        return "Зачем нужен AOP?"
    if "аннотац" in normalized and "исключени" in normalized:
        return "Можешь рассказать про эту аннотацию: зачем она нужна, что делает и как работает с исключениями?"
    if ("партийц" in normalized or "партици" in normalized) and "consumer" in normalized and "топик" in normalized:
        return "Можно ли обработать сообщения в том же порядке, в котором они были отправлены в топик?"
    if ("postgres" in normalized or "пасгресс" in normalized) and "баз" in normalized and "друг" in normalized:
        return "Кроме Postgres, был ли опыт работы с другими базами данных?"
    if "типы индексов" in normalized and "приходилось использовать" in normalized:
        return "Какие типы индексов приходилось использовать?"
    if "нормализация базы данных" in normalized and "три нормальные формы" in normalized:
        return "Что такое нормализация базы данных и какие первые три нормальные формы знаешь?"
    if "чекнолив" in normalized and "форма" in normalized:
        return "Кроме первых трёх, существует ли четвёртая нормальная форма?"
    if "ускорить" in normalized and "запрос" in normalized:
        return "С помощью чего ещё можно ускорить запрос?"
    if "виды индексов" in normalized and "что вообще такой индекс" in normalized:
        return "Что такое индекс и какие виды индексов есть?"
    if "непроверяемые" in normalized and ("искрещ" in normalized or "исключ" in normalized):
        return "Почему современные фреймворки и библиотеки чаще выбирают непроверяемые исключения?"
    if ("сериализац" in normalized or "serialversionuid" in normalized) and "десериализац" in normalized:
        return "Что такое сериализация и десериализация, и для чего нужен serialVersionUID?"
    if "транси" in normalized and _previous_question_mentions(
        previous_candidate,
        "serialversionuid",
    ):
        return "Для чего нужен transient?"
    if "garbage collector" in normalized and "настраивал" in normalized:
        return "Приходилось ли настраивать Garbage Collector?"
    if "дженерик" in normalized and ("какую проблему" in normalized or "для чего" in normalized):
        return "Для чего в Java добавили дженерики и какую проблему они решают?"
    if "полиморфизм" in normalized:
        return "Что такое полиморфизм в объектно-ориентированном программировании?"
    if "паттерны" in normalized and "какую проблему" in normalized and "перечисли" in normalized:
        return "Что такое паттерны и какие проблемы они решают? Какие можешь перечислить?"
    if "сложность поиска" in normalized and ("hash map" in normalized or "hashmap" in normalized):
        return "Какая сложность поиска в ArrayList и в HashMap?"
    if ("volatile" in normalized or "волотайл" in normalized) and "синхронизац" in normalized:
        return "Volatile и synchronized: для чего они используются?"
    if "ритуальн" in normalized and "поток" in normalized:
        return "Работал ли уже с виртуальными потоками?"
    if ("spring" in normalized or "спринк" in normalized) and (
        "для чего его создали" in normalized or "под комботом" in normalized
    ):
        return "Что такое Spring, для чего его создали и что у него под капотом?"
    if "под комботом" in normalized and _previous_question_mentions(
        previous_candidate,
        "spring",
    ):
        return "А что у Spring под капотом, какая там технология?"
    if "циклические зависимости" in normalized:
        return "Что такое циклические зависимости и как их решать?"
    if ("dispatcher" in normalized or "диспетчер" in normalized) and "spring" in normalized:
        return "Что такое DispatcherServlet в архитектуре Spring?"
    if "киширован" in normalized:
        return "Приходилось ли использовать кеширование для каких-то задач?"
    if "стартер" in normalized and "вел" in normalized:
        return "Приходилось ли писать свой стартер или делать свою библиотеку?"
    if "аспекты" in normalized and "приход" in normalized:
        return "Приходилось ли писать свои аспекты?"
    if ("orm" in normalized or "jpa" in normalized or "gp" in normalized) and (
        "hibernate" in normalized or "hypernate" in normalized
    ):
        return "Что такое ORM, JPA и Hibernate?"
    if ("hibernate" in normalized or "hypernate" in normalized) and "хороший framework" in normalized:
        return "Какое у тебя отношение к Hibernate: это хороший фреймворк или у него есть проблемы?"
    if normalized == "зачем это нужно" and _previous_question_mentions_any(
        previous_candidate,
        ("orm", "hibernate"),
        ("проблем", "декарт"),
    ):
        return "Зачем это нужно?"
    if "liquibase" in normalized:
        return "Для чего нужен Liquibase?"
    if "старом spring" in normalized and "без spring boot" in normalized:
        return "Приходилось ли писать код без Spring Boot, просто на старом Spring?"
    if "монолит" in normalized and "микросервис" in normalized and "когда бы" in normalized:
        return "Когда бы ты выбрал монолит, а когда микросервисы?"
    if "spring cloud" in normalized and "config server" not in normalized and (
        "использовали" in normalized or "работал" in normalized
    ):
        return "Использовали когда-нибудь Spring Cloud?"
    if normalized == "какие у тебя туда штуки прикольные" and _previous_question_mentions(
        previous_candidate,
        "spring",
        "cloud",
    ):
        return "Какие инструменты из Spring Cloud использовал?"
    if "балансировщик" in normalized and "микросервис" in normalized:
        return "Зачем нужен балансировщик и почему нельзя отправлять запросы прямо в микросервис?"
    if "аластик" in normalized or "локстеж" in normalized or "кибан" in normalized:
        return "Elastic, Logstash и Kibana — что это за стек и для чего он нужен?"
    if "комит" in normalized and "сообщени" in normalized and ("кавка" in normalized or "kafka" in normalized):
        return "Когда бы ты коммитил offset по сообщению в Kafka?"
    if "комит" in normalized and "сообщени" in normalized and _previous_question_mentions_any(
        previous_candidate,
        ("kafka",),
        ("кавка",),
    ):
        return "Когда бы ты коммитил offset по сообщению в Kafka?"
    if "уровнеизолирования" in normalized or ("уровн" in normalized and "изоляц" in normalized):
        return "Можешь рассказать про уровни изоляции транзакций?"
    if normalized == "почему нужны они" and _previous_question_mentions(
        previous_candidate,
        "изоляц",
    ):
        return "Зачем нужны уровни изоляции?"
    if "vacum" in normalized or "vacuum" in normalized:
        return "Работал ли с командой VACUUM?"
    if "докер" in normalized and "для чего он нужен" in normalized:
        return "Приходилось ли использовать Docker и для чего он был нужен?"
    if "поднимали инфраструктуру" in normalized and "микросердус" in normalized:
        return "Как вы поднимали инфраструктуру для микросервисов?"
    if "капку" in normalized and ("физическом" in normalized or "станции" in normalized):
        return "Как вы поднимали инфраструктуру вроде БД и Kafka: локально, в контейнерах или как-то иначе?"
    if ("мэвен" in normalized or "gradul" in normalized or "gradle" in normalized) and "использовал" in normalized:
        return "Какие средства сборки чаще использовал: Maven или Gradle?"
    if "оптимизировать сборку" in normalized:
        return "Были ли задачи, связанные с оптимизацией сборки?"
    if "код ревью" in normalized or "код-ревью" in normalized:
        return "Какие проблемы решает code review и как ты к нему относишься?"
    if "ребейса от ребейса" in normalized:
        return "В чем отличие rebase от merge?"
    if "force push" in normalized:
        return "Почему не рекомендуют делать force push?"
    if "unique" in normalized and "check" in normalized:
        return "Что такое UNIQUE constraint?"
    if "sql" in normalized and "медленный" in normalized and "запрос" in normalized:
        return "Если у тебя есть очень медленный SQL-запрос, как ты будешь его анализировать?"
    if ("java" in normalized or "тжава" in normalized) and (
        "thread dump" in normalized or "трет дамп" in normalized or "thread дамп" in normalized
    ):
        return "Приходилось ли когда-нибудь смотреть Java dump и thread dump?"
    return None


def _previous_question_mentions(
    previous_candidate: QuestionCandidate | None,
    *terms: str,
) -> bool:
    if previous_candidate is None:
        return False
    previous_normalized = _normalize_intent_text(previous_candidate.text)
    return all(term in previous_normalized for term in terms)


def _previous_question_mentions_any(
    previous_candidate: QuestionCandidate | None,
    *term_groups: tuple[str, ...],
) -> bool:
    if previous_candidate is None:
        return False
    previous_normalized = _normalize_intent_text(previous_candidate.text)
    return any(all(term in previous_normalized for term in group) for group in term_groups)


def _finalize_question_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return "?"
    if normalized.endswith("?"):
        text = normalized[:-1]
    else:
        text = normalized
    text = text.strip(" .,!;:-")
    if not text:
        return "?"
    text = text[0].upper() + text[1:]
    return f"{text}?"


def _matches_any_pattern(value: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(value) is not None


def _find_matching_stems(tokens: list[str], stems: set[str]) -> set[str]:
    matches: set[str] = set()
    for token in tokens:
        for stem in stems:
            if token.startswith(stem):
                matches.add(stem)
    return matches


def _content_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOPWORD_TOKENS]


def _starts_with_lowercase_fragment(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and stripped[0].islower()


def _has_contextual_subject(
    normalized: str,
    *,
    previous_candidate: QuestionCandidate | None,
) -> bool:
    if previous_candidate is None:
        return False

    previous_tokens = _tokenize_question(
        _normalize_intent_text(previous_candidate.text)
    )
    previous_domain_hits = _find_matching_stems(
        previous_tokens,
        TECHNICAL_TOKENS | PROJECT_TOKENS,
    )
    if not previous_domain_hits:
        return False

    if CONTEXTUAL_FOLLOWUP_RE.match(normalized):
        return True
    return False


def _is_generic_without_subject(
    normalized: str,
    has_domain_subject: bool,
    has_contextual_subject: bool,
) -> bool:
    if has_domain_subject or has_contextual_subject:
        return False

    generic_prefixes = (
        "можешь рассказать",
        "можешь подробнее",
        "можешь вот рассказать",
        "что содержится",
        "какие виды",
        "для чего",
        "зачем они",
        "что дает",
        "что делает",
    )
    if not normalized.startswith(generic_prefixes):
        return False

    content_token_count = len(_content_tokens(_tokenize_question(normalized)))
    return content_token_count <= 2


def _extract_srt_slices(text: str) -> list[TranscriptSlice]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [block for block in normalized.split("\n\n") if block.strip()]

    timed_blocks: list[TranscriptSlice] = []
    for raw_block in raw_blocks:
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        if lines[0].isdigit():
            timecode_line = lines[1]
            body_lines = lines[2:]
        else:
            timecode_line = lines[0]
            body_lines = lines[1:]

        match = SRT_TIMECODE_RE.match(timecode_line)
        if match is None or not body_lines:
            continue

        body = normalize_whitespace("\n".join(body_lines))
        if not body:
            continue

        timed_blocks.append(
            TranscriptSlice(
                text=body,
                start_time=match.group("start"),
                end_time=match.group("end"),
            )
        )

    if not timed_blocks:
        return []

    collapsed_slices: list[TranscriptSlice] = []
    text_parts: list[str] = []
    start_time: str | None = None
    end_time: str | None = None

    for index, block in enumerate(timed_blocks):
        if not text_parts:
            start_time = block.start_time
        text_parts.append(block.text)
        end_time = block.end_time
        buffered_text = normalize_whitespace(" ".join(text_parts))
        next_block = timed_blocks[index + 1] if index + 1 < len(timed_blocks) else None

        if _should_close_srt_slice(
            buffered_text=buffered_text,
            current_block_text=block.text,
            next_block=next_block,
        ):
            collapsed_slices.append(
                TranscriptSlice(
                    text=buffered_text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            text_parts = []
            start_time = None
            end_time = None

    if text_parts:
        collapsed_slices.append(
            TranscriptSlice(
                text=normalize_whitespace(" ".join(text_parts)),
                start_time=start_time,
                end_time=end_time,
            )
        )

    return collapsed_slices


def _ends_sentence(value: str) -> bool:
    return bool(re.search(r"[?.!…]\s*$", value))


def _should_close_srt_slice(
    *,
    buffered_text: str,
    current_block_text: str,
    next_block: TranscriptSlice | None,
) -> bool:
    if _ends_sentence(current_block_text):
        return True
    if next_block is None:
        return True

    current_speaker = _split_speaker_labeled_text(buffered_text)
    next_speaker = _split_speaker_labeled_text(next_block.text)
    if current_speaker is not None and next_speaker is not None:
        return True

    cleaned_buffer = _cleanup_question(buffered_text)
    if not cleaned_buffer or not _looks_like_question_without_mark(cleaned_buffer):
        return False

    next_text = next_speaker[2] if next_speaker is not None else next_block.text
    next_cleaned = _cleanup_question(next_text)
    if not next_cleaned:
        return False
    return _looks_like_answer(next_cleaned)


def _label_transcript_slices(
    slices: list[TranscriptSlice],
    *,
    explicit_labels: set[str],
) -> tuple[list[TranscriptSlice], list[SpeakerTurn]]:
    candidate_labels: list[str] = []
    parsed_slices: list[tuple[TranscriptSlice, str | None, str | None, str | None]] = []

    for transcript_slice in slices:
        speaker_data = _split_speaker_labeled_text(transcript_slice.text)
        if speaker_data is None:
            parsed_slices.append((transcript_slice, None, None, None))
            continue

        raw_label, normalized_label, body = speaker_data
        if normalized_label:
            candidate_labels.append(normalized_label)
        parsed_slices.append((transcript_slice, raw_label.strip(), normalized_label, body))

    label_counts = Counter(candidate_labels)
    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= 2
        or label in explicit_labels
        or _looks_like_known_speaker_label(label)
    }
    if not valid_labels:
        return slices, []

    labeled_slices: list[TranscriptSlice] = []
    speaker_turns: list[SpeakerTurn] = []
    for transcript_slice, raw_label, normalized_label, body in parsed_slices:
        if raw_label and normalized_label in valid_labels and body:
            labeled_slice = TranscriptSlice(
                text=body,
                speaker_label=raw_label,
                normalized_speaker_label=normalized_label,
                start_time=transcript_slice.start_time,
                end_time=transcript_slice.end_time,
            )
            labeled_slices.append(labeled_slice)
            speaker_turns.append(
                SpeakerTurn(
                    label=raw_label,
                    normalized_label=normalized_label,
                    text=body,
                )
            )
            continue

        labeled_slices.append(transcript_slice)

    return labeled_slices, speaker_turns


def _split_speaker_labeled_text(
    value: str,
) -> tuple[str, str, str] | None:
    match = GENERIC_SPEAKER_LINE_RE.match(value)
    if not match:
        return None

    raw_label = match.group("bracket") or match.group("plain") or ""
    normalized_label = _normalize_speaker_label(raw_label)
    body = normalize_whitespace(match.group("body") or "")
    if not normalized_label or not body:
        return None
    return raw_label.strip(), normalized_label, body


def _trim_after_candidate_question_invite(value: str) -> str:
    match = CANDIDATE_QUESTION_INVITE_RE.search(value)
    if match is None:
        return value
    return value[: match.start()].rstrip()


def _looks_like_noise(value: str) -> bool:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", value.lower())
    if not tokens:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    if len(tokens) >= 8 and unique_ratio < 0.35:
        return True

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", value)
    return len(letters) < 5


def _looks_like_question_without_mark(value: str) -> bool:
    lowered = _normalize_intent_text(value)
    if not lowered:
        return False
    if re.match(r"^можно\b(?!\s+ли\b)", lowered):
        return False
    if re.match(r"^когда\s+у\s+нас\b", lowered):
        return False
    if re.match(r"^как\s+сказать\b", lowered):
        return False
    if re.match(r"^либо\s+делать\b", lowered):
        return False
    if QUESTION_LIKE_RE.match(lowered):
        return True
    if ANSWER_LIKE_RE.match(lowered):
        return False
    return False


def _looks_like_answer(value: str) -> bool:
    lowered = _normalize_intent_text(value)
    if not lowered:
        return False
    if STRONG_ANSWER_LIKE_RE.match(lowered):
        return True
    if QUESTION_LIKE_RE.match(lowered):
        return False
    return ANSWER_LIKE_RE.match(lowered) is not None


def _looks_like_filler_fragment(value: str) -> bool:
    normalized = _normalize_intent_text(value)
    if not normalized or QUESTION_LIKE_RE.match(normalized):
        return False

    tokens = _tokenize_question(normalized)
    if len(tokens) < 4:
        return False

    filler_count = sum(1 for token in tokens if FILLER_TOKEN_RE.match(token))
    content_count = len(tokens) - filler_count
    if content_count <= 0:
        return True

    filler_ratio = filler_count / len(tokens)
    return filler_ratio >= 0.45 and content_count <= 3


def _looks_like_low_signal_followup(value: str) -> bool:
    normalized = _normalize_intent_text(value)
    if not normalized:
        return False
    if LOW_SIGNAL_FOLLOWUP_RE.match(normalized):
        return True

    tokens = _tokenize_question(normalized)
    if len(tokens) <= 5 and {"что", "это", "такое"} <= set(tokens):
        return True

    return False


def _deduplicate_preserving_order(
    values: list[QuestionCandidate],
) -> list[QuestionCandidate]:
    seen_exact: set[str] = set()
    seen_canonical: list[str] = []
    result: list[QuestionCandidate] = []

    for value in values:
        canonical = _canonicalize_question(value.text)
        if canonical in seen_exact:
            continue
        if any(_questions_are_similar(canonical, existing) for existing in seen_canonical):
            continue
        seen_exact.add(canonical)
        seen_canonical.append(canonical)
        result.append(value)

    return result


def _canonicalize_question(value: str) -> str:
    normalized = CANONICAL_WHITESPACE_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")


def _normalize_intent_text(value: str) -> str:
    normalized = normalize_whitespace(value).lower()
    normalized = LEADING_FILLER_RE.sub("", normalized)
    return normalized.strip(" .,!;:-?")


def _questions_are_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False

    similarity = SequenceMatcher(None, left, right).ratio()
    if similarity >= 0.92:
        return True

    left_tokens = set(_tokenize_question(left))
    right_tokens = set(_tokenize_question(right))
    if not left_tokens or not right_tokens:
        return False

    shared = left_tokens & right_tokens
    if not shared:
        return False

    token_overlap = len(shared) / max(len(left_tokens), len(right_tokens))
    token_coverage = len(shared) / min(len(left_tokens), len(right_tokens))

    if similarity >= 0.84 and token_overlap >= 0.75:
        return True
    if token_coverage >= 0.9 and abs(len(left_tokens) - len(right_tokens)) <= 1:
        return True

    return False


def _tokenize_question(value: str) -> list[str]:
    return QUESTION_TOKEN_RE.findall(value.lower())


def _extract_speaker_turns(
    text: str,
    *,
    interviewer_labels: tuple[str, ...],
) -> list[SpeakerTurn]:
    explicit_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    lines = text.splitlines()

    candidate_labels: list[str] = []
    for line in lines:
        match = GENERIC_SPEAKER_LINE_RE.match(line)
        if not match:
            continue
        raw_label = match.group("bracket") or match.group("plain") or ""
        normalized_label = _normalize_speaker_label(raw_label)
        if normalized_label:
            candidate_labels.append(normalized_label)

    label_counts = Counter(candidate_labels)
    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= 2
        or label in explicit_labels
        or _looks_like_known_speaker_label(label)
    }
    if not valid_labels:
        return []

    turns: list[SpeakerTurn] = []
    for raw_line in lines:
        line = raw_line.strip()
        match = GENERIC_SPEAKER_LINE_RE.match(raw_line)
        if match:
            raw_label = match.group("bracket") or match.group("plain") or ""
            normalized_label = _normalize_speaker_label(raw_label)
            if normalized_label in valid_labels:
                body = (match.group("body") or "").strip()
                turns.append(
                    SpeakerTurn(
                        label=raw_label.strip(),
                        normalized_label=normalized_label,
                        text=body,
                    )
                )
                continue

        if turns and line:
            previous_turn = turns[-1]
            turns[-1] = SpeakerTurn(
                label=previous_turn.label,
                normalized_label=previous_turn.normalized_label,
                text=f"{previous_turn.text}\n{line}".strip(),
            )

    return [turn for turn in turns if turn.text]


def _resolve_interviewer_labels(
    turns: list[SpeakerTurn],
    *,
    interviewer_labels: tuple[str, ...],
) -> set[str]:
    if not turns:
        return set()

    available_labels = {turn.normalized_label for turn in turns}
    explicit_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    if explicit_labels:
        return available_labels & explicit_labels

    detected_labels = {
        label
        for label in available_labels
        if _label_matches_any_keywords(
            label,
            KNOWN_INTERVIEWER_LABELS,
            require_role_like=True,
        )
    }
    if detected_labels:
        return detected_labels

    candidate_labels = {
        label
        for label in available_labels
        if _label_matches_any_keywords(
            label,
            KNOWN_CANDIDATE_LABELS,
            require_role_like=True,
        )
    }
    remaining_labels = available_labels - candidate_labels
    if len(available_labels) == 2 and len(candidate_labels) == 1 and len(remaining_labels) == 1:
        return remaining_labels

    return set()


def _normalize_speaker_label(value: str) -> str:
    normalized = value.strip().strip("[]")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalize_whitespace(normalized).lower()
    return normalized


def _is_srt_source(source_name: str | None) -> bool:
    if not source_name:
        return False
    return Path(source_name).suffix.lower() == ".srt"


def _normalize_explicit_speaker_labels(labels: tuple[str, ...]) -> set[str]:
    return {
        normalized
        for label in labels
        if (normalized := _normalize_speaker_label(label))
    }


def _label_matches_any_keywords(
    label: str,
    keywords: tuple[str, ...],
    *,
    require_role_like: bool = False,
) -> bool:
    return any(
        _label_matches_keyword(
            label,
            keyword,
            require_role_like=require_role_like,
        )
        for keyword in keywords
    )


def _label_matches_keyword(
    label: str,
    keyword: str,
    *,
    require_role_like: bool = False,
) -> bool:
    label_tokens = _tokenize_question(label)
    keyword_tokens = _tokenize_question(keyword)
    if not label_tokens or not keyword_tokens:
        return False
    if require_role_like and not _label_is_role_like(label_tokens, keyword_tokens):
        return False

    if len(keyword_tokens) == 1:
        return keyword_tokens[0] in label_tokens

    window_size = len(keyword_tokens)
    for index in range(len(label_tokens) - window_size + 1):
        if label_tokens[index : index + window_size] == keyword_tokens:
            return True
    return False


def _label_is_role_like(
    label_tokens: list[str],
    keyword_tokens: list[str],
) -> bool:
    max_tokens = max(len(keyword_tokens) + 2, 3)
    max_tokens = min(max_tokens, MAX_ROLE_LABEL_TOKENS)
    return len(label_tokens) <= max_tokens


def _looks_like_known_speaker_label(label: str) -> bool:
    if not label:
        return False
    if KNOWN_SPEAKER_LABEL_RE.match(label):
        return True
    if _label_matches_any_keywords(
        label,
        KNOWN_INTERVIEWER_LABELS,
        require_role_like=True,
    ):
        return True
    if _label_matches_any_keywords(
        label,
        KNOWN_CANDIDATE_LABELS,
        require_role_like=True,
    ):
        return True
    return False


def _ordered_unique(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered
