from __future__ import annotations

import re
from dataclasses import dataclass

from whisper_sift.domain.extraction_text import (
    normalize_intent_text,
    normalize_whitespace,
    tokenize_question,
)
from whisper_sift.domain.questions import QuestionCandidate


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


def classify_question_candidate(
    *,
    question_text: str,
    explicit_question: bool,
    previous_candidate: QuestionCandidate | None,
) -> QuestionClassification:
    normalized = normalize_intent_text(question_text)
    tokens = tokenize_question(normalized)
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
    if looks_like_noise(question_text):
        return QuestionClassification(False, -8, "noise")
    if looks_like_low_signal_followup(question_text):
        return QuestionClassification(False, -7, "low_signal")
    if looks_like_filler_fragment(question_text):
        return QuestionClassification(False, -6, "filler_fragment")
    if looks_like_answer(question_text) and not has_contextual_subject:
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


def cleanup_question(value: str) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = cleaned.strip(" .,!;:-")
    cleaned = cleaned.replace(" ?", "?")
    return cleaned


def looks_like_noise(value: str) -> bool:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", value.lower())
    if not tokens:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    if len(tokens) >= 8 and unique_ratio < 0.35:
        return True

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", value)
    return len(letters) < 5


def looks_like_question_without_mark(value: str) -> bool:
    lowered = normalize_intent_text(value)
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


def looks_like_answer(value: str) -> bool:
    lowered = normalize_intent_text(value)
    if not lowered:
        return False
    if STRONG_ANSWER_LIKE_RE.match(lowered):
        return True
    if QUESTION_LIKE_RE.match(lowered):
        return False
    return ANSWER_LIKE_RE.match(lowered) is not None


def looks_like_filler_fragment(value: str) -> bool:
    normalized = normalize_intent_text(value)
    if not normalized or QUESTION_LIKE_RE.match(normalized):
        return False

    tokens = tokenize_question(normalized)
    if len(tokens) < 4:
        return False

    filler_count = sum(1 for token in tokens if FILLER_TOKEN_RE.match(token))
    content_count = len(tokens) - filler_count
    if content_count <= 0:
        return True

    filler_ratio = filler_count / len(tokens)
    return filler_ratio >= 0.45 and content_count <= 3


def looks_like_low_signal_followup(value: str) -> bool:
    normalized = normalize_intent_text(value)
    if not normalized:
        return False
    if LOW_SIGNAL_FOLLOWUP_RE.match(normalized):
        return True

    tokens = tokenize_question(normalized)
    if len(tokens) <= 5 and {"что", "это", "такое"} <= set(tokens):
        return True

    return False


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

    previous_tokens = tokenize_question(
        normalize_intent_text(previous_candidate.text)
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

    content_token_count = len(_content_tokens(tokenize_question(normalized)))
    return content_token_count <= 2
