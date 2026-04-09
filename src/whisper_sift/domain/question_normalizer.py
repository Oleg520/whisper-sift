from __future__ import annotations

import re

from whisper_sift.domain.extraction_text import normalize_intent_text, normalize_whitespace
from whisper_sift.domain.questions import QuestionCandidate

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


def normalize_question_for_output(
    question_text: str,
    *,
    previous_candidate: QuestionCandidate | None,
) -> str:
    normalized = _apply_readability_replacements(normalize_whitespace(question_text))
    lowered = normalize_intent_text(normalized)

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
    if normalized == "зачем они нужны" and _previous_question_mentions(previous_candidate, "aop", "spring"):
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
    if "транси" in normalized and _previous_question_mentions(previous_candidate, "serialversionuid"):
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
    if "под комботом" in normalized and _previous_question_mentions(previous_candidate, "spring"):
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
    if normalized == "почему нужны они" and _previous_question_mentions(previous_candidate, "изоляц"):
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
    previous_normalized = normalize_intent_text(previous_candidate.text)
    return all(term in previous_normalized for term in terms)


def _previous_question_mentions_any(
    previous_candidate: QuestionCandidate | None,
    *term_groups: tuple[str, ...],
) -> bool:
    if previous_candidate is None:
        return False
    previous_normalized = normalize_intent_text(previous_candidate.text)
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
