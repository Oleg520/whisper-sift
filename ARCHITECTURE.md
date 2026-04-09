# Whisper Sift Architecture

Целевой архитектурный ориентир для следующей итерации развития проекта.

Этот файл нужен, чтобы ответить на три вопроса:

- где сейчас проходят границы слоев
- к какой архитектуре стоит прийти без лишнего усложнения
- в каком порядке лучше проводить рефакторинг

Для трекинга конкретных задач используй [BACKLOG.md](./BACKLOG.md).

Последнее обновление: 2026-04-09

## Текущее состояние

Сейчас Whisper Sift уже имеет хороший базовый каркас:

- `cli.py` отвечает за CLI-поверхность и запуск сценариев
- `application/` содержит use cases и orchestration сценариев
- `domain/` содержит transcript/question model и extraction-логику
- `services/` содержит рабочую логику транскрибации и извлечения вопросов
- `infrastructure/` отвечает за `ffmpeg`
- `runtime/` отвечает за диагностику и bootstrap зависимостей
- `utils/text.py` теперь является совместимой оберткой над domain extraction API

Для MVP это рабочая структура, но по мере роста проявляются архитектурные точки напряжения:

- orchestration use cases уже вынесен из CLI, но пока остаются довольно тонкие result-модели
- extraction-логика уже вынесена в `domain/`, но её ещё можно дробить по политикам и scoring-слою
- часть infrastructure side effects уже вынесена в adapters и reporter hooks, но Whisper backend и bootstrap ещё можно изолировать сильнее
- базовые CLI/extraction defaults уже централизованы в `config.py`, но richer runtime/config-file сценарии ещё впереди
- мало структурированных результатов для будущего `json`-экспорта и richer workflows

## Цели v2

Архитектура следующей версии должна дать:

- тонкий CLI-слой без бизнес-оркестрации
- явный application layer для use cases
- чистую доменную модель transcript/question extraction
- изолированные infrastructure adapters для Whisper, `ffmpeg`, filesystem и environment
- предсказуемую конфигурацию и единые defaults
- возможность добавить `json`, richer outputs и API/GUI без дублирования логики

## Архитектурные принципы

1. CLI должен только парсить аргументы, собирать request-объекты и маппить ошибки в exit codes.
2. Use cases должны жить в application layer и оркестрировать шаги пайплайна.
3. Domain layer должен быть максимально чистым: без IO, `print`, `pip install`, `PATH`-мутаций и прямой работы с файлами.
4. Infrastructure layer должен инкапсулировать side effects: Whisper backend, `ffmpeg`, filesystem, environment/bootstrap.
5. Structured results важнее сырого текста: каждый шаг должен уметь возвращать не только path, но и метаданные выполнения.

## Целевое разбиение пакетов

```text
src/whisper_sift/
├─ __main__.py
├─ cli.py
├─ config.py
├─ paths.py
├─ application/
│  ├─ transcribe.py
│  ├─ extract_questions.py
│  ├─ pipeline.py
│  └─ doctor.py
├─ domain/
│  ├─ transcript.py
│  ├─ questions.py
│  ├─ extraction.py
│  └─ policies.py
├─ infrastructure/
│  ├─ ffmpeg.py
│  ├─ whisper_backend.py
│  ├─ filesystem.py
│  └─ bootstrap.py
├─ runtime/
│  └─ doctor.py
└─ utils/
   └─ text.py
```

Это не означает, что нужно срочно разнести все текущие модули по новым папкам. Лучше идти поэтапно:

- сначала ввести application layer
- затем выделить доменные модели
- затем постепенно сокращать ответственность `utils/text.py`

## Ответственность слоёв

### CLI

CLI должен:

- принимать аргументы
- собирать request DTO
- вызывать use case
- печатать дружелюбный summary
- возвращать codes выхода

CLI не должен:

- решать, в каком порядке выполняются низкоуровневые шаги пайплайна
- напрямую ставить зависимости
- содержать доменные эвристики

### Application

Application layer должен содержать сценарии:

- `TranscribeUseCase`
- `ExtractQuestionsUseCase`
- `PipelineUseCase`
- `DoctorUseCase`

Каждый use case должен получать request-объект и возвращать result-объект.

### Domain

Domain layer должен описывать:

- `TranscriptArtifact`
- `TranscriptSegment`
- `QuestionCandidate`
- `QuestionExtractionResult`
- `TranscriptionResult`

Именно здесь логично держать:

- speaker-aware rules
- question scoring
- deduplication policies
- шумовые фильтры

### Infrastructure

Infrastructure должна отвечать за:

- доступ к Whisper backend
- подготовку и probing `ffmpeg`
- работу с файловой системой
- bootstrap runtime-зависимостей

Infrastructure не должна решать, когда запускать pipeline целиком.

## Целевой поток данных

### Transcribe

1. CLI собирает `TranscriptionRequest`.
2. `TranscribeUseCase` валидирует вход и вызывает backend.
3. Infrastructure-adapter запускает Whisper и возвращает structured result.
4. Filesystem-adapter сохраняет артефакты.
5. CLI печатает summary.

### Extract Questions

1. CLI собирает `QuestionExtractionRequest`.
2. `ExtractQuestionsUseCase` читает transcript.
3. Domain extractor строит candidates и итоговый result.
4. Filesystem-adapter пишет выходной файл или `json`.
5. CLI печатает summary.

### Pipeline

1. `PipelineUseCase` вызывает `TranscribeUseCase`.
2. Затем извлекает `.txt`-артефакты из structured result.
3. Затем вызывает `ExtractQuestionsUseCase`.
4. Возвращает объединенный `PipelineResult`.

## Что менять первым

Наиболее безопасный порядок рефакторинга такой:

1. Ввести более богатые structured result-объекты для транскрибации и extraction.
2. Разделить в domain extraction parsing speaker turns, scoring, filtering и deduplication на более мелкие политики.
3. Добавить `json`-serialization поверх новых result-объектов.
4. Расширить configuration layer до config-file/env-based сценариев.
5. Добавить integration smoke tests на установленный CLI и полный pipeline.

## Что не нужно делать сейчас

- не нужен тяжелый framework
- не нужен отдельный DI-container
- не нужно раскалывать проект на несколько пакетов
- не нужно пытаться сразу внедрить real diarization в рамках архитектурного рефакторинга

Задача v2 не в том, чтобы сделать систему “enterprise”, а в том, чтобы убрать будущие узкие места и сохранить понятную, компактную структуру.
