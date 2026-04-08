# Whisper Sift

Небольшое CLI-приложение для двух задач:

1. Расшифровка аудио и видеофайлов через Whisper.
2. Выделение вопросов интервьюеров из готовых `.txt`-расшифровок.

Приложение собрано как обычный Python-проект с пакетами, а для удобства умеет автоматически подтягивать runtime-зависимости для транскрибации, если их еще нет в окружении.

## Возможности

- Расшифровка локальных медиафайлов в `.txt` и `.srt`
- Поддержка нескольких файлов за один запуск
- Автоматическая подготовка `ffmpeg` через `imageio-ffmpeg`
- Автоматическая установка зависимостей для Whisper при первом запуске
- Извлечение вопросов из `.txt`-расшифровок в отдельные файлы
- Пайплайн "расшифровать + сразу собрать вопросы"
- Обратная совместимость со старым запуском через `transcribe_whisper.py`

План развития проекта: [ROADMAP.md](./ROADMAP.md)  
Трекер задач: [BACKLOG.md](./BACKLOG.md)

## Структура проекта

```text
whisper-sift/
├─ pyproject.toml
├─ README.md
├─ transcribe_whisper.py
└─ src/
   └─ whisper_sift/
      ├─ __main__.py
      ├─ cli.py
      ├─ config.py
      ├─ paths.py
      ├─ infrastructure/
      │  └─ ffmpeg.py
      ├─ runtime/
      │  └─ dependencies.py
      ├─ services/
      │  ├─ questions.py
      │  └─ transcription.py
      └─ utils/
         └─ text.py
```

## Требования

- Python 3.11+
- Windows / Linux / macOS
- Доступ в интернет при первом запуске транскрибации, если пакеты еще не установлены

## Быстрый старт

### Вариант 1. Без предварительной установки проекта

Можно запускать сразу через корневой скрипт:

```powershell
python transcribe_whisper.py transcribe interview_part1.mkv interview_part2.mkv
```

Если пакетов для транскрибации не будет, приложение само попробует установить:

- `torch`
- `openai-whisper`
- `imageio-ffmpeg`

### Вариант 2. Как обычный Python-проект

```powershell
python -m pip install -e .
whisper-sift --help
```

## Команды

### 1. Расшифровка файлов

```powershell
python transcribe_whisper.py transcribe interview_part1.mkv interview_part2.mkv --model small --language ru --output-dir .
```

Параметры:

- `--model` - модель Whisper, по умолчанию `small`
- `--language` - язык расшифровки, по умолчанию `ru`; можно указать `auto`
- `--device` - устройство `auto/cpu/cuda/mps`, по умолчанию `auto`
- `--output-dir` - куда сохранять результаты
- `--formats` - какие форматы сохранить, по умолчанию `txt srt`

По устройству приложение работает так:

- `auto` - автоматически выберет `cuda`, если доступна NVIDIA GPU
- если GPU недоступна, будет использован `cpu`
- при работе на `cuda` автоматически включается `fp16` для более быстрой инференции
- если явно указать `--device cuda`, но CUDA недоступна, приложение мягко откатится на `cpu`

### 2. Извлечение вопросов из готовых расшифровок

```powershell
python transcribe_whisper.py extract-questions interview_part1.txt interview_part2.txt
```

По умолчанию рядом с исходным transcript-файлом будет создан файл:

- `interview_part1_questions.txt`
- `interview_part2_questions.txt`

Дополнительные параметры:

- `--output-dir` - отдельная папка для файлов с вопросами
- `--suffix` - суффикс имени файла, по умолчанию `_questions.txt`
- `--min-length` - минимальная длина вопроса
- `--max-length` - максимальная длина вопроса
- `--no-deduplicate` - не удалять дубликаты вопросов

Важно: выделение вопросов работает эвристически, так как в исходной расшифровке нет speaker labels. Поэтому результат удобен как черновик, но в сложных интервью может потребовать ручной проверки.

### 3. Полный пайплайн

Сначала расшифровать файлы, затем автоматически извлечь вопросы из получившихся `.txt`:

```powershell
python transcribe_whisper.py pipeline interview_part1.mkv interview_part2.mkv --model small --language ru --output-dir . --questions-dir .
```

## Обратная совместимость

Старый короткий запуск тоже работает. Если не указывать подкоманду, приложение считает, что это `transcribe`:

```powershell
python transcribe_whisper.py interview_part1.mkv interview_part2.mkv
```

## Примеры

Расшифровать один файл:

```powershell
python transcribe_whisper.py transcribe interview.mkv --model base --language ru
```

Принудительно использовать GPU:

```powershell
python transcribe_whisper.py transcribe interview.mkv --model small --device cuda
```

Собрать вопросы в отдельную директорию:

```powershell
python transcribe_whisper.py extract-questions interview_part1.txt interview_part2.txt --output-dir questions
```

Запустить через модуль:

```powershell
python -m whisper_sift --help
```

## Как это работает

### Автоустановка зависимостей

Для команд `transcribe` и `pipeline` приложение проверяет наличие runtime-зависимостей. Если чего-то не хватает, запускается:

```powershell
python -m pip install torch openai-whisper imageio-ffmpeg
```

### FFMPEG

`ffmpeg` берется из пакета `imageio-ffmpeg`, затем копируется в `.tools/ffmpeg/ffmpeg.exe`, чтобы Whisper мог стабильно найти бинарник в Windows.

## Что можно улучшить дальше

- Добавить speaker diarization, чтобы точнее понимать, кто задавал вопрос
- Добавить тесты на эвристики извлечения вопросов
- Добавить экспорт в `.json`
- Поддержать отдельный конфиг-файл для запуска пайплайна
