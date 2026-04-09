# Whisper Sift

CLI-инструмент для расшифровки интервью через Whisper и извлечения вопросов интервьюеров из готовых transcript-файлов.

## Возможности

- Расшифровка локальных аудио- и видеофайлов через Whisper
- Сохранение результатов в `.txt` и `.srt`
- Поддержка нескольких файлов за один запуск
- Извлечение вопросов интервьюеров из `.txt`-расшифровок
- Полный пайплайн: расшифровка и сразу сбор вопросов
- Speaker-aware извлечение для speaker-labeled или diarized transcript-файлов
- Диагностическая команда `doctor`
- Автоматическая установка runtime-зависимостей для транскрибации
- Автоматический выбор устройства `auto/cpu/cuda/mps`
- Кроссплатформенная подготовка `ffmpeg` с приоритетом системного бинарника

## Требования

- Python 3.11+
- Windows / Linux / macOS
- Доступ в интернет при первом запуске транскрибации, если зависимости еще не установлены

## Быстрый старт

Без предварительной установки:

```powershell
python transcribe_whisper.py transcribe interview.mkv
```

Как обычный Python-проект:

```powershell
python -m pip install -e .
whisper-sift --help
whisper-sift doctor
```

## Основные команды

Расшифровка файлов:

```powershell
python transcribe_whisper.py transcribe interview_part1.mkv interview_part2.mkv --model small --language ru --output-dir results
```

Извлечение вопросов из готовых расшифровок:

```powershell
python transcribe_whisper.py extract-questions results\interview_part1.txt results\interview_part2.txt --output-dir questions
```

Если нужен структурированный экспорт для автоматизации, можно добавить JSON sidecar:

```powershell
python transcribe_whisper.py extract-questions results\interview_part1.txt --output-dir questions --json
```

Если transcript уже содержит speaker labels, можно явно указать интервьюера:

```powershell
python transcribe_whisper.py extract-questions diarized.txt --interviewer-label SPEAKER_00
```

Полный пайплайн:

```powershell
python transcribe_whisper.py pipeline interview_part1.mkv interview_part2.mkv --model small --language ru --output-dir results --questions-dir questions
```

Если в `pipeline` не указать `txt` в `--formats`, приложение автоматически добавит его, потому что извлечение вопросов требует текстовую расшифровку.

Если `.txt`-расшифровка после пайплайна уже содержит speaker labels, можно так же передать `--interviewer-label` и извлекать вопросы только из нужного спикера.

Для `extract-questions` и `pipeline` опция `--json` сохраняет дополнительный `.json` рядом с файлом вопросов.

## Параметры транскрибации

- `--model` — модель Whisper, например `tiny`, `base`, `small`, `medium`, `large`
- `--language` — код языка, например `ru`; можно указать `auto`
- `--device` — `auto`, `cpu`, `cuda`, `mps`
- `--output-dir` — папка для результатов транскрибации
- `--formats` — выходные форматы, по умолчанию `txt srt`

По устройству приложение работает так:

- `auto` выбирает `cuda`, если доступна NVIDIA GPU
- если `cuda` недоступна, используется `cpu`
- при работе на `cuda` автоматически включается `fp16`
- если явно указать недоступное устройство, CLI мягко откатится на `cpu`

## Диагностика

Проверить окружение, зависимости, `ffmpeg`, Python, `nvidia-smi` и доступность `cuda/mps`:

```powershell
python transcribe_whisper.py doctor
```

Если нужно, можно сразу попробовать установить отсутствующие runtime-зависимости:

```powershell
python transcribe_whisper.py doctor --install-missing
```

Тот же запуск через установленный CLI:

```powershell
whisper-sift doctor
```

## Автоустановка зависимостей

Для команд `transcribe` и `pipeline` приложение проверяет наличие runtime-зависимостей.

Если `torch` еще не установлен, bootstrap выбирает wheel по железу:

- при обнаружении `nvidia-smi` ставится CUDA-сборка `torch` через официальный PyTorch CUDA index
- если NVIDIA GPU не обнаружена, ставится CPU-сборка `torch`

Остальные зависимости ставятся через обычный `pip install`.

Если `torch` уже установлен, приложение не переустанавливает его молча. В этом случае `doctor` покажет, готово ли текущее окружение к CUDA, и при необходимости подскажет рекомендуемую команду.

Например, на машине с NVIDIA GPU bootstrap будет выглядеть так:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install openai-whisper imageio-ffmpeg
```

На CPU-only машине первый шаг будет использовать PyTorch CPU index вместо CUDA.

Если системный `ffmpeg` уже доступен через `PATH`, `imageio-ffmpeg` не является обязательным для bootstrap-проверки.

## FFMPEG

Сначала приложение пытается использовать системный `ffmpeg`, если он уже доступен в `PATH`.

Если системного бинарника нет, используется `imageio-ffmpeg`, а подготовленный алиас сохраняется в пользовательской runtime-директории:

- Windows: `%LOCALAPPDATA%\\whisper-sift\\tools\\ffmpeg\\ffmpeg.exe`
- Linux: `~/.cache/whisper-sift/tools/ffmpeg/ffmpeg`
- macOS: `~/Library/Caches/whisper-sift/tools/ffmpeg/ffmpeg`

При необходимости runtime-директорию можно переопределить через переменную окружения `WHISPER_SIFT_RUNTIME_DIR`.

## Тесты

Python-тесты, включая unit coverage и CLI smoke tests для `transcribe_whisper.py` и `python -m whisper_sift` на сценариях `extract-questions`, `transcribe` и `pipeline`:

```powershell
python -m unittest discover -s tests -v
```

## Как устроен проект

- `src/whisper_sift` — основное Python-приложение и CLI
- `src/whisper_sift/application` — use cases и orchestration сценариев
- `src/whisper_sift/domain` — transcript/question model и чистая extraction-логика
- `src/whisper_sift/infrastructure` — adapters для `ffmpeg`, filesystem и Whisper backend
- `tests` — Python-тесты
- `transcribe_whisper.py` — launcher для прямого запуска без установки пакета

## Что дальше

Ближайшие направления развития:

- улучшение эвристики извлечения вопросов на шумных расшифровках
- `json`-вывод для автоматизации и интеграций
- дальнейшая изоляция runtime/bootstrap слоя
- инженерная обвязка: linting, CI и автоматические проверки
