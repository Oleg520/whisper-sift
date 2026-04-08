# Whisper Sift

CLI-инструмент для расшифровки интервью через Whisper и извлечения вопросов интервьюеров из готовых transcript-файлов.

Проект состоит из двух частей:

- Python-приложение с основным пайплайном транскрибации и обработки текста
- Java CLI-клиент, который использует текущий Python runtime для транскрибации и свою Java-логику для извлечения вопросов

План развития проекта: [ROADMAP.md](./ROADMAP.md)  
Трекер задач: [BACKLOG.md](./BACKLOG.md)

## Возможности

- Расшифровка локальных аудио- и видеофайлов через Whisper
- Сохранение результатов в `.txt` и `.srt`
- Поддержка нескольких файлов за один запуск
- Извлечение вопросов интервьюеров из `.txt`-расшифровок
- Полный пайплайн: расшифровка и сразу сбор вопросов
- Диагностическая команда `doctor`
- Автоматическая установка runtime-зависимостей для транскрибации
- Автоматический выбор устройства `auto/cpu/cuda/mps`
- Кроссплатформенная подготовка `ffmpeg` с приоритетом системного бинарника

## Требования

- Python 3.11+
- Windows / Linux / macOS
- Доступ в интернет при первом запуске транскрибации, если зависимости еще не установлены
- Для Java CLI: Java 21+ и доступный `python` в `PATH` либо явный `--python`

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

Полный пайплайн:

```powershell
python transcribe_whisper.py pipeline interview_part1.mkv interview_part2.mkv --model small --language ru --output-dir results --questions-dir questions
```

Если в `pipeline` не указать `txt` в `--formats`, приложение автоматически добавит его, потому что извлечение вопросов требует текстовую расшифровку.

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

Проверить окружение, зависимости, `ffmpeg`, Python и доступность `cuda/mps`:

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

Для команд `transcribe` и `pipeline` приложение проверяет наличие runtime-зависимостей. Если чего-то не хватает, запускается:

```powershell
python -m pip install torch openai-whisper imageio-ffmpeg
```

Если системный `ffmpeg` уже доступен через `PATH`, `imageio-ffmpeg` не является обязательным для bootstrap-проверки.

## FFMPEG

Сначала приложение пытается использовать системный `ffmpeg`, если он уже доступен в `PATH`.

Если системного бинарника нет, используется `imageio-ffmpeg`, а подготовленный алиас сохраняется в `.tools/ffmpeg/ffmpeg(.exe)` для стабильного запуска Whisper на Windows, Linux и macOS.

## Java CLI

Java-клиент находится в [clients/java-cli](./clients/java-cli).

Сборка:

```powershell
cd clients/java-cli
mvn package
```

Тесты:

```powershell
cd clients/java-cli
mvn test
```

Запуск:

```powershell
java -jar target/whisper-sift-java-cli.jar --help
```

Пример запуска транскрибации:

```powershell
java -jar clients/java-cli/target/whisper-sift-java-cli.jar transcribe interview.mkv --model small --device auto
```

При необходимости можно явно указать Python и корень проекта:

```powershell
java -jar target/whisper-sift-java-cli.jar transcribe interview.mkv --python py --project-root ..\..
```

## Тесты

Python-тесты:

```powershell
python -m unittest discover -s tests -v
```

## Как устроен проект

- `src/whisper_sift` — основное Python-приложение и CLI
- `tests` — Python-тесты
- `transcribe_whisper.py` — launcher для прямого запуска без установки пакета
- `clients/java-cli` — Java-клиент внутри того же репозитория

## Что дальше

Следующие приоритетные шаги описаны в:

- [BACKLOG.md](./BACKLOG.md)
- [ROADMAP.md](./ROADMAP.md)
