from __future__ import annotations

from typing import Protocol


class ProgressReporter(Protocol):
    def emit(self, message: str) -> None:
        """Emit a user-facing progress message."""


class ConsoleReporter:
    def emit(self, message: str) -> None:
        print(message)


def report_progress(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is None:
        return
    reporter.emit(message)

