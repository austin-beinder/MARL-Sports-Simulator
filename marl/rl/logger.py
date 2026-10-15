# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: a minimal CSV logger with fixed
#     columns and a flush after every row (Phase 3 decision E).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
"""Minimal CSV logger (decision E): one row per call, fixed columns, flushed
immediately so a crash keeps the data and a run can be watched while it trains.
The first row fixes the columns; a later row with different keys raises. (A
TensorBoard backend could implement the same ``log(dict)`` interface later.)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._fields: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._fields = list(row)
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            self._writer.writeheader()
        elif list(row) != self._fields:
            raise ValueError(f"{self.path.name}: columns changed from {self._fields} to {list(row)}")
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
