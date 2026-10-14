"""
Simple CSV reporting shared by demo and future RL code.

CSV is intentionally used first because it is easy to inspect, graph, diff,
and load with pandas without introducing another service or dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricsLogger:
    """
    Append fixed-column metric dictionaries to a CSV file.

    The first call defines the columns. Later calls must use the same keys so a
    single metrics.csv stays rectangular and easy to analyze.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            raise ValueError("metrics cannot be empty")

        keys = list(metrics.keys())

        if self._fieldnames is None:
            self._fieldnames = keys
        elif keys != self._fieldnames:
            raise ValueError(
                f"Metric columns changed. "
                f"Expected {self._fieldnames}, got {keys}."
            )

        write_header = not self.path.exists()

        with self.path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._fieldnames,
            )

            if write_header:
                writer.writeheader()

            writer.writerow(metrics)
