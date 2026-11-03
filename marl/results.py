# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this file: the results-loading layer shared by the
#     plots and records (backlog task 2, step 1). The CSV/curve/table helpers
#     were moved here unchanged from marl/plots.py (written 2026-10-11); Arm,
#     run_dirs and eval_mismatch are new.
# Prompt summary: Implement step 1 of the task 2 plotting-refactor plan: a
#   loading layer where an "arm" is a label plus a glob of run directories.
# Human review:   Pending review
# ==============================================================================
"""Loading finished runs for figures and records. Reads only what training and
``scripts/evaluate.py`` wrote (``train.csv``, ``eval.csv``, ``config.json``,
``eval_*.csv`` tables); never re-runs anything.

An **arm** is one experimental condition: a label plus a glob of run directories,
one per training seed of the SAME config. Curves are averaged within an arm, so
the runs of one arm must share an evaluation schedule; different arms need not
(e.g. mock vs PyBullet runs can share a figure).
"""

from __future__ import annotations

import csv
import glob
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVAL_METRICS = ("len_mean", "len_mean_possession", "len_mean_free_ball", "mean_reward")
TRAIN_METRICS = ("entropy", "explained_variance", "approx_kl", "clip_frac")
EVAL_EPISODE_KEYS = ("eval_seed", "eval_episodes")  # same values = the same fixed evaluation episodes


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _col(rows: Sequence[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def eval_curves(run_dirs: Sequence[str | Path], mode: str = "stochastic") -> dict[str, np.ndarray]:
    """Evaluation metrics per run: arrays [n_runs, n_evals] plus the shared ``env_steps``."""
    return _curves(run_dirs, "eval.csv", EVAL_METRICS, lambda r: r["mode"] == mode)


def train_curves(run_dirs: Sequence[str | Path]) -> dict[str, np.ndarray]:
    return _curves(run_dirs, "train.csv", TRAIN_METRICS, lambda r: True)


def _curves(run_dirs, filename, metrics, keep) -> dict[str, np.ndarray]:
    if not run_dirs:
        raise ValueError("no runs given")
    steps: np.ndarray | None = None
    data: dict[str, list[np.ndarray]] = {k: [] for k in metrics}
    for d in run_dirs:
        rows = [r for r in read_csv(Path(d) / filename) if keep(r)]
        s = _col(rows, "env_steps")
        if steps is None:
            steps = s
        elif not np.array_equal(steps, s):
            raise ValueError(f"{d}: its {filename} schedule differs from the other runs (different configs?)")
        for k in metrics:
            data[k].append(_col(rows, k))
    return {"env_steps": steps, **{k: np.vstack(v) for k, v in data.items()}}


def read_table(path: str | Path) -> dict[str, dict[str, float]]:
    """An evaluation table written by ``scripts/evaluate.py``: policy -> numeric columns."""
    table = {}
    for r in read_csv(path):
        table[r["policy"]] = {k: float(v) for k, v in r.items() if k != "policy"}
    return table


def table_path_for(run_dirs: Sequence[str | Path]) -> Path:
    """Where ``scripts/evaluate.py`` writes the table when these runs are passed in this order."""
    first = Path(run_dirs[0])
    return first.parent / f"eval_{first.name}.csv"


def mean_std(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Across runs (axis 0): mean and sample std (ddof=1; 0 for a single run)."""
    sd = np.std(Y, axis=0, ddof=1) if len(Y) > 1 else np.zeros(Y.shape[1])
    return np.mean(Y, axis=0), sd


def smooth(Y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 1, renormalized at the edges."""
    if window <= 1:
        return Y
    kernel = np.ones(window)
    num = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, Y)
    return num / np.convolve(np.ones(Y.shape[1]), kernel, mode="same")


def run_dirs(pattern: str) -> list[Path]:
    """The run directories matching a glob, sorted (files are ignored)."""
    return sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())


def run_config(run_dir: str | Path) -> dict | None:
    """A run's saved ``config.json`` as a dict, or None if it has none."""
    path = Path(run_dir) / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@dataclass(frozen=True)
class Arm:
    """One experimental condition: a label and a glob of its run directories."""

    label: str
    runs: str
    table: Path | None = None  # its evaluation table; None = evaluate.py's naming convention
    color: str | None = None  # None = the next palette slot

    def dirs(self) -> list[Path]:
        found = run_dirs(self.runs)
        if not found:
            raise ValueError(f"arm {self.label!r}: no run directories match {self.runs!r} (nothing to load)")
        return found

    def table_path(self) -> Path:
        return Path(self.table) if self.table is not None else table_path_for(self.dirs())

    def eval_curves(self, mode: str = "stochastic") -> dict[str, np.ndarray]:
        return self._labeled(lambda: eval_curves(self.dirs(), mode))

    def train_curves(self) -> dict[str, np.ndarray]:
        return self._labeled(lambda: train_curves(self.dirs()))

    def _labeled(self, load):
        try:
            return load()
        except ValueError as e:
            raise ValueError(f"arm {self.label!r}: {e}") from e


def eval_mismatch(arms: Sequence[Arm]) -> list[str]:
    """Warnings for arms whose runs were evaluated on different fixed episodes than the
    first checkable arm (``run.eval_seed`` / ``run.eval_episodes``). Arms whose first run
    has no ``config.json`` are skipped. An empty list means the comparison is fair."""
    ref: tuple[str, dict] | None = None
    warnings = []
    for arm in arms:
        cfg = run_config(arm.dirs()[0])
        if cfg is None:
            continue
        episodes = {k: cfg.get("run", {}).get(k) for k in EVAL_EPISODE_KEYS}
        if ref is None:
            ref = (arm.label, episodes)
        elif episodes != ref[1]:
            diff = ", ".join(f"{k} {ref[1][k]} vs {episodes[k]}" for k in EVAL_EPISODE_KEYS if episodes[k] != ref[1][k])
            warnings.append(f"arm {arm.label!r} was evaluated on different episodes than arm {ref[0]!r} ({diff}); "
                            "the comparison isn't on the same fixed episodes")
    return warnings
