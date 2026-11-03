# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this file: the regression suite (TOML) schema
#     and its strict loader, with the reserved seed schema, case/run naming
#     and the per-run --set overrides (backlog task 4, step 1).
#   - 2026-11-01: Added the comparison: CSV (cell/row/column, timing columns
#     ignored, optional tolerance), config.json and weight-fingerprint diffs,
#     the per-case status roll-up, stale record entries and the report text
#     (task 4, step 2).
#   - 2026-11-01: Added the weight fingerprint, the environment stamp and
#     freeze_silver (the --update writer: timing columns stripped, stamp.json
#     with an append-only history) (task 4, step 3).
#   - 2026-11-01: Added the runner (parallel train.py/evaluate.py
#     subprocesses, fingerprints, failure capture), --only/--no-run case
#     selection, the uncovered-config note, exit codes and the CLI main
#     (task 4, step 4).
# Prompt summary: Implement steps 1-5 of the task 4 regression-harness plan: a
#   tracked suite file listing the regression cases (config x env), the
#   comparison of new runs against the frozen record, freezing it, and the
#   one-command runner.
# Human review:   Pending review
# ==============================================================================
"""The regression harness: the silver record (``results/silver/``).

The **suite** (``results/silver/suite.toml``, hand-written and tracked) lists the
regression cases. A case is one config x one env, trained once per suite seed on
a short budget with the config's real PPO settings. Fixed seeds on unchanged code
must reproduce the frozen record exactly, so any delta means the change touched
sim or training results.

Seeds follow the project schema: 0-9 for-score, 90-99 testing. Suite seeds must
lie in 90-99, and the suite's ``eval_seed`` must differ from the for-score default,
so regression runs can't be mixed into for-score tables by accident
(``report._same_task`` refuses runs with different evaluation episodes).

Config paths are relative to the repo root (``configs/keepaway_1v1.toml``).

Comparing: each case's files (per run ``config.json``, ``train.csv``, ``eval.csv``,
``fingerprint.json``; per case the ``evaluate.py`` table and its ``_per_run`` twin)
are compared with the record's. A case is IDENTICAL, CHANGED (a value differs),
SCHEMA (only columns/keys were added or removed: the sim is unchanged, but the
record needs a re-freeze) or NEW (not in the record yet). By default the check is
exact (string-equal cells); a ``Tolerance`` compares numbers with ``math.isclose``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marl.common.config import OPTIONS, Config, config_from_dict, load_config, with_overrides
from marl.manifest import _keys, _str  # the same small validators (and error wording) as manifests

REPO = Path(__file__).resolve().parents[1]
SEED_RANGE = range(90, 100)  # the testing seeds (the student's schema, 2026-11-01)
BUDGET_KEYS = ("total_steps", "eval_every", "eval_episodes")  # run.* keys each case overrides
ENVS = tuple(sorted(OPTIONS["env"][0]))


@dataclass(frozen=True)
class Case:
    config: Path  # absolute
    env: str

    @property
    def name(self) -> str:
        return f"{self.config.stem}_{self.env}"

    def run_names(self, seeds: tuple[int, ...]) -> list[str]:
        """The run folders, named like train.py names them (``<config>_<env>_seed<N>``)."""
        return [f"{self.name}_seed{s}" for s in seeds]


@dataclass(frozen=True)
class Suite:
    name: str
    path: Path  # the suite file
    seeds: tuple[int, ...]
    eval_seed: int
    budget: dict[str, int]
    cases: tuple[Case, ...]

    def set_args(self) -> list[str]:
        """The ``--set`` overrides every run gets (the seed and env have their own flags)."""
        return [f"run.eval_seed={self.eval_seed}"] + [f"run.{k}={self.budget[k]}" for k in BUDGET_KEYS]

    def case_config(self, case: Case, seed: int) -> Config:
        """The config one run trains with (what train.py builds from the same flags)."""
        run = {"seed": seed, "env": case.env, "eval_seed": self.eval_seed, **self.budget}
        return with_overrides(load_config(case.config), {"run": run})


def load_suite(path: str | Path, repo_root: str | Path = REPO) -> Suite:
    """Load and validate a suite; every error names the file."""
    path = Path(os.path.abspath(path))
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    try:
        return _suite(data, path, Path(repo_root))
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


# --- Parsing ------------------------------------------------------------------------------
def _suite(data: dict[str, Any], path: Path, repo_root: Path) -> Suite:
    _keys(data, "suite", {"name", "seeds", "eval_seed", "budget"}, {"case"})  # no cases: the clearer error below
    seeds = data["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list, e.g. [90, 91]")
    if not all(_is_int(s) for s in seeds):
        raise ValueError(f"seeds must be whole numbers, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds has a duplicate: {seeds!r}")
    outside = [s for s in seeds if s not in SEED_RANGE]
    if outside:
        raise ValueError(f"seeds must lie in the testing range {SEED_RANGE.start}-{SEED_RANGE.stop - 1} "
                         f"(0-9 are for-score), got {outside}")
    eval_seed = data["eval_seed"]
    if not _is_int(eval_seed) or eval_seed < 0:
        raise ValueError(f"eval_seed must be a whole number >= 0, got {eval_seed!r}")
    for_score = config_from_dict({}).run.eval_seed
    if eval_seed == for_score:
        raise ValueError(f"eval_seed {eval_seed} is the for-score evaluation seed; reserve a different one")
    budget = _budget(data["budget"])
    raw_cases = data.get("case", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("the suite needs at least one case ([[case]])")
    cases = [c for i, raw in enumerate(raw_cases) for c in _cases(raw, repo_root, f"case[{i}]")]
    seen: set[str] = set()
    for c in cases:
        if c.name in seen:
            raise ValueError(f"duplicate case {c.name!r} (its runs would share folders)")
        seen.add(c.name)
    suite = Suite(_str(data, "name", "suite"), path, tuple(seeds), eval_seed, budget, tuple(cases))
    for c in cases:  # the budget must suit every config (e.g. at least one PPO iteration)
        try:
            suite.case_config(c, seeds[0])
        except ValueError as e:
            raise ValueError(f"case {c.name!r}: {e}") from e
    return suite


def _budget(data: Any) -> dict[str, int]:
    _keys(data, "budget", set(BUDGET_KEYS), set())
    for k in BUDGET_KEYS:
        if not _is_int(data[k]):
            raise ValueError(f"budget.{k} must be a whole number, got {data[k]!r}")
    return {k: data[k] for k in BUDGET_KEYS}


def _cases(data: Any, repo_root: Path, where: str) -> list[Case]:
    _keys(data, where, {"config", "envs"}, set())
    raw = _str(data, "config", where)
    if "\\" in raw:
        raise ValueError(f"{where}: write paths with forward slashes ({raw!r})")
    config = Path(os.path.abspath(repo_root / raw))
    if not config.is_file():
        raise ValueError(f"{where}: no such config {raw!r}")
    envs = data["envs"]
    if not isinstance(envs, list) or not envs:
        raise ValueError(f"{where}: envs must be a non-empty list, e.g. [\"mock\", \"pybullet\"]")
    for env in envs:
        if env not in ENVS:
            raise ValueError(f"{where}: env {env!r} unknown; choose from {list(ENVS)}")
        if envs.count(env) > 1:
            raise ValueError(f"{where}: duplicate env {env!r}")
    return [Case(config, env) for env in envs]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- Comparing ----------------------------------------------------------------------------
IDENTICAL, CHANGED, SCHEMA, NEW, NOTE, STALE = "IDENTICAL", "CHANGED", "SCHEMA", "NEW", "NOTE", "STALE"
FAILED, SKIPPED = "FAILED", "SKIPPED"  # a job crashed; left out by --only
TIMING = frozenset({"sps", "wall_time_s"})  # wall-clock columns of train.csv: never reproducible
RUN_FILES = ("config.json", "train.csv", "eval.csv", "fingerprint.json")
MAX_LISTED = 5  # config keys listed per file before "... (N more)"


@dataclass(frozen=True)
class Tolerance:
    rtol: float = 0.0
    atol: float = 0.0

    @property
    def exact(self) -> bool:
        return self.rtol == 0.0 and self.atol == 0.0

    def __str__(self) -> str:
        return "exact" if self.exact else f"rtol={self.rtol:g}, atol={self.atol:g}"

    def cell(self, a: str, b: str) -> tuple[bool, float | None, float | None]:
        """(equal, |delta|, relative delta). Deltas are None unless both cells are finite numbers."""
        if a == b:
            return True, 0.0, 0.0
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False, None, None  # text: always exact
        if math.isnan(fa) and math.isnan(fb):
            return True, 0.0, 0.0
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False, None, None
        d = abs(fa - fb)
        rel = d / max(abs(fa), abs(fb))  # they differ, so the larger one isn't 0
        return (not self.exact and math.isclose(fa, fb, rel_tol=self.rtol, abs_tol=self.atol)), d, rel


@dataclass(frozen=True)
class FileDiff:
    file: str  # relative to the case's root (e.g. "keepaway_1v1_mock_seed90/train.csv")
    status: str  # IDENTICAL, CHANGED, SCHEMA or NOTE
    detail: str = ""
    n_cells: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0


@dataclass(frozen=True)
class CaseResult:
    name: str
    status: str
    files: tuple[FileDiff, ...] = ()
    headline: tuple[float, float] | None = None  # final stochastic len_mean, record -> new (mean over seeds)
    message: str = ""  # FAILED: which job, and the end of its log


def case_files(case: Case, seeds: tuple[int, ...]) -> list[str]:
    """Every file compared for a case, relative to the folder that holds its runs."""
    runs = case.run_names(seeds)
    return [f"{r}/{f}" for r in runs for f in RUN_FILES] + [f"eval_{runs[0]}.csv", f"eval_{runs[0]}_per_run.csv"]


def compare_csv(new: Path, old: Path, tol: Tolerance, name: str = "") -> FileDiff:
    """Cell by cell over the shared columns (timing columns dropped), rows in order."""
    (nh, nrows), (oh, orows) = _read_csv(new), _read_csv(old)
    shared = [c for c in oh if c in nh]
    added, removed = [c for c in nh if c not in oh], [c for c in oh if c not in nh]
    ni, oi = {c: i for i, c in enumerate(nh)}, {c: i for i, c in enumerate(oh)}
    n_cells, first, max_abs, max_rel = 0, "", 0.0, 0.0
    for r, (nrow, orow) in enumerate(zip(nrows, orows), start=1):
        for c in shared:
            a, b = orow[oi[c]], nrow[ni[c]]
            equal, d, rel = tol.cell(a, b)
            if equal:
                continue
            n_cells += 1
            if d is not None:
                max_abs, max_rel = max(max_abs, d), max(max_rel, rel)
            if not first:
                label = f" ({shared[0]}={orow[oi[shared[0]]]})" if c != shared[0] else ""
                first = f"first: row {r}{label} {c} {a} -> {b}"
    parts = []
    if n_cells:
        parts.append(f"{n_cells} of {len(shared) * min(len(nrows), len(orows))} cells differ")
    if len(nrows) != len(orows):
        parts.append(f"rows {len(orows)} -> {len(nrows)}")
    if added:
        parts.append(f"+columns {added}")
    if removed:
        parts.append(f"-columns {removed}")
    if first:
        parts.append(first)
    if max_abs:
        parts.append(f"max |delta| {max_abs:.3g} (rel {max_rel:.3g})")
    status = CHANGED if n_cells or len(nrows) != len(orows) else SCHEMA if added or removed else IDENTICAL
    return FileDiff(name or new.name, status, "; ".join(parts), n_cells, max_abs, max_rel)


def compare_json(new: Path, old: Path, name: str = "") -> FileDiff:
    """Key by key (nested keys dotted). Config is an input, so this is always exact."""
    n, o = _flatten(json.loads(new.read_text(encoding="utf-8"))), _flatten(json.loads(old.read_text(encoding="utf-8")))
    changed = [f"{k} {json.dumps(o[k])} -> {json.dumps(n[k])}" for k in o if k in n and o[k] != n[k]]
    added = [f"+{k}={json.dumps(n[k])}" for k in n if k not in o]
    removed = [f"-{k}={json.dumps(o[k])}" for k in o if k not in n]
    items = changed + added + removed
    detail = "; ".join(items[:MAX_LISTED]) + (f"; ... ({len(items) - MAX_LISTED} more)" if len(items) > MAX_LISTED else "")
    status = CHANGED if changed else SCHEMA if items else IDENTICAL
    return FileDiff(name or new.name, status, detail, len(items))


def compare_fingerprint(new: Path, old: Path, tol: Tolerance, name: str = "") -> FileDiff:
    """The final weights' hash. A hash can't be "close", so under a tolerance a mismatch is only noted."""
    n, o = json.loads(new.read_text(encoding="utf-8")), json.loads(old.read_text(encoding="utf-8"))
    if n == o:
        return FileDiff(name or new.name, IDENTICAL)
    if tol.exact:
        return FileDiff(name or new.name, CHANGED, "final weights differ")
    return FileDiff(name or new.name, NOTE, "final weights differ (ignored under a tolerance: the hash is exact)")


def compare_case(case: Case, seeds: tuple[int, ...], new_root: Path, silver_root: Path, tol: Tolerance) -> CaseResult:
    """Compare one case's files under ``new_root`` with the record's under ``silver_root``."""
    runs = case.run_names(seeds)
    if not any((silver_root / r).is_dir() for r in runs):
        return CaseResult(case.name, NEW)
    files = []
    for rel in case_files(case, seeds):
        n, o = new_root / rel, silver_root / rel
        if not (n.is_file() and o.is_file()):
            where = "missing from both" if not (n.is_file() or o.is_file()) else (
                "missing from the new run" if not n.is_file() else "not in the record")
            files.append(FileDiff(rel, CHANGED, where))
        elif rel.endswith("fingerprint.json"):
            files.append(compare_fingerprint(n, o, tol, rel))
        elif rel.endswith(".json"):
            files.append(compare_json(n, o, rel))
        else:
            files.append(compare_csv(n, o, tol, rel))
    statuses = {f.status for f in files}
    status = CHANGED if CHANGED in statuses else SCHEMA if SCHEMA in statuses else IDENTICAL
    old_len, new_len = _final_len(silver_root, runs), _final_len(new_root, runs)
    headline = (old_len, new_len) if old_len is not None and new_len is not None else None
    return CaseResult(case.name, status, tuple(files), headline)


def stale_entries(cases: Sequence[Case], seeds: tuple[int, ...], silver_root: Path) -> list[str]:
    """Run folders and files in the record that no case of the suite produces (sorted)."""
    if not silver_root.is_dir():
        return []
    expected = {f.split("/")[0] for c in cases for f in case_files(c, seeds)}
    return sorted(p.name for p in silver_root.iterdir() if p.name not in expected)


def format_report(results: Sequence[CaseResult], tol: Tolerance, env_diffs: Sequence[tuple[str, str, str]] = (),
                  stale: Sequence[str] = (), title: str = "") -> str:
    """The report: a line per case, the differing files under it (config first), and a summary."""
    lines = [title] if title else []
    lines.append(f"Comparison: {tol}")
    if env_diffs:
        lines.append("This machine's environment differs from the record's; deltas may come from the machine, not the code:")
        lines += [f"  {k}: {old} -> {new}" for k, old, new in env_diffs]
    lines.append("")
    for r in results:
        line = f"{r.status:<10} {r.name}"
        if r.status in (CHANGED, SCHEMA) and r.headline:
            line += f"  (final len_mean {r.headline[0]:g} -> {r.headline[1]:g})"
        elif r.status == NEW:
            line += "  (not in the record yet; freeze it with --update)"
        lines.append(line)
        lines += [f"{'':<11}{m}" for m in r.message.splitlines()]
        shown = sorted((f for f in r.files if f.status != IDENTICAL), key=lambda f: not f.file.endswith("config.json"))
        lines += [f"{'':<11}{f.file}: {f.detail}" for f in shown]
    lines += [f"{STALE:<10} {s}  (in the record, not in the suite)" for s in stale]
    counts = [(n, s.lower()) for s in (IDENTICAL, CHANGED, SCHEMA, NEW, FAILED, SKIPPED)
              if (n := sum(r.status == s for r in results))]
    counts += [(len(stale), "stale")] if stale else []
    lines += ["", "Summary: " + ", ".join(f"{n} {s}" for n, s in counts)]
    return "\n".join(lines)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    keep = [i for i, c in enumerate(rows[0]) if c not in TIMING]
    return [rows[0][i] for i in keep], [[row[i] for i in keep] for row in rows[1:]]


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return {prefix: data}
    out: dict[str, Any] = {}
    for k, v in data.items():
        out.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def _final_len(root: Path, runs: Sequence[str]) -> float | None:
    """The mean over runs of the last stochastic evaluation's len_mean (None if any is missing)."""
    vals = []
    for r in runs:
        path = root / r / "eval.csv"
        if not path.is_file():
            return None
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [row for row in csv.DictReader(fh) if row.get("mode") == "stochastic"]
        if not rows:
            return None
        vals.append(float(rows[-1]["len_mean"]))
    return sum(vals) / len(vals)


# --- Fingerprint, environment stamp, freezing --------------------------------------------
STAMP = "stamp.json"


def weights_fingerprint(checkpoint: Path) -> dict[str, Any]:
    """SHA-256 over a checkpoint's actor + critic weights (name, dtype, shape and bytes of each
    tensor, in sorted order). Only the weights count, not the iteration or optimizer state."""
    from marl.rl.checkpoint import load_checkpoint  # torch: only when a fingerprint is needed

    ckpt = load_checkpoint(checkpoint)
    h, n = hashlib.sha256(), 0
    for net in ("actor", "critic"):
        for key in sorted(ckpt[net]):
            t = ckpt[net][key].detach().cpu().contiguous()
            h.update(f"{net}.{key}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.numpy().tobytes())
            n += 1
    return {"sha256": h.hexdigest(), "tensors": n}


def environment_stamp() -> dict[str, str]:
    """What can make bit-exact results differ between machines. torch's own version string keeps
    its build tag (``+cpu``: CPU and CUDA builds run different kernels); pybullet's comes from the
    package metadata, so it isn't imported (that prints its build time)."""
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "pybullet": _version("pybullet"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def environment_diffs(recorded: dict[str, str], current: dict[str, str]) -> list[tuple[str, str, str]]:
    """(key, recorded, current) for every entry that differs, sorted by key."""
    keys = sorted(set(recorded) | set(current))
    return [(k, recorded.get(k, "(not recorded)"), current.get(k, "(not recorded)"))
            for k in keys if recorded.get(k) != current.get(k)]


def load_stamp(record: Path) -> dict[str, Any] | None:
    path = record / STAMP
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def freeze_silver(suite: Suite, new_root: Path, *, reason: str, today: str | None = None,
                  commit: str | None = None) -> list[str]:
    """Replace the record's ``data/`` with the suite's runs under ``new_root`` (``train.csv``
    without its timing columns; everything else copied as is) and rewrite ``stamp.json``,
    appending to its history. Refuses, touching nothing, if any file is missing."""
    if not reason.strip():
        raise ValueError("freezing needs a reason (it goes into stamp.json's history)")
    files = [f for c in suite.cases for f in case_files(c, suite.seeds)]
    missing = [f for f in files if not (new_root / f).is_file()]
    if missing:
        raise ValueError(f"not freezing: missing from {new_root}: {missing}")
    if commit is None:
        from marl.record import git_revision  # marl.record imports matplotlib (via marl.plots)

        commit = git_revision()
    today = today or datetime.date.today().isoformat()
    record = suite.path.parent
    data = record / "data"
    history = (load_stamp(record) or {}).get("updates", [])
    if data.is_dir():
        for entry in data.iterdir():
            _remove(entry)
    for f in files:
        dst = data / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.endswith("/train.csv"):
            header, rows = _read_csv(new_root / f)  # drops the timing columns
            with open(dst, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh, lineterminator="\n").writerows([header] + rows)
        else:
            shutil.copyfile(new_root / f, dst)
    stamp = {
        "suite": suite.name,
        "frozen": today,
        "commit": commit,
        "reason": reason,
        "environment": environment_stamp(),
        "seeds": list(suite.seeds),
        "eval_seed": suite.eval_seed,
        "budget": suite.budget,
        "cases": [c.name for c in suite.cases],
        "updates": history + [{"date": today, "commit": commit, "reason": reason}],
    }
    (record / STAMP).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return files


# --- Running ------------------------------------------------------------------------------
LOG_TAIL = 15  # log lines shown for a failed job
Runner = Callable[[list[str], Path, Path], int]  # (command, working folder, log file) -> exit code


def default_parallel_runs() -> int:
    """Half the logical cores of whoever runs it (about one run per physical core), at least 1."""
    return max(1, (os.cpu_count() or 1) // 2)


def select_cases(suite: Suite, patterns: Sequence[str] | None) -> list[Case]:
    """The cases whose name matches any ``--only`` glob (all cases without one), in suite order."""
    if not patterns:
        return list(suite.cases)
    for p in patterns:
        if not any(fnmatch.fnmatchcase(c.name, p) for c in suite.cases):
            raise ValueError(f"--only {p!r} matches no case; cases: {[c.name for c in suite.cases]}")
    return [c for c in suite.cases if any(fnmatch.fnmatchcase(c.name, p) for p in patterns)]


def uncovered_configs(suite: Suite, repo_root: Path = REPO) -> list[str]:
    """``configs/*.toml`` files no case uses (a reminder to add them to the suite)."""
    used = {c.config for c in suite.cases}
    return sorted(p.relative_to(repo_root).as_posix() for p in (repo_root / "configs").glob("*.toml")
                  if Path(os.path.abspath(p)) not in used)


def train_command(suite: Suite, case: Case, seed: int, out_root: Path) -> list[str]:
    run = f"{case.name}_seed{seed}"
    sets = [a for s in suite.set_args() for a in ("--set", s)]
    return [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(case.config), "--env", case.env,
            "--seed", str(seed), *sets, "--out-dir", str(out_root / run), "--overwrite"]


def evaluate_command(case: Case, seeds: tuple[int, ...]) -> list[str]:
    # Bare run names, run from inside the regression folder: the per-run table records each
    # run as passed, so a full path would make the record machine-specific.
    return [sys.executable, str(REPO / "scripts" / "evaluate.py"), *case.run_names(seeds)]


def run_suite(suite: Suite, cases: Sequence[Case], out_root: Path, parallel: int, runner: Runner | None = None,
              log: Callable[[str], None] = print) -> dict[str, str]:
    """Train every case x seed (``parallel`` jobs at a time), fingerprint each run, then evaluate
    each case whose runs all succeeded. Returns ``{case name: failure message}``."""
    runner = runner or _subprocess
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "logs"
    logs.mkdir(exist_ok=True)
    for c in cases:  # outputs the harness itself writes: an old one must never pass for a new one
        for f in case_files(c, suite.seeds):
            if f.endswith("fingerprint.json") or f.startswith("eval_"):
                (out_root / f).unlink(missing_ok=True)
    log(f"[regress] {len(cases) * len(suite.seeds)} runs ({len(cases)} cases x {len(suite.seeds)} seeds), "
        f"{parallel} at a time -> {out_root}")
    failures: dict[str, str] = {}
    left = {c.name: len(suite.seeds) for c in cases}

    def job(name: str, cmd: list[str], cwd: Path) -> tuple[int, float, Path]:
        t0, path = time.perf_counter(), logs / f"{name}.log"
        return runner(cmd, cwd, path), time.perf_counter() - t0, path

    def failed(case: Case, name: str, why: str, path: Path) -> None:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
        failures.setdefault(case.name, "\n".join([f"{name}: {why}; end of {path}:"] + lines[-LOG_TAIL:]))
        log(f"[regress] FAILED {name} ({why}; see {path})")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures: dict[Future, tuple[str, Case]] = {}
        for c in cases:
            for s in suite.seeds:
                futures[pool.submit(job, f"{c.name}_seed{s}", train_command(suite, c, s, out_root), REPO)] = ("train", c)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                kind, case = futures.pop(fut)
                code, secs, path = fut.result()
                name = path.stem
                if code != 0:
                    failed(case, name, f"exit {code}", path)
                elif kind == "train":
                    try:
                        fp = weights_fingerprint(out_root / name / "checkpoints" / "final.pt")
                    except (OSError, RuntimeError, ValueError) as e:
                        failed(case, name, f"no final checkpoint ({e})", path)
                    else:
                        (out_root / name / "fingerprint.json").write_text(json.dumps(fp, indent=2) + "\n",
                                                                         encoding="utf-8")
                        log(f"[regress] trained {name} ({secs:.0f} s)")
                else:
                    log(f"[regress] evaluated {case.name} ({secs:.0f} s)")
                if kind == "train":
                    left[case.name] -= 1
                    if left[case.name] == 0 and case.name not in failures:
                        cmd = evaluate_command(case, suite.seeds)
                        futures[pool.submit(job, f"eval_{case.name}", cmd, out_root)] = ("evaluate", case)
    return failures


def _subprocess(cmd: list[str], cwd: Path, log: Path) -> int:
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def exit_code(results: Sequence[CaseResult], stale: Sequence[str]) -> int:
    """0 = all identical (or skipped), 1 = any delta, new case or stale entry, 2 = a job failed."""
    if any(r.status == FAILED for r in results):
        return 2
    return 1 if stale or any(r.status in (CHANGED, SCHEMA, NEW) for r in results) else 0


# --- Command line -------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None, *, repo_root: Path = REPO, runner: Runner | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the regression suite on its reserved seeds and compare it with the silver record "
                    "(exit 0 = identical, 1 = deltas, 2 = a failed job or a bad suite).")
    ap.add_argument("--suite", type=Path, help="suite file (default: results/silver/suite.toml)")
    ap.add_argument("--runs-dir", type=Path, help="where runs are written (default: runs/regression)")
    ap.add_argument("--parallel-runs", type=int, default=default_parallel_runs(), metavar="N",
                    help="training runs at once (default: half this machine's logical cores, here %(default)s)")
    ap.add_argument("--only", action="append", metavar="GLOB",
                    help="only the cases whose name (<config>_<env>) matches; repeatable, e.g. --only '*_mock'")
    ap.add_argument("--no-run", action="store_true", help="don't train; compare the runs already in --runs-dir")
    ap.add_argument("--rtol", type=float, default=0.0, help="relative tolerance for numbers (default: exact)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for numbers (default: exact)")
    ap.add_argument("--update", action="store_true",
                    help="re-freeze the record from this run (an intended change); commit it with the change")
    ap.add_argument("--reason", help="why the record changes (required with --update; kept in stamp.json)")
    args = ap.parse_args(argv)
    if args.update and not args.reason:
        ap.error("--update needs --reason \"...\" (why the results changed; it's kept in stamp.json)")
    if args.reason and not args.update:
        ap.error("--reason only goes with --update")
    if args.update and (args.only or args.no_run):
        ap.error("--update freezes the whole suite from a fresh run; drop --only/--no-run")
    if args.update and (args.rtol or args.atol):
        ap.error("--rtol/--atol only apply to comparisons; --update freezes exact outputs")
    if args.parallel_runs < 1:
        ap.error("--parallel-runs must be >= 1")
    if args.rtol < 0 or args.atol < 0:
        ap.error("--rtol/--atol must be >= 0")
    runs_dir = args.runs_dir or repo_root / "runs" / "regression"
    try:
        suite = load_suite(args.suite or repo_root / "results" / "silver" / "suite.toml", repo_root)
        cases = select_cases(suite, args.only)
    except (OSError, ValueError) as e:
        print(f"[regress] error: {e}", file=sys.stderr)
        return 2
    uncovered = uncovered_configs(suite, repo_root)
    if uncovered:
        print(f"[regress] note: configs not in the suite: {uncovered} (add them to {suite.path.name} to cover them)")
    if args.no_run:
        missing = [f for c in cases for f in case_files(c, suite.seeds) if not (runs_dir / f).is_file()]
        if missing:
            print(f"[regress] error: --no-run, but {runs_dir} lacks {len(missing)} file(s), e.g. {missing[0]}; "
                  "run without --no-run first", file=sys.stderr)
            return 2
        failures: dict[str, str] = {}
    else:
        failures = run_suite(suite, cases, runs_dir, args.parallel_runs, runner)
    record = suite.path.parent
    data = record / "data"
    tol = Tolerance(args.rtol, args.atol)
    selected = {c.name for c in cases}
    results = [CaseResult(c.name, SKIPPED) if c.name not in selected
               else CaseResult(c.name, FAILED, message=failures[c.name]) if c.name in failures
               else compare_case(c, suite.seeds, runs_dir, data, tol) for c in suite.cases]
    stale = stale_entries(suite.cases, suite.seeds, data)
    stamp = load_stamp(record)
    env = environment_diffs(stamp["environment"], environment_stamp()) if stamp else []
    shown = record.relative_to(repo_root).as_posix() if record.is_relative_to(repo_root) else str(record)
    title = f"Regression check: {len(suite.cases)} cases x seeds {list(suite.seeds)} vs {shown}"
    text = format_report(results, tol, env, stale, title)
    print("\n" + text)
    (runs_dir / "report.md").write_text(f"```\n{text}\n```\n", encoding="utf-8")
    print(f"\n[regress] report -> {runs_dir / 'report.md'}")
    if args.update:
        if failures:
            print("[regress] not freezing: a job failed (see above)", file=sys.stderr)
            return 2
        written = freeze_silver(suite, runs_dir, reason=args.reason)
        print(f"[regress] froze {len(written)} files into {data}; commit {shown} together with the change")
        return 0
    return exit_code(results, stale)


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _remove(path: Path) -> None:
    """Delete a file or folder, clearing Windows' read-only attribute where needed (OneDrive
    marks synced folders read-only; see train.py's _unlink)."""
    def retry(func: Any, p: str, _exc: BaseException) -> None:
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.is_dir():
        shutil.rmtree(path, onexc=retry)
    else:
        try:
            path.unlink()
        except PermissionError:
            retry(os.unlink, str(path), PermissionError())
