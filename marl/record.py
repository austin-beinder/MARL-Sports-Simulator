# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: freezing a results record in
#     place (copy the manifest's unfrozen inputs into data/, mirrored under
#     runs/; rewrite only those path strings in the manifest's text; append
#     provenance; render the record's figures), validated before any write,
#     plus its command line (backlog task 2, step 6).
# Prompt summary: Implement step 6 of the task 2 plotting-refactor plan: a
#   freeze that keeps the manifest's comments and never half-updates a record.
# Human review:   Pending review
# ==============================================================================
"""Freeze a results record (``results/<record>/``) in place.

The record's ``manifest.toml`` may point outside its folder (``../../runs/...``):
those inputs are *unfrozen*. Freezing copies the files behind them into the
record's ``data/`` folder, mirroring their path under ``runs/`` (so
``runs/Old_mock_env/x_seed0`` becomes ``data/Old_mock_env/x_seed0``), rewrites
just those quoted path strings in the manifest's text (comments and layout are
kept; there is no TOML writer), appends a provenance comment (date, git commit,
the mappings) and renders the figures into ``figures/``.

Copied per run directory: ``config.json`` (if present), ``train.csv``, ``eval.csv``;
per table: the table and its ``_per_run.csv`` twin; never checkpoints or logs.
Every check runs before the first copy, so a failed freeze changes nothing. Files
already in ``data/`` are replaced only with ``overwrite``.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from marl.manifest import PathRef, load_manifest
from marl.plots import clear_outputs, render_manifest
from marl.results import run_dirs, table_path_for

RUNS = Path(__file__).resolve().parents[1] / "runs"  # the repo's runs/ folder
RESULTS = Path("results")
RUN_FILES = (("config.json", False), ("train.csv", True), ("eval.csv", True))  # (file, required)


@dataclass
class FreezeResult:
    record: Path
    mapping: dict[str, str] = field(default_factory=dict)  # path string as written -> its frozen string
    copied: list[Path] = field(default_factory=list)  # destination files
    made: list[Path] = field(default_factory=list)  # rendered figures + table view
    notes: list[str] = field(default_factory=list)


def freeze(manifest_path: str | Path, runs_root: str | Path = RUNS, overwrite: bool = False,
           revision: str | None = None) -> FreezeResult:
    """Freeze a record's unfrozen inputs and render its figures (see the module docstring)."""
    manifest = load_manifest(manifest_path)
    root, runs_root = manifest.root, Path(os.path.abspath(runs_root))
    result = FreezeResult(root)
    plan: dict[Path, Path] = {}  # source -> destination

    def frozen_string(ref: PathRef) -> str:
        if not Path(ref.path).is_relative_to(runs_root):
            raise ValueError(f"{ref.raw!r} is outside {runs_root}; only inputs under runs/ can be frozen")
        result.mapping[ref.raw] = "data/" + Path(os.path.relpath(ref.path, runs_root)).as_posix()
        return result.mapping[ref.raw]

    def add(src: Path) -> None:
        plan[src] = root / "data" / os.path.relpath(src, runs_root)

    def add_file(ref: PathRef, twin: bool = False) -> None:
        frozen_string(ref)
        src = Path(ref.path)
        if not src.is_file():
            raise ValueError(f"{ref.raw!r} not found ({src})")
        add(src)
        per_run = src.with_name(src.stem + "_per_run.csv")
        if twin and per_run.is_file():
            add(per_run)

    def note(text: str) -> None:
        if text not in result.notes:
            result.notes.append(text)

    # 1. Plan every copy (and fail on anything missing) before touching the record.
    for fig in manifest.figures:
        for arm in fig.arms:
            if not arm.runs.frozen:
                frozen_string(arm.runs)
                dirs = run_dirs(arm.runs.path)
                if not dirs:
                    raise ValueError(f"figure {fig.name!r}, arm {arm.label!r}: {arm.runs.raw!r} matches no run directories")
                for d in dirs:
                    for name, required in RUN_FILES:
                        if (d / name).is_file():
                            add(d / name)
                        elif required:
                            raise ValueError(f"{d} has no {name} (is the run finished?)")
                if arm.table is None:  # the table evaluate.py wrote for these runs, by its naming convention
                    table = table_path_for(dirs)
                    if table.is_file():
                        add(table)
                        if (per_run := table.with_name(table.stem + "_per_run.csv")).is_file():
                            add(per_run)
                    else:
                        note(f"warning: arm {arm.label!r} has no evaluation table {table} "
                             "(run scripts/evaluate.py on its runs, then freeze again with --overwrite)")
            if arm.table is not None and not arm.table.frozen:
                add_file(arm.table, twin=True)
        for point in fig.points:
            if not point.table.frozen:
                add_file(point.table)
        for ref in (fig.csv, fig.config):
            if ref is not None and not ref.frozen:
                add_file(ref)
    for t in manifest.tables:
        if t.source is not None and not t.source.frozen:
            add_file(t.source, twin=True)

    # 2. Check the destinations and prepare the manifest's new text.
    existing = [d for d in plan.values() if d.exists()]
    if existing and not overwrite:
        raise ValueError(f"{len(existing)} destination file(s) already exist (e.g. {existing[0]}); "
                         "pass --overwrite to replace them")
    text = manifest.path.read_text(encoding="utf-8")
    for raw, new in result.mapping.items():
        if not any(f"{q}{raw}{q}" in text for q in ('"', "'")):
            raise ValueError(f"can't find the path string {raw!r} in {manifest.path} as a plain quoted string")
        for q in ('"', "'"):
            text = text.replace(f"{q}{raw}{q}", f"{q}{new}{q}")

    # 3. Write: copy, rewrite the manifest with provenance, render.
    for src, dest in plan.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        result.copied.append(dest)
    if result.mapping:
        revision = revision or git_revision()
        lines = [f"# --- frozen {datetime.date.today().isoformat()} by scripts/freeze.py at commit {revision} ---",
                 *(f"#   {raw} -> {new}" for raw, new in result.mapping.items())]
        manifest.path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    figures = root / "figures"
    if figures.exists():
        clear_outputs(figures)
    made, notes = render_manifest(load_manifest(manifest.path), figures)
    result.made = made
    for n in notes:
        note(n)
    return result


def git_revision() -> str:
    """The short commit hash, plus a marker if the working tree has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return head.stdout.strip() + (" + uncommitted changes" if dirty.stdout.strip() else "")


def _manifest_path(record: str) -> Path:
    """A record name (results/<name>/manifest.toml), a record folder, or a manifest file."""
    path = Path(record)
    if path.suffix == ".toml":
        return path
    if path.is_dir():
        return path / "manifest.toml"
    return RESULTS / record / "manifest.toml"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze a results record: copy the inputs its manifest still reads "
                                             "from runs/ into its data/ folder, then render its figures.")
    ap.add_argument("record", help="record name (results/<name>/), record folder, or manifest file")
    ap.add_argument("--overwrite", action="store_true", help="replace files that already exist in the record's data/")
    ap.add_argument("--runs-root", type=Path, default=RUNS, help=f"the runs/ folder inputs must lie under (default: {RUNS})")
    args = ap.parse_args(argv)
    try:
        result = freeze(_manifest_path(args.record), args.runs_root, args.overwrite)
    except ValueError as e:
        print(f"[freeze] error: {e}", file=sys.stderr)
        return 1
    print(f"[freeze] copied {len(result.copied)} file(s) into {result.record / 'data'}")
    for raw, new in result.mapping.items():
        print(f"[freeze]   {raw} -> {new}")
    for path in result.made:
        print(f"[freeze] {path}")
    for note in result.notes:
        print(f"[freeze] {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
