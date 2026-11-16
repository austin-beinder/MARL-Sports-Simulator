# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: the record manifest (TOML)
#     schema, its strict loader, and path resolution relative to the
#     manifest's folder with frozen/unfrozen classification (backlog task 2,
#     step 2).
#   - 2026-11-01: Added ad_hoc_manifest: command-line arms validated by the
#     same rules as a manifest file (task 2, step 4).
# Prompt summary: Implement step 2 of the task 2 plotting-refactor plan: a
#   strictly validated TOML manifest listing each figure and its arms.
# Human review:   Pending review
# ==============================================================================
"""The manifest of a results record (``results/<record>/manifest.toml``).

A manifest lists the figures (``[[figure]]``, each with ``[[figure.arm]]`` or
``[[figure.point]]`` entries) and table sections (``[[table]]``) a record renders.
It is validated strictly, like ``marl/common/config.py``: unknown keys, kinds and
presets are errors that list what is allowed.

Paths are written with forward slashes, relative to the manifest's own folder,
so a record can be moved or renamed. A path inside that folder is **frozen** (its
data lives in the record); a path outside it (e.g. ``../../runs/...``) is
**unfrozen**: work in progress that ``scripts/freeze.py`` copies in. The loader
keeps each path exactly as written (``raw``), because freezing rewrites those
strings in the file's text; there is no TOML writer, so comments survive.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marl.results import Arm

MAX_ARMS = 9  # tab10 without its gray (gray is reserved for the scripted baselines)
COLUMN_PRESETS = ("keepaway", "sweep", "goto")
ARM_KINDS = ("curves", "final", "diagnostics", "goto")
# kind -> (required keys, optional keys), besides "kind", "name" and "title"
KIND_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "curves": (frozenset(), frozenset({"arm", "baselines", "ylim"})),
    "final": (frozenset(), frozenset({"arm", "baselines"})),
    "diagnostics": (frozenset(), frozenset({"arm", "window"})),
    "goto": (frozenset(), frozenset({"arm", "baselines", "ylim"})),
    "sweep": (frozenset(), frozenset({"point", "x_label", "highlight"})),
    "calibration": (frozenset({"csv", "config"}), frozenset()),
}
_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PathRef:
    raw: str  # exactly as written in the manifest (what freezing rewrites)
    path: str  # absolute and normalized; may be a glob
    frozen: bool  # inside the manifest's folder


@dataclass(frozen=True)
class ArmSpec:
    label: str
    runs: PathRef
    table: PathRef | None = None
    color: str | None = None

    def to_arm(self) -> Arm:
        return Arm(self.label, self.runs.path, Path(self.table.path) if self.table else None, self.color)


@dataclass(frozen=True)
class PointSpec:
    x: float
    table: PathRef
    label: str | None = None


@dataclass(frozen=True)
class FigureSpec:
    kind: str
    name: str
    title: str | None = None
    arms: tuple[ArmSpec, ...] = ()
    baselines: str | None = None
    ylim: tuple[float, float] | None = None
    window: int | None = None
    points: tuple[PointSpec, ...] = ()
    x_label: str | None = None
    highlight: float | None = None
    csv: PathRef | None = None
    config: PathRef | None = None

    def paths(self) -> list[PathRef]:
        out: list[PathRef] = []
        for a in self.arms:
            out += [a.runs] + ([a.table] if a.table else [])
        out += [p.table for p in self.points]
        return out + [p for p in (self.csv, self.config) if p is not None]


@dataclass(frozen=True)
class TableSpec:
    title: str
    columns: str
    source: PathRef | None = None
    sweep: str | None = None  # the name of a sweep figure whose points the table aggregates
    first: str | None = None  # the first column's header (default: "policy")


@dataclass(frozen=True)
class Manifest:
    name: str
    path: Path  # the manifest file
    root: Path  # its folder: the base of every relative path
    description: str | None = None
    figures: tuple[FigureSpec, ...] = ()
    tables: tuple[TableSpec, ...] = ()

    def paths(self) -> list[PathRef]:
        """Every path in the manifest, in file order (figures, then tables)."""
        return [p for f in self.figures for p in f.paths()] + [t.source for t in self.tables if t.source]

    def unfrozen(self) -> list[PathRef]:
        return [p for p in self.paths() if not p.frozen]


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a manifest; every error names the file."""
    path = Path(os.path.abspath(path))
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    try:
        return _manifest(data, path)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def ad_hoc_manifest(name: str, arms: list[tuple[str, str]], kinds: list[str], baselines: str | None = None,
                    title: str | None = None) -> Manifest:
    """A manifest for command-line arms ``(label, glob)``: one figure per kind, named
    ``<name>_<kind>``, validated by exactly the rules a manifest file gets. Paths are
    relative to the current folder; backslashes are accepted here and normalized."""
    figure_extra = {k: v for k, v in (("baselines", baselines), ("title", title)) if v is not None}
    data = {"name": name, "figure": [
        {"kind": kind, "name": f"{name}_{kind}", "arm": [{"label": lb, "runs": g.replace("\\", "/")} for lb, g in arms],
         **{k: v for k, v in figure_extra.items() if k == "title" or "baselines" in KIND_KEYS[kind][1]}}
        for kind in kinds
    ]}
    return _manifest(data, Path(os.path.abspath("(ad hoc)")))


# --- Parsing ------------------------------------------------------------------------------
def _manifest(data: dict[str, Any], path: Path) -> Manifest:
    _keys(data, "manifest", {"name"}, {"description", "figure", "table"})
    root = path.parent
    figures = tuple(_figure(f, root, f"figure[{i}]") for i, f in enumerate(_list(data, "figure", "manifest")))
    seen: set[str] = set()
    for f in figures:
        if f.name in seen:
            raise ValueError(f"duplicate figure name {f.name!r} (each figure is written to <name>.png)")
        seen.add(f.name)
    sweeps = {f.name for f in figures if f.kind == "sweep"}
    tables = tuple(_table(t, root, f"table[{i}]", sweeps) for i, t in enumerate(_list(data, "table", "manifest")))
    return Manifest(_str(data, "name", "manifest"), path, root, _opt_str(data, "description", "manifest"), figures, tables)


def _figure(data: Any, root: Path, where: str) -> FigureSpec:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a table, got {type(data).__name__}")
    kind = data.get("kind")
    if kind not in KIND_KEYS:
        raise ValueError(f"{where}: kind {kind!r} unknown; choose from {sorted(KIND_KEYS)}")
    required, optional = KIND_KEYS[kind]
    _keys(data, where, {"kind", "name"} | required, {"title"} | optional)
    name = _str(data, "name", where)
    if not _NAME.fullmatch(name):
        raise ValueError(f"{where}: name {name!r} must use only letters, digits, '_', '.' or '-' (it's a file name)")
    where = f"figure {name!r}"
    arms = tuple(_arm(a, root, f"{where}.arm[{i}]") for i, a in enumerate(_list(data, "arm", where)))
    points = tuple(_point(p, root, f"{where}.point[{i}]") for i, p in enumerate(_list(data, "point", where)))
    if kind in ARM_KINDS:
        if not arms:
            raise ValueError(f"{where}: a {kind} figure needs at least one arm ([[figure.arm]])")
        if len(arms) > MAX_ARMS:
            raise ValueError(f"{where}: {len(arms)} arms; a figure holds at most {MAX_ARMS} (split it into two figures)")
        labels = [a.label for a in arms]
        for label in labels:
            if labels.count(label) > 1:
                raise ValueError(f"{where}: duplicate arm label {label!r}")
    if kind == "sweep" and not points:
        raise ValueError(f"{where}: a sweep figure needs at least one point ([[figure.point]])")
    baselines = _opt_str(data, "baselines", where)
    if baselines is not None and baselines not in [a.label for a in arms]:
        raise ValueError(f"{where}: baselines {baselines!r} names no arm of this figure")
    ylim = data.get("ylim")
    if ylim is not None:
        if not (isinstance(ylim, list) and len(ylim) == 2 and all(_is_num(v) for v in ylim)):
            raise ValueError(f"{where}: ylim must be a list of two numbers, e.g. [50, 175]")
        ylim = (float(ylim[0]), float(ylim[1]))
    window = data.get("window")
    if window is not None and not (isinstance(window, int) and not isinstance(window, bool) and window >= 1):
        raise ValueError(f"{where}: window must be a whole number >= 1")
    return FigureSpec(
        kind=kind, name=name, title=_opt_str(data, "title", where), arms=arms, baselines=baselines, ylim=ylim,
        window=window, points=points, x_label=_opt_str(data, "x_label", where),
        highlight=_opt_num(data, "highlight", where),
        csv=_path(data["csv"], root, f"{where}.csv") if "csv" in data else None,
        config=_path(data["config"], root, f"{where}.config") if "config" in data else None,
    )


def _arm(data: Any, root: Path, where: str) -> ArmSpec:
    _keys(data, where, {"label", "runs"}, {"table", "color"})
    return ArmSpec(
        label=_str(data, "label", where),
        runs=_path(data["runs"], root, f"{where}.runs"),
        table=_path(data["table"], root, f"{where}.table") if "table" in data else None,
        color=_opt_str(data, "color", where),
    )


def _point(data: Any, root: Path, where: str) -> PointSpec:
    _keys(data, where, {"x", "table"}, {"label"})
    if not _is_num(data["x"]):
        raise ValueError(f"{where}: 'x' must be a number, got {data['x']!r}")
    return PointSpec(float(data["x"]), _path(data["table"], root, f"{where}.table"), _opt_str(data, "label", where))


def _table(data: Any, root: Path, where: str, sweeps: set[str]) -> TableSpec:
    _keys(data, where, {"title", "columns"}, {"source", "sweep", "first"})
    columns = _str(data, "columns", where)
    if columns not in COLUMN_PRESETS:
        raise ValueError(f"{where}: columns {columns!r} unknown; choose from {list(COLUMN_PRESETS)}")
    if ("source" in data) == ("sweep" in data):
        raise ValueError(f"{where}: give exactly one of 'source' or 'sweep'")
    sweep = _opt_str(data, "sweep", where)
    if sweep is not None and sweep not in sweeps:
        raise ValueError(f"{where}: sweep {sweep!r} names no sweep figure in this manifest")
    return TableSpec(
        title=_str(data, "title", where), columns=columns,
        source=_path(data["source"], root, f"{where}.source") if "source" in data else None,
        sweep=sweep, first=_opt_str(data, "first", where),
    )


# --- Small validators ----------------------------------------------------------------------
def _keys(data: Any, where: str, required: set[str], optional: set[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a table, got {type(data).__name__}")
    unknown = set(data) - required - optional
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(required | optional)}")
    missing = required - set(data)
    if missing:
        raise ValueError(f"{where}: missing key(s) {sorted(missing)}")


def _list(data: dict[str, Any], key: str, where: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{where}: {key!r} must be a list of tables ([[{key}]] entries)")
    return value


def _str(data: dict[str, Any], key: str, where: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _opt_str(data: dict[str, Any], key: str, where: str) -> str | None:
    return _str(data, key, where) if key in data else None


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _opt_num(data: dict[str, Any], key: str, where: str) -> float | None:
    if key not in data:
        return None
    if not _is_num(data[key]):
        raise ValueError(f"{where}: {key!r} must be a number")
    return float(data[key])


def _path(raw: Any, root: Path, where: str) -> PathRef:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{where}: expected a path string")
    if "\\" in raw:
        raise ValueError(f"{where}: write paths with forward slashes ({raw!r})")
    path = os.path.abspath(os.path.join(root, raw))
    return PathRef(raw, path, Path(path).is_relative_to(root))
