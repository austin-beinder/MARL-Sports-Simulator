# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the report figures (keep-away
#     learning curves per start type, final-results bars, the calibration
#     curve, the goto_target sanity check, training diagnostics) and a
#     markdown table view, styled with a validated light-mode palette and the
#     "emphasis" form (design #23; Phase 3 step 11).
#   - 2026-10-14: Added the PBRS ablation figures (survival-only vs + PBRS
#     learning curves; the coefficient-pilot sweep of closing speed, races won
#     and possession survival), their table views, race columns in the
#     keep-away table, and a validated second series color (slot 2).
#   - 2026-11-01: Moved the CSV/curve/table helpers (read_csv, eval_curves,
#     train_curves, read_table, table_path_for, mean_std, smooth) unchanged
#     to marl/results.py and re-imported them here (task 2, step 1).
#   - 2026-11-01: Reworked the figures into manifest-driven kinds (curves,
#     final, diagnostics, goto, sweep, calibration) that take 1-9 arms, with
#     tab10 colors that follow the arm label; figure builders now return the
#     figure and a render layer loads, saves (PNG only) and writes the table
#     view; replaced the hard-coded command line with --manifest (task 2,
#     step 3).
#   - 2026-11-01: Command line for ad-hoc --arm plots (default kinds per
#     scenario), the --overwrite guard (record renders to their default folder
#     exempt), and warnings for mismatched eval episodes and unfrozen record
#     inputs (task 2, step 4).
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs); later, the task 2 plotting
#   refactor (records, manifests, N-arm figures).
# Human review:   Pending review
# ==============================================================================
"""Report figures. Reads only CSVs that training, ``scripts/evaluate.py`` and
``scripts/calibrate.py`` already wrote, so making figures never re-runs anything.

What to draw comes from a manifest (``marl/manifest.py``): each ``[[figure]]`` has a
``kind`` and its arms (or sweep points). The builders (``fig_*``) take loaded data and
return a matplotlib figure; ``render_figure`` loads a figure's inputs and builds it;
``render_manifest`` saves every figure as PNG plus the ``results_tables.md`` table view.

Visual rules (data-viz method, light mode):

* Each learned arm is one colored series: matplotlib's ``tab10`` minus its gray, in
  fixed order, assigned by arm label so an arm keeps its color in every figure of a
  manifest (up to 9 arms; the student waived the colorblind checks, 2026-11-01).
* Scripted baselines are gray context lines, each direct-labeled with its value.
* The shaded band is +/- 1 standard deviation across training seeds (ddof=1, as
  in the evaluation table); the line is the mean.
* Thin solid lines, hairline solid gridlines, text in ink tokens (never series color).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from marl.common.config import config_from_dict, load_config  # noqa: E402
from marl.manifest import ARM_KINDS, FigureSpec, Manifest, PointSpec, ad_hoc_manifest, load_manifest  # noqa: E402
from marl.results import (  # noqa: E402,F401 (re-exported: callers import these from marl.plots)
    EVAL_METRICS,
    TRAIN_METRICS,
    Arm,
    eval_curves,
    eval_mismatch,
    mean_std,
    read_csv,
    read_table,
    run_config,
    smooth,
    table_path_for,
    train_curves,
)

# Light-mode tokens from the data-viz reference palette.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Arm colors: tab10 without its gray (#7f7f7f), which would read as a baseline.
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf")
CONTEXT = MUTED  # de-emphasis gray for reference/context marks (always direct-labeled)
BAND_ALPHA = 0.12

TRIVIAL = ("random", "still", "untrained")  # baselines that never learn to evade
CURVE_PANELS = (("len_mean_possession", "Possession starts"), ("len_mean_free_ball", "Free-ball starts"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
        }
    )


# --- Data shapes ----------------------------------------------------------------------
@dataclass(frozen=True)
class Series:
    """One arm's loaded curves (arrays [n_runs, n_points] plus ``env_steps``) and its color."""

    label: str
    color: str
    curves: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(next(v for k, v in self.curves.items() if k != "env_steps"))


@dataclass(frozen=True)
class SweepPoint:
    """One x position of a sweep: the PPO (stochastic) rows of its per-run evaluation CSV."""

    x: float
    label: str | None
    rows: list[dict[str, float]]

    @property
    def name(self) -> str:
        return self.label or f"{self.x:g}"


def assign_colors(figures: Sequence[FigureSpec]) -> dict[str, str]:
    """Arm label -> color for a whole manifest. Explicit ``color`` overrides win; the other
    labels take the unused palette slots in order of first appearance."""
    colors: dict[str, str] = {}
    for f in figures:
        for a in f.arms:
            if a.color is not None:
                if colors.get(a.label, a.color) != a.color:
                    raise ValueError(f"arm {a.label!r} is given two colors ({colors[a.label]} and {a.color})")
                colors[a.label] = a.color
    free = [c for c in PALETTE if c not in colors.values()]
    for f in figures:
        for a in f.arms:
            if a.label not in colors:
                if not free:
                    raise ValueError(f"more than {len(PALETTE)} distinct arm labels; give some arms an explicit color")
                colors[a.label] = free.pop(0)
    return colors


# --- Marks ----------------------------------------------------------------------------
def _mean_band(ax, x: np.ndarray, Y: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray]:
    m, sd = mean_std(Y)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, m, color=color, zorder=3)
    return m, sd


def _end_dot(ax, x: float, y: float, color: str) -> None:
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)


def _reference(ax, y: float, text: str, above: bool, right: bool = False) -> None:
    ax.axhline(y, color=CONTEXT, linewidth=1.5, zorder=1)
    ax.annotate(
        text,
        xy=(0.99 if right else 0.01, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4 if above else -4),
        textcoords="offset points",
        ha="right" if right else "left",
        va="bottom" if above else "top",
        fontsize=9,
        color=INK_2,
    )


def _end_label(ax, x: float, y: float, text: str, refs: Sequence[float], series: np.ndarray) -> None:
    """Label the line end where it can't collide: away from a reference line that is
    close (< 8% of the y-range), otherwise on the side the line is NOT arriving from.
    Call after the y-limits are final."""
    lo, hi = ax.get_ylim()
    close = [r for r in refs if abs(r - y) < 0.08 * (hi - lo)]
    if close:
        above = min(close, key=lambda r: abs(r - y)) < y
    else:
        above = len(series) < 2 or series[-1] >= series[-2]  # rising into the end -> label above
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(-8, 9 if above else -9),
        textcoords="offset points",
        ha="right",
        va="bottom" if above else "top",
        fontsize=9.5,
        fontweight="semibold",
        color=INK,
    )


def _end_labels(ax, ends: Sequence[tuple[float, float, str]], refs: Sequence[float]) -> None:
    """Direct-label several line ends at once. Two ends closer than 8% of the y-range
    split: the higher label goes above its line, the lower one below. A lone end avoids
    a nearby reference line (else goes above). Call after the y-limits are final."""
    lo, hi = ax.get_ylim()
    near = 0.08 * (hi - lo)
    for k, (x, y, text) in enumerate(ends):
        rivals = [(e[1], j) for j, e in enumerate(ends) if j != k and abs(e[1] - y) < near]
        if rivals:
            above = (y, k) > max(rivals)
        else:
            close = [r for r in refs if abs(r - y) < near]
            above = not close or min(close, key=lambda r: abs(r - y)) < y
        ax.annotate(text, xy=(x, y), xytext=(-8, 9 if above else -9), textcoords="offset points", ha="right",
                    va="bottom" if above else "top", fontsize=9.5, fontweight="semibold", color=INK)


def _label_ends(ax, ends: Sequence[tuple[float, float, str, np.ndarray]], refs: Sequence[float]) -> None:
    """One arm: label by the line's direction; several: split close ends apart."""
    if len(ends) == 1:
        x, y, text, m = ends[0]
        _end_label(ax, x, y, text, refs, m)
    else:
        _end_labels(ax, [e[:3] for e in ends], refs)


def _suptitle(fig, text: str, fontsize: float = 12) -> None:
    fig.suptitle(text, x=0.01, ha="left", color=INK, fontsize=fontsize, fontweight="semibold")


def _series_legend(fig, series: Sequence[Series], context: str | None = None) -> None:
    handles: list = [(Patch(facecolor=s.color, alpha=BAND_ALPHA, linewidth=0), Line2D([], [], color=s.color, linewidth=2))
                     for s in series]
    labels = [f"{s.label}: mean, shaded ± 1 std over {s.n} seeds" for s in series]
    if context:
        handles.append(Line2D([], [], color=CONTEXT, linewidth=1.5))
        labels.append(context)
    fig.legend(handles, labels, loc="outside lower center", ncols=min(len(labels), 3), fontsize=9)


def _keepaway_refs(ax, baselines: dict[str, dict[str, float]] | None, key: str) -> list[float]:
    """The scripted reference lines of a keep-away panel (whichever the table has)."""
    if baselines is None:
        return []
    refs = []
    if "flee" in baselines:
        flee = baselines["flee"][key]
        _reference(ax, flee, f"flee (scripted): {flee:.0f}", above=True)
        refs.append(flee)
    trivial = [p for p in TRIVIAL if p in baselines]
    if trivial:
        value = float(np.mean([baselines[p][key] for p in trivial]))
        _reference(ax, value, f"{' / '.join(trivial)}: {value:.0f}", above=False)
        refs.append(value)
    return refs


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, metadata={"Software": None})  # no version stamp: fewer byte changes
    plt.close(fig)
    return path


# --- Figure builders (loaded data in, figure out) ---------------------------------------------
def fig_curves(series: Sequence[Series], baselines: dict[str, dict[str, float]] | None = None,
               title: str | None = None, ylim: tuple[float, float] | None = None) -> Figure:
    """Keep-away evaluation curves per start type, one mean +/- std band per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    if ylim:
        axes[0].set_ylim(*ylim)
    pending = []
    for ax, (key, panel) in zip(axes, CURVE_PANELS):
        refs = _keepaway_refs(ax, baselines, key)
        ends = []
        for s in series:
            x = s.curves["env_steps"] / 1e6
            m, sd = _mean_band(ax, x, s.curves[key], s.color)
            _end_dot(ax, x[-1], m[-1], s.color)
            ends.append((x[-1], m[-1], f"{s.label} {m[-1]:.0f} ± {sd[-1]:.0f}", m))
        pending.append((ax, ends, refs))
        ax.set_title(panel)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, max(e[0] for e in ends) * 1.02)
    axes[0].set_ylabel("Mean episode length (control steps)")
    for ax, ends, refs in pending:  # after both panels: the shared y-limits are final
        _label_ends(ax, ends, refs)
    _suptitle(fig, title or "Keep-away: survival on the fixed evaluation episodes during training")
    _series_legend(fig, series, "scripted baselines (values labeled)" if baselines else None)
    return fig


BASELINE_ROWS = (("flee", "flee (scripted)"), ("random", "random"), ("still", "stand still"),
                 ("untrained", "untrained PPO"))


def fig_final(arms: Sequence[tuple[str, str, dict[str, dict[str, float]]]],
              baselines: dict[str, dict[str, float]] | None = None, title: str | None = None) -> Figure:
    """Final evaluation bars: each arm's PPO (stochastic, deterministic), then the baselines."""
    rows = []  # (name, row, color)
    for label, color, table in arms:
        for policy, mode in (("ppo", "stochastic"), ("ppo_deterministic", "deterministic")):
            if policy in table:
                rows.append((f"{label} ({mode})", table[policy], color))
    if baselines is not None:
        rows += [(name, baselines[p], CONTEXT) for p, name in BASELINE_ROWS if p in baselines]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.8, 1.1 + 0.45 * len(rows))), sharey=True, layout="constrained")
    for ax, (key, panel) in zip(axes, CURVE_PANELS):
        vals = np.array([r[key] for _, r, _ in rows])
        stds = np.array([r[f"{key}_seed_std"] for _, r, _ in rows])
        ax.barh(y, vals, height=0.5, color=[c for _, _, c in rows], zorder=2)
        for yi, v, s in zip(y, vals, stds):
            has_spread = math.isfinite(s)
            if has_spread:
                ax.errorbar(v, yi, xerr=s, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
            ax.annotate(
                f"{v:.1f} ± {s:.1f}" if has_spread else f"{v:.1f}",
                xy=(v + (s if has_spread else 0.0), yi),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK_2,
            )
        ax.set_yticks(y, [name for name, _, _ in rows])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(np.nanmax(vals + np.nan_to_num(stds))) * 1.32)
        ax.set_xlabel("Mean episode length (control steps)")
        ax.set_title(panel)
    _suptitle(fig, title or "Keep-away: final evaluation (same fixed episodes for every policy)")
    handles = [Patch(color=color) for _, color, _ in arms]
    labels = [f"{label} (learned); error bars ± 1 std over {int(table['ppo']['n_seeds'])} seeds"
              for label, _, table in arms]
    if baselines is not None:
        handles.append(Patch(color=CONTEXT))
        labels.append("baselines")
    fig.legend(handles, labels, loc="outside lower center", ncols=min(len(labels), 3), fontsize=9)
    return fig


def fig_calibration(rows: Sequence[dict[str, str]], chosen: float, carrier: float, title: str | None = None) -> Figure:
    speeds = sorted({float(r["red_max_speed"]) for r in rows})

    def survival(policies: Sequence[str]) -> np.ndarray:
        return np.array(
            [100 * np.mean([float(r["survival"]) for r in rows if float(r["red_max_speed"]) == v and r["policy"] in policies])
             for v in speeds]
        )

    flee, trivial = survival(["flee"]), survival(["random", "still"])
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout="constrained")
    ax.axhspan(30, 70, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.annotate("target band 30–70%", xy=(0.01, 70), xycoords=("axes fraction", "data"), xytext=(0, -4),
                textcoords="offset points", va="top", fontsize=9, color=INK_2)
    for v, text in ((carrier, f"carrier speed {carrier:g} m/s"), (chosen, f"chosen: {chosen:g} m/s")):
        ax.axvline(v, color=AXIS if v == carrier else INK_2, linewidth=1, zorder=1)
        ax.annotate(text, xy=(v, 98), xytext=(4, 0), textcoords="offset points", va="top", fontsize=9, color=INK_2)
    ax.plot(speeds, trivial, color=CONTEXT, linewidth=1.5, zorder=2)
    ax.plot(speeds, flee, color=PALETTE[0], marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    ax.annotate("flee (scripted)", xy=(speeds[-1], flee[-1]), xytext=(0, 10), textcoords="offset points",
                ha="right", va="bottom", fontsize=9.5, fontweight="semibold", color=INK)
    ax.annotate("random / still", xy=(speeds[-1], trivial[-1]), xytext=(0, -8), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=INK_2)
    if chosen in speeds:
        y = flee[speeds.index(chosen)]
        ax.annotate(f"{y:.0f}%", xy=(chosen, y), xytext=(8, 4), textcoords="offset points", fontsize=9.5,
                    fontweight="semibold", color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(speeds[0] - 0.02, speeds[-1] + 0.02)
    ax.set_xlabel("Red max speed (m/s)")
    ax.set_ylabel("Survival (% of max episode length)")
    ax.set_title(title or "Calibrating red's speed against the scripted flee policy")
    ax.legend(
        [Line2D([], [], color=PALETTE[0], marker="o", ms=7, mec=SURFACE, mew=1.5),
         Line2D([], [], color=CONTEXT, linewidth=1.5), Patch(color=GRID, alpha=0.6)],
        ["flee (scripted)", "random / still (overlap)", "target band"],
        loc="lower left",
        fontsize=9,
    )
    return fig


def fig_goto(series: Sequence[Series], baselines: dict[str, dict[str, float]] | None = None,
             title: str | None = None, ylim: tuple[float, float] | None = None) -> Figure:
    """goto_target: mean distance to the target (= minus the reward) during training."""
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
    refs: list[float] = []
    if baselines is not None:
        oracle = -baselines["oracle"]["mean_reward"]
        trivial = -float(np.mean([baselines[p]["mean_reward"] for p in ("random", "still")]))
        ylim = ylim or (0, trivial * 1.18)
        _reference(ax, oracle, f"oracle (straight to target): {oracle:.2f} m", above=True)
        _reference(ax, trivial, f"random / still: {trivial:.1f} m", above=True)
        refs = [oracle, trivial]
    if ylim:
        ax.set_ylim(*ylim)
    ends = []
    for s in series:
        x = s.curves["env_steps"] / 1e3
        m, sd = _mean_band(ax, x, -s.curves["mean_reward"], s.color)  # reward = -distance (m)
        _end_dot(ax, x[-1], m[-1], s.color)
        ends.append((x[-1], m[-1], f"{s.label} {m[-1]:.2f} ± {sd[-1]:.2f} m", m))
    _label_ends(ax, ends, refs)
    ax.set_xlim(0, max(e[0] for e in ends) * 1.02)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean distance to target (m), lower is better")
    ax.set_title(title or "goto_target sanity check")
    _series_legend(fig, series, "references (values labeled)" if baselines else None)
    return fig


def fig_diagnostics(series: Sequence[Series], title: str | None = None, window: int = 9) -> Figure:
    x_panels = (
        ("entropy", "Policy entropy (nats; max 0 = uniform)"),
        ("explained_variance", "Critic explained variance"),
        ("approx_kl", "Approx. KL(old || new) per update"),
        ("clip_frac", "Fraction of ratios clipped"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, (key, panel) in zip(axes.flat, x_panels):
        for s in series:
            _mean_band(ax, s.curves["env_steps"] / 1e6, smooth(s.curves[key], window), s.color)
        ax.set_title(panel, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Environment steps (millions)")
    counts = {s.n for s in series}
    seeds = f"{counts.pop()} seeds" if len(counts) == 1 else "each arm's seeds"
    _suptitle(fig, f"{title or 'PPO training diagnostics'} (mean ± 1 std over {seeds}; {window}-iteration moving average)",
              fontsize=11)
    if len(series) > 1:
        _series_legend(fig, series)
    return fig


SWEEP_METRICS = ("approach_m_per_step", "approach_alignment", "pickup_rate", "len_mean_possession",
                 "len_mean_free_ball", "env_steps")
SWEEP_PANELS = (  # key, title, y label, scale, value format, y-limits
    ("approach_m_per_step", "Closing speed on the loose ball", "Meters closed per control step", 1.0, "{:.3f}", (0, 0.11)),
    ("pickup_rate", "Races won", "% of free-ball evaluation starts", 100.0, "{:.0f}%", (0, 105)),
    ("len_mean_possession", "Survival in possession starts", "Mean episode length (control steps)", 1.0, "{:.0f}", (50, 175)),
)


def sweep_points(specs: Sequence[PointSpec]) -> tuple[list[SweepPoint], dict[str, float]]:
    """Load each point's per-run evaluation CSV: its PPO (stochastic) rows, plus the flee
    baseline (the same in every file: same fixed episodes). Sorted by x."""
    points, flee = [], None
    for spec in specs:
        rows = read_csv(_need(spec.table.path))
        ppo = [{k: float(v) for k, v in r.items() if k not in ("policy", "run")} for r in rows if r["policy"] == "ppo"]
        if not ppo:
            raise ValueError(f"{spec.table.path}: no PPO rows (was it written by scripts/evaluate.py?)")
        points.append(SweepPoint(spec.x, spec.label, ppo))
        if flee is None:
            flee = next(({k: float(v) for k, v in r.items() if k not in ("policy", "run")}
                         for r in rows if r["policy"] == "flee"), None)
    if flee is None:
        raise ValueError("the sweep files hold no flee rows (were they written by scripts/evaluate.py?)")
    return sorted(points, key=lambda p: p.x), flee


def sweep_table(points: Sequence[SweepPoint]) -> dict[str, dict[str, float]]:
    """Mean and seed std per point, in the shape ``table_markdown`` takes."""
    table = {}
    for p in sorted(points, key=lambda q: q.x):
        out: dict[str, float] = {"n_seeds": len(p.rows)}
        for k in SWEEP_METRICS:
            vals = np.array([r[k] for r in p.rows])
            out[k] = float(vals.mean())
            out[f"{k}_seed_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        table[p.name] = out
    return table


def fig_sweep(points: Sequence[SweepPoint], flee: dict[str, float], highlight: float | None = None,
              title: str | None = None, x_label: str | None = None) -> Figure:
    """One position per x value (ordinal: the values needn't be evenly spaced), seeds as
    faint dots behind the mean; flee as the gray "runs straight at it" reference."""
    color = PALETTE[0]
    xs = [p.x for p in points]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    for ax, (key, panel, ylabel, scale, fmt, ylim) in zip(axes, SWEEP_PANELS):
        ax.set_ylim(*ylim)
        near = 0.08 * (ylim[1] - ylim[0])
        ref = flee[key] * scale
        _reference(ax, ref, f"flee (scripted): {fmt.format(ref)}", above=False, right=True)
        means = []
        for i, p in enumerate(points):
            vals = np.array([r[key] for r in p.rows]) * scale
            # clip_on=False: a 0% point sits ON the baseline and must show whole
            ax.plot(np.full(len(vals), i), vals, "o", ms=5, color=color, alpha=0.35, mew=0, zorder=2, clip_on=False)
            means.append(float(vals.mean()))
            if i == 0 or p.x == highlight:  # label selectively: the first point vs the highlighted one
                # Clear the seed dots: above the cluster, or below it if a reference line is close above.
                above = not 0 <= ref - vals.max() < near
                ax.annotate(fmt.format(means[-1]), xy=(i, vals.max() if above else vals.min()),
                            xytext=(0, 7 if above else -7), textcoords="offset points", ha="center",
                            va="bottom" if above else "top", fontsize=9.5, fontweight="semibold", color=INK)
        ax.plot(range(len(points)), means, color=color, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, clip_on=False)
        # Tick labels wrap at the first space ("0 (survival only)" -> two lines) to stay narrow.
        ax.set_xticks(range(len(points)), [p.name.replace(" ", "\n", 1) + ("\n(chosen)" if p.x == highlight else "")
                                           for p in points])
        if highlight in xs:  # mark the highlighted value on its tick label, in ink (no extra line through the data)
            tick = ax.get_xticklabels()[xs.index(highlight)]
            tick.set_color(INK)
            tick.set_fontweight("semibold")
        ax.set_xlim(-0.5, len(points) - 0.5)
        ax.set_xlabel(x_label or "Swept value")
        ax.set_ylabel(ylabel)
        ax.set_title(panel, fontsize=10)
    _suptitle(fig, title or "Sweep")
    counts = [len(p.rows) for p in points]
    common = max(set(counts), key=counts.count)
    others = [f"{len(p.rows)} at {p.name}" for p in points if len(p.rows) != common]
    n_text = f"{common} per point" + (f"; {', '.join(others)}" if others else "")
    handles = [
        Line2D([], [], color=color, marker="o", ms=7, mec=SURFACE, mew=1.5),
        Line2D([], [], linestyle="", marker="o", ms=5, color=color, alpha=0.35, mew=0),
        Line2D([], [], color=CONTEXT, linewidth=1.5),
    ]
    labels = [f"PPO, stochastic eval: mean over training seeds ({n_text})", "individual training seeds",
              "flee (scripted): runs straight at a loose ball"]
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    return fig


# --- Table view -----------------------------------------------------------------------
def table_markdown(title: str, table: dict[str, dict[str, float]], columns: Sequence[tuple[str, ...]],
                   first: str = "policy") -> str:
    """Columns are (key, name) or (key, name, format spec); the default format is ".2f"."""

    def cell(row: dict[str, float], key: str, fmt: str) -> str:
        v, s = row.get(key, float("nan")), row.get(f"{key}_seed_std", float("nan"))
        if not math.isfinite(v):
            return "-"
        return f"{v:{fmt}} ± {s:{fmt}}" if math.isfinite(s) else f"{v:{fmt}}"

    head = f"| {first} | seeds | " + " | ".join(c[1] for c in columns) + " |"
    sep = "|---|---|" + "---|" * len(columns)
    body = [f"| {p} | {int(r['n_seeds'])} | " + " | ".join(cell(r, c[0], c[2] if len(c) > 2 else ".2f") for c in columns) + " |"
            for p, r in table.items()]
    return "\n".join([f"### {title}", "", head, sep, *body, ""])


KEEPAWAY_COLUMNS = (
    ("len_mean", "overall length"), ("len_mean_possession", "possession starts"),
    ("len_mean_free_ball", "free-ball starts"), ("frac_truncated", "fraction truncated"),
    ("pickup_rate", "free-ball races won (0-1)"), ("approach_m_per_step", "closing speed (m/step)", ".4f"),
)
SWEEP_COLUMNS = (
    ("approach_m_per_step", "closing speed (m/step)", ".4f"), ("approach_alignment", "alignment (cos)"),
    ("pickup_rate", "races won (0-1)"), ("len_mean_possession", "possession starts"),
    ("len_mean_free_ball", "free-ball starts"), ("env_steps", "env steps", ".0f"),
)
COLUMNS = {"keepaway": KEEPAWAY_COLUMNS, "sweep": SWEEP_COLUMNS, "goto": (("mean_reward", "mean reward (-m)"),)}


# --- Rendering (manifest in, files out) ---------------------------------------------------
class MissingInput(FileNotFoundError):
    """An input a figure or table needs hasn't been written yet (e.g. no evaluate.py table)."""


def _need(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise MissingInput(f"{path} not found")
    return path


def _arm_table(arm: Arm) -> dict[str, dict[str, float]]:
    path = arm.table_path()
    if not path.exists():
        raise MissingInput(f"arm {arm.label!r} has no evaluation table {path}; "
                           f"run `python scripts/evaluate.py \"{arm.runs}\"` first")
    return read_table(path)


def _shared_config(path: str | Path):
    """The ``shared`` config of a run's config.json (or of a TOML config file)."""
    path = _need(path)
    if path.suffix == ".toml":
        return load_config(path).shared
    return config_from_dict(json.loads(path.read_text(encoding="utf-8"))).shared


def render_figure(spec: FigureSpec, colors: dict[str, str]) -> Figure:
    """Load one figure's inputs and build it. Raises MissingInput before drawing anything."""
    arms = [a.to_arm() for a in spec.arms]
    base = _arm_table(next(a for a in arms if a.label == spec.baselines)) if spec.baselines else None

    def series(load) -> list[Series]:
        return [Series(a.label, colors[a.label], load(a)) for a in arms]

    if spec.kind == "curves":
        return fig_curves(series(lambda a: a.eval_curves()), base, spec.title, spec.ylim)
    if spec.kind == "final":
        return fig_final([(a.label, colors[a.label], _arm_table(a)) for a in arms], base, spec.title)
    if spec.kind == "diagnostics":
        return fig_diagnostics(series(lambda a: a.train_curves()), spec.title, spec.window or 9)
    if spec.kind == "goto":
        return fig_goto(series(lambda a: a.eval_curves()), base, spec.title, spec.ylim)
    if spec.kind == "sweep":
        points, flee = sweep_points(spec.points)
        return fig_sweep(points, flee, spec.highlight, spec.title, spec.x_label)
    if spec.kind == "calibration":
        rows = read_csv(_need(spec.csv.path))
        cfg = _shared_config(spec.config.path)
        return fig_calibration(rows, cfg.red_max_speed, cfg.max_speed * cfg.carry_speed_factor, spec.title)
    raise ValueError(f"unknown figure kind {spec.kind!r}")  # the manifest loader already rejects these


def render_tables(manifest: Manifest) -> tuple[list[str], list[str]]:
    """The ``results_tables.md`` sections in manifest order, plus notes for skipped ones."""
    sweeps = {f.name: f for f in manifest.figures if f.kind == "sweep"}
    sections, notes = [], []
    for t in manifest.tables:
        try:
            if t.source is not None:
                table = read_table(_need(t.source.path))
            else:
                table = sweep_table(sweep_points(sweeps[t.sweep].points)[0])
        except MissingInput as e:
            notes.append(f"skipped table {t.title!r}: {e}")
            continue
        title = t.title
        if "{n_runs}" in title and "ppo" in table:
            title = title.replace("{n_runs}", str(int(table["ppo"]["n_seeds"])))
        sections.append(table_markdown(title, table, COLUMNS[t.columns], first=t.first or "policy"))
    return sections, notes


def render_manifest(manifest: Manifest, out_dir: Path) -> tuple[list[Path], list[str]]:
    """Write every figure (PNG) and the table view into ``out_dir``. A figure or table whose
    inputs are missing is skipped with a note; anything else wrong is an error."""
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = assign_colors(manifest.figures)
    made: list[Path] = []
    notes: list[str] = []
    for spec in manifest.figures:
        for warning in eval_mismatch([a.to_arm() for a in spec.arms]):
            if f"warning: {warning}" not in notes:  # ad-hoc figures share their arms: say it once
                notes.append(f"warning: {warning}")
        try:
            fig = render_figure(spec, colors)
        except MissingInput as e:
            notes.append(f"skipped {spec.name}: {e}")
            continue
        made.append(_save(fig, out_dir, spec.name))
    sections, table_notes = render_tables(manifest)
    notes += table_notes
    if sections:
        path = out_dir / "results_tables.md"
        path.write_text("\n".join(sections), encoding="utf-8")
        made.append(path)
    return made, notes


# --- Command line ---------------------------------------------------------------------
GOLDEN = Path("results/golden/manifest.toml")
PLOTS = Path("runs/plots")
DEFAULT_KINDS = {"goto_target": ("goto", "diagnostics")}  # any other scenario: keep-away's kinds
KEEPAWAY_KINDS = ("curves", "final", "diagnostics")


def _arm_arg(text: str) -> tuple[str, str]:
    label, sep, pattern = text.partition("=")
    if not sep or not label or not pattern:
        raise argparse.ArgumentTypeError(f"expected LABEL=GLOB, got {text!r}")
    return label, pattern


def _default_kinds(pattern: str) -> tuple[str, ...]:
    """By the scenario in the first matching run's config.json (keep-away if unknown)."""
    first = Arm("", pattern).dirs()[0]
    scenario = ((run_config(first) or {}).get("shared") or {}).get("scenario")
    return DEFAULT_KINDS.get(scenario, KEEPAWAY_KINDS)


def clear_outputs(out: Path) -> None:
    """Remove what a render writes (PNGs + the table view), so dropped figures don't linger."""
    for path in [*out.glob("*.png"), out / "results_tables.md"]:
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render a results record (default: the golden record) or ad-hoc plots of any runs.",
        epilog='ad hoc: plot.py --arm "new reward=runs/new_reward_seed*" [--arm ...] [--kind curves ...]',
    )
    ap.add_argument("--manifest", type=Path, help=f"record manifest to render (default: {GOLDEN})")
    ap.add_argument("--arm", action="append", type=_arm_arg, metavar="LABEL=GLOB",
                    help="an ad-hoc arm: a label and a glob of its run directories (repeatable, up to 9)")
    ap.add_argument("--kind", nargs="+", choices=ARM_KINDS,
                    help="ad-hoc figure kinds (default: curves final diagnostics; goto diagnostics for goto_target)")
    ap.add_argument("--baselines", metavar="LABEL", help="ad hoc: the arm whose table supplies the gray scripted lines")
    ap.add_argument("--name", default="adhoc", help="ad hoc: output folder and figure-name prefix (default: adhoc)")
    ap.add_argument("--title", help="ad hoc: figure title")
    ap.add_argument("--out", type=Path, help="output folder (default: runs/plots/<name>/)")
    ap.add_argument("--overwrite", action="store_true", help="replace the figures in a non-empty output folder")
    args = ap.parse_args(argv)
    if args.arm and args.manifest:
        ap.error("give either --manifest or --arm, not both")
    if not args.arm and (args.kind or args.baselines or args.title):
        ap.error("--kind, --baselines and --title go with --arm (a manifest sets them per figure)")

    if args.arm:
        kinds = args.kind or _default_kinds(args.arm[0][1])
        manifest = ad_hoc_manifest(args.name, args.arm, list(kinds), args.baselines, args.title)
        exempt = False
    else:
        path = args.manifest or GOLDEN
        if not path.is_file():
            print(f"[plot] no manifest at {path}; pass --manifest PATH, or --arm for ad-hoc plots", file=sys.stderr)
            return 1
        manifest = load_manifest(path)
        exempt = args.out is None  # a record's own default folder: pure re-derivation from its data
    out = args.out or PLOTS / manifest.name

    if out.exists() and any(out.iterdir()) and not (exempt or args.overwrite):
        print(f"[plot] {out} already holds files; pass --overwrite to replace them", file=sys.stderr)
        return 1
    if out.exists():
        clear_outputs(out)
    made, notes = render_manifest(manifest, out)
    if not args.arm and (unfrozen := manifest.unfrozen()):
        n = len(unfrozen)
        notes.append(f"warning: {n} input{'s' if n != 1 else ''} not frozen yet; "
                     f"run `python scripts/freeze.py {manifest.root.name}` to copy them into the record")
    for path in made:
        print(f"[plot] {path}")
    for note in notes:
        print(f"[plot] {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
