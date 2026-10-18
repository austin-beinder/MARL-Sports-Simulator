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
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs).
# Human review:   Pending review
# ==============================================================================
"""Report figures (step 11). Reads only CSVs that training, ``scripts/evaluate.py``
and ``scripts/calibrate.py`` already wrote, so making figures never re-runs anything.

Visual rules (data-viz method, light mode):

* **Emphasis form**: the learned policy (PPO) is the one accent-colored series;
  scripted baselines are gray context lines, each direct-labeled with its value.
* The shaded band is +/- 1 standard deviation across training seeds (ddof=1, as
  in the evaluation table); the line is the mean.
* Thin solid lines, hairline solid gridlines, text in ink tokens (never series color).
* Every figure has a table view twin: ``results_tables.md`` (plus the source CSVs).
* The PBRS comparison has TWO learned arms: survival-only keeps the accent (slot 1,
  as in every other figure); + PBRS takes categorical slot 2, validated against it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from marl.common.config import load_config  # noqa: E402

# Light-mode tokens from the data-viz reference palette.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"  # categorical slot 1; validator: band, chroma and >= 3:1 on SURFACE all PASS
# Categorical slot 2. Validator, pair with ACCENT on SURFACE (light): band, chroma PASS;
# CVD dE 24.7 (target >= 8); normal-vision dE 33.6 (floor 15); contrast >= 3:1 PASS.
SERIES_2 = "#eb6834"
CONTEXT = MUTED  # de-emphasis gray for reference/context marks (always direct-labeled)
BAND_ALPHA = 0.12

EVAL_METRICS = ("len_mean", "len_mean_possession", "len_mean_free_ball", "mean_reward")
TRAIN_METRICS = ("entropy", "explained_variance", "approx_kl", "clip_frac")
TRIVIAL = ("random", "still", "untrained")  # baselines that never learn to evade


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


# --- Data -----------------------------------------------------------------------------
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


# --- Marks ----------------------------------------------------------------------------
def _mean_band(ax, x: np.ndarray, Y: np.ndarray, color: str = ACCENT) -> tuple[np.ndarray, np.ndarray]:
    m, sd = mean_std(Y)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, m, color=color, zorder=3)
    return m, sd


def _end_dot(ax, x: float, y: float, color: str = ACCENT) -> None:
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


def _legend(fig, n_seeds: int, context_label: str) -> None:
    handles = [
        (Patch(facecolor=ACCENT, alpha=BAND_ALPHA, linewidth=0), Line2D([], [], color=ACCENT, linewidth=2)),
        Line2D([], [], color=CONTEXT, linewidth=1.5),
    ]
    labels = [f"PPO: mean, shaded ± 1 std over {n_seeds} seeds", context_label]
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    return path


# --- Figures --------------------------------------------------------------------------
def fig_keepaway_curves(curves: dict[str, np.ndarray], table: dict[str, dict[str, float]], out_dir: Path) -> Path:
    x = curves["env_steps"] / 1e6
    n = len(curves["len_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    axes[0].set_ylim(50, 175)
    for ax, key, title in zip(axes, ("len_mean_possession", "len_mean_free_ball"), ("Possession starts", "Free-ball starts")):
        m, sd = _mean_band(ax, x, curves[key])
        _end_dot(ax, x[-1], m[-1])
        flee = table["flee"][key]
        trivial = float(np.mean([table[p][key] for p in TRIVIAL]))
        _reference(ax, flee, f"flee (scripted): {flee:.0f}", above=True)
        _reference(ax, trivial, f"random / still / untrained: {trivial:.0f}", above=False)
        _end_label(ax, x[-1], m[-1], f"PPO {m[-1]:.0f} ± {sd[-1]:.0f}", [flee, trivial], m)
        ax.set_title(title)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, x[-1] * 1.02)
    axes[0].set_ylabel("Mean episode length (control steps)")
    fig.suptitle("Keep-away 1v1: survival on the fixed evaluation episodes during training", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="semibold")
    _legend(fig, n, "scripted baselines (values labeled)")
    return _save(fig, out_dir, "keepaway_learning_curves")


FINAL_ORDER = (
    ("ppo", "PPO (stochastic)"),
    ("ppo_deterministic", "PPO (deterministic)"),
    ("flee", "flee (scripted)"),
    ("random", "random"),
    ("still", "stand still"),
    ("untrained", "untrained PPO"),
)


def fig_keepaway_final(table: dict[str, dict[str, float]], out_dir: Path) -> Path:
    rows = [(p, name) for p, name in FINAL_ORDER if p in table]
    y = np.arange(len(rows))[::-1]
    n_seeds = int(table["ppo"]["n_seeds"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ("len_mean_possession", "len_mean_free_ball"), ("Possession starts", "Free-ball starts")):
        vals = np.array([table[p][key] for p, _ in rows])
        stds = np.array([table[p][f"{key}_seed_std"] for p, _ in rows])
        colors = [ACCENT if p.startswith("ppo") else CONTEXT for p, _ in rows]
        ax.barh(y, vals, height=0.5, color=colors, zorder=2)
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
        ax.set_yticks(y, [name for _, name in rows])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(np.nanmax(vals + np.nan_to_num(stds))) * 1.32)
        ax.set_xlabel("Mean episode length (control steps)")
        ax.set_title(title)
    fig.suptitle("Keep-away 1v1: final evaluation (same fixed episodes for every policy)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="semibold")
    fig.legend(
        [Patch(color=ACCENT), Patch(color=CONTEXT)],
        [f"PPO (learned); error bars ± 1 std over {n_seeds} seeds", "baselines"],
        loc="outside lower center",
        ncols=2,
        fontsize=9,
    )
    return _save(fig, out_dir, "keepaway_final_results")


def fig_calibration(rows: Sequence[dict[str, str]], chosen: float, carrier: float, out_dir: Path) -> Path:
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
    ax.plot(speeds, flee, color=ACCENT, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
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
    ax.set_title("Calibrating red's speed against the scripted flee policy")
    ax.legend(
        [Line2D([], [], color=ACCENT, marker="o", ms=7, mec=SURFACE, mew=1.5), Line2D([], [], color=CONTEXT, linewidth=1.5),
         Patch(color=GRID, alpha=0.6)],
        ["flee (scripted)", "random / still (overlap)", "target band"],
        loc="lower left",
        fontsize=9,
    )
    return _save(fig, out_dir, "calibration_red_speed")


def fig_goto(curves: dict[str, np.ndarray], table: dict[str, dict[str, float]], out_dir: Path) -> Path:
    x = curves["env_steps"] / 1e3
    n = len(curves["mean_reward"])
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
    oracle = -table["oracle"]["mean_reward"]
    trivial = -float(np.mean([table[p]["mean_reward"] for p in ("random", "still")]))
    ax.set_ylim(0, trivial * 1.18)
    m, sd = _mean_band(ax, x, -curves["mean_reward"])  # reward = -distance (m)
    _end_dot(ax, x[-1], m[-1])
    _reference(ax, oracle, f"oracle (straight to target): {oracle:.2f} m", above=True)
    _reference(ax, trivial, f"random / still: {trivial:.1f} m", above=True)
    _end_label(ax, x[-1], m[-1], f"PPO {m[-1]:.2f} ± {sd[-1]:.2f} m", [oracle, trivial], m)
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean distance to target (m), lower is better")
    ax.set_title("goto_target sanity check")
    _legend(fig, n, "references (values labeled)")
    return _save(fig, out_dir, "goto_target_curve")


def fig_diagnostics(curves: dict[str, np.ndarray], out_dir: Path, window: int = 9) -> Path:
    x = curves["env_steps"] / 1e6
    n = len(curves["entropy"])
    panels = (
        ("entropy", "Policy entropy (nats; max 0 = uniform)"),
        ("explained_variance", "Critic explained variance"),
        ("approx_kl", "Approx. KL(old || new) per update"),
        ("clip_frac", "Fraction of ratios clipped"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, (key, title) in zip(axes.flat, panels):
        _mean_band(ax, x, smooth(curves[key], window))
        ax.set_title(title, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Environment steps (millions)")
    fig.suptitle(f"Keep-away PPO training diagnostics (mean ± 1 std over {n} seeds; {window}-iteration moving average)",
                 x=0.01, ha="left", color=INK, fontsize=11, fontweight="semibold")
    return _save(fig, out_dir, "keepaway_training_diagnostics")


# --- PBRS ablation (acquisition shaping) -------------------------------------------------
def fig_pbrs_curves(ctl: dict[str, np.ndarray], pbrs: dict[str, np.ndarray], table: dict[str, dict[str, float]],
                    coef: float, out_dir: Path) -> Path:
    """Survival-only vs + PBRS: evaluation curves per start type (same training seeds,
    same fixed evaluation episodes); scripted baselines as labeled gray context."""
    arms = ((ctl, ACCENT, "survival only"), (pbrs, SERIES_2, f"+ PBRS (coef {coef:g})"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    axes[0].set_ylim(50, 175)
    for ax, key, title in zip(axes, ("len_mean_possession", "len_mean_free_ball"), ("Possession starts", "Free-ball starts")):
        flee = table["flee"][key]
        trivial = float(np.mean([table[p][key] for p in TRIVIAL]))
        _reference(ax, flee, f"flee (scripted): {flee:.0f}", above=True)
        _reference(ax, trivial, f"random / still / untrained: {trivial:.0f}", above=False)
        ends = []
        for curves, color, name in arms:
            x = curves["env_steps"] / 1e6
            m, sd = _mean_band(ax, x, curves[key], color)
            _end_dot(ax, x[-1], m[-1], color)
            ends.append((x[-1], m[-1], f"{name} {m[-1]:.0f} ± {sd[-1]:.0f}"))
        _end_labels(ax, ends, [flee, trivial])
        ax.set_title(title)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_xlim(0, max(e[0] for e in ends) * 1.02)
    axes[0].set_ylabel("Mean episode length (control steps)")
    fig.suptitle(f"Keep-away 1v1: survival-only vs + PBRS acquisition shaping (coef {coef:g}), fixed evaluation episodes",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    handles = [(Patch(facecolor=c, alpha=BAND_ALPHA, linewidth=0), Line2D([], [], color=c, linewidth=2)) for _, c, _ in arms]
    labels = [f"{name}: mean, shaded ± 1 std over {len(curves['len_mean'])} seeds" for curves, _, name in arms]
    fig.legend([*handles, Line2D([], [], color=CONTEXT, linewidth=1.5)], [*labels, "scripted baselines (values labeled)"],
               loc="outside lower center", ncols=3, fontsize=9)
    return _save(fig, out_dir, "pbrs_learning_curves")


SWEEP_METRICS = ("approach_m_per_step", "approach_alignment", "pickup_rate", "len_mean_possession",
                 "len_mean_free_ball", "env_steps")
SWEEP_PANELS = (  # key, title, y label, scale, value format, y-limits
    ("approach_m_per_step", "Closing speed on the loose ball", "Meters closed per control step", 1.0, "{:.3f}", (0, 0.11)),
    ("pickup_rate", "Races won", "% of free-ball evaluation starts", 100.0, "{:.0f}%", (0, 105)),
    ("len_mean_possession", "Survival in possession starts", "Mean episode length (control steps)", 1.0, "{:.0f}", (50, 175)),
)


def sweep_rows(paths: Sequence[str | Path]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """The PPO (stochastic) rows of the pilot's per-run evaluation CSVs, one per (coef, seed),
    plus the flee baseline (the same in every file: same fixed episodes)."""
    rows: list[dict[str, float]] = []
    flee: dict[str, float] | None = None
    for p in paths:
        for r in read_csv(p):
            values = {k: float(v) for k, v in r.items() if k not in ("policy", "run")}
            if r["policy"] == "ppo":
                rows.append(values)
            elif r["policy"] == "flee" and flee is None:
                flee = values
    if not rows or flee is None:
        raise ValueError("the sweep files hold no PPO or flee rows (were they written by scripts/evaluate.py?)")
    return rows, flee


def sweep_table(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and seed std per coefficient, in the shape ``table_markdown`` takes."""
    table = {}
    for c in sorted({r["pbrs_coef"] for r in rows}):
        sel = [r for r in rows if r["pbrs_coef"] == c]
        out: dict[str, float] = {"n_seeds": len(sel)}
        for k in SWEEP_METRICS:
            vals = np.array([r[k] for r in sel])
            out[k] = float(vals.mean())
            out[f"{k}_seed_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        table["0 (survival only)" if c == 0 else f"{c:g}"] = out
    return table


def fig_pbrs_sweep(rows: Sequence[dict[str, float]], flee: dict[str, float], chosen: float | None, out_dir: Path) -> Path:
    """The coefficient pilot: one point per coefficient (ordinal x: the coefficients double),
    seeds as faint dots behind the mean; flee as the gray "runs straight at it" reference."""
    coefs = sorted({r["pbrs_coef"] for r in rows})
    pos = {c: i for i, c in enumerate(coefs)}
    seeds = {c: sum(r["pbrs_coef"] == c for r in rows) for c in coefs}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    for ax, (key, title, ylabel, scale, fmt, ylim) in zip(axes, SWEEP_PANELS):
        ax.set_ylim(*ylim)
        near = 0.08 * (ylim[1] - ylim[0])
        ref = flee[key] * scale
        _reference(ax, ref, f"flee (scripted): {fmt.format(ref)}", above=False, right=True)
        means = []
        for c in coefs:
            vals = np.array([r[key] for r in rows if r["pbrs_coef"] == c]) * scale
            # clip_on=False: a 0% point sits ON the baseline and must show whole
            ax.plot(np.full(len(vals), pos[c]), vals, "o", ms=5, color=ACCENT, alpha=0.35, mew=0, zorder=2, clip_on=False)
            means.append(float(vals.mean()))
            if c in (coefs[0], chosen):  # label selectively: no shaping vs the chosen weight
                # Clear the seed dots: above the cluster, or below it if a reference line is close above.
                above = not 0 <= ref - vals.max() < near
                ax.annotate(fmt.format(means[-1]), xy=(pos[c], vals.max() if above else vals.min()),
                            xytext=(0, 7 if above else -7), textcoords="offset points", ha="center",
                            va="bottom" if above else "top", fontsize=9.5, fontweight="semibold", color=INK)
        ax.plot(range(len(coefs)), means, color=ACCENT, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3, clip_on=False)
        ax.set_xticks(range(len(coefs)), [("0\n(none)" if c == 0 else f"{c:g}") + ("\n(chosen)" if c == chosen else "")
                                          for c in coefs])
        if chosen in pos:  # mark the chosen weight on its tick label, in ink (no extra line through the data)
            tick = ax.get_xticklabels()[pos[chosen]]
            tick.set_color(INK)
            tick.set_fontweight("semibold")
        ax.set_xlim(-0.5, len(coefs) - 0.5)
        ax.set_xlabel("PBRS coefficient")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    fig.suptitle("PBRS coefficient pilot (~200k steps each): stronger shaping speeds the approach but doesn't win races",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
    counts = list(seeds.values())
    common = max(set(counts), key=counts.count)
    others = [f"{n} at coef {c:g}" for c, n in seeds.items() if n != common]
    n_text = f"{common} per coef" + (f"; {', '.join(others)}" if others else "")
    handles = [
        Line2D([], [], color=ACCENT, marker="o", ms=7, mec=SURFACE, mew=1.5),
        Line2D([], [], linestyle="", marker="o", ms=5, color=ACCENT, alpha=0.35, mew=0),
        Line2D([], [], color=CONTEXT, linewidth=1.5),
    ]
    labels = [f"PPO, stochastic eval: mean over training seeds ({n_text})", "individual training seeds",
              "flee (scripted): runs straight at a loose ball"]
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    return _save(fig, out_dir, "pbrs_coef_sweep")


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


# --- Command line ---------------------------------------------------------------------
def _dirs(pattern: str) -> list[str]:
    return sorted(p for p in glob.glob(pattern) if Path(p).is_dir())


def _arm_coef(table_path: Path) -> float:
    """The PBRS coefficient of the runs behind an evaluation table (its per-run CSV records it)."""
    rows = read_csv(table_path.with_name(table_path.stem + "_per_run.csv"))
    return float(next(r["pbrs_coef"] for r in rows if r["policy"] == "ppo"))


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
SWEEP_DEFAULT = ("runs/eval_keepaway_1v1_seed0_iter_00100_per_run.csv", "runs/eval_pilot_pbrs_coef*_seed10_per_run.csv")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Make the report figures from finished runs (step 11).")
    ap.add_argument("--keepaway", default="runs/keepaway_1v1_seed*", help="glob of keep-away run directories")
    ap.add_argument("--goto", default="runs/goto_target_seed*", help="glob of goto_target run directories")
    ap.add_argument("--pbrs", default="runs/keepaway_1v1_pbrs_seed*", help="glob of the PBRS-ablation run directories")
    ap.add_argument("--sweep", nargs="+", default=list(SWEEP_DEFAULT),
                    help="per-run evaluation CSVs of the PBRS coefficient pilot (globs allowed)")
    ap.add_argument("--calibration", type=Path, default=Path("runs/calibration_keepaway_1v1.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/keepaway_1v1.toml"),
                    help="keep-away config (for the chosen red speed and the carrier speed)")
    ap.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = ap.parse_args(argv)

    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    notes: list[str] = []
    tables_md: list[str] = []

    keep = _dirs(args.keepaway)
    if keep:
        made.append(fig_diagnostics(train_curves(keep), args.out))
        tpath = table_path_for(keep)
        if tpath.exists():
            table = read_table(tpath)
            made.append(fig_keepaway_curves(eval_curves(keep), table, args.out))
            made.append(fig_keepaway_final(table, args.out))
            tables_md.append(table_markdown(f"Keep-away ({len(keep)} runs; mean episode length, control steps)", table,
                                            KEEPAWAY_COLUMNS))
        else:
            notes.append(f"skipped keep-away curves/results: run `python scripts/evaluate.py {' '.join(keep)}` first")
    else:
        notes.append(f"no keep-away runs match {args.keepaway!r}")

    pbrs, chosen = _dirs(args.pbrs), None
    if pbrs and keep and table_path_for(pbrs).exists() and table_path_for(keep).exists():
        ppath = table_path_for(pbrs)
        chosen = _arm_coef(ppath)
        made.append(fig_pbrs_curves(eval_curves(keep), eval_curves(pbrs), read_table(table_path_for(keep)), chosen, args.out))
        tables_md.append(table_markdown(
            f"Keep-away + PBRS acquisition shaping, coef {chosen:g} ({len(pbrs)} runs; mean episode length, control steps)",
            read_table(ppath), KEEPAWAY_COLUMNS))
    elif pbrs:
        notes.append(f"skipped the PBRS comparison: run `python scripts/evaluate.py {' '.join(pbrs)}` "
                     "(and on the keep-away runs) first")

    sweep = sorted({p for pattern in args.sweep for p in glob.glob(pattern)})
    if sweep:
        rows, flee = sweep_rows(sweep)
        made.append(fig_pbrs_sweep(rows, flee, chosen, args.out))
        tables_md.append(table_markdown("PBRS coefficient pilot (PPO, stochastic eval; lengths in control steps)",
                                        sweep_table(rows), SWEEP_COLUMNS, first="PBRS coef"))

    goto = _dirs(args.goto)
    gpath = table_path_for(goto) if goto else None
    if goto and gpath.exists():
        gtable = read_table(gpath)
        made.append(fig_goto(eval_curves(goto), gtable, args.out))
        tables_md.append(table_markdown(f"goto_target ({len(goto)} runs; mean reward = minus mean distance, m)", gtable,
                                        (("mean_reward", "mean reward (-m)"),)))
    elif goto:
        notes.append(f"skipped goto_target curve: run `python scripts/evaluate.py {' '.join(goto)}` first")

    if args.calibration.exists():
        cfg = load_config(args.config).shared
        made.append(fig_calibration(read_csv(args.calibration), cfg.red_max_speed, cfg.max_speed * cfg.carry_speed_factor, args.out))
    else:
        notes.append(f"skipped calibration curve: {args.calibration} not found (run scripts/calibrate.py)")

    if tables_md:
        (args.out / "results_tables.md").write_text("\n".join(tables_md), encoding="utf-8")
    for path in made:
        print(f"[plot] {path}")
    for note in notes:
        print(f"[plot] {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
