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
def _mean_band(ax, x: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, sd = mean_std(Y)
    ax.fill_between(x, m - sd, m + sd, color=ACCENT, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(x, m, color=ACCENT, zorder=3)
    return m, sd


def _end_dot(ax, x: float, y: float) -> None:
    ax.plot([x], [y], "o", ms=8, color=ACCENT, mec=SURFACE, mew=2, zorder=5)


def _reference(ax, y: float, text: str, above: bool) -> None:
    ax.axhline(y, color=CONTEXT, linewidth=1.5, zorder=1)
    ax.annotate(
        text,
        xy=(0.01, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4 if above else -4),
        textcoords="offset points",
        ha="left",
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


# --- Table view -----------------------------------------------------------------------
def table_markdown(title: str, table: dict[str, dict[str, float]], columns: Sequence[tuple[str, str]]) -> str:
    def cell(row: dict[str, float], key: str) -> str:
        v, s = row.get(key, float("nan")), row.get(f"{key}_seed_std", float("nan"))
        if not math.isfinite(v):
            return "-"
        return f"{v:.2f} ± {s:.2f}" if math.isfinite(s) else f"{v:.2f}"

    head = "| policy | seeds | " + " | ".join(name for _, name in columns) + " |"
    sep = "|---|---|" + "---|" * len(columns)
    body = [f"| {p} | {int(r['n_seeds'])} | " + " | ".join(cell(r, k) for k, _ in columns) + " |" for p, r in table.items()]
    return "\n".join([f"### {title}", "", head, sep, *body, ""])


# --- Command line ---------------------------------------------------------------------
def _dirs(pattern: str) -> list[str]:
    return sorted(p for p in glob.glob(pattern) if Path(p).is_dir())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Make the report figures from finished runs (step 11).")
    ap.add_argument("--keepaway", default="runs/keepaway_1v1_seed*", help="glob of keep-away run directories")
    ap.add_argument("--goto", default="runs/goto_target_seed*", help="glob of goto_target run directories")
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
            cols = (("len_mean", "overall length"), ("len_mean_possession", "possession starts"),
                    ("len_mean_free_ball", "free-ball starts"), ("frac_truncated", "fraction truncated"))
            tables_md.append(table_markdown(f"Keep-away ({len(keep)} runs; mean episode length, control steps)", table, cols))
        else:
            notes.append(f"skipped keep-away curves/results: run `python scripts/evaluate.py {' '.join(keep)}` first")
    else:
        notes.append(f"no keep-away runs match {args.keepaway!r}")

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
