# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-27
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-27: Generated this file: ad-hoc figures across team sizes for the
#     golden_teams_and_goto_ball_warm_starts record (survival, survival
#     normalized between untrained and flee, free-ball races won, kicks and
#     passes per episode; scratch vs goto warm start), read from the record's
#     frozen data (backlog task 16).
# ==============================================================================
"""Ad-hoc figures that compare team sizes, for a team-size results record.

The manifest's figure kinds can't mix team sizes: each size is a different game, with its own
scripted baselines. These figures put the sizes on one x axis instead, one line per arm
(scratch, goto warm start), one faint dot per seed:

* ``team_size_survival``: mean episode length (overall, possession starts, free-ball starts).
* ``team_size_survival_normalized``: the same, scaled per size so untrained = 0 and flee = 1,
  which makes sizes comparable: ``(ppo - untrained) / (flee - untrained)``.
* ``team_size_races_won``: the share of free-ball evaluation starts that blue wins.
* ``team_size_kicks_passes``: kicks and completed passes per episode.

Survival and races come from the final evaluation tables (``evaluate.py``, the stochastic
``ppo`` rows, the fixed evaluation episodes). The eval tables don't count kicks, so kicks and
passes come from ``train.csv``: ``kicks_per_step * ep_len_mean`` and ``ep_passes_mean``,
averaged over the last 10% of training iterations (training episodes, stochastic policy).

Reads only the record's frozen ``data/``, so it re-draws from any checkout. Run from the repo
root with the venv active:

    python scripts/plot_team_size.py                    # the default record below
    python scripts/plot_team_size.py results/<record>   # -> results/<record>/figures_adhoc/

It writes to ``figures_adhoc/``, not ``figures/``: freezing clears and re-renders ``figures/``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from marl.manifest import load_manifest  # noqa: E402  pylint: disable=wrong-import-position
from marl.plots import CONTEXT, INK, INK_2, apply_style, assign_colors  # noqa: E402  pylint: disable=wrong-import-position
from marl.results import read_csv  # noqa: E402  pylint: disable=wrong-import-position

RECORD = Path("results/golden_teams_and_goto_ball_warm_starts")
SIZES = ("1v1", "2v1", "2v2", "3v2", "3v3")
ARMS = (("scratch", "keepaway_kick_{}"), ("goto warm start", "keepaway_kick_warm_{}"))  # label, run stem
PANELS = (("len_mean", "All evaluation starts"), ("len_mean_possession", "Possession starts"),
          ("len_mean_free_ball", "Free-ball starts"))
LAST = 0.10  # kicks/passes: the last 10% of training iterations


def eval_rows(data: Path, stem: str) -> list[dict[str, str]]:
    """The per-seed rows evaluate.py wrote for one arm (PPO seeds and the baselines)."""
    return read_csv(data / f"eval_{stem}_pybullet_seed0_per_run.csv")


def column(rows, policy: str, key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r["policy"] == policy])


def train_tail(data: Path, stem: str, key) -> np.ndarray:
    """Per seed: the mean of key(row) over the last LAST of train.csv's iterations."""
    out = []
    for run in sorted(data.glob(f"{stem}_pybullet_seed*")):
        rows = read_csv(run / "train.csv")
        tail = rows[int(len(rows) * (1 - LAST)):]
        out.append(float(np.mean([key(r) for r in tail])))

    return np.array(out)


def gather(data: Path) -> dict:
    """label -> metric -> list over sizes of per-seed arrays."""
    out = {label: {} for label, _ in ARMS}
    for label, pattern in ARMS:
        m = out[label]
        for size in SIZES:
            stem = pattern.format(size)
            rows = eval_rows(data, stem)
            for key, _ in PANELS:
                ppo = column(rows, "ppo", key)
                untrained = column(rows, "untrained", key).mean()
                flee = column(rows, "flee", key).mean()
                m.setdefault(key, []).append(ppo)
                m.setdefault(key + "_norm", []).append((ppo - untrained) / (flee - untrained))

            m.setdefault("pickup_rate", []).append(column(rows, "ppo", "pickup_rate") * 100)
            m.setdefault("kicks", []).append(train_tail(
                data, stem, lambda r: float(r["kicks_per_step"]) * float(r["ep_len_mean"])))
            m.setdefault("passes", []).append(train_tail(data, stem, lambda r: float(r["ep_passes_mean"])))

    return out


def draw(ax, series: dict, colors: dict, key: str, fmt: str, label_at=(0, len(SIZES) - 1)) -> None:
    """One line per arm through the seed means, faint dots for the seeds. Value labels only at
    the sizes in label_at (selective, not every point), beside the point (left of the first size,
    right of the last), nudged apart vertically: the higher arm's label up, the lower one's down."""
    x = np.arange(len(SIZES))
    means = {label: np.array([v.mean() for v in series[label][key]]) for label, _ in ARMS}
    for i, (label, _) in enumerate(ARMS):
        for j, v in enumerate(series[label][key]):
            ax.plot(np.full(len(v), x[j] + (i - 0.5) * 0.08), v, "o", ms=4, color=colors[label],
                    alpha=0.3, mew=0, zorder=2, clip_on=False)

        ax.plot(x, means[label], color=colors[label], lw=2, marker="o", ms=8, mec="white", mew=2, zorder=3,
                clip_on=False, label=label)

    for j in label_at:
        side = -1 if j == 0 else 1
        for label, _ in ARMS:
            others = [means[o][j] for o, _ in ARMS if o != label]
            dy = 6 if all(means[label][j] >= o for o in others) else -6
            ax.annotate(fmt.format(means[label][j]), xy=(x[j], means[label][j]), xytext=(side * 10, dy),
                        textcoords="offset points", ha="right" if side < 0 else "left", va="center",
                        fontsize=9, fontweight="semibold", color=INK)

    ax.set_xticks(x, SIZES)
    ax.set_xlim(-0.75, len(x) - 0.25)
    ax.set_xlabel("team size (blue v red)")


def finish(fig, axes, title: str, note: str, out: Path, name: str, left: float = 0.06) -> Path:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=12, fontweight="semibold", color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, note + "\nLines: the mean of 5 seeds; faint dots: each seed.", fontsize=9,
             color=INK_2, ha="left", va="top")
    fig.subplots_adjust(top=0.76, bottom=0.2, left=left, right=0.99, wspace=0.18)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)
    return path


def survival(s, colors, out, normalized: bool) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (key, panel) in zip(axes, PANELS):
        draw(ax, s, colors, key + ("_norm" if normalized else ""), "{:.2f}" if normalized else "{:.0f}")
        ax.set_title(panel, loc="left", fontsize=10.5, color=INK)
        if normalized:  # the references, labeled mid-panel (clear of the end labels): flee above, untrained below
            for y, text, dy in ((0, "untrained", -4), (1, "flee (scripted)", 4)):
                ax.axhline(y, color=CONTEXT, lw=1.2, zorder=1)
                ax.annotate(text, xy=(0.5, y), xycoords=("axes fraction", "data"), xytext=(0, dy),
                            textcoords="offset points", ha="center", va="top" if dy < 0 else "bottom",
                            fontsize=8.5, color=CONTEXT)

    if normalized:
        axes[0].set_ylim(bottom=-0.25)
        axes[0].set_ylabel("(PPO - untrained) / (flee - untrained)")
        return finish(fig, axes, "Keep-away with kicking: survival relative to the scripted references, by team size",
                      "Per team size and start type, 0 = an untrained network and 1 = the scripted flee policy "
                      "on the same fixed evaluation episodes. Final evaluation, stochastic policy.",
                      out, "team_size_survival_normalized")

    axes[0].set_ylabel("mean episode length (control steps)")
    return finish(fig, axes, "Keep-away with kicking: survival by team size, from scratch vs goto_target warm start",
                  "Final evaluation on the fixed episodes, stochastic policy. Each size is a different game "
                  "(its own red count), so the levels across sizes are not a like-for-like comparison.",
                  out, "team_size_survival")


def races(s, colors, out) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    draw(ax, s, colors, "pickup_rate", "{:.0f}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("free-ball evaluation starts won (%)")
    return finish(fig, [ax], "Free-ball races won, by team size",
                  "Blue reaches the loose ball first. Final evaluation, stochastic policy.",
                  out, "team_size_races_won", left=0.11)


def kicks_passes(s, colors, out) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    draw(axes[0], s, colors, "kicks", "{:.1f}")
    draw(axes[1], s, colors, "passes", "{:.2f}", label_at=(len(SIZES) - 1,))
    for ax, panel in zip(axes, ("Kicks per episode", "Completed passes per episode (1v1: no teammate, so 0)")):
        ax.set_title(panel, loc="left", fontsize=10.5, color=INK)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("per episode (whole blue team)")
    return finish(fig, axes, "Kicks and completed passes per episode, by team size",
                  f"Training episodes, the last {LAST:.0%} of training (eval tables don't count kicks): "
                  "kicks = kicks_per_step x ep_len_mean, passes = ep_passes_mean.",
                  out, "team_size_kicks_passes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ad-hoc team-size figures for a team-size results record.")
    ap.add_argument("record", nargs="?", type=Path, default=RECORD, help=f"record folder (default: {RECORD})")
    args = ap.parse_args(argv)
    data, out = args.record / "data" / "task16", args.record / "figures_adhoc"
    if not data.is_dir():
        print(f"[plot_team_size] no frozen data at {data}", file=sys.stderr)
        return 1

    apply_style()
    colors = assign_colors(load_manifest(args.record / "manifest.toml").figures)  # same colors as the record
    out.mkdir(exist_ok=True)
    s = gather(data)
    for path in (survival(s, colors, out, False), survival(s, colors, out, True), races(s, colors, out),
                 kicks_passes(s, colors, out)):
        print(f"[plot_team_size] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
