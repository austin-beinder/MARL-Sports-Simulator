# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-07
# Author:  Austin Beinder
# Usage:
#   - 2026-11-07: Generated this entire file: plots episode length over
#     training from the text a training or curriculum run prints.
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters and
#     removed the unused math import; no behaviour change. (Austin Beinder)
# ==============================================================================
# pylint: disable=line-too-long
# (the docstring quotes a printed training line verbatim)
"""
Plot episode length over training from the lines a training run prints (copied from the
terminal), so you don't need the run's CSV files.

Save the terminal text to a file and give it the file:

    python scripts/plot_episode_length.py my_log.txt
    python scripts/plot_episode_length.py my_log.txt --threshold 0.5 --out free_ball.png

or paste it straight in (end with Ctrl+Z then Enter on Windows, Ctrl+D elsewhere):

    python scripts/plot_episode_length.py -

It reads two kinds of lines and ignores everything else:

    it  140/732 | steps   286,720 | ep_len  135.8 (n= 15, trunc  13%) | ... | gate 0.44 (low 0.36) lvl 0 = 1
      eval@ 140 [stoch] len  113.2 +/-  55.5 (trunc   6%) | ...

What is drawn:
- training: the mean length of the episodes that ended in that printed iteration. They follow the
  curriculum's current level, not the real task. Only every few iterations get printed, so this
  is a sample of the run, and it is noisy.
- evaluation: the stochastic policy on the fixed evaluation episodes, which always measure the
  stage's real task. The band is +/- one standard deviation across those episodes.
- --threshold (optional): the episode length the stage's gate asks for, threshold x the time
  limit. The gate itself checks the mean minus 2 standard errors, so it passes a bit above it.
- Level changes (the "lvl" at the end of curriculum lines, e.g. max_ball_dist) and stage
  changes ("[curriculum] stage" lines) are marked with vertical lines.

The figure goes next to the log file (my_log.png), or to episode_length.png when reading pasted
text, unless --out says otherwise.
"""
# pylint: enable=line-too-long

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file; no window needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# The validated default palette's first two categorical slots, and its light-mode ink and surface
TRAIN_COLOR = "#2a78d6"  # blue
EVAL_COLOR = "#eb6834"  # orange
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

TRAIN_LINE = re.compile(
    r"^\s*it\s+(\d+)/(\d+)\s*\|\s*steps\s+([\d,]+)\s*\|\s*ep_len\s+([\d.]+|nan)")
LEVEL = re.compile(r"\blvl\s+(\d+)\s*=\s*([\d.]+)")
EVAL_LINE = re.compile(r"^\s*eval@\s*(\d+)\s*\[stoch\]\s*len\s+([\d.]+|nan)\s*\+/-\s*([\d.]+|nan)")
STAGE_LINE = re.compile(r"^\[curriculum\] stage (\d+)/\d+ (\S+):")


class Log:
    """What the printed lines say, on one step axis (later stages continue where earlier ones
    ended)."""

    def __init__(self):
        self.train_steps, self.train_len = [], []
        self.eval_steps, self.eval_len, self.eval_std = [], [], []
        self.level_changes = []  # (step, value in metres; 0 = anywhere)
        self.stage_starts = []  # (step, stage name)
        self.time_limit = None

    def read(self, lines):
        offset = 0  # steps before the current stage
        last_steps = 0
        steps_per_iteration = None
        level = None
        # eval lines can come before the stage's first "it" line says how long an iteration is
        pending_evals = []

        for line in lines:
            if m := STAGE_LINE.match(line):
                offset += last_steps
                last_steps, level, steps_per_iteration = 0, None, None
                self.stage_starts.append((offset, m.group(2)))
                continue

            if m := TRAIN_LINE.match(line):
                iteration, steps = int(m.group(1)), int(m.group(3).replace(",", ""))
                # a new run without a stage line: keep going after the last one
                if steps < last_steps:
                    offset += last_steps
                    level = None
                steps_per_iteration = steps / iteration
                last_steps = steps
                self.train_steps.append(offset + steps)
                self.train_len.append(float(m.group(4)))
                if lv := LEVEL.search(line):
                    value = float(lv.group(2))
                    if value != level:
                        self.level_changes.append((offset + steps, value))
                        level = value
                for it, length, std in pending_evals:
                    self._add_eval(offset + it * steps_per_iteration, length, std)
                pending_evals = []
                continue

            if m := EVAL_LINE.match(line):
                iteration, length, std = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if steps_per_iteration is None:
                    pending_evals.append((iteration, length, std))
                else:
                    self._add_eval(offset + iteration * steps_per_iteration, length, std)

        if not self.train_steps:
            raise SystemExit("found no training lines (\"it 12/488 | steps ... | ep_len ...\") "
                             "in the input")

    def _add_eval(self, step, length, std):
        self.eval_steps.append(step)
        self.eval_len.append(length)
        self.eval_std.append(std)


def plot(log: Log, title: str, time_limit: int, threshold: float | None, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    top = time_limit * 1.1
    x_max = max(log.train_steps + log.eval_steps)

    # Evaluation: line + markers, with a +/- 1 standard deviation band
    if log.eval_steps:
        lo = [m - s for m, s in zip(log.eval_len, log.eval_std)]
        hi = [m + s for m, s in zip(log.eval_len, log.eval_std)]
        ax.fill_between(log.eval_steps, lo, hi, color=EVAL_COLOR, alpha=0.12, linewidth=0)
        ax.plot(log.eval_steps, log.eval_len, color=EVAL_COLOR, linewidth=2.0, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="evaluation, real task (mean ± 1 sd)")

    # Training: the printed iterations
    ax.plot(log.train_steps, log.train_len, color=TRAIN_COLOR, linewidth=2.0, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="training episodes (printed iterations)")

    if threshold is not None:
        gate = threshold * time_limit
        ax.axhline(gate, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(x_max, gate, f"  gate {gate:.0f} ({threshold:g})", color=INK_SECONDARY,
                fontsize=8, va="center")

    for step, name in log.stage_starts:
        if step > 0:
            ax.axvline(step, color=INK_SECONDARY, linewidth=1.0)
        ax.text(step, top, f" {name}", color=INK, fontsize=9, va="top")

    for step, value in log.level_changes:
        ax.axvline(step, color=GRID, linewidth=1.0, linestyle=":")
        label = "ball anywhere" if value == 0 else f"ball ≤ {value:g} m"
        ax.text(step, time_limit * 0.98, f" {label}", color=INK_SECONDARY, fontsize=7, va="top",
                rotation=90)

    ax.set_ylim(0, top)
    ax.set_xlim(0, x_max * 1.02)
    ax.set_xlabel("environment steps", color=INK_SECONDARY)
    ax.set_ylabel(f"episode length (control steps; {time_limit} = time limit)",
                  color=INK_SECONDARY)
    ax.set_title(f"Episode length over training{': ' + title if title else ''}", color=INK,
                 loc="left", fontsize=12)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot episode length from a training run's printed output.")
    ap.add_argument("log", help="a text file with the copied terminal output, or - to paste it in")
    ap.add_argument("--threshold", type=float,
                    help="draw the stage's gate (e.g. 0.5 -> 150 of 300 steps)")
    ap.add_argument("--time-limit", type=int, default=300, help="max_episode_steps (default 300)")
    ap.add_argument("--title", default=None,
                    help="text after 'Episode length over training:' (default: the file name)")
    ap.add_argument("--out", type=Path,
                    help="default: next to the log file, or episode_length.png for pasted text")
    args = ap.parse_args(argv)

    if args.log == "-":
        lines, title, out = sys.stdin.read().splitlines(), "", Path("episode_length.png")
    else:
        path = Path(args.log)
        lines, title, out = (path.read_text(encoding="utf-8").splitlines(), path.stem,
                             path.with_suffix(".png"))
    if args.title is not None:
        title = args.title

    log = Log()
    log.read(lines)
    plot(log, title, args.time_limit, args.threshold, args.out or out)
    print(f"read {len(log.train_steps)} training and {len(log.eval_steps)} evaluation lines; "
          f"wrote {args.out or out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
