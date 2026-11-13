<!--
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-13
Author:  Austin Beinder
Usage:
  2026-11-13: Generated this entire file: how to plot a curriculum run's
    progress with scripts/plot.py --curriculum.
Prompt summary: Put the curriculum progress plots in scripts/plot.py as options,
  with a short readme of the commands.
Human review:   Pending review
-->

# Plotting

Run from the repo root with the venv active. Figures go to `runs/figures/`
(change it with `--out`), as PNG and PDF.

## A curriculum run's progress

```bash
python scripts/plot.py --curriculum runs/curriculum_passing_3v1_pybullet_seed0
```

It makes two figures, named after the run folder:

- `<run>_progress`: every stage end to end on one step axis.
  - Episode length: training, and eval on the stage's real task.
  - Each stage's gate score against its threshold, with level changes.
  - Reward per step.
- `<run>_catch`: only if the run has `pass_practice_moving` stages, such as
  `catch_moving`.
  - Passes per episode.
  - Share of kicks caught, with the gate.
  - Shaping reward and kicks per step.

A stage that hasn't finished is drawn to its step cap, with a dotted line where
it is now, so you can plot a run while it trains. A stage that `--start-at`
left from an earlier attempt is left out.

Give several run folders to plot each one:

```bash
python scripts/plot.py --curriculum runs/curriculum_passing_3v1_pybullet_seed0 runs/curriculum_keepaway_1v1_pybullet_seed0
```

## The report figures

Without `--curriculum`, `plot.py` makes the report figures from the 5-seed
keep-away, goto_target and PBRS runs. Run `scripts/evaluate.py` on those runs
first. `python scripts/plot.py --help` lists every option.

## From a terminal log

`plot_episode_length.py` plots episode length from the text a training or
curriculum run printed, when you don't have the run's CSV files:

```bash
python scripts/plot_episode_length.py my_log.txt
```
