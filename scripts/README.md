<!--
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-13
Author:  Austin Beinder
Usage:
  2026-11-13: Generated this entire file: how to plot a curriculum run's
    progress with scripts/plot.py --curriculum.
  2026-11-27: The report figures now come from the golden records; a section
    on plot_team_size.py, the team-size record's ad-hoc figures (backlog
    task 16). (JC Stinson III)
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

Without `--curriculum`, `plot.py` renders the golden records (every
`results/golden*/`, each to `runs/plots/<record>/`), or ad-hoc plots of any
runs with `--arm`. Run `scripts/evaluate.py` on the runs first. `python
scripts/plot.py --help` lists every option; the README's "Results & plots"
explains records.

## Across team sizes

`plot_team_size.py` draws four figures that compare team sizes (1v1 to 3v3),
from scratch vs goto_target warm start, from a team-size record's frozen data:
survival, survival scaled so untrained = 0 and flee = 1, free-ball races won,
and kicks and passes per episode. A record's own figures can't mix sizes, since
each size has its own scripted baselines.

```bash
python scripts/plot_team_size.py   # -> results/golden_teams_and_goto_ball_warm_starts/figures_adhoc/
```

## From a terminal log

`plot_episode_length.py` plots episode length from the text a training or
curriculum run printed, when you don't have the run's CSV files:

```bash
python scripts/plot_episode_length.py my_log.txt
```
