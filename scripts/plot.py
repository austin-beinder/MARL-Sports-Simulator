# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.plots.main.
#   - 2026-11-01: Rewrote the docstring for records and ad-hoc plots (task 2).
#   - 2026-11-13: Docstring: the --curriculum option. (Austin Beinder)
#   - 2026-11-27: Docstring: a plain plot.py renders every golden record
#     (backlog task 16). (JC Stinson III)
# ==============================================================================
"""Make figures from finished runs. Run from the repo root with the venv active.

    python scripts/plot.py                               # every results/golden*/ record -> runs/plots/<record>/
    python scripts/plot.py --manifest results/X/manifest.toml   # any record -> runs/plots/X/
    python scripts/plot.py --arm "new reward=runs/new_reward_pybullet_seed*" [--arm "golden=results/golden_1v1_initial/data/..."]
                           [--kind curves final diagnostics] [--baselines LABEL] [--name N] [--overwrite]
                                                         # ad hoc -> runs/plots/<name>/ (default: adhoc)

Figures are PNG, plus results_tables.md for a record's [[table]] entries. Run
scripts/evaluate.py on a set of runs first: its table feeds the final-results bars and
the gray scripted-baseline lines (--baselines). A non-empty output folder needs
--overwrite, except a record rendered into its own default folder. See --help.
    python scripts/plot.py

Defaults: keep-away runs runs/keepaway_1v1_seed*, goto runs runs/goto_target_seed*,
calibration runs/calibration_keepaway_1v1.csv, output runs/figures/ (PNG + PDF +
results_tables.md). Run scripts/evaluate.py on the runs first (the tables feed
the reference lines). See --help for every option.

### AI GENERATED (2026-11-13)
For a curriculum run, plot its progress instead (stage by stage; see scripts/README.md):

    python scripts/plot.py --curriculum runs/curriculum_passing_3v1_pybullet_seed0
### END AI GENERATED
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.plots import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
