# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.plots.main.
#   - 2026-11-01: Rewrote the docstring for records and ad-hoc plots (task 2).
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs).
# Human review:   Pending review
# ==============================================================================
"""Make figures from finished runs. Run from the repo root with the venv active.

    python scripts/plot.py                               # the golden record -> runs/plots/golden/
    python scripts/plot.py --manifest results/X/manifest.toml   # any record -> runs/plots/X/
    python scripts/plot.py --arm "new reward=runs/new_reward_pybullet_seed*" [--arm "golden=results/golden/data/..."]
                           [--kind curves final diagnostics] [--baselines LABEL] [--name N] [--overwrite]
                                                         # ad hoc -> runs/plots/<name>/ (default: adhoc)

Figures are PNG, plus results_tables.md for a record's [[table]] entries. Run
scripts/evaluate.py on a set of runs first: its table feeds the final-results bars and
the gray scripted-baseline lines (--baselines). A non-empty output folder needs
--overwrite, except a record rendered into its own default folder. See --help.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.plots import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
