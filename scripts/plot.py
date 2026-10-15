# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.plots.main.
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs).
# Human review:   Pending review
# ==============================================================================
"""Make the report figures from finished runs. Run from the repo root with the venv active:

    python scripts/plot.py

Defaults: keep-away runs runs/keepaway_1v1_seed*, goto runs runs/goto_target_seed*,
calibration runs/calibration_keepaway_1v1.csv, output runs/figures/ (PNG + PDF +
results_tables.md). Run scripts/evaluate.py on the runs first (the tables feed
the reference lines). See --help for every option.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.plots import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
