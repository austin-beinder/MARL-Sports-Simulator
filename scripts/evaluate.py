# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.rl.report.main.
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
"""Evaluate trained runs + baselines on the fixed evaluation episodes (design #23). Run from the repo root with the venv active:

    python scripts/evaluate.py runs/keepaway_1v1_seed0 runs/keepaway_1v1_seed1 runs/keepaway_1v1_seed2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.rl.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
