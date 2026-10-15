# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.calibrate.main.
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
"""Calibrate red_max_speed against the scripted flee policy (design #22). Run from the repo root with the venv active:

    python scripts/calibrate.py --config configs/keepaway_1v1.toml
    python scripts/calibrate.py --speeds 1.6:1.9:0.05 --episodes 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.calibrate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
