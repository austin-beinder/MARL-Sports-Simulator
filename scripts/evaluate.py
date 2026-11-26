# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.rl.report.main.
#   - 2026-11-25: Citation cleanup (backlog task 9): dropped the Prompt summary
#     and Human review lines; no behaviour change.
# ==============================================================================
"""Evaluate trained runs + baselines on the fixed evaluation episodes (design #23). Run from the repo root with the venv active:

    python scripts/evaluate.py runs/keepaway_1v1_pybullet_seed0 runs/keepaway_1v1_pybullet_seed1 runs/keepaway_1v1_pybullet_seed2
    python scripts/evaluate.py "runs/keepaway_1v1_pybullet_seed*"     # same; the glob is expanded here, sorted
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.rl.report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
