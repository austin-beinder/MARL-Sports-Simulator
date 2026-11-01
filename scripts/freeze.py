# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Created this file: a thin command-line wrapper around
#     marl.record.main (backlog task 2, step 6).
# Prompt summary: Implement step 6 of the task 2 plotting-refactor plan
#   (freezing results records).
# Human review:   Pending review
# ==============================================================================
"""Freeze a results record. Run from the repo root with the venv active:

    python scripts/freeze.py golden               # results/golden/manifest.toml
    python scripts/freeze.py golden --overwrite   # replace files already in results/golden/data/

Copies the inputs the record's manifest still reads from runs/ (paths like
"../../runs/new_reward_seed*") into results/<record>/data/, rewrites those path
strings in the manifest (your comments stay), appends a provenance comment, and
renders the record's figures into results/<record>/figures/. With nothing left to
copy, it just re-renders the figures. Commit the record afterwards.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
