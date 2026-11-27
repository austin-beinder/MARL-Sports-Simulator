# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Created this file: a thin command-line wrapper around
#     marl.record.main (backlog task 2, step 6).
#   - 2026-11-25: Citation cleanup (backlog task 9): dropped the Prompt summary
#     and Human review lines; its docstring says why freezing is a separate
#     script from plot.py; no behaviour change.
#   - 2026-11-27: Docstring examples use golden_1v1_initial, the renamed golden
#     record (backlog task 16).
# ==============================================================================
"""Freeze a results record. Run from the repo root with the venv active:

    python scripts/freeze.py golden_1v1_initial               # results/golden_1v1_initial/manifest.toml
    python scripts/freeze.py golden_1v1_initial --overwrite   # replace files already in its data/

Copies the inputs the record's manifest still reads from runs/ (paths like
"../../runs/new_reward_pybullet_seed*") into results/<record>/data/, rewrites those path
strings in the manifest (your comments stay), appends a provenance comment, and
renders the record's figures into results/<record>/figures/. With nothing left to
copy, it just re-renders the figures. Commit the record afterwards.

Why a separate script from scripts/plot.py: plotting only reads runs and draws figures, while
freezing changes a tracked record. It moves the record's data out of runs/ (gitignored, so local
and disposable) into results/<record>/, where it is committed, so the record's figures can be
redrawn from any checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
