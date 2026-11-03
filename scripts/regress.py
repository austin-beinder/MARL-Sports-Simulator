# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Created this file: a thin command-line wrapper around
#     marl.regression.main (backlog task 4, step 4).
# Prompt summary: Implement step 4 of the task 4 regression-harness plan
#   (the one-command regression runner).
# Human review:   Pending review
# ==============================================================================
"""Regression check against the silver record. Run from the repo root with the venv active:

    python scripts/regress.py                          # run the suite, compare, report
    python scripts/regress.py --only "*_mock"          # just the mock cases
    python scripts/regress.py --no-run --rtol 1e-9     # re-compare the last run, near-match
    python scripts/regress.py --update --reason "..."  # an intended change: re-freeze the record

Trains every case of results/silver/suite.toml (config x env) on its reserved seeds
(90-91) and short budget into runs/regression/, evaluates each case, and compares
everything with results/silver/data/. Unchanged code must reproduce it exactly.
Exit 0 = identical, 1 = deltas (see runs/regression/report.md), 2 = a failed job.
After --update, commit results/silver/ together with the change that caused it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.regression import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
