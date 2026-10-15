# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Created this file: a thin command-line wrapper around
#     marl.rl.train.main.
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
"""Train PPO/MAPPO on the mock env. Run from the repo root with the venv active:

    python scripts/train.py --config configs/goto_target.toml
    python scripts/train.py --config configs/keepaway_1v1.toml --seed 1
    python scripts/train.py --config configs/keepaway_1v1.toml --set shared.red_max_speed=1.6

Outputs go to runs/<config name>_seed<seed>/ (config.json, train.csv, eval.csv,
checkpoints/). See ``python scripts/train.py --help`` for every option.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.rl.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
