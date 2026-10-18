# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Created this file: a thin command-line wrapper around
#     marl.watch.main.
# Prompt summary: Make a watch script that plays games with the latest trained
#   model in the PyBullet GUI.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
"""Watch a trained policy play keep-away in the PyBullet GUI. Run from the repo root with the venv active:

    python scripts/watch.py runs/keepaway_1v1_pybullet_seed0
    python scripts/watch.py runs/keepaway_1v1_pybullet_seed0 --episodes 5 --seed 3
    python scripts/watch.py --baseline flee

See ``python scripts/watch.py --help`` for every option.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `marl` importable

from marl.watch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
### END AI GENERATED
