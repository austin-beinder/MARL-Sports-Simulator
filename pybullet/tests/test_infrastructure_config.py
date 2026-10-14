from pathlib import Path
import sys

# Existing repository files use sibling imports (from ball import Ball, etc.),
# so tests explicitly put the pybullet source directory on sys.path.
PYBULLET_SRC = Path(__file__).resolve().parents[1]
if str(PYBULLET_SRC) not in sys.path:
    sys.path.insert(0, str(PYBULLET_SRC))

from config import load_config


def test_default_config_loads():
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "configs" / "default.yaml")

    assert cfg.project.seed == 42
    assert cfg.game.name == "keep_away"
    assert cfg.game.config["action_repeat"] == 8
    assert cfg.training.algorithm == "ippo"
