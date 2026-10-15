# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: calibration tests (selection
#     rule, speed parsing, a small real sweep, dedicated calibration seeds,
#     the CLI).
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
import csv
from pathlib import Path

import pytest

from marl.calibrate import calibration_seeds, main, parse_speeds, pick_speed, survival_table
from marl.common.config import config_from_dict
from marl.rl.evaluate import eval_env_seeds

ROOT = Path(__file__).resolve().parents[1]


def flee_rows(pairs):
    rows = [{"policy": "flee", "red_max_speed": v, "survival": s} for v, s in pairs]
    return rows + [{"policy": "still", "red_max_speed": v, "survival": 0.5} for v, _ in pairs]  # ignored


def test_pick_speed_takes_the_band_center():
    rows = flee_rows([(1.5, 0.83), (1.6, 0.67), (1.7, 0.56), (1.8, 0.49), (1.9, 0.44)])
    assert pick_speed(rows) == 1.8


def test_pick_speed_breaks_ties_toward_the_slower_red():
    assert pick_speed(flee_rows([(1.7, 0.55), (1.8, 0.45)])) == 1.7


def test_pick_speed_refuses_when_nothing_is_in_the_band():
    with pytest.raises(ValueError, match="widen"):
        pick_speed(flee_rows([(1.2, 0.95), (1.3, 0.85)]))


def test_parse_speeds():
    assert parse_speeds("1.5:2.0:0.25") == [1.5, 1.75, 2.0]
    grid = parse_speeds("1.5:2.0:0.05")
    assert len(grid) == 11 and grid[0] == 1.5 and grid[-1] == 2.0
    assert parse_speeds("1.6, 1.8") == [1.6, 1.8]
    for bad in ("2.0:1.0:0.1", "1.0:2.0:0", ""):
        with pytest.raises(ValueError):
            parse_speeds(bad)


def test_small_sweep_survival_falls_as_red_gets_faster():
    rows = survival_table(config_from_dict({}), [1.5, 2.2], episodes=6)
    assert [(r["red_max_speed"], r["policy"]) for r in rows] == [
        (1.5, "flee"), (1.5, "random"), (1.5, "still"), (2.2, "flee"), (2.2, "random"), (2.2, "still")
    ]
    flee = {r["red_max_speed"]: r["survival"] for r in rows if r["policy"] == "flee"}
    assert flee[1.5] > flee[2.2] and all(0.0 < r["survival"] <= 1.0 for r in rows)


def test_calibration_uses_its_own_seeds_not_the_eval_seeds():
    assert calibration_seeds(10) != eval_env_seeds(config_from_dict({}).run.eval_seed, 10)


def test_rejects_goto_target():
    cfg = config_from_dict({"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}})
    with pytest.raises(ValueError):
        survival_table(cfg, [1.7], episodes=1)


def test_cli_writes_the_table_and_recommends(tmp_path, capsys):
    out = tmp_path / "cal.csv"
    assert main(["--config", str(ROOT / "configs" / "keepaway_1v1.toml"), "--speeds", "1.7,1.8", "--episodes", "4", "--out", str(out)]) == 0
    with open(out, newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 6
    assert "recommended red_max_speed" in capsys.readouterr().out
