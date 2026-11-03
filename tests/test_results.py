# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: tests for the results-loading
#     layer (arm glob resolution, per-arm schedule check, table lookup,
#     eval-episode mismatch warnings) for backlog task 2, step 1.
# Prompt summary: Implement step 1 of the task 2 plotting-refactor plan
#   (marl/results.py), tests first.
# Human review:   Pending review
# ==============================================================================
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from marl.results import Arm, eval_mismatch, mean_std, run_dirs, smooth, table_path_for


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_run(root, name, iterations=(0, 10, 20), eval_seed=7, eval_episodes=6, config=True):
    """A run directory with the files plotting reads: eval.csv, train.csv, config.json."""
    d = root / name
    write_csv(d / "eval.csv", [
        {"iteration": it, "env_steps": it * 2048, "mode": mode, "len_mean": 80 + it, "len_mean_possession": 80 + it,
         "len_mean_free_ball": 90.0, "mean_reward": 1.0}
        for it in iterations for mode in ("stochastic", "deterministic")
    ])
    write_csv(d / "train.csv", [
        {"iteration": i, "env_steps": i * 2048, "entropy": -0.1 * i, "explained_variance": 0.9, "approx_kl": 0.005,
         "clip_frac": 0.05}
        for i in range(1, 6)
    ])
    if config:
        cfg = {"shared": {"scenario": "keepaway"}, "rl": {},
               "run": {"seed": 0, "eval_seed": eval_seed, "eval_episodes": eval_episodes}}
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def test_run_dirs_are_sorted_directories_only(tmp_path):
    for k in (2, 0, 1):
        fake_run(tmp_path, f"x_seed{k}")
    (tmp_path / "x_seed9.csv").write_text("not a run", encoding="utf-8")
    assert [p.name for p in run_dirs(str(tmp_path / "x_seed*"))] == ["x_seed0", "x_seed1", "x_seed2"]


def test_an_arm_matching_nothing_names_itself(tmp_path):
    with pytest.raises(ValueError, match="'new reward'.*nothing"):
        Arm("new reward", str(tmp_path / "missing_seed*")).dirs()


def test_arm_table_defaults_to_evaluate_naming_and_honors_an_explicit_path(tmp_path):
    fake_run(tmp_path, "a_seed0")
    fake_run(tmp_path, "a_seed1")
    assert Arm("a", str(tmp_path / "a_seed*")).table_path() == tmp_path / "eval_a_seed0.csv"
    explicit = tmp_path / "elsewhere.csv"
    assert Arm("a", str(tmp_path / "a_seed*"), table=explicit).table_path() == explicit


def test_arms_on_different_schedules_load_separately(tmp_path):
    for k in range(2):
        fake_run(tmp_path, f"mock_seed{k}", iterations=(0, 10, 20))
        fake_run(tmp_path, f"pyb_seed{k}", iterations=(0, 5, 10, 15))
    mock = Arm("mock", str(tmp_path / "mock_seed*")).eval_curves()
    pyb = Arm("pybullet", str(tmp_path / "pyb_seed*")).eval_curves()
    assert mock["len_mean"].shape == (2, 3) and pyb["len_mean"].shape == (2, 4)
    assert Arm("mock", str(tmp_path / "mock_seed*")).train_curves()["entropy"].shape == (2, 5)


def test_a_mixed_schedule_within_one_arm_is_refused_with_the_arm_label(tmp_path):
    fake_run(tmp_path, "m_seed0", iterations=(0, 10, 20))
    fake_run(tmp_path, "m_seed1", iterations=(0, 5, 20))
    with pytest.raises(ValueError, match="'mixed'.*schedule"):
        Arm("mixed", str(tmp_path / "m_seed*")).eval_curves()


def test_eval_mismatch_warns_only_when_the_fixed_episodes_differ(tmp_path):
    fake_run(tmp_path, "a_seed0")
    fake_run(tmp_path, "b_seed0", iterations=(0, 5, 10))  # another schedule, same episodes: fine
    fake_run(tmp_path, "c_seed0", eval_seed=8)
    fake_run(tmp_path, "d_seed0", config=False)  # no config.json: nothing to compare
    a, b, c, d = (Arm(n, str(tmp_path / f"{n}_seed*")) for n in "abcd")
    assert eval_mismatch([a, b, d]) == []
    [msg] = eval_mismatch([a, b, c])
    assert "'c'" in msg and "eval_seed" in msg


def test_moved_helpers_still_behave():
    Y = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    np.testing.assert_allclose(mean_std(Y)[0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(smooth(np.full((2, 7), 3.0), 3), 3.0)
    assert table_path_for([Path("runs/k_seed0")]) == Path("runs/eval_k_seed0.csv")
