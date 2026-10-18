# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: evaluation-table tests (the table
#     reproduces the training run's own iteration-0 and final evaluations
#     exactly, aggregation across seeds, refusing mixed tasks, the CLI).
#   - 2026-10-11: Added a goto_target reproduction test: keep-away lengths are
#     too coarse to tell two untrained inits apart (a mutation check showed it).
#   - 2026-10-14: Added tests for the race metrics and arm columns, pbrs_coef,
#     and the --checkpoint option (it never overwrites the final table).
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
import csv
import math

import pytest

from marl.common.config import config_from_dict
from marl.rl.report import evaluate_runs, format_table, main, pbrs_coef
from marl.rl.train import train

KEYS = ("len_mean", "frac_truncated", "mean_reward", "len_mean_possession", "len_mean_free_ball")


def tiny_run(tmp_path, seed=0, goto=False, **shared):
    rl = {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}}
    if goto:
        shared |= {"scenario": "goto_target", "n_red": 0, "max_episode_steps": 40}
        rl["reward"] = {"base": "goto_target"}
    cfg = config_from_dict({
        "shared": shared,
        "rl": rl,
        "run": {"seed": seed, "total_steps": 128, "eval_every": 2, "eval_episodes": 4},
    })
    return train(cfg, tmp_path / f"run_s{seed}{'_goto' if goto else ''}", log=lambda _: None).out_dir


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def same(a, b):
    for k in KEYS:
        x, y = float(a[k]), float(b[k])
        assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-12), k


def test_table_reproduces_the_training_runs_own_evaluations(tmp_path):
    run = tiny_run(tmp_path)
    per_run, table = evaluate_runs([run])
    by = {r["policy"]: r for r in per_run}
    assert set(by) == {"ppo", "ppo_deterministic", "untrained", "random", "still", "flee"}
    logged = csv_rows(run / "eval.csv")
    same(by["untrained"], logged[0])  # iteration 0: rebuilt init == the logged untrained policy
    assert logged[0]["iteration"] == "0"
    final_stoch = [r for r in logged if r["mode"] == "stochastic"][-1]
    final_det = [r for r in logged if r["mode"] == "deterministic"][-1]
    same(by["ppo"], final_stoch)
    same(by["ppo_deterministic"], final_det)
    assert [r["policy"] for r in table][:3] == ["ppo", "ppo_deterministic", "untrained"]


def test_untrained_policy_is_rebuilt_exactly(tmp_path):
    # goto_target's reward (-distance) is continuous, so ANY difference in the rebuilt
    # initial network would show up in mean_reward; keep-away lengths are too coarse.
    run = tiny_run(tmp_path, goto=True)
    per_run, _ = evaluate_runs([run])
    by = {r["policy"]: r for r in per_run}
    assert set(by) == {"ppo", "ppo_deterministic", "untrained", "random", "still", "oracle"}
    logged = csv_rows(run / "eval.csv")
    assert float(by["untrained"]["mean_reward"]) == float(logged[0]["mean_reward"])  # bit-exact
    same(by["ppo"], [r for r in logged if r["mode"] == "stochastic"][-1])


def test_aggregates_across_training_seeds(tmp_path):
    runs = [tiny_run(tmp_path, seed=0), tiny_run(tmp_path, seed=1)]
    _, table = evaluate_runs(runs)
    by = {r["policy"]: r for r in table}
    assert by["ppo"]["n_seeds"] == 2 and math.isfinite(by["ppo"]["len_mean_seed_std"])
    assert by["flee"]["n_seeds"] == 1 and math.isnan(by["flee"]["len_mean_seed_std"])
    assert by["ppo"]["survival"] == pytest.approx(by["ppo"]["len_mean"] / 300)
    text = format_table(table)
    assert "ppo_deterministic" in text and "flee" in text


def test_refuses_to_mix_different_tasks(tmp_path):
    runs = [tiny_run(tmp_path, seed=0), tiny_run(tmp_path, seed=1, red_max_speed=1.9)]
    with pytest.raises(ValueError, match="different task"):
        evaluate_runs(runs)


def test_cli_writes_both_tables(tmp_path):
    run = tiny_run(tmp_path)
    out = tmp_path / "table.csv"
    assert main([str(run), "--out", str(out)]) == 0
    assert len(csv_rows(out)) == 6 and len(csv_rows(tmp_path / "table_per_run.csv")) == 6


def test_rows_carry_race_metrics_and_the_arm(tmp_path):
    run = tiny_run(tmp_path, start={"start_mix": 0.0})  # every eval episode is a race
    per_run, table = evaluate_runs([run])
    by = {r["policy"]: r for r in per_run}
    assert (by["ppo"]["pbrs_coef"], by["ppo"]["env_steps"], by["untrained"]["env_steps"]) == (0.0, 128, 0)
    assert math.isnan(by["flee"]["pbrs_coef"]) and math.isnan(by["flee"]["env_steps"])
    assert all(r["races"] == 4 for r in per_run)
    assert by["flee"]["pickup_rate"] == 1.0 and by["flee"]["approach_m_per_step"] == pytest.approx(0.1)
    tab = {r["policy"]: r for r in table}
    assert tab["ppo"]["pickup_rate"] == by["ppo"]["pickup_rate"]  # aggregated like every metric
    assert "pickup" in format_table(table)


def test_pbrs_coef_identifies_the_arm():
    def shaped(enabled):
        term = {"name": "pbrs_ball_acquisition", "enabled": enabled, "coef": 4.0}
        return config_from_dict({"rl": {"reward": {"shaping": [term]}}})

    assert (pbrs_coef(config_from_dict({})), pbrs_coef(shaped(True)), pbrs_coef(shaped(False))) == (0.0, 4.0, 0.0)


def test_cli_checkpoint_option_never_overwrites_the_final_table(tmp_path):
    run = tiny_run(tmp_path)  # 2 iterations, eval_every 2 -> checkpoints iter_00002.pt + final.pt
    assert main([str(run), "--checkpoint", "iter_00002.pt"]) == 0
    assert (tmp_path / f"eval_{run.name}_iter_00002.csv").exists()
    assert not (tmp_path / f"eval_{run.name}.csv").exists()  # the final table's name stays free
    rows = csv_rows(tmp_path / f"eval_{run.name}_iter_00002_per_run.csv")
    assert rows[0]["env_steps"] == "128"
