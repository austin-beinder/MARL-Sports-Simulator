# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: plotting tests on synthetic run
#     directories (every figure + the table view is written, schedule
#     mismatch is refused, missing inputs are skipped, helper math).
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs).
# Human review:   Pending review
# ==============================================================================
import csv
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from marl.plots import eval_curves, main, mean_std, smooth, table_path_for
from marl.rl.evaluate import summarize
from marl.rl.report import aggregate
from marl.rl.rollout import EpisodeRecord

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records(rng, scenario, scale):
    if scenario == "goto_target":
        return [EpisodeRecord(300, False, True, "target", -scale * rng.uniform(1, 2)) for _ in range(6)]
    return [EpisodeRecord(int(scale * rng.uniform(60, 90)), True, False, st, 1.0) for st in ("possession", "free_ball") * 3]


def fake_run(root, name, seed, scenario="keepaway", iterations=(0, 10, 20)):
    rng = np.random.default_rng(seed)
    d = root / name
    ev = []
    for it in iterations:
        row = {"iteration": it, "env_steps": it * 2048, "mode": "stochastic", **summarize(records(rng, scenario, 1 + it / 20))}
        ev.append(row)
    ev.append({**ev[-1], "mode": "deterministic"})
    write_csv(d / "eval.csv", ev)
    write_csv(d / "train.csv", [
        {"iteration": i, "env_steps": i * 2048, "entropy": -0.1 * i, "explained_variance": 0.9,
         "approx_kl": 0.005, "clip_frac": 0.05}
        for i in range(1, 21)
    ])
    return d


def fake_table(path, runs, scenario):
    rng = np.random.default_rng(0)
    per_run = [{"policy": p, **summarize(records(rng, scenario, 1.5))} for p in ("ppo", "ppo_deterministic", "untrained")
               for _ in runs]
    third = "flee" if scenario == "keepaway" else "oracle"
    per_run += [{"policy": p, **summarize(records(rng, scenario, 1.2))} for p in ("random", "still", third)]
    write_csv(path, aggregate(per_run, 300))


@pytest.fixture
def workspace(tmp_path):
    keep = [fake_run(tmp_path, f"keepaway_1v1_seed{k}", k) for k in range(3)]
    goto = [fake_run(tmp_path, f"goto_target_seed{k}", 10 + k, "goto_target") for k in range(2)]
    fake_table(table_path_for(keep), keep, "keepaway")
    fake_table(table_path_for(goto), goto, "goto_target")
    write_csv(tmp_path / "calibration.csv", [
        {"red_max_speed": v, "policy": p, "survival": s}
        for v, f in ((1.7, 0.56), (1.8, 0.50), (1.9, 0.45)) for p, s in (("flee", f), ("random", 0.28), ("still", 0.28))
    ])
    return tmp_path


def args(ws, out):
    return ["--keepaway", str(ws / "keepaway_1v1_seed*"), "--goto", str(ws / "goto_target_seed*"),
            "--calibration", str(ws / "calibration.csv"), "--config", str(ROOT / "configs" / "keepaway_1v1.toml"),
            "--out", str(out)]


def test_every_figure_and_the_table_view_are_written(workspace, capsys):
    out = workspace / "figures"
    assert main(args(workspace, out)) == 0
    names = ("keepaway_learning_curves", "keepaway_final_results", "calibration_red_speed",
             "goto_target_curve", "keepaway_training_diagnostics")
    for name in names:
        for ext in ("png", "pdf"):
            assert (out / f"{name}.{ext}").stat().st_size > 1000, f"{name}.{ext}"
    md = (out / "results_tables.md").read_text(encoding="utf-8")
    assert "| ppo |" in md and "| flee |" in md and "| oracle |" in md
    assert "skipped" not in capsys.readouterr().out
    assert matplotlib.get_backend().lower() == "agg"  # never opens a window


def test_missing_inputs_are_skipped_not_fatal(tmp_path, capsys):
    fake_run(tmp_path, "keepaway_1v1_seed0", 0)  # a run, but no evaluate.py table and no calibration
    rc = main(["--keepaway", str(tmp_path / "keepaway_1v1_seed*"), "--goto", str(tmp_path / "nothing*"),
               "--calibration", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "fig")])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "scripts/evaluate.py" in printed and "calibration" in printed
    assert (tmp_path / "fig" / "keepaway_training_diagnostics.png").exists()  # needs only train.csv


def test_runs_with_different_eval_schedules_are_refused(tmp_path):
    a = fake_run(tmp_path, "a", 0, iterations=(0, 10, 20))
    b = fake_run(tmp_path, "b", 1, iterations=(0, 5, 20))
    with pytest.raises(ValueError, match="schedule"):
        eval_curves([a, b])


def test_helpers():
    Y = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    m, sd = mean_std(Y)
    np.testing.assert_allclose(m, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(sd, np.std(Y, axis=0, ddof=1))  # sample std, as in the eval table
    np.testing.assert_allclose(mean_std(Y[:1])[1], 0.0)
    np.testing.assert_allclose(smooth(np.full((2, 7), 3.0), 3), 3.0)  # edges renormalized
    assert table_path_for([Path("runs/keepaway_1v1_seed0")]) == Path("runs/eval_keepaway_1v1_seed0.csv")
