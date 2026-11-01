# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: plotting tests on synthetic run
#     directories (every figure + the table view is written, schedule
#     mismatch is refused, missing inputs are skipped, helper math).
#   - 2026-10-14: Added the PBRS figures to the synthetic workspace (fake PBRS
#     runs, per-run tables with race metrics, sweep files) plus tests for the
#     sweep aggregation, per-column table formats, and end-label collisions.
#   - 2026-11-01: Rewrote the figure/CLI tests for manifest-driven rendering
#     (task 2, step 3): every kind renders from a manifest, PNG only with no
#     version stamp, missing tables skip their figures, 9 arms and mixed
#     schedules share a figure, label-stable colors, no baselines means no
#     gray lines, sweep points. The synthetic-run helpers and the mark tests
#     are kept.
#   - 2026-11-01: Added tests for the demo fixes (stacked close labels, early
#     ends labeled to the right, single-seed labels and legends).
# Prompt summary: Implement Phase 3 step 11 of the issue #7 plan (plots for the
#   midterm from the 5-seed keep-away runs); later, task 2's plotting refactor.
# Human review:   Pending review
# ==============================================================================
import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from marl.common.config import config_to_dict, load_config
from marl.manifest import load_manifest
from marl.plots import (
    CONTEXT,
    PALETTE,
    Series,
    SweepPoint,
    _end_labels,
    assign_colors,
    eval_curves,
    fig_curves,
    main,
    mean_std,
    read_table,
    render_figure,
    render_manifest,
    smooth,
    sweep_points,
    sweep_table,
    table_markdown,
    table_path_for,
)
from marl.rl.evaluate import summarize
from marl.rl.report import aggregate
from marl.rl.rollout import EpisodeRecord

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("keepaway_learning_curves", "pbrs_learning_curves", "keepaway_final_results", "keepaway_training_diagnostics",
           "goto_target_curve", "pbrs_coef_sweep", "calibration_red_speed")


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


def race_stats(rng):
    return {"races": 3, "pickup_rate": rng.uniform(0, 1), "pickup_step_mean": 50.0,
            "approach_m_per_step": rng.uniform(0, 0.03), "approach_alignment": rng.uniform(0, 0.4)}


def fake_table(path, runs, scenario, coef=0.0):
    """An evaluation table plus its per-run CSV, shaped like scripts/evaluate.py's output."""
    rng = np.random.default_rng(0)
    nan = float("nan")
    per_run = [{"policy": p, "pbrs_coef": coef, "env_steps": 40960, **summarize(records(rng, scenario, 1.5)),
                **race_stats(rng)} for p in ("ppo", "ppo_deterministic", "untrained") for _ in runs]
    third = "flee" if scenario == "keepaway" else "oracle"
    per_run += [{"policy": p, "pbrs_coef": nan, "env_steps": nan, **summarize(records(rng, scenario, 1.2)),
                 **race_stats(rng)} for p in ("random", "still", third)]
    write_csv(path, aggregate(per_run, 300))
    write_csv(path.with_name(path.stem + "_per_run.csv"), per_run)


WORKSPACE_MANIFEST = """
name = "test"

[[figure]]
kind      = "curves"
name      = "keepaway_learning_curves"
baselines = "PPO"
ylim      = [0, 300]
  [[figure.arm]]
  label = "PPO"
  runs  = "keepaway_1v1_seed*"

[[figure]]
kind      = "curves"
name      = "pbrs_learning_curves"
baselines = "survival only"
  [[figure.arm]]
  label = "survival only"
  runs  = "keepaway_1v1_seed*"
  [[figure.arm]]
  label = "+ PBRS"
  runs  = "keepaway_1v1_pbrs_seed*"

[[figure]]
kind      = "final"
name      = "keepaway_final_results"
baselines = "PPO"
  [[figure.arm]]
  label = "PPO"
  runs  = "keepaway_1v1_seed*"

[[figure]]
kind = "diagnostics"
name = "keepaway_training_diagnostics"
  [[figure.arm]]
  label = "PPO"
  runs  = "keepaway_1v1_seed*"

[[figure]]
kind      = "goto"
name      = "goto_target_curve"
baselines = "goto PPO"
  [[figure.arm]]
  label = "goto PPO"
  runs  = "goto_target_seed*"

[[figure]]
kind      = "sweep"
name      = "pbrs_coef_sweep"
highlight = 4
x_label   = "PBRS coefficient"
  [[figure.point]]
  x     = 0
  label = "0 (survival only)"
  table = "sweep_c0_per_run.csv"
  [[figure.point]]
  x     = 16
  table = "sweep_c16_per_run.csv"
  [[figure.point]]
  x     = 4
  table = "sweep_c4_per_run.csv"

[[figure]]
kind   = "calibration"
name   = "calibration_red_speed"
csv    = "calibration.csv"
config = "keepaway_1v1_seed0/config.json"

[[table]]
title   = "Keep-away ({n_runs} runs)"
source  = "eval_keepaway_1v1_seed0.csv"
columns = "keepaway"

[[table]]
title   = "PBRS coefficient pilot"
sweep   = "pbrs_coef_sweep"
columns = "sweep"
first   = "PBRS coef"

[[table]]
title   = "goto_target ({n_runs} runs)"
source  = "eval_goto_target_seed0.csv"
columns = "goto"
"""


@pytest.fixture
def workspace(tmp_path):
    keep = [fake_run(tmp_path, f"keepaway_1v1_seed{k}", k) for k in range(3)]
    pbrs = [fake_run(tmp_path, f"keepaway_1v1_pbrs_seed{k}", 20 + k) for k in range(3)]
    goto = [fake_run(tmp_path, f"goto_target_seed{k}", 10 + k, "goto_target") for k in range(2)]
    fake_table(table_path_for(keep), keep, "keepaway")
    fake_table(table_path_for(pbrs), pbrs, "keepaway", coef=4.0)
    for c in (0.0, 4.0, 16.0):  # the coefficient pilot's per-run files
        fake_table(tmp_path / f"sweep_c{c:g}.csv", keep, "keepaway", coef=c)
    fake_table(table_path_for(goto), goto, "goto_target")
    write_csv(tmp_path / "calibration.csv", [
        {"red_max_speed": v, "policy": p, "survival": s}
        for v, f in ((1.7, 0.56), (1.8, 0.50), (1.9, 0.45)) for p, s in (("flee", f), ("random", 0.28), ("still", 0.28))
    ])
    cfg = config_to_dict(load_config(ROOT / "configs" / "keepaway_1v1.toml"))
    (keep[0] / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "manifest.toml").write_text(WORKSPACE_MANIFEST, encoding="utf-8")
    return tmp_path


def manifest_with(tmp_path, body):
    path = tmp_path / "m.toml"
    path.write_text('name = "m"\n' + body, encoding="utf-8")
    return load_manifest(path)


def test_every_figure_and_the_table_view_are_written(workspace):
    out = workspace / "figures"
    made, notes = render_manifest(load_manifest(workspace / "manifest.toml"), out)
    assert notes == []
    for name in FIGURES:
        assert (out / f"{name}.png").stat().st_size > 1000, name
    assert not list(out.glob("*.pdf"))  # PNG only
    md = (out / "results_tables.md").read_text(encoding="utf-8")
    assert "### Keep-away (3 runs)" in md and "### goto_target (2 runs)" in md  # {n_runs} filled from the table
    assert "| ppo |" in md and "| flee |" in md and "| oracle |" in md
    assert "| PBRS coef |" in md and "| 0 (survival only) | 3 |" in md and "| 16 | 3 |" in md
    assert md.index("| 4 | 3 |") < md.index("| 16 | 3 |")  # sweep rows sorted by x, not manifest order
    assert matplotlib.get_backend().lower() == "agg"  # never opens a window


def test_pngs_carry_no_matplotlib_version_stamp(workspace):
    out = workspace / "figures"
    render_manifest(load_manifest(workspace / "manifest.toml"), out)
    assert b"Matplotlib" not in (out / "keepaway_learning_curves.png").read_bytes()


def test_missing_tables_skip_their_figures_not_fatal(tmp_path):
    fake_run(tmp_path, "k_seed0", 0)  # a run, but no evaluate.py table
    m = manifest_with(tmp_path, """
[[figure]]
kind = "curves"
name = "with_baselines"
baselines = "a"
  [[figure.arm]]
  label = "a"
  runs = "k_seed*"
[[figure]]
kind = "curves"
name = "without_baselines"
  [[figure.arm]]
  label = "a"
  runs = "k_seed*"
[[figure]]
kind = "diagnostics"
name = "diag"
  [[figure.arm]]
  label = "a"
  runs = "k_seed*"
""")
    made, notes = render_manifest(m, tmp_path / "fig")
    assert [p.name for p in made] == ["without_baselines.png", "diag.png"]  # these need no table
    [note] = notes
    assert note.startswith("skipped with_baselines") and "scripts/evaluate.py" in note


def test_an_arm_matching_nothing_is_an_error_not_a_skip(tmp_path):
    m = manifest_with(tmp_path, '[[figure]]\nkind = "diagnostics"\nname = "d"\n'
                                '[[figure.arm]]\nlabel = "typo"\nruns = "no_such_seed*"\n')
    with pytest.raises(ValueError, match="'typo'"):
        render_manifest(m, tmp_path / "fig")


def test_nine_arms_on_mixed_schedules_share_one_figure(tmp_path):
    body = ""
    for k in range(9):
        fake_run(tmp_path, f"arm{k}_seed0", k, iterations=(0, 10, 20) if k % 2 else (0, 5, 10, 15))
        body += f'  [[figure.arm]]\n  label = "arm {k}"\n  runs = "arm{k}_seed*"\n'
    m = manifest_with(tmp_path, '[[figure]]\nkind = "curves"\nname = "many"\n' + body)
    fig = render_figure(m.figures[0], assign_colors(m.figures))
    assert [t.get_text().split(":")[0] for t in fig.legends[0].texts] == [f"arm {k}" for k in range(9)]
    plt.close(fig)


def test_colors_follow_the_label_across_figures_and_honor_overrides(tmp_path):
    def arm(label, color=None):
        return f'  [[figure.arm]]\n  label = "{label}"\n  runs = "x*"\n' + (f'  color = "{color}"\n' if color else "")

    m = manifest_with(tmp_path, '[[figure]]\nkind = "curves"\nname = "f1"\n' + arm("a") + arm("b")
                      + '[[figure]]\nkind = "curves"\nname = "f2"\n' + arm("c", "#000000") + arm("b") + arm("d"))
    assert assign_colors(m.figures) == {"a": PALETTE[0], "b": PALETTE[1], "c": "#000000", "d": PALETTE[2]}
    clash = manifest_with(tmp_path, '[[figure]]\nkind = "curves"\nname = "f1"\n' + arm("a", "#111111")
                          + '[[figure]]\nkind = "curves"\nname = "f2"\n' + arm("a", "#222222"))
    with pytest.raises(ValueError, match="two colors"):
        assign_colors(clash.figures)
    ten = "".join(f'[[figure]]\nkind = "curves"\nname = "f{k}"\n' + arm(f"l{k}") for k in range(10))
    with pytest.raises(ValueError, match="more than 9 distinct arm labels"):
        assign_colors(manifest_with(tmp_path, ten).figures)


def test_no_baselines_means_no_gray_lines(workspace):
    series = [Series("PPO", PALETTE[0], eval_curves(sorted(workspace.glob("keepaway_1v1_seed*"))))]
    base = read_table(workspace / "eval_keepaway_1v1_seed0.csv")

    def grays(fig):
        return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_color() == CONTEXT]

    without, with_ = fig_curves(series), fig_curves(series, base)
    assert grays(without) == [] and len(grays(with_)) == 4  # flee + trivial, in each of the 2 panels
    plt.close(without)
    plt.close(with_)


def test_sweep_points_are_sorted_and_aggregated_per_point(workspace):
    m = load_manifest(workspace / "manifest.toml")
    points, flee = sweep_points(next(f for f in m.figures if f.kind == "sweep").points)
    assert [p.x for p in points] == [0.0, 4.0, 16.0] and [p.name for p in points] == ["0 (survival only)", "4", "16"]
    assert "approach_m_per_step" in flee

    def row(v):
        return {"approach_m_per_step": v, "approach_alignment": 0.1, "pickup_rate": 0.5,
                "len_mean_possession": 130.0, "len_mean_free_ball": 95.0, "env_steps": 200_000.0}

    t = sweep_table([SweepPoint(4.0, None, [row(0.02)]), SweepPoint(0.0, "none", [row(0.01), row(0.03)])])
    assert list(t) == ["none", "4"]  # sorted by x
    assert t["none"]["n_seeds"] == 2 and t["none"]["approach_m_per_step"] == pytest.approx(0.02)
    assert t["none"]["approach_m_per_step_seed_std"] == pytest.approx(np.std([0.01, 0.03], ddof=1))
    assert math.isnan(t["4"]["approach_m_per_step_seed_std"])  # one seed: no spread


def test_sweep_files_need_ppo_and_flee_rows(tmp_path):
    write_csv(tmp_path / "x_per_run.csv", [{"policy": "random", "pbrs_coef": 0.0, "run": ""}])
    m = manifest_with(tmp_path, '[[figure]]\nkind = "sweep"\nname = "s"\n[[figure.point]]\nx = 0\n'
                                'table = "x_per_run.csv"\n')
    with pytest.raises(ValueError, match="no PPO rows"):
        sweep_points(m.figures[0].points)


def test_main_renders_a_manifest_into_runs_plots_by_default(workspace, monkeypatch, capsys):
    monkeypatch.chdir(workspace)
    assert main(["--manifest", "manifest.toml"]) == 0
    assert (workspace / "runs" / "plots" / "test" / "pbrs_coef_sweep.png").exists()
    assert "[plot]" in capsys.readouterr().out


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


def test_table_markdown_formats_per_column():
    row = {"n_seeds": 2, "x": 0.01234, "x_seed_std": 0.001, "y": 3.0}
    md = table_markdown("T", {"a": row}, (("x", "X", ".4f"), ("y", "Y")), first="arm")
    assert "| arm | seeds | X | Y |" in md and "| a | 2 | 0.0123 ± 0.0010 | 3.00 |" in md


def test_close_line_ends_get_labels_on_opposite_sides():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    _end_labels(ax, [(1.0, 50.0, "low"), (1.0, 52.0, "high"), (1.0, 90.0, "alone")], refs=[88.0])
    va = {t.get_text(): t.get_va() for t in ax.texts}
    plt.close(fig)
    # the close pair splits (higher above); the lone end steps away from the reference just below it
    assert va == {"low": "top", "high": "bottom", "alone": "bottom"}


def test_three_close_ends_stack_and_early_ends_label_to_the_right():
    # Found in the task 2 demo: arms of different budgets end at different x, and three
    # finals near 94 overlapped; early-ending labels ran left over the y axis.
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 100)
    _end_labels(ax, [(1.0, 94.0, "long"), (0.1, 94.2, "short a"), (0.1, 93.9, "short b"), (1.0, 93.8, "long b")],
                refs=[])
    texts = {t.get_text(): t for t in ax.texts}
    offsets = {name: t.xyann[1] for name, t in texts.items()}
    plt.close(fig)
    # The right half keeps the split (higher above, lower below); the left half, where the
    # reference-line labels sit below their lines, stacks upward from the lowest end.
    assert offsets["long"] > 0 > offsets["long b"]
    assert offsets["short a"] > offsets["short b"] > 0
    assert texts["long"].get_ha() == "right" and texts["short a"].get_ha() == "left"


def test_single_seed_arms_show_no_spread_and_legends_count_seeds_correctly(tmp_path):
    one = [Series("solo", PALETTE[0], eval_curves([fake_run(tmp_path, "solo_seed0", 0)]))]
    fig = fig_curves(one)
    labels = [t.get_text() for t in fig.axes[0].texts]
    legend = [t.get_text() for t in fig.legends[0].texts]
    plt.close(fig)
    assert any(lb.startswith("solo ") and "±" not in lb for lb in labels)
    assert legend == ["solo: mean over 1 seed"]
