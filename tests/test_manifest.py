# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: tests for the record manifest
#     loader (strict validation, relative paths, frozen vs unfrozen inputs)
#     for backlog task 2, step 2.
# Prompt summary: Implement step 2 of the task 2 plotting-refactor plan
#   (marl/manifest.py), tests first.
# Human review:   Pending review
# ==============================================================================
import os
from pathlib import Path

import pytest

from marl.manifest import MAX_ARMS, load_manifest

FULL = """
name        = "golden"
description = "a test record"

[[figure]]
kind      = "curves"
name      = "pbrs_learning_curves"
title     = "survival-only vs + PBRS"
baselines = "survival only"
ylim      = [50, 175]
  [[figure.arm]]
  label = "survival only"
  runs  = "data/Old_mock_env/keepaway_1v1_seed*"
  [[figure.arm]]
  label = "+ PBRS"
  runs  = "../../runs/keepaway_1v1_pbrs_seed*"
  table = "../../runs/eval_keepaway_1v1_pbrs_seed0.csv"
  color = "#d62728"

[[figure]]
kind   = "diagnostics"
name   = "keepaway_training_diagnostics"
window = 9
  [[figure.arm]]
  label = "survival only"
  runs  = "data/Old_mock_env/keepaway_1v1_seed*"

[[figure]]
kind      = "sweep"
name      = "pbrs_coef_sweep"
x_label   = "PBRS coefficient"
highlight = 4
  [[figure.point]]
  x     = 0
  label = "0 (survival only)"
  table = "data/Old_mock_env/eval_c0_per_run.csv"
  [[figure.point]]
  x     = 4
  table = "data/Old_mock_env/eval_c4_per_run.csv"

[[figure]]
kind   = "calibration"
name   = "calibration_red_speed"
csv    = "data/Old_mock_env/calibration_keepaway_1v1.csv"
config = "data/Old_mock_env/keepaway_1v1_seed0/config.json"

[[table]]
title   = "Keep-away ({n_runs} runs)"
source  = "data/Old_mock_env/eval_keepaway_1v1_seed0.csv"
columns = "keepaway"

[[table]]
title   = "PBRS coefficient pilot"
sweep   = "pbrs_coef_sweep"
columns = "sweep"
first   = "PBRS coef"
"""


def write(tmp_path, text, record="golden"):
    d = tmp_path / "results" / record
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.toml"
    path.write_text(text, encoding="utf-8")
    return path


def arms_toml(n):
    return "".join(f'\n  [[figure.arm]]\n  label = "arm {k}"\n  runs = "data/a{k}_seed*"\n' for k in range(n))


def test_a_full_manifest_loads_with_paths_resolved_against_its_own_folder(tmp_path):
    path = write(tmp_path, FULL)
    m = load_manifest(path)
    root = path.parent
    assert m.name == "golden" and m.root == root
    assert [f.kind for f in m.figures] == ["curves", "diagnostics", "sweep", "calibration"]
    curves = m.figures[0]
    assert curves.baselines == "survival only" and curves.ylim == (50.0, 175.0)
    first, second = curves.arms
    assert first.runs.raw == "data/Old_mock_env/keepaway_1v1_seed*"
    assert first.runs.path == os.path.abspath(root / "data/Old_mock_env/keepaway_1v1_seed*")
    assert first.runs.frozen and not second.runs.frozen and not second.table.frozen
    assert second.runs.path == os.path.abspath(tmp_path / "runs/keepaway_1v1_pbrs_seed*")
    assert second.color == "#d62728" and first.color is None
    arm = second.to_arm()
    assert arm.label == "+ PBRS" and arm.runs == second.runs.path and arm.table == Path(second.table.path)
    assert m.figures[1].window == 9
    sweep = m.figures[2]
    assert [(p.x, p.label) for p in sweep.points] == [(0.0, "0 (survival only)"), (4.0, None)]
    assert sweep.highlight == 4.0 and sweep.x_label == "PBRS coefficient"
    assert m.figures[3].config.raw.endswith("config.json")
    assert [(t.source is not None, t.sweep, t.first) for t in m.tables] == [(True, None, None),
                                                                         (False, "pbrs_coef_sweep", "PBRS coef")]


def test_unfrozen_lists_the_paths_outside_the_record_in_manifest_order(tmp_path):
    m = load_manifest(write(tmp_path, FULL))
    assert [p.raw for p in m.unfrozen()] == ["../../runs/keepaway_1v1_pbrs_seed*",
                                             "../../runs/eval_keepaway_1v1_pbrs_seed0.csv"]
    assert len(m.paths()) == 9  # 3 arm runs + 1 arm table + 2 point tables + csv + config + 1 [[table]] source
    assert all(p.frozen for p in m.paths() if p.raw.startswith("data/"))


def test_a_table_only_manifest_is_valid(tmp_path):
    m = load_manifest(write(tmp_path, 'name = "silver"\n[[table]]\ntitle = "t"\nsource = "data/e.csv"\n'
                                      'columns = "goto"\n'))
    assert m.figures == () and len(m.tables) == 1


@pytest.mark.parametrize("text, match", [
    ('name = "x"\ncolour = "red"\n', r"unknown key\(s\) \['colour'\]; allowed"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\npoints = 3\n' + arms_toml(1), r"unknown key\(s\) \['points'\]"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n[[figure.arm]]\nlabel = "a"\nruns = "data/a*"\ncolr = "r"\n',
     r"unknown key\(s\) \['colr'\]"),
    ('name = "x"\n[[figure]]\nkind = "histogram"\nname = "f"\n', r"kind 'histogram'.*curves"),
    ('name = "x"\n[[table]]\ntitle = "t"\nsource = "data/e.csv"\ncolumns = "wide"\n', r"columns 'wide'"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n' + arms_toml(MAX_ARMS + 1), r"10 arms.*at most 9"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n[[figure.arm]]\nlabel = "a"\nruns = "data/a*"\n'
     '[[figure.arm]]\nlabel = "a"\nruns = "data/b*"\n', r"duplicate arm label 'a'"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\nbaselines = "nope"\n' + arms_toml(1),
     r"baselines 'nope' names no arm"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n[[figure.arm]]\nlabel = "a"\nruns = "data\\\\a*"\n',
     r"forward slashes"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n[[figure.arm]]\nlabel = "a"\n', r"missing key\(s\) \['runs'\]"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n', r"at least one arm"),
    ('name = "x"\n[[figure]]\nkind = "calibration"\nname = "c"\ncsv = "data/c.csv"\n', r"missing key\(s\) \['config'\]"),
    ('name = "x"\n[[figure]]\nkind = "diagnostics"\nname = "f"\n' + arms_toml(1) + '[[figure]]\nkind = "final"\n'
     'name = "f"\n' + arms_toml(1), r"duplicate figure name 'f'"),
    ('name = "x"\n[[table]]\ntitle = "t"\ncolumns = "sweep"\nsweep = "missing"\n', r"sweep 'missing' names no sweep figure"),
    ('name = "x"\n[[table]]\ntitle = "t"\ncolumns = "keepaway"\n', r"exactly one of 'source' or 'sweep'"),
    ('name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\nylim = [1]\n' + arms_toml(1), r"ylim.*two numbers"),
    ('name = "x"\n[[figure]]\nkind = "sweep"\nname = "s"\n[[figure.point]]\nx = "four"\ntable = "data/t.csv"\n',
     r"'x' must be a number"),
    ('description = "no name"\n', r"missing key\(s\) \['name'\]"),
])
def test_invalid_manifests_are_rejected_with_a_specific_message(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_manifest(write(tmp_path, text))


def test_nine_arms_are_allowed(tmp_path):
    m = load_manifest(write(tmp_path, 'name = "x"\n[[figure]]\nkind = "curves"\nname = "f"\n' + arms_toml(MAX_ARMS)))
    assert len(m.figures[0].arms) == MAX_ARMS


def test_errors_name_the_manifest_file(tmp_path):
    path = write(tmp_path, 'name = "x"\nbogus = 1\n')
    with pytest.raises(ValueError, match="manifest.toml"):
        load_manifest(path)
