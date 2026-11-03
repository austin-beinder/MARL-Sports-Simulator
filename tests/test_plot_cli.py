# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: tests for scripts/plot.py's
#     command line (ad-hoc --arm plots, default kinds per scenario, the
#     --overwrite guard and its record-render exemption, the eval-mismatch
#     and unfrozen-input warnings) for backlog task 2, step 4.
#   - 2026-11-01: Added a test for the demo-found bug: a record render must
#     not clear ad-hoc output that shares its folder (task 2, step 7).
# Prompt summary: Implement step 4 of the task 2 plotting-refactor plan (the
#   plot command line), tests first.
# Human review:   Pending review
# ==============================================================================
import json

import pytest

from marl.plots import main
from tests.test_plots import fake_run, fake_table
from marl.results import table_path_for


def with_config(run_dir, scenario="keepaway", eval_seed=7):
    cfg = {"shared": {"scenario": scenario}, "rl": {}, "run": {"eval_seed": eval_seed, "eval_episodes": 6}}
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return run_dir


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """runs/new_seed{0,1} (evaluated), runs/old_seed{0,1} (another eval_seed), runs/goto_seed0."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    new = [with_config(fake_run(root, f"new_seed{k}", k)) for k in range(2)]
    old = [with_config(fake_run(root, f"old_seed{k}", 5 + k), eval_seed=8) for k in range(2)]
    goto = [with_config(fake_run(root, "goto_seed0", 9, "goto_target"), "goto_target")]
    fake_table(table_path_for(new), new, "keepaway")
    fake_table(table_path_for(old), old, "keepaway")
    fake_table(table_path_for(goto), goto, "goto_target")
    return tmp_path


def pngs(folder):
    return sorted(p.name for p in folder.glob("*.png"))


def test_ad_hoc_arms_make_the_default_keepaway_kinds(runs):
    assert main(["--arm", "new reward=runs/new_seed*", "--name", "nr"]) == 0
    assert pngs(runs / "runs/plots/nr") == ["nr_curves.png", "nr_diagnostics.png", "nr_final.png"]


def test_goto_arms_default_to_the_goto_kinds_and_name_defaults_to_adhoc(runs):
    assert main(["--arm", "g=runs/goto_seed*"]) == 0
    assert pngs(runs / "runs/plots/adhoc") == ["adhoc_diagnostics.png", "adhoc_goto.png"]


def test_explicit_kinds_title_and_baselines(runs, capsys):
    assert main(["--arm", "a=runs/new_seed*", "--arm", "b=runs/old_seed*", "--kind", "curves", "--baselines", "a",
                 "--name", "cmp", "--title", "A vs B"]) == 0
    assert pngs(runs / "runs/plots/cmp") == ["cmp_curves.png"]


def test_the_overwrite_guard_refuses_a_non_empty_output_folder(runs, capsys):
    args = ["--arm", "a=runs/new_seed*", "--kind", "diagnostics", "--name", "g"]
    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert main([*args, "--overwrite"]) == 0


def test_mismatched_eval_episodes_warn_but_still_plot(runs, capsys):
    assert main(["--arm", "a=runs/new_seed*", "--arm", "b=runs/old_seed*", "--kind", "curves", "--name", "w"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "'b'" in out and "eval_seed" in out
    assert (runs / "runs/plots/w/w_curves.png").exists()


def test_bad_arm_syntax_and_mixing_modes_are_usage_errors(runs):
    with pytest.raises(SystemExit):
        main(["--arm", "no-equals-sign"])
    with pytest.raises(SystemExit):
        main(["--manifest", "m.toml", "--arm", "a=runs/new_seed*"])


def test_ad_hoc_arms_get_the_manifest_rules(runs):
    with pytest.raises(ValueError, match="baselines 'zzz' names no arm"):
        main(["--arm", "a=runs/new_seed*", "--baselines", "zzz"])
    assert main(["--arm", "a=runs\\new_seed*", "--kind", "diagnostics", "--name", "bs"]) == 0  # backslashes normalized


RECORD = """
name = "rec"
[[figure]]
kind = "diagnostics"
name = "diag"
  [[figure.arm]]
  label = "frozen"
  runs  = "data/new_seed*"
  [[figure.arm]]
  label = "work in progress"
  runs  = "../../runs/old_seed*"
"""


def make_record(root):
    rec = root / "results" / "rec"
    for k in range(2):
        fake_run(rec / "data", f"new_seed{k}", k)
    (rec / "manifest.toml").write_text(RECORD, encoding="utf-8")
    return rec / "manifest.toml"


def test_record_renders_overwrite_their_default_folder_and_warn_about_unfrozen_inputs(runs, capsys):
    manifest = make_record(runs)
    assert main(["--manifest", str(manifest)]) == 0
    assert main(["--manifest", str(manifest)]) == 0  # exempt from the guard: pure re-derivation
    out = capsys.readouterr().out
    assert "1 input not frozen yet" in out and "scripts/freeze.py" in out
    assert pngs(runs / "runs/plots/rec") == ["diag.png"]


def test_a_record_render_never_clears_another_render_in_its_default_folder(runs, capsys):
    # Found in the task 2 demo: ad-hoc `--name rec` and the record "rec" share runs/plots/rec/,
    # and the record render (exempt from the guard) deleted the ad-hoc figures.
    assert main(["--arm", "a=runs/new_seed*", "--kind", "diagnostics", "--name", "rec"]) == 0
    manifest = make_record(runs)
    assert main(["--manifest", str(manifest)]) == 1  # the folder holds someone else's figures
    assert (runs / "runs/plots/rec/rec_diagnostics.png").exists()
    assert "--overwrite" in capsys.readouterr().err
    assert main(["--manifest", str(manifest), "--overwrite"]) == 0
    assert main(["--manifest", str(manifest)]) == 0  # now the folder is this record's own render: exempt


def test_a_record_rendered_to_an_explicit_folder_is_guarded(runs, capsys):
    manifest = make_record(runs)
    assert main(["--manifest", str(manifest), "--out", "elsewhere"]) == 0
    assert main(["--manifest", str(manifest), "--out", "elsewhere"]) == 1


def test_a_missing_manifest_is_a_clear_error(runs, capsys):
    assert main([]) == 1  # no results/golden/ yet
    assert "no manifest" in capsys.readouterr().err


def test_plain_plot_renders_golden(runs, capsys):
    golden = runs / "results" / "golden"
    for k in range(2):
        fake_run(golden / "data", f"new_seed{k}", k)
    (golden / "manifest.toml").write_text('name = "golden"\n[[figure]]\nkind = "diagnostics"\nname = "diag"\n'
                                          '[[figure.arm]]\nlabel = "PPO"\nruns = "data/new_seed*"\n', encoding="utf-8")
    assert main([]) == 0
    assert pngs(runs / "runs/plots/golden") == ["diag.png"]
