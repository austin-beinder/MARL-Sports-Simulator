# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: tests for freezing a record in
#     place (manifest-driven copy mirrored under data/, only the unfrozen path
#     strings rewritten, provenance, validation before any write, the
#     --overwrite guard) for backlog task 2, step 6.
# Prompt summary: Implement step 6 of the task 2 plotting-refactor plan
#   (marl/record.py + scripts/freeze.py), tests first.
# Human review:   Pending review
# ==============================================================================
import json

import pytest

from marl.manifest import load_manifest
from marl.record import freeze, main
from marl.results import table_path_for
from tests.test_plots import fake_run, fake_table

MANIFEST = """# My golden record. This comment must survive freezing.
name = "rec"

[[figure]]
kind      = "curves"
name      = "cmp"
baselines = "new"
  [[figure.arm]]
  label = "old"            # already frozen
  runs  = "data/old/k_seed*"
  [[figure.arm]]
  label = "new"
  runs  = "../../runs/exp/k_seed*"

[[figure]]
kind = "diagnostics"
name = "diag"
  [[figure.arm]]
  label = "new"
  runs  = '../../runs/exp/k_seed*'

[[figure]]
kind   = "calibration"
name   = "calib"
csv    = "../../runs/calibration.csv"
config = "../../runs/exp/k_seed0/config.json"
"""


@pytest.fixture
def ws(tmp_path):
    runs = tmp_path / "runs"
    new = [fake_run(runs / "exp", f"k_seed{k}", k) for k in range(2)]
    fake_table(table_path_for(new), new, "keepaway")
    for d in new:
        (d / "checkpoints").mkdir()
        (d / "checkpoints" / "final.pt").write_bytes(b"weights")
        (d / "train.log").write_text("console output", encoding="utf-8")
    cfg = {"shared": {"scenario": "keepaway", "red_max_speed": 1.8, "max_speed": 2.0, "carry_speed_factor": 0.75},
           "run": {"eval_seed": 7, "eval_episodes": 6}}
    (new[0] / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (runs / "calibration.csv").write_text("red_max_speed,policy,survival\n1.8,flee,0.5\n1.8,random,0.3\n"
                                          "1.8,still,0.3\n1.9,flee,0.4\n1.9,random,0.3\n1.9,still,0.3\n",
                                          encoding="utf-8")
    rec = tmp_path / "results" / "rec"
    fake_run(rec / "data" / "old", "k_seed0", 9)
    (rec / "manifest.toml").write_text(MANIFEST, encoding="utf-8")
    return tmp_path


def files_under(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def do_freeze(ws, **kw):
    return freeze(ws / "results" / "rec" / "manifest.toml", runs_root=ws / "runs", revision="abc1234", **kw)


def test_freeze_copies_only_what_the_manifest_reads_mirrored_under_data(ws):
    do_freeze(ws)
    assert files_under(ws / "results" / "rec" / "data") == [
        "calibration.csv",
        "exp/eval_k_seed0.csv", "exp/eval_k_seed0_per_run.csv",  # the arm's table (by convention) + its twin
        "exp/k_seed0/config.json", "exp/k_seed0/eval.csv", "exp/k_seed0/train.csv",
        "exp/k_seed1/eval.csv", "exp/k_seed1/train.csv",
        "old/k_seed0/eval.csv", "old/k_seed0/train.csv",  # already frozen: untouched
    ]  # never checkpoints/ or *.log


def test_freeze_rewrites_only_the_unfrozen_strings_and_appends_provenance(ws):
    do_freeze(ws)
    text = (ws / "results" / "rec" / "manifest.toml").read_text(encoding="utf-8")
    assert text.startswith("# My golden record. This comment must survive freezing.")
    assert '"data/old/k_seed*"' in text and "# already frozen" in text
    assert '  runs  = "data/exp/k_seed*"' in text and "  runs  = 'data/exp/k_seed*'" in text  # both quote styles
    assert 'csv    = "data/calibration.csv"' in text
    assert "frozen 20" in text and "abc1234" in text
    assert "#   ../../runs/exp/k_seed* -> data/exp/k_seed*" in text
    assert load_manifest(ws / "results" / "rec" / "manifest.toml").unfrozen() == []


def test_freeze_renders_the_record_figures(ws):
    result = do_freeze(ws)
    figures = ws / "results" / "rec" / "figures"
    assert sorted(p.name for p in figures.glob("*.png")) == ["calib.png", "cmp.png", "diag.png"]
    assert result.notes == []


def test_a_second_freeze_with_nothing_unfrozen_only_re_renders(ws):
    do_freeze(ws)
    path = ws / "results" / "rec" / "manifest.toml"
    before = path.read_text(encoding="utf-8")
    result = do_freeze(ws)
    assert result.copied == [] and path.read_text(encoding="utf-8") == before


def test_an_existing_destination_needs_overwrite_and_nothing_changes_without_it(ws):
    path = ws / "results" / "rec" / "manifest.toml"
    do_freeze(ws)
    path.write_text(MANIFEST, encoding="utf-8")  # point at runs/ again, e.g. after re-running those seeds
    before = files_under(ws / "results" / "rec")
    with pytest.raises(ValueError, match="already exist.*--overwrite"):
        do_freeze(ws)
    assert path.read_text(encoding="utf-8") == MANIFEST and files_under(ws / "results" / "rec") == before
    do_freeze(ws, overwrite=True)
    assert load_manifest(path).unfrozen() == []


@pytest.mark.parametrize("bad, match", [
    ("../../runs/exp/nothing_seed*", "matches no run directories"),
    ("../../elsewhere/k_seed*", "outside"),
])
def test_bad_inputs_fail_before_anything_is_copied(ws, bad, match):
    path = ws / "results" / "rec" / "manifest.toml"
    text = MANIFEST.replace('"../../runs/exp/k_seed*"', f'"{bad}"')
    path.write_text(text, encoding="utf-8")
    (ws / "elsewhere" / "k_seed0").mkdir(parents=True)
    with pytest.raises(ValueError, match=match):
        do_freeze(ws)
    assert not (ws / "results" / "rec" / "data" / "exp").exists() and path.read_text(encoding="utf-8") == text


def test_a_run_without_its_csvs_is_refused(ws):
    (ws / "runs" / "exp" / "k_seed1" / "train.csv").unlink()
    with pytest.raises(ValueError, match="train.csv"):
        do_freeze(ws)


def test_a_missing_arm_table_is_only_a_note(ws):
    for p in (ws / "runs" / "exp").glob("eval_*.csv"):
        p.unlink()
    result = do_freeze(ws)
    assert any("no evaluation table" in n for n in result.notes)
    assert any(n.startswith("skipped cmp") for n in result.notes)  # the figure that needs it says so too


def test_cli_takes_a_record_name(ws, monkeypatch, capsys):
    monkeypatch.chdir(ws)
    assert main(["rec", "--runs-root", "runs"]) == 0
    out = capsys.readouterr().out
    assert "[freeze]" in out and "copied" in out
    assert load_manifest(ws / "results" / "rec" / "manifest.toml").unfrozen() == []
