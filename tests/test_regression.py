# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: tests for the regression
#     harness's suite loader (strict validation, the reserved seed schema,
#     case and run naming, the per-run --set overrides) for backlog task 4,
#     step 1.
#   - 2026-11-01: Added the comparison tests: CSV cell/row/column diffs with
#     timing columns ignored, tolerances, config.json and fingerprint diffs,
#     the per-case roll-up, stale runs and the report text (task 4, step 2).
#   - 2026-11-01: Added the freeze tests: the weight fingerprint, the
#     environment stamp, and --update's record writer (timing columns
#     stripped, stamp history, stale entries removed) (task 4, step 3).
#   - 2026-11-01: Added the runner and CLI tests: case selection, uncovered
#     configs, the job commands (fake runner), failures, --no-run/--only/
#     --rtol/--update rules, exit codes, and one end-to-end run through the
#     real train.py/evaluate.py subprocesses (task 4, step 4).
# Prompt summary: Implement steps 1-5 of the task 4 regression-harness plan
#   (marl/regression.py), tests first.
# Human review:   Pending review
# ==============================================================================
import json
from pathlib import Path

import pytest
import torch

from marl.common.config import config_from_dict
from marl.regression import (
    CHANGED, IDENTICAL, NEW, NOTE, SCHEMA, Case, Tolerance, case_files, compare_case, compare_csv,
    compare_fingerprint, compare_json, default_parallel_runs, environment_diffs, environment_stamp, format_report,
    freeze_silver, load_suite, run_suite, select_cases, stale_entries, uncovered_configs, weights_fingerprint,
)
from marl.regression import main as regress_main
from marl.rl.checkpoint import save_checkpoint
from marl.rl.networks import seeded_networks
from marl.rl.ppo import PPOUpdater
from marl.rl.rollout import Featurizer

SUITE = """
name      = "silver"
seeds     = [90, 91]
eval_seed = 20261101

[budget]
total_steps   = 6144
eval_every    = 1
eval_episodes = 10

[[case]]
config = "configs/keep.toml"
envs   = ["mock", "pybullet"]

[[case]]
config = "configs/goto.toml"
envs   = ["mock"]
"""


@pytest.fixture
def repo(tmp_path):
    """A fake repo root with two configs; the suite file lives in results/silver/."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "keep.toml").write_text("[run]\ntotal_steps = 1_000_000\n", encoding="utf-8")
    (tmp_path / "configs" / "goto.toml").write_text(
        '[shared]\nscenario = "goto_target"\nn_blue = 1\nn_red = 0\n[rl.reward]\nbase = "goto_target"\n',
        encoding="utf-8",
    )
    (tmp_path / "results" / "silver").mkdir(parents=True)
    return tmp_path


def write(repo, text):
    path = repo / "results" / "silver" / "suite.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load(repo, text=SUITE):
    return load_suite(write(repo, text), repo_root=repo)


def test_loads_seeds_budget_and_cases(repo):
    s = load(repo)
    assert s.name == "silver" and s.seeds == (90, 91) and s.eval_seed == 20261101
    assert s.budget == {"total_steps": 6144, "eval_every": 1, "eval_episodes": 10}
    assert [c.name for c in s.cases] == ["keep_mock", "keep_pybullet", "goto_mock"]
    assert s.cases[0].config == repo / "configs" / "keep.toml"
    assert s.cases[1].env == "pybullet"
    assert s.cases[0].run_names(s.seeds) == ["keep_mock_seed90", "keep_mock_seed91"]


def test_set_args_apply_the_budget_and_eval_seed(repo):
    s = load(repo)
    args = s.set_args()
    assert args == ["run.eval_seed=20261101", "run.total_steps=6144", "run.eval_every=1", "run.eval_episodes=10"]
    cfg = s.case_config(s.cases[1], seed=91)
    assert (cfg.run.seed, cfg.run.env, cfg.run.eval_seed) == (91, "pybullet", 20261101)
    assert (cfg.run.total_steps, cfg.run.eval_every, cfg.run.eval_episodes) == (6144, 1, 10)


@pytest.mark.parametrize("old, new, match", [
    ('name      = "silver"\n', 'name = "silver"\ncolor = "grey"\n', r"unknown key\(s\) \['color'\]"),
    ('eval_seed = 20261101\n', "", r"missing key\(s\) \['eval_seed'\]"),
    ("seeds     = [90, 91]", "seeds = []", "seeds.*non-empty"),
    ("seeds     = [90, 91]", "seeds = [90, 90]", "seeds.*duplicate"),
    ("seeds     = [90, 91]", "seeds = [90, true]", "seeds.*whole numbers"),
    ("seeds     = [90, 91]", "seeds = [3, 90]", r"seeds.*90-99.*\[3\]"),
    ("eval_seed = 20261101", "eval_seed = 20261019", "eval_seed.*for-score"),
    ("eval_seed = 20261101", "eval_seed = -1", "eval_seed.*>= 0"),
    ("eval_every    = 1\n", "", r"budget: missing key\(s\) \['eval_every'\]"),
    ("eval_every    = 1\n", "eval_every = 1\nlr = 0.1\n", r"budget: unknown key\(s\) \['lr'\]"),
    ("eval_every    = 1", "eval_every = 1.5", "budget.eval_every.*whole number"),
    ("total_steps   = 6144", "total_steps = 100", "keep_mock.*total_steps must be >= rl.ppo.rollout_steps"),
    ('envs   = ["mock"]', "envs = []", "case\\[1\\].*envs.*non-empty"),
    ('envs   = ["mock"]', 'envs = ["mock", "mock"]', "case\\[1\\].*duplicate env 'mock'"),
    ('envs   = ["mock"]', 'envs = ["unity"]', "case\\[1\\].*env 'unity'.*\\['mock', 'pybullet'\\]"),
    ('config = "configs/goto.toml"', 'config = "configs/nope.toml"', "case\\[1\\].*no such config"),
    ('config = "configs/goto.toml"', 'config = "configs\\\\goto.toml"', "case\\[1\\].*forward slashes"),
    ('config = "configs/goto.toml"', 'config = "configs/keep.toml"', "duplicate case 'keep_mock'"),
])
def test_rejects_bad_suites(repo, old, new, match):
    assert old in SUITE
    with pytest.raises(ValueError, match=match):
        load(repo, SUITE.replace(old, new))


def test_errors_name_the_file(repo):
    with pytest.raises(ValueError, match=r"suite\.toml: .*eval_seed"):
        load(repo, SUITE.replace("eval_seed = 20261101", "eval_seed = 20261019"))


def test_needs_at_least_one_case(repo):
    head = SUITE.split("[[case]]")[0]
    with pytest.raises(ValueError, match=r"at least one case \(\[\[case\]\]\)"):
        load(repo, head)


def test_two_configs_with_one_name_would_share_run_dirs(repo):
    (repo / "configs" / "other").mkdir()
    (repo / "configs" / "other" / "keep.toml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case 'keep_mock'"):
        load(repo, SUITE + '\n[[case]]\nconfig = "configs/other/keep.toml"\nenvs = ["mock"]\n')


# --- Step 2: comparing a new run against the record -------------------------------------
EXACT = Tolerance()


def csv_file(path, header, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]) + "\n", encoding="utf-8")
    return path


TRAIN_H = ["iteration", "policy_loss", "sps", "wall_time_s"]


def test_csv_identical_ignores_timing_columns(tmp_path):
    new = csv_file(tmp_path / "new.csv", TRAIN_H, (1, 0.5, 900.1, 2.2), (2, 0.25, 870.3, 4.5))
    old = csv_file(tmp_path / "old.csv", ["iteration", "policy_loss"], (1, 0.5), (2, 0.25))  # frozen: stripped
    d = compare_csv(new, old, EXACT)
    assert d.status == IDENTICAL and d.n_cells == 0


def test_csv_changed_cell_reports_the_first_difference_and_largest_delta(tmp_path):
    old = csv_file(tmp_path / "old.csv", ["iteration", "policy_loss", "kl"], (1, 0.5, 0.01), (2, 0.25, 0.02))
    new = csv_file(tmp_path / "new.csv", ["iteration", "policy_loss", "kl"], (1, 0.5, 0.01), (2, 0.35, 0.03))
    d = compare_csv(new, old, EXACT)
    assert d.status == CHANGED and d.n_cells == 2
    assert d.max_abs == pytest.approx(0.1) and d.max_rel == pytest.approx(1 / 3)
    assert "2 of 6 cells differ" in d.detail
    assert "first: row 2 (iteration=2) policy_loss 0.25 -> 0.35" in d.detail


def test_csv_row_count_change_is_a_change(tmp_path):
    old = csv_file(tmp_path / "old.csv", ["iteration", "x"], (1, 1), (2, 2))
    new = csv_file(tmp_path / "new.csv", ["iteration", "x"], (1, 1), (2, 2), (3, 3))
    d = compare_csv(new, old, EXACT)
    assert d.status == CHANGED and "rows 2 -> 3" in d.detail


def test_csv_column_added_or_removed_alone_is_schema(tmp_path):
    old = csv_file(tmp_path / "old.csv", ["iteration", "x", "gone"], (1, 1, 7))
    new = csv_file(tmp_path / "new.csv", ["iteration", "x", "extra"], (1, 1, 9))
    d = compare_csv(new, old, EXACT)
    assert d.status == SCHEMA and "+columns ['extra']" in d.detail and "-columns ['gone']" in d.detail


@pytest.mark.parametrize("a, b, exact, loose", [
    ("1.0", "1.0000000001", CHANGED, IDENTICAL),  # a last-digits drift: only a tolerance forgives it
    ("nan", "nan", IDENTICAL, IDENTICAL),
    ("nan", "1.0", CHANGED, CHANGED),
    ("stochastic", "deterministic", CHANGED, CHANGED),  # text is always compared exactly
    ("1.0", "1.1", CHANGED, CHANGED),
])
def test_tolerance(tmp_path, a, b, exact, loose):
    old = csv_file(tmp_path / "old.csv", ["k", "v"], ("r", a))
    new = csv_file(tmp_path / "new.csv", ["k", "v"], ("r", b))
    assert compare_csv(new, old, EXACT).status == exact
    assert compare_csv(new, old, Tolerance(rtol=1e-6)).status == loose


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_config_diff_names_the_dotted_key(tmp_path):
    old = json_file(tmp_path / "old.json", {"rl": {"ppo": {"gae_lambda": 0.95, "lr": 3e-4}}})
    new = json_file(tmp_path / "new.json", {"rl": {"ppo": {"gae_lambda": 0.9, "lr": 3e-4}}})
    d = compare_json(new, old)
    assert d.status == CHANGED and "rl.ppo.gae_lambda 0.95 -> 0.9" in d.detail


def test_config_key_added_alone_is_schema(tmp_path):
    old = json_file(tmp_path / "old.json", {"run": {"seed": 90}})
    new = json_file(tmp_path / "new.json", {"run": {"seed": 90, "print_every": 1}})
    d = compare_json(new, old)
    assert d.status == SCHEMA and "+run.print_every=1" in d.detail


def test_fingerprint_is_exact_and_only_noted_under_a_tolerance(tmp_path):
    a = json_file(tmp_path / "a.json", {"sha256": "aa"})
    b = json_file(tmp_path / "b.json", {"sha256": "bb"})
    assert compare_fingerprint(a, json_file(tmp_path / "a2.json", {"sha256": "aa"}), EXACT).status == IDENTICAL
    assert compare_fingerprint(a, b, EXACT).status == CHANGED
    noted = compare_fingerprint(a, b, Tolerance(rtol=1e-6))
    assert noted.status == NOTE and "ignored under a tolerance" in noted.detail


# A fake case: what the runner (new) or the record (silver) holds for one config x env.
CASE = Case(Path("configs/keep.toml"), "mock")
SEEDS = (90, 91)


def fake_case(root, final_len=(100.0, 110.0), loss=0.5, case=CASE, timing=False):
    """``timing``: train.csv keeps the wall-clock columns, as a fresh run's does (a record's has none)."""
    runs = case.run_names(SEEDS)
    for run, length in zip(runs, final_len):
        json_file(root / run / "config.json", {"run": {"seed": int(run[-2:])}})
        csv_file(root / run / "train.csv", ["iteration", "policy_loss"] + (["sps", "wall_time_s"] if timing else []),
                 (1, loss) + ((812.5, 3.25) if timing else ()))
        csv_file(root / run / "eval.csv", ["iteration", "mode", "len_mean"],
                 (0, "stochastic", 50.0), (1, "stochastic", length), (1, "deterministic", length + 1))
        json_file(root / run / "fingerprint.json", {"sha256": "aa"})
    csv_file(root / f"eval_{runs[0]}.csv", ["policy", "len_mean"], ("ppo", sum(final_len) / 2))
    csv_file(root / f"eval_{runs[0]}_per_run.csv", ["policy", "run", "len_mean"],
             ("ppo", runs[0], final_len[0]), ("ppo", runs[1], final_len[1]))
    return root


def test_case_files_lists_every_compared_file():
    assert case_files(CASE, SEEDS) == [
        "keep_mock_seed90/config.json", "keep_mock_seed90/train.csv", "keep_mock_seed90/eval.csv",
        "keep_mock_seed90/fingerprint.json",
        "keep_mock_seed91/config.json", "keep_mock_seed91/train.csv", "keep_mock_seed91/eval.csv",
        "keep_mock_seed91/fingerprint.json",
        "eval_keep_mock_seed90.csv", "eval_keep_mock_seed90_per_run.csv",
    ]


def test_case_identical(tmp_path):
    r = compare_case(CASE, SEEDS, fake_case(tmp_path / "new"), fake_case(tmp_path / "silver"), EXACT)
    assert r.status == IDENTICAL and r.headline == (105.0, 105.0)
    assert all(f.status == IDENTICAL for f in r.files) and len(r.files) == 10


def test_case_changed_lists_the_changed_files(tmp_path):
    r = compare_case(CASE, SEEDS, fake_case(tmp_path / "new", final_len=(100.0, 120.0), loss=0.6),
                     fake_case(tmp_path / "silver"), EXACT)
    assert r.status == CHANGED and r.headline == (105.0, 110.0)
    changed = {f.file for f in r.files if f.status == CHANGED}
    assert "keep_mock_seed91/eval.csv" in changed and "keep_mock_seed90/train.csv" in changed
    assert "keep_mock_seed90/eval.csv" not in changed


def test_case_schema_only_and_noted_fingerprints_roll_up(tmp_path):
    new, silver = fake_case(tmp_path / "new"), fake_case(tmp_path / "silver")
    csv_file(new / "keep_mock_seed90" / "train.csv", ["iteration", "policy_loss", "extra"], (1, 0.5, 3))
    json_file(new / "keep_mock_seed91" / "fingerprint.json", {"sha256": "bb"})
    assert compare_case(CASE, SEEDS, new, silver, Tolerance(rtol=1e-9)).status == SCHEMA
    assert compare_case(CASE, SEEDS, new, silver, EXACT).status == CHANGED  # the hash counts when exact


def test_case_missing_from_the_record_is_new(tmp_path):
    r = compare_case(CASE, SEEDS, fake_case(tmp_path / "new"), tmp_path / "silver", EXACT)
    assert r.status == NEW and r.files == ()


def test_a_missing_file_is_a_change(tmp_path):
    new, silver = fake_case(tmp_path / "new"), fake_case(tmp_path / "silver")
    (new / "keep_mock_seed91" / "fingerprint.json").unlink()
    (silver / "eval_keep_mock_seed90_per_run.csv").unlink()
    r = compare_case(CASE, SEEDS, new, silver, EXACT)
    details = {f.file: f.detail for f in r.files if f.status == CHANGED}
    assert r.status == CHANGED
    assert details == {"keep_mock_seed91/fingerprint.json": "missing from the new run",
                       "eval_keep_mock_seed90_per_run.csv": "not in the record"}


def test_stale_entries_are_record_runs_and_tables_the_suite_no_longer_lists(tmp_path):
    silver = fake_case(tmp_path / "silver")
    (silver / "old_mock_seed90").mkdir()
    csv_file(silver / "eval_old_mock_seed90.csv", ["policy"], ("ppo",))
    assert stale_entries([CASE], SEEDS, silver) == ["eval_old_mock_seed90.csv", "old_mock_seed90"]
    assert stale_entries([CASE], SEEDS, tmp_path / "nothing") == []


def test_report_lists_statuses_config_first_and_a_summary(tmp_path):
    new, silver = fake_case(tmp_path / "new", loss=0.6), fake_case(tmp_path / "silver")
    # seed91's config: in file order it comes after seed90's train.csv, so only the sort puts it first
    json_file(new / "keep_mock_seed91" / "config.json", {"run": {"seed": 91, "lr": 1}})
    changed = compare_case(CASE, SEEDS, new, silver, EXACT)
    fresh = compare_case(Case(Path("configs/keep.toml"), "pybullet"), SEEDS, tmp_path / "x", tmp_path / "y", EXACT)
    text = format_report([changed, fresh], EXACT, env_diffs=[("torch", "2.14.1+cpu", "2.15.0+cpu")],
                         stale=["old_mock_seed90"])
    lines = text.splitlines()
    assert any(l.startswith("CHANGED") and "keep_mock" in l and "final len_mean 105 -> 105" in l for l in lines)
    assert any(l.startswith("NEW") and "keep_pybullet" in l for l in lines)
    assert any(l.startswith("STALE") and "old_mock_seed90" in l for l in lines)
    config_at = next(i for i, l in enumerate(lines) if "config.json" in l)
    train_at = next(i for i, l in enumerate(lines) if "train.csv" in l)
    assert config_at < train_at  # a config diff usually explains the rest, so it comes first
    assert "deltas may come from the machine, not the code" in text and "torch: 2.14.1+cpu -> 2.15.0+cpu" in text
    assert "Comparison: exact" in text
    assert lines[-1] == "Summary: 1 changed, 1 new, 1 stale"


# --- Step 3: fingerprint, environment stamp, freezing -----------------------------------
def checkpoint(path, seed=90, iteration=3, nudge=0.0):
    cfg = config_from_dict({"run": {"seed": seed, "env": "mock"}})
    f = Featurizer(cfg)
    actor, critic = seeded_networks(cfg, f.obs_dim, f.state_dim)
    with torch.no_grad():
        next(iter(critic.parameters())).view(-1)[0] += nudge
    save_checkpoint(path, actor=actor, critic=critic, updater=PPOUpdater(actor, critic, cfg.rl.ppo),
                    iteration=iteration, env_steps=iteration * 2048, cfg=cfg)
    return path


def test_fingerprint_hashes_only_the_weights(tmp_path):
    a = weights_fingerprint(checkpoint(tmp_path / "a.pt"))
    assert a == weights_fingerprint(checkpoint(tmp_path / "b.pt", iteration=7))  # metadata doesn't count
    assert a["tensors"] > 0 and len(a["sha256"]) == 64
    assert a != weights_fingerprint(checkpoint(tmp_path / "c.pt", seed=91))
    assert a != weights_fingerprint(checkpoint(tmp_path / "d.pt", nudge=1e-7))  # one critic weight, barely


def test_environment_stamp_names_the_versions_without_importing_pybullet(capsys):
    stamp = environment_stamp()
    assert set(stamp) == {"python", "numpy", "torch", "pybullet", "platform", "processor"}
    assert stamp["torch"] == torch.__version__ and all(isinstance(v, str) for v in stamp.values())
    assert capsys.readouterr().out == ""  # importing pybullet prints its build time


def test_environment_diffs():
    old = {"python": "3.14.7", "torch": "2.14.1+cpu", "platform": "Windows-10"}
    new = {"python": "3.14.7", "torch": "2.15.0+cpu", "platform": "Windows-10", "numpy": "2.5.3"}
    assert environment_diffs(old, new) == [("numpy", "(not recorded)", "2.5.3"), ("torch", "2.14.1+cpu", "2.15.0+cpu")]


def record_suite(repo):
    """Two mock cases: keep_mock and goto_mock."""
    return load(repo, SUITE.replace('envs   = ["mock", "pybullet"]', 'envs   = ["mock"]'))


def fresh_runs(root, suite):
    for case in suite.cases:
        fake_case(root, case=case, timing=True)
    return root


def test_freeze_writes_the_record_and_round_trips_identical(repo, tmp_path):
    suite = record_suite(repo)
    new = fresh_runs(tmp_path / "new", suite)
    written = freeze_silver(suite, new, reason="silver v1", today="2026-11-01", commit="abc1234")
    data = repo / "results" / "silver" / "data"
    assert sorted(written) == sorted(f for c in suite.cases for f in case_files(c, suite.seeds))
    header = (data / "keep_mock_seed90" / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "iteration,policy_loss"  # wall-clock columns stripped
    for rel in ("keep_mock_seed90/eval.csv", "eval_goto_mock_seed90_per_run.csv", "keep_mock_seed91/config.json"):
        assert (data / rel).read_bytes() == (new / rel).read_bytes()  # everything else is copied as is
    assert all(compare_case(c, suite.seeds, new, data, EXACT).status == IDENTICAL for c in suite.cases)
    stamp = json.loads((repo / "results" / "silver" / "stamp.json").read_text(encoding="utf-8"))
    assert stamp["frozen"] == "2026-11-01" and stamp["commit"] == "abc1234" and stamp["reason"] == "silver v1"
    assert stamp["environment"] == environment_stamp()
    assert stamp["seeds"] == [90, 91] and stamp["eval_seed"] == 20261101 and stamp["budget"]["total_steps"] == 6144
    assert stamp["cases"] == ["keep_mock", "goto_mock"]
    assert stamp["updates"] == [{"date": "2026-11-01", "commit": "abc1234", "reason": "silver v1"}]


def test_refreeze_appends_history_and_drops_entries_the_suite_no_longer_lists(repo, tmp_path):
    suite = record_suite(repo)
    new = fresh_runs(tmp_path / "new", suite)
    freeze_silver(suite, new, reason="v1", today="2026-11-01", commit="aaa")
    data = repo / "results" / "silver" / "data"
    (data / "old_mock_seed90").mkdir()
    (data / "old_mock_seed90" / "train.csv").write_text("x\n", encoding="utf-8")
    freeze_silver(suite, new, reason="gae_lambda 0.9 on purpose", today="2026-11-02", commit="bbb")
    stamp = json.loads((repo / "results" / "silver" / "stamp.json").read_text(encoding="utf-8"))
    assert [u["reason"] for u in stamp["updates"]] == ["v1", "gae_lambda 0.9 on purpose"]
    assert stamp["reason"] == "gae_lambda 0.9 on purpose" and stamp["commit"] == "bbb"
    assert not (data / "old_mock_seed90").exists()
    assert stale_entries(suite.cases, suite.seeds, data) == []


def test_freeze_refuses_incomplete_runs_and_leaves_the_record_alone(repo, tmp_path):
    suite = record_suite(repo)
    new = fresh_runs(tmp_path / "new", suite)
    freeze_silver(suite, new, reason="v1", today="2026-11-01", commit="aaa")
    before = (repo / "results" / "silver" / "stamp.json").read_text(encoding="utf-8")
    (new / "goto_mock_seed91" / "fingerprint.json").unlink()
    with pytest.raises(ValueError, match=r"missing.*goto_mock_seed91/fingerprint\.json"):
        freeze_silver(suite, new, reason="v2", today="2026-11-02", commit="bbb")
    assert (repo / "results" / "silver" / "stamp.json").read_text(encoding="utf-8") == before
    assert (repo / "results" / "silver" / "data" / "goto_mock_seed91" / "fingerprint.json").is_file()


def test_freeze_needs_a_reason(repo, tmp_path):
    suite = record_suite(repo)
    with pytest.raises(ValueError, match="reason"):
        freeze_silver(suite, fresh_runs(tmp_path / "new", suite), reason="  ", today="2026-11-01", commit="aaa")


# --- Step 4: the runner and the command line --------------------------------------------
def test_select_cases(repo):
    s = load(repo)
    assert select_cases(s, None) == list(s.cases)
    assert [c.name for c in select_cases(s, ["*_mock"])] == ["keep_mock", "goto_mock"]
    assert [c.name for c in select_cases(s, ["goto_*", "keep_pybullet"])] == ["keep_pybullet", "goto_mock"]
    with pytest.raises(ValueError, match=r"--only 'nope\*' matches no case; cases: \['keep_mock'"):
        select_cases(s, ["nope*"])


def test_uncovered_configs(repo):
    s = load(repo)
    assert uncovered_configs(s, repo) == []
    (repo / "configs" / "new_idea.toml").write_text("", encoding="utf-8")
    assert uncovered_configs(s, repo) == ["configs/new_idea.toml"]


class FakeRunner:
    """Stands in for the subprocesses: records each command, 'trains' by writing a checkpoint,
    and fails the jobs whose command contains ``fail``."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], fail

    def __call__(self, cmd, cwd, log):
        self.calls.append((cmd, cwd))
        log.write_text("line 1\nTraceback: boom\n", encoding="utf-8")
        if any(f in " ".join(cmd) for f in self.fail):
            return 1
        if "--out-dir" in cmd:
            out = Path(cmd[cmd.index("--out-dir") + 1])
            (out / "checkpoints").mkdir(parents=True, exist_ok=True)
            checkpoint(out / "checkpoints" / "final.pt", seed=int(cmd[cmd.index("--seed") + 1]))
        return 0


def test_run_suite_trains_every_seed_then_evaluates_each_case(repo, tmp_path):
    suite = record_suite(repo)
    out = tmp_path / "regression"
    (out / "keep_mock_seed90").mkdir(parents=True)
    (out / "keep_mock_seed90" / "fingerprint.json").write_text("{}", encoding="utf-8")  # an old run's
    (out / "eval_keep_mock_seed90.csv").write_text("old\n", encoding="utf-8")
    fake = FakeRunner()
    assert run_suite(suite, list(suite.cases), out, parallel=3, runner=fake, log=lambda _m: None) == {}
    trains = [cmd for cmd, _ in fake.calls if "--out-dir" in cmd]
    evals = [(cmd, cwd) for cmd, cwd in fake.calls if "--out-dir" not in cmd]
    assert len(trains) == 4 and len(evals) == 2
    cmd = next(c for c in trains if str(out / "goto_mock_seed91") in c)
    assert cmd[1].endswith("train.py") and cmd[cmd.index("--config") + 1] == str(repo / "configs" / "goto.toml")
    assert cmd[cmd.index("--env") + 1] == "mock" and cmd[cmd.index("--seed") + 1] == "91" and "--overwrite" in cmd
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--set"] == suite.set_args()
    # evaluate.py runs inside the regression folder on bare run names, so its per-run table
    # records "keep_mock_seed90", not a machine-specific path
    assert sorted((c[2:], cwd) for c, cwd in evals) == [
        (["goto_mock_seed90", "goto_mock_seed91"], out), (["keep_mock_seed90", "keep_mock_seed91"], out)]
    assert all(c[1].endswith("evaluate.py") for c, _ in evals)
    fp = json.loads((out / "keep_mock_seed90" / "fingerprint.json").read_text(encoding="utf-8"))
    assert fp == weights_fingerprint(out / "keep_mock_seed90" / "checkpoints" / "final.pt")
    assert not (out / "eval_keep_mock_seed90.csv").exists()  # the old table can't pass for a new one


def test_a_failed_run_fails_its_case_only(repo, tmp_path):
    suite = record_suite(repo)
    fake = FakeRunner(fail=("keep_mock_seed91",))
    failures = run_suite(suite, list(suite.cases), tmp_path / "r", parallel=2, runner=fake, log=lambda _m: None)
    assert list(failures) == ["keep_mock"]
    assert "keep_mock_seed91: exit 1" in failures["keep_mock"] and "Traceback: boom" in failures["keep_mock"]
    evaluated = [cmd[2:] for cmd, _ in fake.calls if "--out-dir" not in cmd]
    assert evaluated == [["goto_mock_seed90", "goto_mock_seed91"]]  # never evaluate a half-trained case


@pytest.fixture
def frozen(repo, tmp_path):
    """A record frozen from fake runs, plus those runs in a regression folder (as after a run)."""
    suite = record_suite(repo)
    runs = fresh_runs(tmp_path / "runs", suite)
    freeze_silver(suite, runs, reason="v1", today="2026-11-01", commit="aaa")
    return suite, runs


def cli(suite, runs, *extra, repo=None, runner=None):
    return regress_main(["--suite", str(suite.path), "--runs-dir", str(runs), *extra],
                        repo_root=repo or suite.path.parents[2], runner=runner)


def test_no_run_compares_what_is_there(frozen, capsys):
    suite, runs = frozen
    assert cli(suite, runs, "--no-run") == 0
    out = capsys.readouterr().out
    assert "Regression check: 2 cases x seeds [90, 91]" in out and "Summary: 2 identical" in out
    assert "Summary: 2 identical" in (runs / "report.md").read_text(encoding="utf-8")


def test_a_delta_exits_1_and_a_tolerance_can_forgive_it(frozen, capsys):
    suite, runs = frozen
    csv_file(runs / "goto_mock_seed90" / "train.csv", ["iteration", "policy_loss", "sps", "wall_time_s"],
             (1, 0.5000000001, 1, 1))
    assert cli(suite, runs, "--no-run") == 1
    assert "CHANGED    goto_mock" in capsys.readouterr().out
    assert cli(suite, runs, "--no-run", "--rtol", "1e-6") == 0


def test_only_skips_the_other_cases(frozen, capsys):
    suite, runs = frozen
    csv_file(runs / "keep_mock_seed90" / "train.csv", ["iteration", "policy_loss"], (1, 9.9))
    assert cli(suite, runs, "--no-run", "--only", "goto_*") == 0
    out = capsys.readouterr().out
    assert "SKIPPED    keep_mock" in out and "Summary: 1 identical, 1 skipped" in out


def test_a_machine_difference_is_flagged_above_the_report(frozen, capsys):
    suite, runs = frozen
    stamp_path = suite.path.parent / "stamp.json"
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["environment"]["torch"] = "1.0.0"
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    assert cli(suite, runs, "--no-run") == 0  # identical results still pass
    assert f"torch: 1.0.0 -> {torch.__version__}" in capsys.readouterr().out


def test_stale_record_entries_exit_1(frozen, capsys):
    suite, runs = frozen
    (suite.path.parent / "data" / "retired_mock_seed90").mkdir()
    assert cli(suite, runs, "--no-run") == 1
    assert "STALE      retired_mock_seed90" in capsys.readouterr().out


def test_no_record_yet_reports_new_cases(repo, tmp_path, capsys):
    suite = record_suite(repo)
    assert cli(suite, fresh_runs(tmp_path / "runs", suite), "--no-run") == 1
    assert "NEW        keep_mock" in capsys.readouterr().out


def test_no_run_without_outputs_is_an_error(repo, tmp_path, capsys):
    assert cli(record_suite(repo), tmp_path / "empty", "--no-run") == 2
    assert "--no-run" in capsys.readouterr().err


def test_a_failed_job_exits_2_and_never_freezes(frozen, capsys):
    suite, runs = frozen
    before = (suite.path.parent / "stamp.json").read_text(encoding="utf-8")
    assert cli(suite, runs, runner=FakeRunner(fail=("goto_mock_seed90",))) == 2
    assert "FAILED     goto_mock" in capsys.readouterr().out
    assert cli(suite, runs, "--update", "--reason", "x", runner=FakeRunner(fail=("goto_mock_seed90",))) == 2
    assert (suite.path.parent / "stamp.json").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("args, match", [
    (["--update"], "--update needs --reason"),
    (["--reason", "x"], "--reason only goes with --update"),
    (["--update", "--reason", "x", "--only", "goto_*"], "--update freezes the whole suite"),
    (["--update", "--reason", "x", "--no-run"], "--update freezes the whole suite"),
    (["--update", "--reason", "x", "--rtol", "1e-6"], "--rtol/--atol only apply to comparisons"),
    (["--parallel-runs", "0"], "--parallel-runs must be >= 1"),
    (["--rtol", "-1"], "--rtol/--atol must be >= 0"),
])
def test_flag_rules(frozen, capsys, args, match):
    suite, runs = frozen
    with pytest.raises(SystemExit) as e:
        cli(suite, runs, *args)
    assert e.value.code == 2 and match in capsys.readouterr().err


def test_parallel_runs_defaults_to_half_the_logical_cores(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert default_parallel_runs() == 8
    monkeypatch.setattr("os.cpu_count", lambda: 1)
    assert default_parallel_runs() == 1
    monkeypatch.setattr("os.cpu_count", lambda: None)  # "can't tell"
    assert default_parallel_runs() == 1


TINY_CONFIG = """
[rl.ppo]
rollout_steps   = 64
num_minibatches = 4
update_epochs   = 1
"""

TINY_SUITE = """
name      = "silver"
seeds     = [90, 91]
eval_seed = 20261101

[budget]
total_steps   = 128
eval_every    = 1
eval_episodes = 2

[[case]]
config = "configs/tiny.toml"
envs   = ["mock"]
"""


def test_end_to_end_freeze_then_reproduce_in_fresh_processes(tmp_path, capsys):
    # The real path: train.py and evaluate.py subprocesses. Freezing, then a second full run in
    # new processes, must reproduce the record exactly.
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tiny.toml").write_text(TINY_CONFIG, encoding="utf-8")
    (tmp_path / "results" / "silver").mkdir(parents=True)
    path = tmp_path / "results" / "silver" / "suite.toml"
    path.write_text(TINY_SUITE, encoding="utf-8")
    base = ["--suite", str(path), "--runs-dir", str(tmp_path / "runs" / "regression"), "--parallel-runs", "2"]
    assert regress_main(base + ["--update", "--reason", "tiny v1"], repo_root=tmp_path) == 0
    data = tmp_path / "results" / "silver" / "data"
    assert "sps" not in (data / "tiny_mock_seed90" / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    per_run = (data / "eval_tiny_mock_seed90_per_run.csv").read_text(encoding="utf-8")
    assert ",tiny_mock_seed91," in per_run and "\\" not in per_run  # bare run names, no paths
    capsys.readouterr()
    assert regress_main(base, repo_root=tmp_path) == 0
    assert "IDENTICAL  tiny_mock" in capsys.readouterr().out
