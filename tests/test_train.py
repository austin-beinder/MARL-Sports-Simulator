# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: end-to-end tests of the training
#     loop on tiny configs (outputs, checkpoint reload, determinism,
#     evaluation isolation, the same eval seeds for every training seed,
#     output-directory safety, --set parsing, the CLI script).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import marl.rl.evaluate as evaluate_mod
from marl.common.config import config_from_dict
from marl.rl.checkpoint import load_actor, load_checkpoint
from marl.rl.networks import MOVE
from marl.rl.train import parse_overrides, train

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "shared": {"scenario": "goto_target", "n_red": 0, "max_episode_steps": 20},
    "rl": {"reward": {"base": "goto_target"}, "ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 2}},
    "run": {"total_steps": 256, "eval_every": 2, "eval_episodes": 3},
}
TIMING = {"sps", "wall_time_s"}


def tiny(**run):
    data = json.loads(json.dumps(TINY))
    data["run"] |= run
    return config_from_dict(data)


def quiet(_msg):
    pass


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_timing(table):
    return [{k: v for k, v in r.items() if k not in TIMING} for r in table]


def test_outputs_and_checkpoint_reload(tmp_path):
    cfg = tiny()
    res = train(cfg, tmp_path / "run", log=quiet)
    out = res.out_dir
    assert config_from_dict(json.loads((out / "config.json").read_text())) == cfg
    tr = rows(out / "train.csv")
    assert [int(r["iteration"]) for r in tr] == [1, 2, 3, 4]
    assert [int(r["env_steps"]) for r in tr] == [64, 128, 192, 256]
    ev = rows(out / "eval.csv")
    assert [(int(r["iteration"]), r["mode"]) for r in ev] == [
        (0, "stochastic"), (2, "stochastic"), (4, "stochastic"), (4, "deterministic")
    ]
    ckpts = sorted(p.name for p in (out / "checkpoints").iterdir())
    assert ckpts == ["final.pt", "iter_00002.pt", "iter_00004.pt"]
    assert load_checkpoint(out / "checkpoints" / "final.pt")["env_steps"] == 256

    _, actor, _ = load_actor(out / "checkpoints" / "final.pt")
    obs = torch.randn(5, actor.trunk[0].in_features)
    for name in ("concentration1", "concentration0"):
        torch.testing.assert_close(
            getattr(actor.distributions(obs)[MOVE], name), getattr(res.actor.distributions(obs)[MOVE], name)
        )
    assert res.final_eval["mode"] == "stochastic" and res.final_eval_deterministic["mode"] == "deterministic"


def test_same_seed_same_run(tmp_path):
    a = train(tiny(), tmp_path / "a", log=quiet)
    b = train(tiny(), tmp_path / "b", log=quiet)
    assert strip_timing(rows(a.out_dir / "train.csv")) == strip_timing(rows(b.out_dir / "train.csv"))
    assert rows(a.out_dir / "eval.csv") == rows(b.out_dir / "eval.csv")
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(pa, pb)
    c = train(tiny(seed=1), tmp_path / "c", log=quiet)
    assert strip_timing(rows(c.out_dir / "train.csv")) != strip_timing(rows(a.out_dir / "train.csv"))


def test_evaluation_never_changes_training(tmp_path):
    # Evaluating every iteration vs. only at the end: identical training (RNG and env isolation).
    often = train(tiny(eval_every=1), tmp_path / "often", log=quiet)
    rarely = train(tiny(eval_every=4), tmp_path / "rarely", log=quiet)
    assert strip_timing(rows(often.out_dir / "train.csv")) == strip_timing(rows(rarely.out_dir / "train.csv"))
    for pa, pb in zip(often.actor.parameters(), rarely.actor.parameters()):
        assert torch.equal(pa, pb)


def test_every_training_seed_is_evaluated_on_the_same_episodes(tmp_path, monkeypatch):
    seen = []
    real = evaluate_mod.run_episodes

    def spy(env, policy, cfg, reward_fn, seeds):
        seen.append(list(seeds))
        return real(env, policy, cfg, reward_fn, seeds)

    monkeypatch.setattr(evaluate_mod, "run_episodes", spy)
    train(tiny(seed=0, total_steps=64), tmp_path / "s0", log=quiet)
    train(tiny(seed=1, total_steps=64), tmp_path / "s1", log=quiet)
    assert len(seen) == 6 and all(s == seen[0] for s in seen)


def test_keepaway_run_works(tmp_path):
    cfg = config_from_dict({
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 2},
    })
    res = train(cfg, tmp_path / "ka", log=quiet)
    assert res.final_eval["episodes"] == 2 and res.final_eval["len_mean"] > 0


def test_output_directory_safety(tmp_path):
    out = tmp_path / "run"
    train(tiny(total_steps=64), out, log=quiet)
    with pytest.raises(FileExistsError, match="--overwrite"):
        train(tiny(total_steps=64), out, log=quiet)
    train(tiny(total_steps=64), out, overwrite=True, log=quiet)  # replaces the run's own files
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="refusing"):
        train(tiny(total_steps=64), out, overwrite=True, log=quiet)
    assert (out / "notes.txt").read_text() == "keep me"  # never deletes unrecognized files


def test_parse_overrides():
    got = parse_overrides(
        ["shared.red_max_speed=1.6", "rl.critic_input=local", "rl.ppo.hidden_sizes=[32, 32]", "run.seed=3", "rl.kick_enabled=false"]
    )
    assert got == {
        "shared": {"red_max_speed": 1.6},
        "rl": {"critic_input": "local", "ppo": {"hidden_sizes": [32, 32]}, "kick_enabled": False},
        "run": {"seed": 3},
    }
    for bad in ["no_equals_sign", "=3", "a..b=1"]:
        with pytest.raises(ValueError):
            parse_overrides([bad])


def test_cli_script(tmp_path):
    out = tmp_path / "cli"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(ROOT / "configs" / "goto_target.toml"),
        "--out-dir", str(out), "--total-steps", "128",
        "--set", "rl.ppo.rollout_steps=64", "--set", "rl.ppo.num_minibatches=4", "--set", "rl.ppo.update_epochs=1",
        "--set", "run.eval_episodes=2", "--set", "shared.max_episode_steps=20",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)  # cwd elsewhere: the script finds marl
    assert r.returncode == 0, r.stderr
    assert "[train] done" in r.stdout
    assert len(rows(out / "train.csv")) == 2
