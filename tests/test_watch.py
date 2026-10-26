# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Generated this entire file: tests of the watch script
#     (newest-checkpoint selection, episode playback and pacing, argument
#     checks, --start). The GUI itself isn't opened here.
# Prompt summary: Make a watch script that plays games with the latest trained
#   model in the PyBullet GUI.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
from types import SimpleNamespace

import pytest

from marl.baselines import make_baseline
from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.evaluate import actor_policy
from marl.rl.networks import seeded_networks
from marl.rl.rollout import Featurizer
from marl.watch import latest_checkpoint, main, watch_episodes


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_latest_checkpoint_prefers_final_then_highest_iteration(tmp_path):
    ck = tmp_path / "checkpoints"
    for name in ("iter_00002.pt", "iter_00010.pt", "iter_00009.pt", "notes.pt"):
        touch(ck / name)
    assert latest_checkpoint(tmp_path).name == "iter_00010.pt"  # numeric, not alphabetical order
    touch(ck / "final.pt")
    assert latest_checkpoint(tmp_path).name == "final.pt"


def test_latest_checkpoint_explains_an_empty_run(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        latest_checkpoint(tmp_path)


def test_watch_plays_each_seed_and_paces_to_real_time():
    cfg = config_from_dict({})
    sleeps, lines = [], []
    results = watch_episodes(
        MockEnv(cfg.shared), make_baseline("still", cfg.shared), [3, 4], cfg,
        log=lines.append, sleep=sleeps.append,
    )
    assert [r["seed"] for r in results] == [3, 4]
    assert all(r["terminated"] or r["truncated"] for r in results)
    assert len(lines) == 2 and "seed 3" in lines[0]
    # one sleep per non-final step, plus a pause after each game; none longer than a control step
    assert len(sleeps) == sum(r["steps"] - 1 for r in results) + 2
    pauses = [s for s in sleeps if s == 1.0]
    assert len(pauses) == 2 and max(s for s in sleeps if s != 1.0) <= cfg.shared.dt_control


def test_watch_replays_a_stochastic_policy_exactly():
    cfg = config_from_dict({})
    f = Featurizer(cfg)
    actor, _ = seeded_networks(cfg, f.obs_dim, f.state_dim)

    def play():
        return watch_episodes(MockEnv(cfg.shared), actor_policy(actor, f), [11], cfg,
                              speed=float("inf"), log=lambda _m: None, sleep=lambda _s: None)

    assert play() == play()


@pytest.mark.parametrize("start, mix", [("free", 0.0), ("possession", 1.0)])
def test_start_flag_forces_the_start_type(monkeypatch, start, mix):
    seen = {}
    monkeypatch.setattr("marl.watch.make_env", lambda cfg, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("marl.watch.watch_episodes", lambda env, policy, seeds, cfg, **kw: seen.update(cfg=cfg))
    main(["--baseline", "still", "--start", start, "--seed", "0"])
    assert seen["cfg"].shared.start.start_mix == mix


@pytest.mark.parametrize("argv", [[], ["runs/x", "--baseline", "flee"], ["--baseline", "oracle"], ["--baseline", "flee", "--speed", "0"]])
def test_bad_arguments_exit(argv):
    with pytest.raises(SystemExit):
        main(argv)
### END AI GENERATED
