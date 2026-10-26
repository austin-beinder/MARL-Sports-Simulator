# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Generated this entire file: tests of the run.env selector
#     (make_env, config validation) and the evaluators (the subprocess one
#     matches the in-process one exactly; training end to end on PyBullet)
#     and the --render / --debug flags.
# Prompt summary: Add an env selector to train/report/calibrate and run
#   evaluation in its own process for the shared pybullet simulation.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
import pytest
import torch

from marl.common.config import config_from_dict
from marl.envs import make_env
from marl.envs.mock_env import MockEnv
from marl.rl.eval_worker import InProcessEvaluator, SubprocessEvaluator, make_evaluator
from marl.rl.networks import seeded_networks
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer
from marl.rl.train import main, train

SMALL = {
    "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
    "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
}


def small(env, **shared):
    return config_from_dict({**SMALL, "shared": shared, "run": {**SMALL["run"], "env": env}})


def test_make_env_follows_run_env():
    from marl.envs.pybullet_env import PyBulletEnv

    assert isinstance(make_env(small("mock")), MockEnv)
    assert isinstance(make_env(small("pybullet")), PyBulletEnv)


def test_default_env_is_mock_and_unknown_env_is_rejected():
    assert config_from_dict({}).run.env == "mock"
    with pytest.raises(ValueError, match="run.env|env="):
        config_from_dict({"run": {"env": "bogus"}})


def test_shared_sim_env_evaluates_in_a_subprocess():
    cfg = small("mock")
    f, r = Featurizer(cfg), RewardModule(cfg)
    assert isinstance(make_evaluator(cfg, f, r), InProcessEvaluator)
    ev = make_evaluator(small("pybullet"), f, r)
    try:
        assert isinstance(ev, SubprocessEvaluator)
    finally:
        ev.close()


@pytest.mark.parametrize("deterministic", [False, True])
def test_subprocess_evaluation_matches_in_process(deterministic):
    cfg = small("mock")  # the mock, so the worker path is checked against a cheap reference
    f, r = Featurizer(cfg), RewardModule(cfg)
    actor, _ = seeded_networks(cfg, f.obs_dim, f.state_dim)
    torch.manual_seed(123)
    for p in actor.parameters():  # move off the init, so the weights really have to be sent
        p.data.add_(0.1 * torch.randn_like(p))
    ev = SubprocessEvaluator(cfg)
    try:
        assert ev(actor, deterministic) == InProcessEvaluator(cfg, f, r)(actor, deterministic)
    finally:
        ev.close()


def test_worker_setup_errors_surface_in_the_trainer():
    cfg = small("pybullet", n_blue=2, n_red=2)  # the PyBullet adapter is 1v1 only
    with pytest.raises(RuntimeError, match="1v1 only"):
        SubprocessEvaluator(cfg)


def test_training_on_pybullet_runs_end_to_end(tmp_path):
    res = train(small("pybullet"), tmp_path / "pb", log=lambda _m: None)
    assert res.final_eval["episodes"] == 3 and res.final_eval["len_mean"] > 0
    assert res.final_eval_deterministic["episodes"] == 3


def test_cli_env_flag_sets_run_env_and_out_dir(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("marl.rl.train.train", lambda cfg, out_dir, **kw: seen.update(cfg=cfg, out_dir=out_dir))
    main(["--config", "configs/keepaway_1v1.toml", "--env", "pybullet", "--set", f"run.out_dir={tmp_path.as_posix()}"])
    assert seen["cfg"].run.env == "pybullet"
    assert seen["out_dir"].name == "keepaway_1v1_pybullet_seed0"


def test_cli_debug_and_render_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("marl.rl.train.train", lambda cfg, out_dir, **kw: seen.update(kw))
    base = ["--config", "configs/keepaway_1v1.toml", "--set", f"run.out_dir={tmp_path.as_posix()}"]
    main(base + ["--env", "pybullet", "--render", "--debug"])
    assert seen["render"] and seen["print_collisions"]
    main(base + ["--env", "pybullet"])
    assert not seen["render"] and not seen["print_collisions"]
    with pytest.raises(SystemExit):  # the mock has no GUI
        main(base + ["--render"])


@pytest.mark.parametrize("on", [False, True])
def test_print_collisions_switch(capsys, on):
    env = make_env(small("pybullet"), print_collisions=on)
    env.reset(seed=0, options={"start_mix": 0.0})  # free ball: red runs into it
    still = {"blue": [{"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}]}
    while True:
        _, terminated, truncated, _ = env.step(still)
        if terminated or truncated:
            break
    printed = "collided with" in capsys.readouterr().out
    assert printed == on
### END AI GENERATED
