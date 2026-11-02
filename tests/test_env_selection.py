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
#   - 2026-10-31: Train end to end on PyBullet at 2v2 and goto_target too, and
#     use a physics_dt mismatch for the worker setup error (2v2 now works).
#   - 2026-10-31: PyBullet evaluates in-process too; added tests that several
#     PyBullet envs run side by side in one process.
# Prompt summary: Add an env selector to train/report/calibrate and run
#   evaluation in its own process for the shared pybullet simulation.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
import numpy as np
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


def test_every_env_evaluates_in_process():
    # 2026-10-31: each PyBullet env has its own simulation, so it no longer needs a worker process.
    for env in ("mock", "pybullet"):
        cfg = small(env)
        ev = make_evaluator(cfg, Featurizer(cfg), RewardModule(cfg))
        try:
            assert isinstance(ev, InProcessEvaluator)
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
    cfg = small("pybullet", physics_dt=1 / 120)  # pybullet's time step is fixed at 1/240 s
    with pytest.raises(RuntimeError, match="physics_dt"):
        SubprocessEvaluator(cfg)


# 2026-10-31: 2v2 and goto_target train on PyBullet too.
@pytest.mark.parametrize("shared, rl", [({}, {}), ({"n_blue": 2, "n_red": 2}, {}),
                                        ({"scenario": "goto_target", "n_red": 0}, {"reward": {"base": "goto_target"}})])
def test_training_on_pybullet_runs_end_to_end(tmp_path, shared, rl):
    cfg = config_from_dict({**SMALL, "shared": shared, "rl": {**SMALL["rl"], **rl}, "run": {**SMALL["run"], "env": "pybullet"}})
    res = train(cfg, tmp_path / "pb", log=lambda _m: None)
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


# --- Several PyBullet envs in one process (2026-10-31) -------------------------------------
def _episode(env, seed):
    """Yield the reset and every step of one episode with a seeded random policy."""
    rng = np.random.default_rng(seed)
    yield env.reset(seed=seed)
    while True:
        a = {"blue": [{"move": [float(x) for x in rng.uniform(-1, 1, 2)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}]}
        step = env.step(a)
        yield step
        if step[1] or step[2]:
            return


def test_interleaved_pybullet_envs_match_running_alone():
    from marl.envs.pybullet_env import PyBulletEnv

    sc = config_from_dict({}).shared
    alone = {seed: list(_episode(PyBulletEnv(sc), seed)) for seed in (5, 6)}

    a, b = PyBulletEnv(sc), PyBulletEnv(sc)
    runs = {5: _episode(a, 5), 6: _episode(b, 6)}
    got = {5: [], 6: []}
    while runs:  # step the two envs in turn, both mid-episode
        for seed in list(runs):
            step = next(runs[seed], None)
            if step is None:
                del runs[seed]
            else:
                got[seed].append(step)
    assert got == alone


def test_closing_one_pybullet_env_leaves_the_others_running():
    from marl.envs.pybullet_env import PyBulletEnv

    sc = config_from_dict({}).shared
    a, b = PyBulletEnv(sc), PyBulletEnv(sc)
    a.reset(seed=0)
    b.reset(seed=1)
    a.close()
    still = {"blue": [{"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}]}
    s, *_ = b.step(still)
    assert s["step"] == 1
    a.close()  # a second close is a no-op: it must not disconnect a simulation that reused a's id
    c = PyBulletEnv(sc)  # may get a's freed connection id
    del a  # garbage-collecting a must not close c either
    c.reset(seed=2)
    c.step(still)
    b.step(still)


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
