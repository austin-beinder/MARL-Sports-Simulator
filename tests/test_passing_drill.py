# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-08
# Author:  Austin Beinder
# Usage:
#   - 2026-11-08: Generated this entire file: tests for the passing drill (the
#     spawn, frozen blues in the PyBullet env, the masked movement head, the
#     pass memory and reward terms, and training end to end).
#   - 2026-11-08: Every drill episode starts with a blue holding the ball.
#   - 2026-11-08: The hold reward grows by 1 with each catch.
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters and fixed
#     the new warnings; no behaviour change. (Austin Beinder)
# ==============================================================================
import math

import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict, with_overrides
from marl.common.spawn import START_POSSESSION, sample_spawn
from marl.envs.mock_env import MockEnv
from marl.envs.pybullet_env import PyBulletEnv
from marl.rl.networks import MOVE, Actor, build_networks
from marl.rl.obs import OWN_POSSESSION
from marl.rl.reward import PassMemory, RewardModule
from marl.rl.train import train

# pytest passes the fixtures (cfg, env) in as arguments of the same name.
# pylint: disable=redefined-outer-name

DRILL = {
    "shared": {"scenario": "passing_drill", "n_blue": 3, "red_max_speed": 0.6},
    "rl": {
        "kick_enabled": True,
        "reward": {
            "base": "passing_drill",
            "shaping": [
                {"name": "pass_progress", "enabled": True, "coef": 1.0},
                {"name": "pass_completed", "enabled": True, "coef": 3.0},
            ],
        },
    },
    "run": {"env": "pybullet"},
}


@pytest.fixture
def cfg():
    return config_from_dict(DRILL)


@pytest.fixture
def env(cfg):
    env = PyBulletEnv(cfg.shared)
    env.reset(seed=0)
    yield env
    env.close()


def actions(moves, kicker=None, kick_vec=(0.0, 0.0)):
    return {"blue": [{"move": list(m), "kick_trigger": int(i == kicker), "kick_vec": list(kick_vec)}
                     for i, m in enumerate(moves)]}


# --- Spawn -----------------------------------------------------------------------
def drill_spawn(cfg, seed, **options):
    sc = cfg.shared
    return sample_spawn(np.random.default_rng(seed), scenario="passing_drill", n_blue=sc.n_blue,
                        n_red=sc.n_red, field_size=sc.field_size, player_radius=sc.player_radius,
                        possession_radius=sc.possession_radius,
                        options={**sc.start.to_reset_options(), **options})


def test_spawn_puts_the_blues_round_red_with_the_ball(cfg):
    sc, r = cfg.shared, cfg.shared.start.drill_radius
    half = np.asarray(sc.field_size) / 2 - sc.player_radius
    for seed in range(200):
        sp = drill_spawn(cfg, seed)
        blue, red = np.array(sp.blue_pos), np.array(sp.red_pos)
        assert sp.start_type == START_POSSESSION and sp.ball_pos == sp.blue_pos[sp.holder]
        assert np.all(np.abs(np.concatenate([blue, red])) <= half + 1e-9)  # inside the walls
        players = np.concatenate([blue, red])
        d = np.linalg.norm(players[:, None] - players[None],
                           axis=-1)[np.triu_indices(len(players), 1)]
        assert d.min() >= sc.start.min_clearance
        # radius 0.85-1.15 r, red within 0.3 x radius of the centre -> 0.5 r to 1.5 r from red
        assert np.all((np.linalg.norm(blue - red, axis=-1) >= 0.5 * r)
                      & (np.linalg.norm(blue - red, axis=-1) <= 1.5 * r))
        # every pass is within a full kick's roll (ln(1 + 6) / 0.4 = 4.9 m)
        passes = np.linalg.norm(blue[:, None] - blue[None], axis=-1)
        assert passes.max() < math.log(1 + sc.max_kick_speed) / sc.ball_damping


def test_spawn_is_seeded_and_follows_drill_radius(cfg):
    assert drill_spawn(cfg, 7) == drill_spawn(cfg, 7) and drill_spawn(cfg, 7) != drill_spawn(cfg, 8)

    def mean_red_distance(radius):
        return np.mean([np.linalg.norm(np.subtract(sp.blue_pos, sp.red_pos[0]), axis=-1).mean()
                        for sp in (drill_spawn(cfg, s, drill_radius=radius) for s in range(100))])

    assert mean_red_distance(3.0) == pytest.approx(1.5 * mean_red_distance(2.0), rel=0.1)


def test_spawn_ignores_the_keepaway_only_options(cfg):
    # red starts about one radius (2 m) from the carrier, well inside min_red_ball_dist (3 m)
    sp = drill_spawn(cfg, 0, start_mix=0.0, min_red_ball_dist=3.0)
    assert sp.start_type == START_POSSESSION


# --- Env ---------------------------------------------------------------------------
def test_every_episode_starts_with_a_blue_holding_the_ball(env):
    still = actions([(0.0, 0.0)] * 3)
    for seed in range(30):
        # a keepaway-only option can't loosen it
        s, _ = env.reset(seed=seed, options={"start_mix": 0.0})
        assert sum(b["has_possession"] for b in s["blue"]) == 1
        s, term, _, _ = env.step(still)  # nobody kicks: the holder keeps it
        assert not term and sum(b["has_possession"] for b in s["blue"]) == 1


def test_blues_ignore_move_but_red_chases(env, cfg):
    s0 = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[[-5, -3]],
                                   holder=("blue", 0))
    s, *_ = env.step(actions([(1.0, 0.0), (-1.0, 1.0), (0.0, -1.0)]))
    for b0, b in zip(s0["blue"], s["blue"]):
        assert b["pos"] == pytest.approx(b0["pos"], abs=1e-3) and np.hypot(*b["vel"]) < 1e-3
    assert np.hypot(*s["red"][0]["vel"]) == pytest.approx(cfg.shared.red_max_speed, rel=0.05)


def test_a_kick_to_a_teammate_is_a_pass(env):
    env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[[-5, -3]],
                              holder=("blue", 0))
    memory, still = PassMemory(), [(0.0, 0.0)] * 3
    _, term, _, step_info = env.step(actions(still, kicker=0, kick_vec=(1.0, 0.0)))
    steps = [memory.update(step_info)]
    for _ in range(40):
        s, term, _, step_info = env.step(actions(still))
        steps.append(memory.update(step_info))
        if s["blue"][1]["has_possession"]:
            break
    assert s["blue"][1]["has_possession"] and not term
    assert [st.completed for st in steps].count(True) == 1 and steps[-1].completed
    assert memory.passes == 1 and memory.kicker is None
    assert all(st.kicker == 0 for st in steps)  # blue 0's kick was in play the whole way


def test_mock_refuses_the_drill(cfg):
    with pytest.raises(NotImplementedError, match="passing_drill"):
        MockEnv(cfg.shared)


# --- Masked movement head -------------------------------------------------------------
def test_frozen_actor_ignores_its_move_samples():
    torch.manual_seed(0)
    actor = Actor(17, kick_enabled=True, move_enabled=False)
    obs = torch.randn(4, 17)
    obs[:, OWN_POSSESSION] = 1.0
    samples, log_prob = actor.act(obs)
    other = {**samples, MOVE: torch.full_like(samples[MOVE], 0.01)}  # a very unlikely move
    lp, entropy, head_entropy = actor.evaluate(obs, other)
    assert torch.allclose(lp, log_prob)  # the move sample changed, the log-prob didn't
    assert torch.all(head_entropy[MOVE] == 0)
    assert torch.allclose(entropy, head_entropy["kick_trigger"] + head_entropy["kick_vec"])


def test_drill_networks_are_frozen_and_load_like_any_other(cfg):
    keepaway = with_overrides(cfg, {"shared": {"scenario": "keepaway"},
                                    "rl": {"reward": {"base": "survival", "shaping": []}}})
    drill_actor, _ = build_networks(cfg, 41, 20)
    normal_actor, _ = build_networks(keepaway, 41, 20)
    assert not drill_actor.move_enabled and normal_actor.move_enabled
    normal_actor.load_state_dict(drill_actor.state_dict())  # a flag, not a weight: same layout


# --- Reward ---------------------------------------------------------------------------
def state(holder=None, ball=(0.0, 0.0), red_holds=False):
    blues = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]
    return {
        "step": 0,
        "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": i == holder}
                 for i, p in enumerate(blues)],
        "red": [{"id": 0, "pos": [-1.0, -1.0], "vel": [0.0, 0.0], "has_possession": red_holds}],
        "ball": {"pos": list(blues[holder] if holder is not None else ball), "vel": [0.0, 0.0]},
    }


def info(*events):
    return {"events": list(events), "kick_applied": [False] * 3}


def kick(i):
    return {"type": "kick", "agent": ("blue", i)}


def poss(frm, to):
    return {"type": "possession_change", "from": frm, "to": to}


def test_holding_pays_only_after_the_first_pass(cfg):
    rm, memory = RewardModule(cfg), PassMemory()
    held = rm.parts(state(0), {}, state(0), False, False, info(), memory)
    assert held.base == 0 and held.shaping == 0  # the opening hold earns nothing
    rm.parts(state(0), {}, state(ball=(1.0, 0.0)), False, False,
             info(kick(0), poss(("blue", 0), None)), memory)
    loose = rm.parts(state(ball=(1.0, 0.0)), {}, state(ball=(2.0, 0.0)), False, False, info(),
                     memory)
    assert loose.base == 0  # a loose ball earns nothing
    caught = rm.parts(state(ball=(2.0, 0.0)), {}, state(1), False, False,
                      info(poss(None, ("blue", 1))), memory)
    assert caught.base == 1 and memory.passes == 1
    # from now on holding pays
    assert rm.parts(state(1), {}, state(1), False, False, info(), memory).base == 1


def test_each_catch_makes_holding_worth_one_more(cfg):
    rm, memory = RewardModule(cfg), PassMemory()
    for n, (kicker, catcher) in enumerate([(0, 1), (1, 2), (2, 0)], start=1):
        rm.parts(state(kicker), {}, state(ball=(1.0, 1.0)), False, False,
                 info(kick(kicker), poss(("blue", kicker), None)), memory)
        rm.parts(state(ball=(1.0, 1.0)), {}, state(catcher), False, False,
                 info(poss(None, ("blue", catcher))), memory)
        assert rm.parts(state(catcher), {}, state(catcher), False, False, info(), memory).base == n


def test_progress_and_the_pass_bonus(cfg):
    rm, memory = RewardModule(cfg), PassMemory()
    # the kick step: from the kicker (3 m from blue 1 and from blue 2) to 1 m along the line to
    # blue 1
    k = rm.parts(state(0), {}, state(ball=(1.0, 0.0)), False, False,
                 info(kick(0), poss(("blue", 0), None)), memory)
    assert k.shaping == pytest.approx(1.0 * (3.0 - 2.0))
    away = rm.parts(state(ball=(1.0, 0.0)), {}, state(ball=(0.5, -0.5)), False, False, info(),
                    memory)
    assert away.shaping < 0  # rolling away from both teammates costs
    caught = rm.parts(state(ball=(2.9, 0.0)), {}, state(1), False, False,
                      info(poss(None, ("blue", 1))), memory)
    assert caught.shaping == pytest.approx(1.0 * 0.1 + 3.0)  # the last 0.1 m, plus the pass bonus
    # no kick in play
    assert rm.parts(state(1), {}, state(1), False, False, info(), memory).shaping == 0


def test_the_kicker_picking_it_up_again_is_no_pass(cfg):
    rm, memory = RewardModule(cfg), PassMemory()
    rm.parts(state(0), {}, state(ball=(0.6, 0.0)), False, False,
             info(kick(0), poss(("blue", 0), None)), memory)
    back = rm.parts(state(ball=(0.6, 0.0)), {}, state(0), False, False,
                    info(poss(None, ("blue", 0))), memory)
    assert memory.passes == 0 and back.base == 0
    assert back.shaping == pytest.approx(2.4 - 3.0)  # the progress made comes back off


def test_red_stealing_a_pass_pays_nothing(cfg):
    rm, memory = RewardModule(cfg), PassMemory()
    rm.parts(state(0), {}, state(ball=(1.0, 0.0)), False, False,
             info(kick(0), poss(("blue", 0), None)), memory)
    stolen = rm.parts(state(ball=(1.0, 0.0)), {}, state(ball=(1.2, 0.0), red_holds=True), True,
                      False, info(poss(None, ("red", 0))), memory)
    assert stolen.base == 0 and stolen.shaping == 0 and memory.passes == 0


def test_pass_terms_need_the_episode_memory(cfg):
    with pytest.raises(ValueError, match="memory"):
        RewardModule(cfg).parts(state(0), {}, state(0), False, False, info())
    survival = RewardModule(config_from_dict({}))
    assert not survival.needs_memory and survival.parts({}, {}, {}, False, False, {}).base == 1


# --- End to end -----------------------------------------------------------------------
def test_training_the_drill_runs_end_to_end(tmp_path):
    small = with_overrides(config_from_dict(DRILL), {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(small, tmp_path / "drill", log=lambda *_: None)
    assert result.iterations == 2 and (tmp_path / "drill" / "checkpoints" / "final.pt").is_file()
