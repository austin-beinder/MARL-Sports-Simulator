# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-31
# Author:  Austin Beinder
# Usage:
#   - 2026-10-31: Generated this entire file: the game behaviour both envs
#     must share (speed cap, carry slowdown, walls, possession, termination,
#     teammates, goto_target), run on the mock and on PyBullet through their
#     set_state_for_testing hooks.
# Prompt summary: Close the PyBullet integration gaps: test coverage for the
#   behaviour the mock's own tests (test_mock_env.py) only check on the mock.
# Human review:   Pending review
# ==============================================================================
"""Game behaviour both envs must share, run on the mock and on PyBullet.

``test_mock_env.py`` pins the mock's exact kinematics. These tests check the same rules with
tolerances that also fit the physics sim: PyBullet's damping takes about 1.6% off every speed, and
its possession is contact-based (``player_radius`` + the ball's 0.15 m radius to pick up a loose
ball, two player radii for red to reach the carrier) instead of ``possession_radius``.
"""

import math

import numpy as np
import pytest

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.envs.pybullet_env import PyBulletEnv

ENVS = {"mock": MockEnv, "pybullet": PyBulletEnv}
REL = 0.03  # speed tolerance: PyBullet's damping costs ~1.6%
GOTO = {"scenario": "goto_target", "n_red": 0}


@pytest.fixture(params=sorted(ENVS))
def make(request):
    def make(shared=None, rl=None):
        cfg = config_from_dict({"shared": shared or {}, "rl": rl or {}}).shared
        env = ENVS[request.param](cfg)
        env.reset(seed=0)
        return env, cfg

    return make


def act(*moves):
    return {"blue": [{"move": [float(mx), float(my)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]} for mx, my in moves]}


def speed(ent):
    return float(np.hypot(*ent["vel"]))


def steps(env, actions, n):
    """Step n times (stopping early at episode end); return the last result and all events."""
    events = []
    for _ in range(n):
        s, term, trunc, info = env.step(actions)
        events += info["events"]
        if term or trunc:
            break
    return s, term, trunc, events


# --- Movement (ICD §6) ---------------------------------------------------------------
@pytest.mark.parametrize("deg", [0, 30, 45, 90, 135, 225])
def test_speed_does_not_depend_on_heading(make, deg):
    # 2-norm cap: a unit move vector gives max_speed in every direction, diagonals included.
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[-6, -4], holder=None)
    heading = [math.cos(math.radians(deg)), math.sin(math.radians(deg))]
    s, *_ = steps(env, act(heading), 2)
    b = s["blue"][0]
    assert speed(b) == pytest.approx(cfg.max_speed, rel=REL)
    assert np.dot(b["vel"], heading) / speed(b) == pytest.approx(1.0, abs=1e-3)  # and in that direction


def test_over_long_move_is_capped_to_unit_length(make):
    # parse_actions accepts each component in [-1, 1] but caps the vector's length at 1.
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = steps(env, act((1.0, 1.0)), 2)
    assert speed(s["blue"][0]) == pytest.approx(cfg.max_speed, rel=REL)


def test_partial_move_scales_speed(make):
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = steps(env, act((0.5, 0.0)), 2)
    assert speed(s["blue"][0]) == pytest.approx(0.5 * cfg.max_speed, rel=REL)


def test_carry_slowdown_and_the_ball_follows_the_carrier(make):
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    s, *_ = steps(env, act((-1.0, 0.0)), 2)  # away from red
    b = s["blue"][0]
    assert b["has_possession"]
    assert speed(b) == pytest.approx(cfg.carry_speed_factor * cfg.max_speed, rel=REL)
    assert s["ball"]["pos"] == pytest.approx(b["pos"], abs=0.02)


def test_walls_stop_players(make):
    env, cfg = make()
    x_max = cfg.field_size[0] / 2 - cfg.player_radius
    env.set_state_for_testing(blue_pos=[[x_max - 0.01, 0]], red_pos=[[-7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = steps(env, act((1.0, 1.0)), 2)
    b = s["blue"][0]
    assert b["pos"][0] == pytest.approx(x_max, abs=0.01)
    assert abs(b["vel"][0]) < 0.05 and b["vel"][1] > 0  # sliding along the wall


# --- Possession and termination (ICD §3, §7, §8) -----------------------------------------
def test_loose_ball_pickup_event(make):
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[0.6, 0], holder=None)
    s, term, trunc, events = steps(env, act((1.0, 0.0)), 5)
    assert s["blue"][0]["has_possession"] and not term
    assert events == [{"type": "possession_change", "from": None, "to": ("blue", 0)}]


def test_red_reaching_a_loose_ball_terminates(make):
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[-6, 0]], red_pos=[[0.6, 0]], ball_pos=[0, 0], holder=None)
    s, term, trunc, events = steps(env, act((0.0, 0.0)), 20)
    assert term and not trunc and s["red"][0]["has_possession"]
    assert events[-1] == {"type": "possession_change", "from": None, "to": ("red", 0)}


def test_red_reaching_the_carrier_terminates_mid_block(make):
    # Red starts just out of reach of a still carrier, so it gets there within the first control step.
    env, cfg = make()
    r = cfg.possession_radius
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[r + 0.02, 0]], holder=("blue", 0))
    s, term, trunc, info = env.step(act((0.0, 0.0)))
    assert term and s["step"] == 1 and s["red"][0]["has_possession"]
    assert info["events"] == [{"type": "possession_change", "from": ("blue", 0), "to": ("red", 0)}]
    # Stopped on the tick red got the ball, not at the end of the block (red would have pushed on).
    assert np.hypot(*np.subtract(s["red"][0]["pos"], s["blue"][0]["pos"])) <= r + 0.02


def test_teammate_cannot_take_the_ball_from_the_carrier(make):
    # Only a kick releases the ball (ICD §8): a teammate running into the carrier changes nothing.
    env, cfg = make({"n_blue": 2, "n_red": 1})
    env.set_state_for_testing(blue_pos=[[0, 0], [0.8, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    s, term, trunc, events = steps(env, act((0.0, 0.0), (-1.0, 0.0)), 5)
    assert s["blue"][0]["has_possession"] and not s["blue"][1]["has_possession"]
    assert events == [] and not term


def test_red_steals_from_the_second_blue(make):
    # Ids are mapped per team: the events name the right players when it isn't blue 0 / red 0.
    env, cfg = make({"n_blue": 2, "n_red": 2})
    r = cfg.possession_radius
    env.set_state_for_testing(
        blue_pos=[[-6, -3], [0, 0]], red_pos=[[6, 3], [r + 0.02, 0]], ball_pos=[0, 0], holder=("blue", 1)
    )
    s, term, trunc, info = env.step(act((0.0, 0.0), (0.0, 0.0)))
    assert term and s["red"][1]["has_possession"]
    assert info["events"] == [{"type": "possession_change", "from": ("blue", 1), "to": ("red", 1)}]


# --- goto_target (ICD §9) ------------------------------------------------------------------
def test_target_is_immobile_and_untouchable(make):
    env, cfg = make(GOTO, {"reward": {"base": "goto_target"}})
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[], ball_pos=[1.0, 0], holder=None)
    s, term, trunc, events = steps(env, act((1.0, 0.0)), 15)  # 1.5 m: straight through the target
    assert s["ball"]["pos"] == [1.0, 0.0] and s["ball"]["vel"] == [0.0, 0.0]
    assert s["blue"][0]["pos"][0] > 1.2  # passed through it
    assert not s["blue"][0]["has_possession"] and events == [] and not term


# --- API errors (ICD §4, §9) -------------------------------------------------------------
def test_no_auto_reset_and_step_before_reset(make):
    env, cfg = make()
    fresh = type(env)(cfg)
    with pytest.raises(RuntimeError):
        fresh.step(act((0.0, 0.0)))
    env.reset(seed=0)
    env.set_state_for_testing(blue_pos=[[-6, 0]], red_pos=[[0.6, 0]], ball_pos=[0, 0], holder=None)
    _, term, _, _ = steps(env, act((0.0, 0.0)), 20)
    assert term
    with pytest.raises(RuntimeError):
        env.step(act((0.0, 0.0)))


def test_kick_with_possession_not_implemented_and_state_untouched(make):
    env, cfg = make()
    before = env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    with pytest.raises(NotImplementedError):
        env.step({"blue": [{"move": [1.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 0.0]}]})
    assert env._state() == before


def test_unknown_reset_option_and_bad_seed(make):
    env, cfg = make()
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"start_mixx": 0.5})
    with pytest.raises(ValueError):
        env.reset(seed=-1)
