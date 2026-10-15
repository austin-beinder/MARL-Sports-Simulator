# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: mock-specific dynamics tests
#     (speed cap, carry slowdown, walls, acquisition, mid-block termination,
#     red pursuit, no auto-reset), beyond the generic ICD conformance suite.
# Prompt summary: Implement Phase 3 step 3 of the issue #7 plan (mock env).
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv


def make(shared=None):
    cfg = config_from_dict({"shared": shared or {}}).shared
    env = MockEnv(cfg)
    env.reset(seed=0)
    return env, cfg


def act(mx, my, n=1):
    return {"blue": [{"move": [mx, my], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}] * n}


def speed(ent):
    return float(np.hypot(*ent["vel"]))


def test_two_norm_speed_cap():
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = env.step(act(1.0, 1.0))
    assert speed(s["blue"][0]) == pytest.approx(cfg.max_speed)  # diagonal isn't faster
    assert s["blue"][0]["pos"][0] == pytest.approx(cfg.max_speed / np.sqrt(2) * cfg.dt_control)


def test_partial_move_scales_speed():
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = env.step(act(0.5, 0.0))
    assert speed(s["blue"][0]) == pytest.approx(0.5 * cfg.max_speed)


def test_carry_slowdown():
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    s, *_ = env.step(act(1.0, 0.0))
    assert speed(s["blue"][0]) == pytest.approx(cfg.carry_speed_factor * cfg.max_speed)
    assert s["ball"]["pos"] == s["blue"][0]["pos"]  # the ball follows the carrier


def test_walls_clamp_and_zero_into_wall_velocity():
    env, cfg = make()
    x_max = cfg.field_size[0] / 2 - cfg.player_radius
    env.set_state_for_testing(blue_pos=[[x_max - 0.01, 0]], red_pos=[[-7, 4]], ball_pos=[-6, -4], holder=None)
    s, *_ = env.step(act(1.0, 1.0))
    b = s["blue"][0]
    assert b["pos"][0] == pytest.approx(x_max)
    assert b["vel"][0] == 0.0 and b["vel"][1] > 0  # sliding along the wall


def test_loose_ball_acquisition_event():
    env, cfg = make()
    # One control step = 0.05 s, so blue covers 0.1 m: 0.55 -> 0.45 < radius 0.5.
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], ball_pos=[0.55, 0], holder=None)
    s, term, trunc, info = env.step(act(1.0, 0.0))
    assert s["blue"][0]["has_possession"] and not term
    assert info["events"] == [{"type": "possession_change", "from": None, "to": ("blue", 0)}]


def test_red_reaches_loose_ball_first_terminates():
    env, cfg = make()
    # Red covers 1.7 * 0.05 = 0.085 m per control step: 0.55 -> 0.465 < 0.5.
    env.set_state_for_testing(blue_pos=[[-6, 0]], red_pos=[[0.55, 0]], ball_pos=[0, 0], holder=None)
    s, term, trunc, info = env.step(act(0.0, 0.0))
    assert term and s["red"][0]["has_possession"]


def test_mid_block_termination_stops_at_first_tick():
    # Red starts just outside the radius and runs at the stationary carrier.
    env, cfg = make()
    r, v, dt = cfg.possession_radius, cfg.red_max_speed, cfg.physics_dt
    env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[r + 2.5 * v * dt, 0]], holder=("blue", 0))
    s, term, trunc, info = env.step(act(0.0, 0.0))
    assert term and s["step"] == 1
    d = np.hypot(*np.subtract(s["red"][0]["pos"], s["ball"]["pos"]))
    # Stopped on the tick it crossed the radius, not at the end of the 12-tick block.
    assert r - v * dt - 1e-9 <= d <= r


def test_red_does_not_overshoot_ball():
    env, cfg = make()
    env.set_state_for_testing(blue_pos=[[-6, 0]], red_pos=[[0.06, 0]], ball_pos=[0, 0], holder=None)
    env._possession = False  # isolate the pursuit rule from termination
    s, *_ = env.step(act(0.0, 0.0))
    assert s["red"][0]["pos"] == pytest.approx([0.0, 0.0])


def test_no_auto_reset_and_step_before_reset():
    env = MockEnv(config_from_dict({}).shared)
    with pytest.raises(RuntimeError):
        env.step(act(0.0, 0.0))
    env.reset(seed=0)
    env.set_state_for_testing(blue_pos=[[-6, 0]], red_pos=[[0.55, 0]], ball_pos=[0, 0], holder=None)
    _, term, *_ = env.step(act(0.0, 0.0))
    assert term
    with pytest.raises(RuntimeError):
        env.step(act(0.0, 0.0))


def test_kick_with_possession_not_implemented_and_state_untouched():
    env, cfg = make()
    before = env.set_state_for_testing(blue_pos=[[0, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    with pytest.raises(NotImplementedError):
        env.step({"blue": [{"move": [1.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 0.0]}]})
    assert env._state() == before


def test_unknown_reset_option_and_bad_seed():
    env = MockEnv(config_from_dict({}).shared)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"start_mixx": 0.5})
    with pytest.raises(ValueError):
        env.reset(seed=-1)


def test_teammate_cannot_steal_from_carrier():
    env, cfg = make({"n_blue": 2, "n_red": 1})
    env.set_state_for_testing(blue_pos=[[0, 0], [0.3, 0]], red_pos=[[7, 4]], holder=("blue", 0))
    s, term, trunc, info = env.step(act(0.0, 0.0, n=2))
    assert s["blue"][0]["has_possession"] and not s["blue"][1]["has_possession"]
    assert info["events"] == []
