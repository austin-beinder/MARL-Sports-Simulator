# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: tests for strict state parsing,
#     the flattening spec (#11) and the obs builder (#13), including
#     hand-computed vectors, distance ordering, N=2/M=2 shapes, permutation
#     equivariance and N=1 information equivalence.
#   - 2026-10-11: Updated the expected red velocity for the max_speed scaling
#     amendment to #11.
# Prompt summary: Implement Phase 3 step 4 of the issue #7 plan (state
#   flattener + obs builder).
# Human review:   Pending review
# ==============================================================================
import copy

import numpy as np
import pytest

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.obs import ObsBuilder
from marl.rl.state import StateFlattener, parse_state

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


def shared(n_blue=1, n_red=1):
    return config_from_dict({"shared": {"n_blue": n_blue, "n_red": n_red}}).shared


def ent(i, pos, vel=(0.0, 0.0), poss=False):
    return {"id": i, "pos": list(pos), "vel": list(vel), "has_possession": poss}


def make_s(blue, red, ball_pos, ball_vel=(0.0, 0.0), step=0):
    return {
        "step": step,
        "blue": [ent(i, *b) for i, b in enumerate(blue)],
        "red": [ent(j, *r) for j, r in enumerate(red)],
        "ball": {"pos": list(ball_pos), "vel": list(ball_vel)},
    }


# Defaults: field 16 x 10 -> half (8, 5); max_speed 2.0; red_max_speed 1.7.
S_1V1 = make_s(
    blue=[((4.0, -2.5), (1.0, 0.0), True)],
    red=[((-8.0, 5.0), (0.0, -1.7), False)],
    ball_pos=(4.0, -2.5),
    ball_vel=(1.0, 0.0),
    step=17,
)


# --- Flattener (#11) ----------------------------------------------------------------
def test_flatten_hand_computed_1v1():
    sc = shared()
    flat = StateFlattener(sc)(parse_state(S_1V1, 1, 1))
    expected = [0.5, -0.5, 0.5, 0.0, 1.0,  # blue 0: pos/half, vel/max_speed, possession
                -1.0, 1.0, 0.0, -0.85, 0.0,  # red 0: pos/half, vel/max_speed (1.7/2), possession
                0.5, -0.5, 0.5, 0.0]  # ball: pos/half, vel/max_speed
    assert flat.dtype == np.float32 and flat.shape == (14,)
    np.testing.assert_allclose(flat, expected, atol=1e-7)


def test_flatten_drops_step_and_id():
    sc = shared()
    a = StateFlattener(sc)(parse_state(S_1V1, 1, 1))
    s2 = copy.deepcopy(S_1V1)
    s2["step"] = 999
    np.testing.assert_array_equal(a, StateFlattener(sc)(parse_state(s2, 1, 1)))


def test_flatten_key_order_blue_then_red_then_ball():
    sc = shared(2, 2)
    s = make_s(
        blue=[((1.0, 0.0), (0, 0), False), ((2.0, 0.0), (0, 0), False)],
        red=[((3.0, 0.0), (0, 0), False), ((4.0, 0.0), (0, 0), False)],
        ball_pos=(5.0, 0.0),
    )
    flat = StateFlattener(sc)(parse_state(s, 2, 2))
    assert flat.shape == (24,)
    xs = flat[[0, 5, 10, 15, 20]] * 8.0  # x positions, unscaled
    np.testing.assert_allclose(xs, [1, 2, 3, 4, 5], atol=1e-6)


# --- Strict parsing ----------------------------------------------------------------
def _mutations():
    def top_unknown(s): s["ball_spin"] = 0.0
    def ent_unknown(s): s["red"][0]["cooldown"] = 3
    def ball_unknown(s): s["ball"]["spin"] = [0.0, 0.0]
    def missing(s): del s["blue"][0]["vel"]
    def bad_id(s): s["blue"][0]["id"] = 1
    def wrong_len(s): s["red"].append(ent(1, (0, 0)))
    def poss_int(s): s["blue"][0]["has_possession"] = 1
    def nan_pos(s): s["ball"]["pos"] = [float("nan"), 0.0]
    def short_vec(s): s["blue"][0]["pos"] = [1.0]
    return [top_unknown, ent_unknown, ball_unknown, missing, bad_id, wrong_len, poss_int, nan_pos, short_vec]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_parse_rejects_malformed_state(mutate):
    s = copy.deepcopy(S_1V1)
    mutate(s)
    with pytest.raises((TypeError, ValueError)):
        parse_state(s, 1, 1)


def test_unknown_key_message_points_to_icd():
    s = copy.deepcopy(S_1V1)
    s["ball_spin"] = 0.0
    with pytest.raises(ValueError, match="ICD §5"):
        parse_state(s, 1, 1)


# --- Obs builder (#13) ----------------------------------------------------------------
def test_obs_hand_computed_1v1():
    sc = shared()
    o = ObsBuilder(sc)(parse_state(S_1V1, 1, 1))
    expected = [0.5, -0.5, 0.5, 0.0, 1.0,  # self: pos/half, vel/max_speed, own possession
                -1.5, 1.5, -0.5, -0.85, 0.0, 1.0,  # red slot: rel pos, rel vel, possession, visible
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0]  # ball slot: rel pos, rel vel, held-by-anyone, visible
    assert o.dtype == np.float32 and o.shape == (1, 17)
    np.testing.assert_allclose(o[0], expected, atol=1e-6)


@pytest.mark.parametrize("n_blue, n_red, dim", [(1, 1, 17), (2, 2, 29), (3, 1, 29), (1, 3, 29)])
def test_obs_dims(n_blue, n_red, dim):
    assert ObsBuilder(shared(n_blue, n_red)).dim == dim


def test_obs_distance_ordering_and_ties():
    sc = shared(1, 2)
    b = ObsBuilder(sc)
    near_second = make_s([((0, 0),)], [((6.0, 0.0),), ((1.0, 0.0),)], ball_pos=(0, 3))
    o = b(parse_state(near_second, 1, 2))[0]
    assert o[5] == pytest.approx(1.0 / 8) and o[11] == pytest.approx(6.0 / 8)  # nearest red (id 1) first
    tie = make_s([((0, 0),)], [((2.0, 0.0),), ((-2.0, 0.0),)], ball_pos=(0, 3))
    o = b(parse_state(tie, 1, 2))[0]
    assert o[5] == pytest.approx(2.0 / 8) and o[11] == pytest.approx(-2.0 / 8)  # tie -> lower id first


def test_obs_flags():
    sc = shared(2, 1)
    s = make_s(
        blue=[((0, 0), (0, 0), False), ((1.0, 0), (0, 0), True)],
        red=[((5.0, 0),)],
        ball_pos=(1.0, 0),
    )
    o = ObsBuilder(sc)(parse_state(s, 2, 1))
    # agent 0: own possession 0; teammate slot flag 1; ball held 1
    assert o[0, 4] == 0 and o[0, 5 + 4] == 1 and o[0, -2] == 1
    # agent 1: own possession 1; teammate slot flag 0
    assert o[1, 4] == 1 and o[1, 5 + 4] == 0
    # every visible flag is 1 under full observability
    vis = o[:, [5 + 5, 11 + 5, 17 + 5]]
    assert np.all(vis == 1)
    loose = make_s([((0, 0),), ((1.0, 0),)], [((5.0, 0),)], ball_pos=(3.0, 0))
    assert np.all(ObsBuilder(sc)(parse_state(loose, 2, 1))[:, -2] == 0)  # loose ball -> held 0


def test_obs_self_first_is_permutation_equivariant():
    # Swapping two blue agents' ids swaps their observations: slot 0 is always "me",
    # so the shared actor needs no agent ID (#4, #13).
    sc = shared(2, 2)
    s = make_s(
        blue=[((1.0, 2.0), (0.5, 0), False), ((-3.0, -1.0), (0, 1.0), True)],
        red=[((5.0, 0),), ((-6.0, 4.0),)],
        ball_pos=(-3.0, -1.0),
    )
    swapped = copy.deepcopy(s)
    swapped["blue"] = [dict(s["blue"][1], id=0), dict(s["blue"][0], id=1)]
    b = ObsBuilder(sc)
    o, o_sw = b(parse_state(s, 2, 2)), b(parse_state(swapped, 2, 2))
    np.testing.assert_array_equal(o[0], o_sw[1])
    np.testing.assert_array_equal(o[1], o_sw[0])


def test_obs_n1_carries_the_global_state():
    # At N = 1, o_1 holds the same information as flattened s (#1): absolute
    # positions are recoverable from self pos + relative pos.
    sc = shared()
    o = ObsBuilder(sc)(parse_state(S_1V1, 1, 1))[0]
    half = np.array([8.0, 5.0])
    self_pos = o[0:2] * half
    red_pos = self_pos + o[5:7] * half
    ball_pos = self_pos + o[11:13] * half
    np.testing.assert_allclose(red_pos, S_1V1["red"][0]["pos"], atol=1e-5)
    np.testing.assert_allclose(ball_pos, S_1V1["ball"]["pos"], atol=1e-5)


@pytest.mark.parametrize("data", [{"shared": {"n_blue": 2, "n_red": 2}}, {}, GOTO])
def test_pipeline_on_env_states(data):
    sc = config_from_dict(data).shared
    env = MockEnv(sc)
    flat, obs = StateFlattener(sc), ObsBuilder(sc)
    rng = np.random.default_rng(0)
    s, _ = env.reset(seed=4)
    for _ in range(30):
        st = parse_state(s, sc.n_blue, sc.n_red)
        f, o = flat(st), obs(st)
        assert f.shape == (flat.dim,) and o.shape == (sc.n_blue, obs.dim)
        assert np.all(np.isfinite(f)) and np.all(np.isfinite(o))
        assert np.all(np.abs(f) <= 1.0 + 1e-6)  # flattened features stay in [-1, 1] (milestone 1)
        acts = [{"move": [float(x) for x in rng.uniform(-1, 1, 2)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}
                for _ in range(sc.n_blue)]
        s, term, trunc, _ = env.step({"blue": acts})
        if term or trunc:
            break


def test_goto_target_dims_with_empty_red_block():
    sc = config_from_dict(GOTO).shared
    assert StateFlattener(sc).dim == 9 and ObsBuilder(sc).dim == 11


def test_deferred_modes_raise():
    sc = shared()
    with pytest.raises(NotImplementedError):
        ObsBuilder(sc, obs_ordering="index")
    with pytest.raises(NotImplementedError):
        ObsBuilder(sc, obs_mode="radius")
    with pytest.raises(NotImplementedError):
        StateFlattener(sc, horizon_mode="finite")
