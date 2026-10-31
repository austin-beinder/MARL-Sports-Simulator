# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-18: Added the PyBullet adapter to ENV_FACTORIES and skip
#     configs a factory doesn't support. (Austin Beinder)
#   - 2026-10-10: Generated this entire file: the ICD §11 conformance suite,
#     parametrized over env factories so the real env can be plugged in.
#   - 2026-10-14: Added the returned-states-are-snapshots test (the PBRS reward
#     reads Phi(s) after step(), so an env must not mutate a returned state).
# Prompt summary: Implement Phase 3 step 3 of the issue #7 plan (mock env +
#   ICD conformance tests).
# Human review:   Pending review
# ==============================================================================
"""ICD (docs/ICD.md v0.2) conformance tests.

These use ONLY the public API (reset/step + the shared config), so they double as
the acceptance test for the real env: add its constructor to ``ENV_FACTORIES``.
"""

import copy

import numpy as np
import pytest

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv

### AI GENERATED
from marl.envs.pybullet_env import PyBulletEnv  # noqa: E402

# name -> callable(SharedConfig) -> env. A factory that raises NotImplementedError for a config
# (e.g. the 1v1-only PyBullet env with n_blue = 2) skips that test instead of failing it.
ENV_FACTORIES = {"mock": MockEnv, "pybullet": PyBulletEnv}
### END AI GENERATED

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


@pytest.fixture(params=sorted(ENV_FACTORIES))
def make_env(request):
    factory = ENV_FACTORIES[request.param]

    def make(data=None):
        cfg = config_from_dict(data or {})
        ### AI GENERATED
        try:
            return factory(cfg.shared), cfg.shared
        except NotImplementedError as e:
            pytest.skip(f"{request.param}: {e}")
        ### END AI GENERATED

    return make


def zero_actions(n):
    return {"blue": [{"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]} for _ in range(n)]}


def random_actions(rng, n):
    return {
        "blue": [
            {"move": [float(x) for x in rng.uniform(-1, 1, 2)], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}
            for _ in range(n)
        ]
    }


def assert_plain(obj, where="value"):
    """ICD §2: everything crossing the interface is plain Python."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert type(k) is str, f"{where}: key {k!r}"
            assert_plain(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_plain(v, f"{where}[{i}]")
    else:
        assert obj is None or type(obj) in (bool, int, float, str), f"{where}: {type(obj).__name__}"


def run_until_done(env, actions_fn, limit=100_000):
    steps = 0
    while True:
        s, term, trunc, info = env.step(actions_fn())
        steps += 1
        if term or trunc:
            return s, term, trunc, info, steps
        assert steps < limit


# --- Reset / state format (ICD §5, §9) ---------------------------------------------
@pytest.mark.parametrize("shared", [{}, {"n_blue": 2, "n_red": 2}])
def test_reset_state_format(make_env, shared):
    env, sc = make_env({"shared": shared})
    s, info = env.reset(seed=0, options=sc.start.to_reset_options())
    assert set(s) == {"step", "blue", "red", "ball"} and s["step"] == 0
    assert [b["id"] for b in s["blue"]] == list(range(sc.n_blue))
    assert [r["id"] for r in s["red"]] == list(range(sc.n_red))
    for ent in s["blue"] + s["red"]:
        assert set(ent) == {"id", "pos", "vel", "has_possession"}
        assert len(ent["pos"]) == 2 and len(ent["vel"]) == 2
    assert set(s["ball"]) == {"pos", "vel"}
    assert sum(e["has_possession"] for e in s["blue"] + s["red"]) <= 1
    assert info == {"events": [], "kick_applied": [False] * sc.n_blue}
    assert_plain(s, "s")
    assert_plain(info, "info")


def test_step_counter_and_plain_types(make_env):
    env, sc = make_env({"shared": {"n_blue": 2, "n_red": 2}})
    env.reset(seed=1, options=sc.start.to_reset_options())
    rng = np.random.default_rng(0)
    for t in range(1, 6):
        s, term, trunc, info = env.step(random_actions(rng, 2))
        assert_plain(s, "s")
        assert_plain(info, "info")
        assert type(term) is bool and type(trunc) is bool
        assert len(info["kick_applied"]) == 2
        if term or trunc:
            break
        assert s["step"] == t


def test_returned_states_are_never_mutated_later(make_env):
    # ICD §5: a returned s is a snapshot. The RL side keeps s while it steps the env: the
    # PBRS reward reads Phi(s) AFTER step() returns s' (marl/rl/reward.py). An env that
    # updated a returned dict in place would silently corrupt that.
    env, sc = make_env({"shared": {"n_blue": 2, "n_red": 2}})
    options = sc.start.to_reset_options()
    s, _ = env.reset(seed=3, options=options)
    kept = [(s, copy.deepcopy(s))]
    rng = np.random.default_rng(1)
    for _ in range(20):
        s, term, trunc, _ = env.step(random_actions(rng, 2))
        kept.append((s, copy.deepcopy(s)))
        if term or trunc:
            break
    env.reset(seed=4, options=options)  # a reset must not touch old states either
    for i, (state, snapshot) in enumerate(kept):
        assert state == snapshot, f"state {i} changed after it was returned"


# --- Determinism (ICD §2) ---------------------------------------------------------
def test_same_seed_same_episode(make_env):
    def episode(seed):
        env, sc = make_env({"shared": {"n_blue": 2, "n_red": 2}})
        out = [env.reset(seed=seed, options=sc.start.to_reset_options())]
        rng = np.random.default_rng(99)
        for _ in range(80):
            step = env.step(random_actions(rng, 2))
            out.append(step)
            if step[1] or step[2]:
                break
        return out

    assert episode(5) == episode(5)
    assert episode(5)[0] != episode(6)[0]


# --- Malformed actions raise (ICD §6) --------------------------------------------------
GOOD = {"move": [0.0, 0.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]}


@pytest.mark.parametrize(
    "actions",
    [
        [GOOD],  # not a dict
        {"blue": [GOOD, GOOD]},  # wrong length (n_blue = 1)
        {"blue": []},
        {"blue": [GOOD], "red": [GOOD]},  # red is scripted
        {"blue": [{"move": [0.0, 0.0], "kick_trigger": 0}]},  # missing key
        {"blue": [{**GOOD, "move": [0.0]}]},
        {"blue": [{**GOOD, "move": [1.5, 0.0]}]},  # out of range
        {"blue": [{**GOOD, "move": [float("nan"), 0.0]}]},
        {"blue": [{**GOOD, "move": np.zeros(2)}]},  # numpy, not a list
        {"blue": [{**GOOD, "move": [np.float32(0.1), 0.0]}]},
        {"blue": [{**GOOD, "kick_trigger": 2}]},
        {"blue": [{**GOOD, "kick_trigger": True}]},  # bool, not int
        {"blue": [{**GOOD, "kick_trigger": 0.0}]},
        {"blue": [{**GOOD, "kick_vec": [0.0, -1.01]}]},
    ],
)
def test_malformed_actions_raise(make_env, actions):
    env, sc = make_env()
    env.reset(seed=0, options=sc.start.to_reset_options())
    with pytest.raises((TypeError, ValueError)):
        env.step(actions)


def test_boundary_actions_are_accepted(make_env):
    env, sc = make_env()
    env.reset(seed=0, options=sc.start.to_reset_options())
    env.step({"blue": [{"move": [1.0, -1.0], "kick_trigger": 0, "kick_vec": [0, 0]}]})  # ints in kick_vec ok


# --- Episode end (ICD §7) ---------------------------------------------------------------
def standstill_capture(make_env, seed, max_steps=100_000):
    env, sc = make_env({"shared": {"max_episode_steps": max_steps, "start": {"start_mix": 1.0}}})
    env.reset(seed=seed, options=sc.start.to_reset_options())
    return run_until_done(env, lambda: zero_actions(sc.n_blue))


@pytest.mark.parametrize("seed", range(5))
def test_terminated_implies_red_possession_and_true_final_state(make_env, seed):
    s, term, trunc, info, steps = standstill_capture(make_env, seed)
    assert term and not trunc
    assert any(r["has_possession"] for r in s["red"])
    assert not any(b["has_possession"] for b in s["blue"])
    assert s["step"] == steps  # the true final state, not a reset state
    assert any(e["type"] == "possession_change" and e["to"][0] == "red" for e in info["events"])


@pytest.mark.parametrize("seed", range(3))
def test_terminated_takes_priority_over_truncated(make_env, seed):
    # Find the capture step L, then set max_episode_steps = L: both conditions hit at once.
    *_, steps = standstill_capture(make_env, seed)
    s, term, trunc, info, steps2 = standstill_capture(make_env, seed, max_steps=steps)
    assert steps2 == steps and term and not trunc


def test_truncation_at_max_episode_steps(make_env):
    env, sc = make_env({**GOTO, "shared": {**GOTO["shared"], "max_episode_steps": 50}})
    env.reset(seed=0, options=sc.start.to_reset_options())
    s, term, trunc, info, steps = run_until_done(env, lambda: zero_actions(1))
    assert trunc and not term and steps == 50 and s["step"] == 50


# --- Kicks (ICD §3, §6) ---------------------------------------------------------------
def test_kick_without_possession_is_not_applied(make_env):
    env, sc = make_env({"shared": {"start": {"start_mix": 0.0}}})  # free-ball start
    env.reset(seed=0, options=sc.start.to_reset_options())
    s, term, trunc, info = env.step({"blue": [{"move": [0.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 0.0]}]})
    assert info["kick_applied"] == [False]


def test_kick_with_possession_is_applied(make_env):
    env, sc = make_env({"shared": {"start": {"start_mix": 1.0}}})  # blue holds the ball
    env.reset(seed=0, options=sc.start.to_reset_options())
    try:
        s, term, trunc, info = env.step({"blue": [{"move": [0.0, 0.0], "kick_trigger": 1, "kick_vec": [1.0, 0.0]}]})
    except NotImplementedError:
        pytest.skip("this env doesn't implement kicking yet (milestone 1 is movement-only)")
    assert info["kick_applied"] == [True]


# --- Spawning (ICD §8, §9) ------------------------------------------------------------
@pytest.mark.parametrize("shared", [{}, {"n_blue": 2, "n_red": 2}])
def test_spawn_constraints_and_no_red_possession(make_env, shared):
    env, sc = make_env({"shared": shared})
    opt = sc.start.to_reset_options()
    lim = np.array(sc.field_size) / 2
    starts = set()
    for seed in range(200):
        s, _ = env.reset(seed=seed, options=opt)
        blue = np.array([b["pos"] for b in s["blue"]])
        red = np.array([r["pos"] for r in s["red"]])
        ball = np.array(s["ball"]["pos"])
        players = np.concatenate([blue, red])
        assert np.all(np.abs(players) <= lim)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                assert np.linalg.norm(players[i] - players[j]) >= opt["min_clearance"]
        assert not any(r["has_possession"] for r in s["red"])
        assert np.linalg.norm(red - ball, axis=1).min() >= opt["min_red_ball_dist"]
        holders = [b for b in s["blue"] if b["has_possession"]]
        starts.add(bool(holders))
        if holders:
            assert np.allclose(holders[0]["pos"], ball)
        elif opt["require_blue_closer"]:
            assert np.linalg.norm(blue - ball, axis=1).min() < np.linalg.norm(red - ball, axis=1).min()
    assert starts == {True, False}  # both start types occur at start_mix = 0.5


@pytest.mark.parametrize("mix, has_holder", [(0.0, False), (1.0, True)])
def test_start_mix_controls_start_type(make_env, mix, has_holder):
    env, sc = make_env()
    opt = sc.start.to_reset_options() | {"start_mix": mix}
    for seed in range(50):
        s, _ = env.reset(seed=seed, options=opt)
        assert any(b["has_possession"] for b in s["blue"]) == has_holder


# --- goto_target, n_red = 0 (ICD §9) ------------------------------------------------------
def test_goto_target_n_red_zero(make_env):
    env, sc = make_env(GOTO)
    s0, _ = env.reset(seed=3, options=sc.start.to_reset_options())
    assert s0["red"] == []
    rng = np.random.default_rng(0)
    while True:
        s, term, trunc, info = env.step(random_actions(rng, 1))
        assert not term
        assert s["ball"]["pos"] == s0["ball"]["pos"]  # immobile target
        assert not s["blue"][0]["has_possession"]  # possession disabled
        if trunc:
            break
    assert s["step"] == sc.max_episode_steps
