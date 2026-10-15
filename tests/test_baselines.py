# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: baseline policy tests (valid ICD
#     actions over whole episodes, flee behavior in hand-built states, flee
#     outlasting still/random, the oracle landing on the target, seeding,
#     errors).
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest

from marl.baselines import FleePolicy, baselines_for, make_baseline
from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.evaluate import eval_env_seeds, run_episodes, summarize
from marl.rl.reward import RewardModule

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}
X_MAX, Y_MAX = 8.0 - 0.25, 5.0 - 0.25  # wall limits for a player center (default field 16 x 10)


def state(blue, red, ball):
    """blue/red: lists of (pos, has_possession)."""
    ent = lambda i, p, poss: {"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": poss}
    return {
        "step": 0,
        "blue": [ent(i, p, poss) for i, (p, poss) in enumerate(blue)],
        "red": [ent(j, p, poss) for j, (p, poss) in enumerate(red)],
        "ball": {"pos": list(ball), "vel": [0.0, 0.0]},
    }


@pytest.mark.parametrize(
    "data, names",
    [
        ({}, ("random", "still", "flee")),
        ({"shared": {"n_blue": 2, "n_red": 2}}, ("random", "still", "flee")),
        (GOTO, ("random", "still", "oracle")),
    ],
)
def test_baselines_play_whole_episodes_with_valid_actions(data, names):
    cfg = config_from_dict(data).shared
    for name in names:
        env, policy = MockEnv(cfg), make_baseline(name, cfg, seed=0)
        s, _ = env.reset(seed=3)
        for _ in range(cfg.max_episode_steps):
            s, term, trunc, _ = env.step(policy(s))  # the env validates actions strictly
            if term or trunc:
                break


def test_flee_runs_straight_for_a_loose_ball():
    cfg = config_from_dict({}).shared
    s = state(blue=[((0.0, 0.0), False)], red=[((6.0, 4.0), False)], ball=(3.0, -4.0))
    np.testing.assert_allclose(FleePolicy(cfg)(s)["blue"][0]["move"], [0.6, -0.8])


def test_flee_carrier_runs_away_from_red_in_the_open():
    cfg = config_from_dict({}).shared
    s = state(blue=[((0.0, 0.0), True)], red=[((3.0, 0.0), False)], ball=(0.0, 0.0))
    mx, _ = FleePolicy(cfg)(s)["blue"][0]["move"]
    assert mx < -0.9  # (nearly) due west, straight away from red


def test_flee_escapes_along_the_wall_instead_of_into_the_corner():
    # Carrier near the north-east corner, red approaching along the north wall from the west.
    cfg = config_from_dict({}).shared
    me = (X_MAX - 0.2, Y_MAX - 0.2)
    s = state(blue=[(me, True)], red=[((me[0] - 3.0, me[1]), False)], ball=me)
    _, my = FleePolicy(cfg)(s)["blue"][0]["move"]
    assert my < -0.5  # heads south along the east wall; the corner is a trap


def test_flee_outlasts_still_and_random():
    cfg = config_from_dict({})
    seeds, rf = eval_env_seeds(123, 20), RewardModule(cfg)
    length = {
        name: summarize(run_episodes(MockEnv(cfg.shared), make_baseline(name, cfg.shared, 0), cfg, rf, seeds))["len_mean"]
        for name in ("flee", "still", "random")
    }
    assert length["flee"] > 1.5 * max(length["still"], length["random"])


def test_oracle_lands_exactly_on_the_target():
    cfg = config_from_dict(GOTO).shared
    oracle = make_baseline("oracle", cfg)
    # One control step at full speed covers 2.0 m/s * 0.05 s = 0.1 m; the target is 0.05 m away.
    near = state(blue=[((0.0, 0.0), False)], red=[], ball=(0.05, 0.0))
    np.testing.assert_allclose(oracle(near)["blue"][0]["move"], [0.5, 0.0])
    far = state(blue=[((0.0, 0.0), False)], red=[], ball=(0.0, -4.0))
    np.testing.assert_allclose(oracle(far)["blue"][0]["move"], [0.0, -1.0])
    env = MockEnv(cfg)
    s, _ = env.reset(seed=1)
    for _ in range(cfg.max_episode_steps):
        s, *_ = env.step(oracle(s))
    assert s["blue"][0]["pos"] == pytest.approx(s["ball"]["pos"], abs=1e-9)


def test_random_is_seeded():
    cfg = config_from_dict({}).shared
    s = state(blue=[((0.0, 0.0), True)], red=[((3.0, 0.0), False)], ball=(0.0, 0.0))
    seq = lambda p: [p(s)["blue"][0]["move"] for _ in range(5)]
    a, b, c = (make_baseline("random", cfg, seed=k) for k in (0, 0, 1))
    assert seq(a) == seq(b) and seq(make_baseline("random", cfg, seed=0)) != seq(c)


def test_still_never_moves():
    cfg = config_from_dict({}).shared
    s = state(blue=[((0.0, 0.0), True)], red=[((3.0, 0.0), False)], ball=(0.0, 0.0))
    assert make_baseline("still", cfg)(s)["blue"][0]["move"] == [0.0, 0.0]


def test_make_baseline_errors_and_scenario_lists():
    keep, goto = config_from_dict({}).shared, config_from_dict(GOTO).shared
    with pytest.raises(ValueError, match="unknown"):
        make_baseline("greedy", keep)
    with pytest.raises(ValueError):
        make_baseline("flee", goto)
    with pytest.raises(ValueError):
        make_baseline("oracle", keep)
    assert baselines_for("keepaway") == ("random", "still", "flee")
    assert baselines_for("goto_target") == ("random", "still", "oracle")
