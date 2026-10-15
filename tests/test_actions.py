# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: action adapter tests (mapping,
#     plain types, bounds, bad input) and an actor -> adapter -> mock env
#     integration test at N=2/M=2.
# Prompt summary: Implement Phase 3 step 6 of the issue #7 plan (actor/critic
#   + Beta movement head + action adapter).
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest
import torch

from marl.common.config import config_from_dict
from marl.envs.mock_env import MockEnv
from marl.rl.actions import commanded_speed_fraction, to_env_actions
from marl.rl.networks import MOVE, Actor
from marl.rl.obs import ObsBuilder
from marl.rl.state import parse_state


def test_mapping_and_icd_format():
    acts = to_env_actions(np.array([[0.75, 0.25], [0.5, 1.0]], dtype=np.float32))
    assert acts == {
        "blue": [
            {"move": [0.5, -0.5], "kick_trigger": 0, "kick_vec": [0.0, 0.0]},
            {"move": [0.0, 1.0], "kick_trigger": 0, "kick_vec": [0.0, 0.0]},
        ]
    }
    a = acts["blue"][0]
    assert type(a["move"][0]) is float and type(a["kick_trigger"]) is int  # plain Python (ICD §2)


def test_clamp_bounds_map_strictly_inside_the_action_range():
    eps = np.float32(1e-6)
    acts = to_env_actions(np.array([[eps, np.float32(1) - eps]], dtype=np.float32))
    mx, my = acts["blue"][0]["move"]
    assert -1.0 < mx < -0.99999 and 0.99999 < my < 1.0


def test_accepts_torch_tensors():
    acts = to_env_actions(torch.tensor([[0.5, 0.5]]))
    assert acts["blue"][0]["move"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "bad",
    [np.array([0.5, 0.5]), np.zeros((1, 3)), np.array([[1.5, 0.5]]), np.array([[-0.1, 0.5]]), np.array([[np.nan, 0.5]])],
)
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        to_env_actions(bad)


def test_commanded_speed_fraction_matches_the_env_cap():
    x = np.array([[1.0, 1.0], [0.5, 0.5], [0.75, 0.5], [1.0, 0.5]])
    np.testing.assert_allclose(commanded_speed_fraction(x), [1.0, 0.0, 0.5, 1.0])


def test_actor_to_env_integration_n2_m2():
    torch.manual_seed(0)
    sc = config_from_dict({"shared": {"n_blue": 2, "n_red": 2}}).shared
    env, builder = MockEnv(sc), ObsBuilder(sc)
    actor = Actor(builder.dim)
    s, _ = env.reset(seed=0)
    for _ in range(50):
        obs = builder(parse_state(s, 2, 2))
        samples, _ = actor.act(torch.as_tensor(obs))
        s, term, trunc, _ = env.step(to_env_actions(samples[MOVE].numpy()))  # the env validates strictly
        if term or trunc:
            s, _ = env.reset(seed=1)
