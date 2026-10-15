# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: reward module tests (registry,
#     (1 - gamma) scaling, survival return = 1 - gamma^L, goto_target uses
#     s_next, deferred/unknown terms, effective_terminal truth table).
# Prompt summary: Implement Phase 3 step 5 of the issue #7 plan (reward module).
# Human review:   Pending review
# ==============================================================================
import dataclasses

import pytest

from marl.common.config import (
    BASE_REWARD_TERMS,
    SHAPING_TERMS_BUILT,
    SHAPING_TERMS_DEFERRED,
    ShapingTerm,
    config_from_dict,
)
from marl.rl.reward import BASE_TERMS, SHAPING_TERMS, RewardModule, effective_terminal

GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


def s_with(blue_pos, ball_pos):
    return {
        "step": 0,
        "blue": [{"id": 0, "pos": list(blue_pos), "vel": [0.0, 0.0], "has_possession": False}],
        "red": [],
        "ball": {"pos": list(ball_pos), "vel": [0.0, 0.0]},
    }


def with_reward(cfg, **reward_changes):
    """Bypass config validation, to test the module's own checks (defense in depth)."""
    rw = dataclasses.replace(cfg.rl.reward, **reward_changes)
    return dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, reward=rw))


def test_registry_matches_config_names():
    assert set(BASE_TERMS) == BASE_REWARD_TERMS
    assert set(SHAPING_TERMS) == SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED


def test_survival_is_scaled_and_paid_on_terminal_step():
    cfg = config_from_dict({})
    rm = RewardModule(cfg)
    assert rm.scale == pytest.approx(0.01)
    assert rm({}, {}, {}, False, False, {}) == pytest.approx(0.01)
    assert rm({}, {}, {}, True, False, {}) == pytest.approx(0.01)


@pytest.mark.parametrize("L", [1, 10, 96, 300])
def test_survival_return_is_one_minus_gamma_to_the_L(L):
    cfg = config_from_dict({})
    rm, g = RewardModule(cfg), cfg.gamma
    ret = sum(g**t * rm({}, {}, {}, t == L - 1, False, {}) for t in range(L))
    assert ret == pytest.approx(1 - g**L)  # in [0, 1); 0.95 at L = 300 (#15)


def test_goto_target_is_negative_distance_at_s_next():
    cfg = config_from_dict(GOTO)
    rm = RewardModule(cfg)
    s = s_with((-7.0, 0.0), (3.0, 4.0))  # far away before the step...
    s_next = s_with((0.0, 0.0), (3.0, 4.0))  # ...5 m away after it
    assert rm(s, {}, s_next, False, False, {}) == pytest.approx(-5.0 * 0.01)


def test_unknown_terms_raise_value_error():
    cfg = config_from_dict({})
    with pytest.raises(ValueError):
        RewardModule(with_reward(cfg, base="win"))
    with pytest.raises(ValueError):
        RewardModule(with_reward(cfg, shaping=(ShapingTerm("touch_bonus", True, 1.0),)))


def test_enabled_deferred_term_raises_not_implemented():
    cfg = config_from_dict({})
    with pytest.raises(NotImplementedError):
        RewardModule(with_reward(cfg, shaping=(ShapingTerm("pbrs_red_distance", True, 0.1),)))


def test_disabled_deferred_term_has_no_effect():
    cfg = config_from_dict({"rl": {"reward": {"shaping": [{"name": "pbrs_red_distance", "enabled": False}]}}})
    assert RewardModule(cfg)({}, {}, {}, False, False, {}) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "terminated, truncated, mode, expected",
    [
        (False, False, "infinite", False),
        (True, False, "infinite", True),
        (False, True, "infinite", False),  # time-out -> bootstrap, not terminal
        (True, True, "infinite", True),
        (False, True, "finite", True),  # finite mode relabels time-outs as terminal
        (False, False, "finite", False),
    ],
)
def test_effective_terminal(terminated, truncated, mode, expected):
    assert effective_terminal(terminated, truncated, mode) is expected
