# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: tests for config loading and
#     validation (plan "Verification": config, derived gamma).
#   - 2026-10-14: Added tests for the built pbrs_ball_acquisition term and the
#     non-negative shaping coef check, and a test that the PBRS ablation config
#     differs from the survival-only control config only in the reward.
# Prompt summary: Implement Phase 3 step 1 of the issue #7 plan (config +
#   validation) with expected-input, unexpected-input and corner-case tests.
# Human review:   Pending review
# ==============================================================================
import dataclasses
from pathlib import Path

import pytest

from marl.common.config import (
    Config,
    config_from_dict,
    config_to_dict,
    load_config,
    with_overrides,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GOTO = {"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}}


# --- Expected input ----------------------------------------------------------
def test_defaults_are_valid():
    cfg = config_from_dict({})
    assert cfg == Config()
    assert cfg.shared.n_blue == 1 and cfg.shared.n_red == 1
    assert cfg.shared.start.start_mix == 0.5


def test_derived_gamma_defaults_to_0_99():
    cfg = config_from_dict({})
    assert cfg.shared.dt_control == pytest.approx(0.05)
    assert cfg.gamma == pytest.approx(0.99)


def test_gamma_follows_control_k():
    # Halving control_k halves dt, so 1 - gamma halves too (gamma counts steps, #19).
    cfg = config_from_dict({"shared": {"control_k": 6}})
    assert cfg.gamma == pytest.approx(0.995)


@pytest.mark.parametrize("name", ["keepaway_1v1.toml", "goto_target.toml"])
def test_shipped_config_files_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.gamma == pytest.approx(0.99)


def test_keepaway_file_equals_defaults():
    # The file documents every default; drift between the two would be confusing.
    assert load_config(CONFIGS / "keepaway_1v1.toml") == Config()


def test_int_accepted_for_float_field():
    cfg = config_from_dict({"shared": {"max_speed": 3}})
    assert cfg.shared.max_speed == 3.0 and isinstance(cfg.shared.max_speed, float)


def test_round_trip_and_overrides():
    cfg = config_from_dict(GOTO)
    assert config_from_dict(config_to_dict(cfg)) == cfg
    cfg2 = with_overrides(cfg, {"run": {"seed": 7}})
    assert cfg2.run.seed == 7 and cfg2.shared == cfg.shared


def test_reset_options_match_icd_keys():
    opts = config_from_dict({}).shared.start.to_reset_options()
    assert set(opts) == {"start_mix", "require_blue_closer", "min_clearance", "min_red_ball_dist"}


def test_disabled_deferred_shaping_term_is_allowed():
    cfg = config_from_dict({"rl": {"reward": {"shaping": [{"name": "pbrs_red_distance", "enabled": False}]}}})
    assert cfg.rl.reward.shaping[0].name == "pbrs_red_distance"


def test_pbrs_config_differs_from_the_control_only_in_the_reward():
    # keepaway_1v1_pbrs.toml relies on the defaults equaling keepaway_1v1.toml, so the
    # ablation's two arms differ ONLY in the reward. This fails if either file drifts.
    ctl, pbrs = load_config(CONFIGS / "keepaway_1v1.toml"), load_config(CONFIGS / "keepaway_1v1_pbrs.toml")
    assert dataclasses.replace(pbrs, rl=dataclasses.replace(pbrs.rl, reward=ctl.rl.reward)) == ctl
    assert [(t.name, t.enabled, t.coef) for t in pbrs.rl.reward.shaping] == [("pbrs_ball_acquisition", True, 4.0)]


def test_built_pbrs_term_can_be_enabled():
    cfg = config_from_dict({"rl": {"reward": {"shaping": [{"name": "pbrs_ball_acquisition", "enabled": True, "coef": 2}]}}})
    term = cfg.rl.reward.shaping[0]
    assert (term.name, term.enabled, term.coef) == ("pbrs_ball_acquisition", True, 2.0)


def test_n2_m2_is_valid():
    cfg = config_from_dict({"shared": {"n_blue": 2, "n_red": 2}})
    assert cfg.shared.n_blue == 2


# --- Unexpected input: ValueError ----------------------------------------------
@pytest.mark.parametrize(
    "data",
    [
        {"bogus": 1},  # unknown top-level section
        {"shared": {"n_blues": 2}},  # misspelled key
        {"shared": {"start": {"start_mixx": 0.5}}},  # nested misspelling
        {"rl": {"ppo": {"learning_rate": 1e-3}}},
    ],
)
def test_unknown_key_raises(data):
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"shared": {"n_blue": 1.5}},  # float for int
        {"shared": {"n_blue": True}},  # bool is not an int here
        {"shared": {"max_speed": "fast"}},
        {"rl": {"kick_enabled": 0}},  # int is not a bool
        {"shared": {"field_size": [16.0]}},  # wrong length
        {"shared": {"start": 0.5}},  # scalar for a table
    ],
)
def test_wrong_type_raises(data):
    with pytest.raises(ValueError):
        config_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"shared": {"scenario": "soccer"}},
        {"rl": {"critic_input": "central"}},
        {"rl": {"horizon_mode": "episodic"}},
        {"rl": {"reward": {"base": "win"}}},
        {"rl": {"reward": {"shaping": [{"name": "touch_bonus"}]}}},
    ],
)
def test_unknown_option_raises_value_error(data):
    with pytest.raises(ValueError):
        config_from_dict(data)


@pytest.mark.parametrize(
    "data, match",
    [
        ({"shared": {"control_k": 1200}}, "horizon_seconds"),  # dt = 5 s = H -> gamma = 0
        ({"rl": {"horizon_seconds": 0.05}}, "horizon_seconds"),  # dt == H exactly
        ({"shared": {"start": {"min_red_ball_dist": 0.5}}}, "possession_radius"),  # == radius
        ({"shared": {"start": {"min_clearance": 0.4}}}, "overlapping"),
        ({"shared": {"start": {"start_mix": 1.5}}}, "start_mix"),
        ({"shared": {"n_red": 0}}, "keepaway"),
        ({"shared": {"scenario": "goto_target"}, "rl": {"reward": {"base": "goto_target"}}}, "goto_target"),  # n_red=1
        ({"rl": {"reward": {"base": "goto_target"}}}, "needs rl.reward.base"),  # mismatch with keepaway
        ({"rl": {"ppo": {"num_minibatches": 30}}}, "divisible"),
        ({"shared": {"carry_speed_factor": 0.0}}, "carry_speed_factor"),
        ({"shared": {"max_speed": float("nan")}}, "finite"),
        ({"run": {"total_steps": 100}}, "total_steps"),
        (
            {"rl": {"reward": {"shaping": [{"name": "pbrs_red_distance"}, {"name": "pbrs_red_distance"}]}}},
            "duplicate",
        ),
        (
            {"rl": {"reward": {"shaping": [{"name": "pbrs_ball_acquisition", "enabled": True, "coef": -1.0}]}}},
            "coef must be >= 0",
        ),
    ],
)
def test_invalid_values_raise(data, match):
    with pytest.raises(ValueError, match=match):
        config_from_dict(data)


# --- Deferred options: NotImplementedError ------------------------------------
@pytest.mark.parametrize(
    "data",
    [
        {"rl": {"horizon_mode": "finite"}},
        {"rl": {"obs_ordering": "index"}},
        {"rl": {"obs_mode": "radius"}},
        {"rl": {"kick_enabled": True}},
        {"rl": {"reward": {"shaping": [{"name": "pbrs_red_distance", "enabled": True, "coef": 0.1}]}}},
    ],
)
def test_deferred_option_raises_not_implemented(data):
    with pytest.raises(NotImplementedError):
        config_from_dict(data)


def test_overrides_are_validated_too():
    cfg = config_from_dict({})
    with pytest.raises(ValueError):
        with_overrides(cfg, {"shared": {"n_red": 0}})
