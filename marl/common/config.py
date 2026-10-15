# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: the config schema (shared ICD §10
#     keys + RL-only keys), TOML loading, strict unknown-key rejection,
#     validation at load, and the derived discount factor (design #19, #21).
#   - 2026-10-11: Set the red_max_speed default to the calibrated 1.8 (step 10).
# Prompt summary: Implement Phase 3 step 1 of the issue #7 plan (config +
#   validation) following the student's design decisions.
# Human review:   Pending review
# ==============================================================================
"""Configuration schema, TOML loading and validation.

The config has three parts:

* ``shared`` -- values the env and the RL layer must agree on (ICD §10).
* ``rl``     -- RL-only settings (design decisions #1, #11-#14, #19, #21; PPO defaults D).
* ``run``    -- seeds, training budget, evaluation schedule, output directory.

Everything is validated at load (#21): an unknown key or option value raises
``ValueError``; an option that exists in the interface but is not built yet
raises ``NotImplementedError``.

This module is torch-free on purpose, so the env owner can import it.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Option values ---------------------------------------------------------
# name -> (built values, interface-only values). Interface-only values raise
# NotImplementedError ("design the interface, defer the implementation", #21).
OPTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "scenario": (frozenset({"keepaway", "goto_target"}), frozenset()),
    "critic_input": (frozenset({"local", "global"}), frozenset()),
    "horizon_mode": (frozenset({"infinite"}), frozenset({"finite"})),
    "obs_ordering": (frozenset({"distance"}), frozenset({"index"})),
    "obs_mode": (frozenset({"full"}), frozenset({"radius"})),
}

# Reward term names (#14). The registry in marl/rl/reward.py must match these
# (a test checks it); they live here so the config can be validated at load.
BASE_REWARD_TERMS: frozenset[str] = frozenset({"survival", "goto_target"})
SHAPING_TERMS_BUILT: frozenset[str] = frozenset()
SHAPING_TERMS_DEFERRED: frozenset[str] = frozenset({"pbrs_red_distance"})  # #15

# The base reward each scenario needs (survival is constant in goto_target,
# and -distance is meaningless in keep-away).
SCENARIO_BASE_REWARD: dict[str, str] = {"keepaway": "survival", "goto_target": "goto_target"}


# --- Schema ----------------------------------------------------------------
@dataclass(frozen=True)
class StartConfig:
    """Start-distribution options; these are also the ``reset()`` options (ICD §9)."""

    start_mix: float = 0.5  # fraction of possession starts (decision F)
    require_blue_closer: bool = True
    min_clearance: float = 1.0  # meters between any two players at spawn
    min_red_ball_dist: float = 3.0  # meters; must exceed possession_radius

    def to_reset_options(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SharedConfig:
    """Values both sides must agree on (ICD §10). Mock defaults = decision G."""

    scenario: str = "keepaway"
    n_blue: int = 1
    n_red: int = 1
    physics_dt: float = 1.0 / 240.0
    control_k: int = 12
    max_episode_steps: int = 300
    field_size: tuple[float, float] = (16.0, 10.0)  # FULL extents [Lx, Ly], meters
    max_speed: float = 2.0
    red_max_speed: float = 1.8  # calibrated 2026-10-11: flee survives 50% of max steps (#22)
    carry_speed_factor: float = 0.75
    max_kick_speed: float = 6.0  # unused at milestone 1
    possession_radius: float = 0.5
    player_radius: float = 0.25
    start: StartConfig = field(default_factory=StartConfig)

    @property
    def dt_control(self) -> float:
        """Seconds per control step (one ``env.step`` call)."""
        return self.control_k * self.physics_dt


@dataclass(frozen=True)
class ShapingTerm:
    name: str
    enabled: bool = False
    coef: float = 0.0


@dataclass(frozen=True)
class RewardConfig:
    base: str = "survival"
    shaping: tuple[ShapingTerm, ...] = ()


@dataclass(frozen=True)
class PPOConfig:
    """PPO defaults (open item D, accepted 2026-10-10)."""

    rollout_steps: int = 2048  # T, control steps per rollout
    num_minibatches: int = 32  # minibatch size = T * N / num_minibatches
    update_epochs: int = 10
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    lr: float = 3e-4
    adam_eps: float = 1e-5
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class RLConfig:
    horizon_seconds: float = 5.0  # gamma is derived from this (#19)
    critic_input: str = "global"
    horizon_mode: str = "infinite"
    obs_ordering: str = "distance"
    obs_mode: str = "full"
    kick_enabled: bool = False
    beta_eps: float = 1e-6
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0  # training seed (env + torch)
    total_steps: int = 1_000_000  # control steps
    eval_every: int = 10  # PPO iterations between evaluations
    eval_episodes: int = 50
    # Fixed across training seeds, so every run is evaluated on the same episodes (#23).
    eval_seed: int = 20261019
    out_dir: str = "runs"


@dataclass(frozen=True)
class Config:
    shared: SharedConfig = field(default_factory=SharedConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def gamma(self) -> float:
        """Discount factor derived from the control rate and the horizon (#19)."""
        return 1.0 - self.shared.dt_control / self.rl.horizon_seconds


# --- Loading ---------------------------------------------------------------
def load_config(path: str | Path) -> Config:
    """Load a TOML config file, fill in defaults, and validate it."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build a validated Config from nested dicts (missing keys take defaults)."""
    cfg = _build(Config, data, "config")
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Plain nested dicts (tuples become lists), e.g. for saving with a run."""
    return _plain(dataclasses.asdict(cfg))


def with_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a new validated Config with nested overrides applied."""
    merged = _deep_merge(config_to_dict(cfg), overrides)
    return config_from_dict(merged)


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a table, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    hints = typing.get_type_hints(cls)
    return cls(**{k: _coerce(hints[k], v, f"{path}.{k}") for k, v in data.items()})


def _coerce(tp: Any, value: Any, path: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if typing.get_origin(tp) is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    # bool is a subclass of int in Python, so check it explicitly everywhere.
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported schema type {tp!r}")  # a bug in the schema


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --- Validation ------------------------------------------------------------
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _check_option(name: str, value: str) -> None:
    built, deferred = OPTIONS[name]
    if value in deferred:
        raise NotImplementedError(f"{name}={value!r} is interface-only and not built yet (#21)")
    _require(value in built, f"{name}={value!r} is not a valid option; choose from {sorted(built | deferred)}")


def _check_finite(obj: Any, path: str) -> None:
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            _check_finite(v, f"{path}.{f.name}")
        elif isinstance(v, tuple):
            for i, x in enumerate(v):
                if dataclasses.is_dataclass(x):
                    _check_finite(x, f"{path}.{f.name}[{i}]")
                elif isinstance(x, float):
                    _require(math.isfinite(x), f"{path}.{f.name}[{i}] must be finite")
        elif isinstance(v, float):
            _require(math.isfinite(v), f"{path}.{f.name} must be finite")


def validate(cfg: Config) -> None:
    """Raise ValueError / NotImplementedError if the config is invalid."""
    _check_finite(cfg, "config")
    _validate_shared(cfg.shared)
    _validate_rl(cfg)
    _validate_run(cfg)


def _validate_shared(s: SharedConfig) -> None:
    _check_option("scenario", s.scenario)
    _require(s.n_blue >= 1, "shared.n_blue must be >= 1")
    _require(s.n_red >= 0, "shared.n_red must be >= 0")
    if s.scenario == "keepaway":
        _require(s.n_red >= 1, "keepaway needs n_red >= 1 (termination = red possession)")
    else:  # goto_target (ICD §9)
        _require(s.n_blue == 1 and s.n_red == 0, "goto_target needs n_blue = 1, n_red = 0")
    _require(s.physics_dt > 0, "shared.physics_dt must be > 0")
    _require(s.control_k >= 1, "shared.control_k must be >= 1")
    _require(s.max_episode_steps >= 1, "shared.max_episode_steps must be >= 1")
    for name in ("max_speed", "red_max_speed", "max_kick_speed", "possession_radius", "player_radius"):
        _require(getattr(s, name) > 0, f"shared.{name} must be > 0")
    _require(0 < s.carry_speed_factor <= 1, "shared.carry_speed_factor must be in (0, 1]")
    _require(
        all(L > 2 * s.player_radius for L in s.field_size),
        "shared.field_size must exceed the player diameter on both axes",
    )
    st = s.start
    _require(0.0 <= st.start_mix <= 1.0, "shared.start.start_mix must be in [0, 1]")
    _require(
        st.min_clearance >= 2 * s.player_radius,
        "shared.start.min_clearance must be >= 2 * player_radius (no overlapping spawns)",
    )
    _require(
        st.min_red_ball_dist > s.possession_radius,
        "shared.start.min_red_ball_dist must be > possession_radius (no start inside red's reach)",
    )


def _validate_rl(cfg: Config) -> None:
    rl, s = cfg.rl, cfg.shared
    for name in ("critic_input", "horizon_mode", "obs_ordering", "obs_mode"):
        _check_option(name, getattr(rl, name))
    if rl.kick_enabled:
        raise NotImplementedError("rl.kick_enabled=true: the kick heads come after milestone 1 (#21)")
    _require(rl.horizon_seconds > 0, "rl.horizon_seconds must be > 0")
    _require(
        s.dt_control < rl.horizon_seconds,
        f"control_k * physics_dt ({s.dt_control:.4g} s) must be < horizon_seconds "
        f"({rl.horizon_seconds:.4g} s), or gamma would be <= 0 (#19)",
    )
    _require(0 < rl.beta_eps <= 1e-2, "rl.beta_eps must be in (0, 0.01]")

    rw = rl.reward
    _require(rw.base in BASE_REWARD_TERMS, f"rl.reward.base={rw.base!r} unknown; choose from {sorted(BASE_REWARD_TERMS)}")
    want = SCENARIO_BASE_REWARD[s.scenario]
    _require(rw.base == want, f"scenario {s.scenario!r} needs rl.reward.base = {want!r}, got {rw.base!r}")
    seen: set[str] = set()
    for term in rw.shaping:
        known = SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED
        _require(term.name in known, f"rl.reward.shaping: unknown term {term.name!r}; choose from {sorted(known)}")
        _require(term.name not in seen, f"rl.reward.shaping: duplicate term {term.name!r}")
        seen.add(term.name)
        if term.enabled and term.name in SHAPING_TERMS_DEFERRED:
            raise NotImplementedError(f"shaping term {term.name!r} is interface-only for milestone 1 (#15)")

    p = rl.ppo
    for name in ("rollout_steps", "num_minibatches", "update_epochs"):
        _require(getattr(p, name) >= 1, f"rl.ppo.{name} must be >= 1")
    batch = p.rollout_steps * s.n_blue
    _require(
        batch % p.num_minibatches == 0,
        f"rollout_steps * n_blue ({batch}) must be divisible by num_minibatches ({p.num_minibatches})",
    )
    _require(0 < p.clip_eps < 1, "rl.ppo.clip_eps must be in (0, 1)")
    _require(0 <= p.gae_lambda <= 1, "rl.ppo.gae_lambda must be in [0, 1]")
    for name in ("lr", "adam_eps", "max_grad_norm"):
        _require(getattr(p, name) > 0, f"rl.ppo.{name} must be > 0")
    _require(p.ent_coef >= 0, "rl.ppo.ent_coef must be >= 0")
    _require(len(p.hidden_sizes) >= 1 and all(h >= 1 for h in p.hidden_sizes), "rl.ppo.hidden_sizes must be positive")


def _validate_run(cfg: Config) -> None:
    r = cfg.run
    _require(r.seed >= 0 and r.eval_seed >= 0, "run.seed and run.eval_seed must be >= 0")
    _require(
        r.total_steps >= cfg.rl.ppo.rollout_steps,
        "run.total_steps must be >= rl.ppo.rollout_steps (at least one iteration)",
    )
    _require(r.eval_every >= 1, "run.eval_every must be >= 1")
    _require(r.eval_episodes >= 1, "run.eval_episodes must be >= 1")
    _require(bool(r.out_dir), "run.out_dir must not be empty")
