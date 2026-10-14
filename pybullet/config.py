"""
Infrastructure configuration loader for MARL-Sports-Simulator.

TEAM NOTES
----------
This file is intentionally limited to application/infrastructure configuration.
It does not define Player/Ball/Field behavior and it does not implement RL.

The existing KeepAwayEnv already has its own KeepAwayConfig dataclass. Rather
than duplicating that schema here, YAML values under:

    game:
      config:

are kept as a dictionary and later passed directly into KeepAwayConfig by
runner.py.

That keeps this infrastructure layer aligned with keep_away_env.py and avoids
creating two competing sources of truth for environment parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Top-level infrastructure configuration sections.
#
# These dataclasses correspond to major sections of the YAML config. Their
# purpose is to give the rest of the project clean attribute access such as:
#
#     cfg.simulation.gui
#     cfg.training.learning_rate
#
# rather than repeatedly indexing nested dictionaries.
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str = "marl-sports-simulator"
    seed: int = 42


@dataclass
class RunConfig:
    # "demo" exercises the existing KeepAwayEnv without requiring an RL
    # implementation. "rl" delegates to the RL integration entrypoint.
    mode: str = "demo"  # demo | rl


@dataclass
class SimulationConfig:
    # KeepAwayEnv itself owns p.connect(...). This setting is translated into
    # render_mode="human" or render_mode=None by runner.py.
    gui: bool = False

    # If true, demo mode sleeps so the GUI approximately matches simulation
    # time. Training should generally leave this false.
    real_time: bool = False


@dataclass
class DemoConfig:
    # Demo mode exists so Issue #10 can be tested before PPO/IPPO/MAPPO exists.
    episodes: int = 1

    # "random" sends random 2-D velocity commands.
    # "stationary" sends [0, 0].
    policy: str = "random"  # random | stationary


@dataclass
class TrainingConfig:
    # This value is metadata for now. The actual RL implementation is owned by
    # the RL issue.
    algorithm: str = "ippo"

    # Dynamic function path used by rl_bridge.py.
    # Example future value: "train:run"
    entrypoint: str | None = None

    # Common RL hyperparameters are stored here so experiments can be
    # reproduced from YAML. The future RL trainer may consume these.
    total_episodes: int = 10_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class ModelConfig:
    # Path to an existing checkpoint, if a future RL trainer wants to resume.
    resume_from: str | None = None

    # Infrastructure supplies checkpoint locations/helpers; the RL trainer
    # decides when and what model objects to save.
    checkpoint_every_episodes: int = 500


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"


@dataclass
class CameraConfig:
    # None means "derive a reasonable camera distance from the field size".
    distance: float | None = None
    yaw: float = 0.0
    pitch: float = -89.9
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class DebugConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    draw_labels: bool = True


@dataclass
class GameConfig:
    # Only keep_away is implemented by the current repository.
    name: str = "keep_away"

    # IMPORTANT:
    # Keys inside this dictionary are expected to match KeepAwayConfig exactly.
    # runner.py converts this dict into:
    #
    #     KeepAwayConfig(**cfg.game.config)
    #
    # When the 2v2 environment is implemented, this section can be extended or
    # replaced by a new game-specific config path.
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Root configuration object passed through the infrastructure pipeline."""

    project: ProjectConfig
    run: RunConfig
    simulation: SimulationConfig
    game: GameConfig
    demo: DemoConfig
    training: TrainingConfig
    model: ModelConfig
    logging: LoggingConfig
    debug: DebugConfig

    # Keep the original YAML in case future tooling needs unprocessed values.
    raw: dict[str, Any] = field(repr=False)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """
    Validate that a YAML section is a mapping/object.

    This produces an early, readable error for malformed YAML instead of
    allowing a confusing failure later in the startup process.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping/object.")
    return value


def load_config(path: str | Path) -> AppConfig:
    """
    Load YAML -> AppConfig, then run basic infrastructure validation.

    Unknown game-specific keys are intentionally not validated here because
    KeepAwayConfig is the authority for the current game's parameter names.
    runner.py will produce a clear error if game.config contains an unsupported
    KeepAwayConfig argument.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("The top-level YAML value must be a mapping/object.")

    debug_raw = _mapping(raw.get("debug"), "debug")
    camera_raw = _mapping(debug_raw.get("camera"), "debug.camera")
    game_raw = _mapping(raw.get("game"), "game")

    cfg = AppConfig(
        project=ProjectConfig(**_mapping(raw.get("project"), "project")),
        run=RunConfig(**_mapping(raw.get("run"), "run")),
        simulation=SimulationConfig(**_mapping(raw.get("simulation"), "simulation")),
        game=GameConfig(
            name=game_raw.get("name", "keep_away"),
            config=_mapping(game_raw.get("config"), "game.config"),
        ),
        demo=DemoConfig(**_mapping(raw.get("demo"), "demo")),
        training=TrainingConfig(**_mapping(raw.get("training"), "training")),
        model=ModelConfig(**_mapping(raw.get("model"), "model")),
        logging=LoggingConfig(**_mapping(raw.get("logging"), "logging")),
        debug=DebugConfig(
            camera=CameraConfig(**camera_raw),
            draw_labels=debug_raw.get("draw_labels", True),
        ),
        raw=raw,
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    """
    Validate rules owned by infrastructure.

    This is intentionally not a physics/game validator. Game-specific validity
    belongs with the environment implementation.
    """
    if cfg.run.mode not in {"demo", "rl"}:
        raise ValueError("run.mode must be 'demo' or 'rl'.")

    if cfg.game.name != "keep_away":
        raise ValueError(
            "The current repository only implements game.name='keep_away'. "
            "Add another environment branch when the 2v2 environment exists."
        )

    if cfg.demo.episodes < 1:
        raise ValueError("demo.episodes must be >= 1.")

    if cfg.demo.policy not in {"random", "stationary"}:
        raise ValueError("demo.policy must be 'random' or 'stationary'.")

    if cfg.training.total_episodes < 1:
        raise ValueError("training.total_episodes must be >= 1.")

    if cfg.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0.")

    if not 0.0 <= cfg.training.gamma <= 1.0:
        raise ValueError("training.gamma must be in [0, 1].")

    if cfg.model.checkpoint_every_episodes < 1:
        raise ValueError("model.checkpoint_every_episodes must be >= 1.")
