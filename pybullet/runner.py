"""
Run orchestration for the current repository.

TEAM NOTES
----------
The existing KeepAwayEnv already owns:
- the PyBullet connection
- environment reset
- physics stepping
- reward/termination logic
- Player/Ball/Field creation

Therefore this infrastructure layer must NOT create a second PyBullet
connection or duplicate the environment loop internally.

This module's job is to:
1. convert YAML game config into KeepAwayConfig,
2. create KeepAwayEnv,
3. select demo mode or the future RL entrypoint,
4. guarantee env.close() is called.
"""

from __future__ import annotations

import time

import numpy as np

from config import AppConfig
from debug_tools import apply_debug_view
from reporting import MetricsLogger
from rl_bridge import invoke_rl_loop
from runtime import RunContext


def create_environment(cfg: AppConfig):
    """
    Construct the environment already implemented by the repository.

    The game.config YAML keys are passed directly into KeepAwayConfig, so this
    creates a useful integration contract between Issue #10 and the environment
    issue without redefining the environment schema here.
    """
    if cfg.game.name != "keep_away":
        raise ValueError(f"Unsupported game: {cfg.game.name}")

    # Lazy imports keep pure infrastructure tests lightweight and preserve the
    # repository's current sibling-import structure.
    from keep_away_env import KeepAwayConfig, KeepAwayEnv

    try:
        env_cfg = KeepAwayConfig(**cfg.game.config)
    except TypeError as exc:
        raise ValueError(
            "game.config contains a key that KeepAwayConfig does not accept. "
            "Keep YAML parameter names aligned with keep_away_env.py."
        ) from exc

    # KeepAwayEnv itself calls p.connect(...); infrastructure only chooses
    # whether the environment should request GUI or DIRECT mode.
    render_mode = "human" if cfg.simulation.gui else None

    return KeepAwayEnv(
        config=env_cfg,
        render_mode=render_mode,
    )


def run(cfg: AppConfig, context: RunContext) -> None:
    """
    Create one environment and dispatch to the selected run mode.

    env.close() lives in finally so the PyBullet connection is cleaned up even
    if the demo, future trainer, or user interrupts with an exception.
    """
    env = create_environment(cfg)

    try:
        if cfg.run.mode == "demo":
            run_demo(env, cfg, context)
        elif cfg.run.mode == "rl":
            invoke_rl_loop(env, cfg, context)
        else:
            raise ValueError(f"Unsupported run mode: {cfg.run.mode}")
    finally:
        env.close()


def run_demo(env, cfg: AppConfig, context: RunContext) -> None:
    """
    Exercise the real KeepAwayEnv before the RL implementation exists.

    This is intentionally NOT a learning algorithm. It exists so teammates can
    test the complete infrastructure pipeline today:

        YAML -> main -> environment -> metrics -> output directory

    The action shape matches KeepAwayEnv.step(): [vx, vy].
    """
    logger = MetricsLogger(context.metrics_path)

    # np.random.default_rng gives demo actions their own reproducible generator.
    rng = np.random.default_rng(cfg.project.seed)

    for episode in range(1, cfg.demo.episodes + 1):

        # KeepAwayEnv.reset(seed=...) seeds its game-level randomness.
        observation, _ = env.reset(
            seed=cfg.project.seed + episode - 1
        )

        # The environment already sets a camera in human mode. This helper
        # applies Issue #10's optional debug overrides after reset.
        apply_debug_view(cfg, env)

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        while not (terminated or truncated):
            if cfg.demo.policy == "stationary":
                action = np.zeros(2, dtype=float)
            else:
                # Generate the same 2-D desired-velocity action shape the RL
                # layer will eventually produce. KeepAwayEnv performs its own
                # final speed clipping.
                action = rng.uniform(
                    low=-env.config.blue_max_speed,
                    high=env.config.blue_max_speed,
                    size=2,
                )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            # Training should normally run as fast as possible. For a human
            # demo, real_time=true makes the GUI easier to watch.
            if cfg.simulation.gui and cfg.simulation.real_time:
                time.sleep(env.control_dt)

        # Team is an Enum in the current code, so prefer its .name for CSV.
        winner = getattr(
            info.get("winner"),
            "name",
            str(info.get("winner")),
        )

        logger.log({
            "episode": episode,
            "winner": winner,
            "reward": total_reward,
            "game_time": info.get("game_time", 0.0),
            "blue_total_possession_time": info.get(
                "blue_total_possession_time",
                0.0,
            ),
        })

        print(
            f"Episode {episode}/{cfg.demo.episodes}: "
            f"winner={winner}, reward={total_reward:+.0f}, "
            f"time={info.get('game_time', 0.0):.2f}s"
        )

    print(f"Run output: {context.run_dir}")
