# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-23
# Author:  Austin Beinder
# Usage:
#   - 2026-11-23: Generated this entire file from docs/soccer_env_plan.md §3:
#     the ICD adapter around the PyBullet soccer game
#     (pybulletGame.SoccerEnv), a subclass of the keep-away adapter.
#   - 2026-11-23: Passes shared.soccer.red_policy to the game
#     (docs/red_dribble_plan.md). (Austin Beinder)
# ==============================================================================
### AI GENERATED
"""ICD adapter around the PyBullet soccer game (``pybulletGame.SoccerEnv``).

Everything but the game it builds is the keep-away adapter's (``PyBulletEnv``): the seeded
spawn (drawn exactly like keepaway), the action translation, the state ``s``, the kick and
possession events, ``s["ball"]["passer"]``, the calls and the test hook. Soccer differs in:

* the game: a field with a goal at each end; blue defends the west goal and attacks east;
* red possession is an ordinary event, not the end of the episode (ICD §7, §8);
* ``terminated`` = a goal by either team on this step; ``truncated`` = the time limit without one;
* one more event, after the kicks and possession changes (ICD §9)::

      {"type": "goal", "team": "blue" | "red", "scorer": ("blue", i) | ("red", j) | None,
       "own_goal": bool, "carried": bool}

  ``team`` gets the point (any ball in a goal scores for the side that doesn't defend it);
  ``scorer`` is whoever last held the ball.

Red's fixed policy is the game's, chosen by ``shared.soccer.red_policy``: ``"chase"`` (every red
chases the ball) or ``"dribble"`` (a red with the ball carries it into blue's goal). The RL side computes the reward from
the goal event (``rl.reward.base = "goal"``).
"""

from __future__ import annotations

from typing import Any

from marl.common.config import SharedConfig
from marl.envs.pybullet_env import PyBulletEnv
from pybulletGame import SoccerConfig, SoccerEnv, Team


class PyBulletSoccerEnv(PyBulletEnv):
    """ICD-conformant wrapper around ``SoccerEnv``."""

    SCENARIOS = ("soccer",)

    @staticmethod
    def _make_game(cfg: SharedConfig, render_mode: str | None) -> Any:
        return SoccerEnv(
            SoccerConfig(
                blue_max_speed=cfg.max_speed,
                red_max_speed=cfg.red_max_speed,
                possession_speed_multiplier=cfg.carry_speed_factor,
                free_speed_multiplier=cfg.free_speed_factor,
                player_radius=cfg.player_radius,
                max_kick_speed=cfg.max_kick_speed,
                ball_damping=cfg.ball_damping,
                max_game_time=cfg.max_episode_steps * cfg.control_k * cfg.physics_dt,
                field_length=cfg.field_size[0],
                field_width=cfg.field_size[1],
                goal_width=cfg.soccer.goal_width,
                goal_depth=cfg.soccer.goal_depth,
                action_repeat=cfg.control_k,
                n_blue=cfg.n_blue,
                n_red=cfg.n_red,
                red_policy=cfg.soccer.red_policy,  # AI GENERATED (2026-11-23)
            ),
            render_mode=render_mode,
        )

    def _game_events(self, game_info: dict) -> list[dict]:
        goal = game_info["goal"]
        if goal is None:
            return []
        return [{
            "type": "goal",
            "team": "blue" if goal.scoring_team == Team.BLUE else "red",
            "scorer": self._holder_id(goal.scorer),
            "own_goal": bool(goal.own_goal),
            "carried": bool(goal.carried),
        }]
### END AI GENERATED
