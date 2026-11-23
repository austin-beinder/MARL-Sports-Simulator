"""
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-23
Author:  Austin Beinder
Usage:
  - 2026-11-23: Generated this entire file from docs/soccer_env_plan.md: SoccerConfig, SoccerEnv
    and the joystick demo. Modelled on keep_away_env.py (whose static helpers it reuses), with
    the goals from goal.py. Design decisions (red possession isn't terminal, the first goal ends
    the game, red just goes to the ball, blue attacks east) by Austin Beinder.

The "gym" layer for soccer. Like keep_away_env.KeepAwayEnv, it owns its own pybullet simulation
and is the only thing the RL code needs to talk to:
- reset() builds the field with a goal at each end, the players and the ball.
- step(action, kicks) routes the RL action to the blue players, moves the fixed-policy red players,
  ticks pybullet forward and checks the goals every physics tick.

Rules of soccer (this version):
- Blue is the learner, red follows a fixed policy: every red chases the ball. A red that gets the
  ball is standing on it, so it keeps still until a blue takes it back. (A red that dribbles to
  goal comes in a later version, through red_policy.)
- Blue defends the west goal (x = -field_length / 2) and attacks east.
- Possession is keep-away's: touching a loose ball takes it, and an opponent touching the carrier
  (or the ball) takes it from them, either way round. Only a kick releases the ball; teammates
  can't hand it over. Red getting the ball doesn't end the game.
- A blue player holding the ball can kick it (step's kicks), exactly as in keep-away.
- The first goal ends the game (terminated): any ball in a goal is a point for the team that
  doesn't defend it, kicked, carried in, or an own goal. Reward +1 if blue scored, -1 if red did.
- If nobody scores within max_game_time, the game is cut off (truncated) as a draw, reward 0.
- Blue moves slower while it has the ball (possession_speed_multiplier); its speed without the
  ball can be scaled too (free_speed_multiplier).
- By default the two teams line up kickoff_distance either side of the centre spot, spread across
  the field spawn_separation apart, with the ball on the spot. reset's options override that.
"""

### AI GENERATED
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pybullet as p

from ball import Ball
from entity import Entity
from field import Field
from goal import PANEL_THICKNESS, standard_goals
from keep_away_env import KeepAwayEnv
from player import Player
from teams import Team

RESET_OPTIONS = frozenset({"blue_position", "red_position", "blue_positions", "red_positions",
                           "ball_position", "blue_has_ball", "ball_holder", "red_ball_holder"})


@dataclass
class SoccerConfig:
    """Tunable parameters for the soccer game. Speeds are in m/s, times in seconds, lengths in
    metres. The defaults are keep-away's (KeepAwayConfig); training passes its own (SharedConfig)."""

    # Players
    blue_max_speed: float = 3.0
    red_max_speed: float = 1.8
    red_gain: float = 240.0  # red's P-controller gain toward the ball (see KeepAwayConfig.red_gain)
    possession_speed_multiplier: float = 0.75  # blue's speed is scaled by this while it holds the ball
    free_speed_multiplier: float = 1.0  # ...and by this while it doesn't
    player_radius: float = 0.3

    # Ball: a kick's speed is capped at max_kick_speed, also the ball's top speed. A ball sent off
    # at v m/s rolls about ln(1 + v) / ball_damping metres.
    max_kick_speed: float = 6.0
    ball_damping: float = 0.4

    # Rules: the game is a draw if nobody scores within max_game_time
    max_game_time: float = 60.0

    # Field and goals (goal.standard_goals: one in the middle of each end wall)
    field_length: float = 21.0
    field_width: float = 13.6
    goal_width: float = 2.0
    goal_depth: float = 0.5

    # Default spawn: each team in a line across the field, kickoff_distance from the centre spot
    # on its own half, players spawn_separation apart; the ball on the centre spot
    kickoff_distance: float = 2.0
    spawn_separation: float = 2.0

    # Physics steps (1/240 s each) per call to step()
    action_repeat: int = 8

    # Team sizes. n_red = 0 is an empty net.
    n_blue: int = 1
    n_red: int = 1


class SoccerEnv:

    def __init__(self, config: Optional[SoccerConfig] = None, render_mode: Optional[str] = None):
        """render_mode: "human" opens the pybullet GUI, None runs headless (much faster, for
        training). Like KeepAwayEnv, every game has its own pybullet simulation, and pybullet
        allows only one GUI per process."""
        self.config = config or SoccerConfig()
        cfg = self.config
        if cfg.n_blue < 1 or cfg.n_red < 0:
            raise ValueError(f"need n_blue >= 1 and n_red >= 0, got {cfg.n_blue} and {cfg.n_red}")
        self.render_mode = render_mode
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("pybullet couldn't connect; it allows only one GUI (render_mode='human') per process")
        self._connected = True  # close() disconnects once: pybullet reuses freed connection ids
        self.rng = random.Random()  # the steal roll's generator, per game

        self.time_step = p.getPhysicsEngineParameters(physicsClientId=self.client)["fixedTimeStep"]
        self.control_dt = cfg.action_repeat * self.time_step
        self.max_game_steps = max(1, round(cfg.max_game_time / self.time_step))

        # Blue defends the west goal, red the east one
        self.goal_specs = standard_goals(cfg.field_length, cfg.field_width, width=cfg.goal_width,
                                         depth=cfg.goal_depth, west_team=Team.BLUE)

        self.field: Optional[Field] = None
        self.ball: Optional[Ball] = None
        self.blues: list = []
        self.reds: list = []
        self.kicks_applied: list = []
        self.possession_events: list = []
        self.game_steps = 0
        self.winner = Team.NONE
        self.last_goal = None  # the GoalEvent that ended the game
        self.score = {Team.BLUE: 0, Team.RED: 0}

    # ---------- Gym interface ----------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new game. Rebuilds the whole simulation so nothing carries over from the last game.
        seed seeds this game's random generator (self.rng), which decides whether steals succeed.

        options can override the spawn (anything left out keeps the default spawn):
        - "blue_position", "red_position": (x, y) in metres, for a team of one player
        - "blue_positions", "red_positions": a list of (x, y), one per player, for a team of any size
        - "ball_position": (x, y) in metres
        - "blue_has_ball": True to start the game with the first blue holding the ball
        - "ball_holder": the index of the blue player who starts with the ball
        - "red_ball_holder": the index of the red player who starts with the ball

        Returns (observation, info).
        """
        if seed is not None:
            self.rng.seed(seed)
        options = dict(options or {})
        unknown = set(options) - RESET_OPTIONS
        if unknown:
            raise ValueError(f"unknown reset option(s) {sorted(unknown)}")
        holders = [key for key in ("ball_holder", "blue_has_ball", "red_ball_holder") if key in options]
        if len(holders) > 1:
            raise ValueError(f"give at most one of {holders}")
        cfg = self.config
        blue_holder = options.get("ball_holder", 0 if options.get("blue_has_ball") else None)
        red_holder = options.get("red_ball_holder")
        if blue_holder is not None and not 0 <= blue_holder < cfg.n_blue:
            raise ValueError(f"ball_holder must be a blue index below {cfg.n_blue}, got {blue_holder}")
        if red_holder is not None and not 0 <= red_holder < cfg.n_red:
            raise ValueError(f"red_ball_holder must be a red index below {cfg.n_red}, got {red_holder}")

        c = self.client
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=c)  # don't draw while rebuilding
        p.resetSimulation(physicsClientId=c)
        Entity.clear_registry(c)  # only this game's entities (goals included)

        self.field = Field(length=cfg.field_length, width=cfg.field_width, physics_client=c,
                           goals=self.goal_specs)
        blue_starts = self._team_spawn(options, "blue", -cfg.kickoff_distance, cfg.n_blue)
        red_starts = self._team_spawn(options, "red", cfg.kickoff_distance, cfg.n_red)
        if blue_holder is not None:
            ball_start = blue_starts[blue_holder].copy()
        elif red_holder is not None:
            ball_start = red_starts[red_holder].copy()
        elif "ball_position" in options:
            ball_start = np.array([*options["ball_position"][:2], 0.0], dtype=float)
        else:
            ball_start = np.zeros(3)

        self.blues = [Player(team=Team.BLUE, radius=cfg.player_radius, initial_position=start,
                             maxJointVelocity=cfg.blue_max_speed, physics_client=c) for start in blue_starts]
        self.reds = [Player(team=Team.RED, radius=cfg.player_radius, initial_position=start,
                            maxJointVelocity=cfg.red_max_speed, physics_client=c) for start in red_starts]
        self.ball = Ball(initial_position=ball_start, maxJointVelocity=cfg.max_kick_speed,
                         linear_damping=cfg.ball_damping, physics_client=c)
        self.ball.rng = self.rng
        self.ball.allow_teammate_handoff = False  # only a kick releases the ball (ICD §8)
        self.ball.play_area = (cfg.field_length / 2, cfg.field_width / 2)  # kicks stay inside the walls...
        self.ball.goal_lanes = list(self.goal_specs)  # ...or go into a goal
        if blue_holder is not None:
            self.ball._player_takes_possession(self.blues[blue_holder])
        elif red_holder is not None:
            self.ball._player_takes_possession(self.reds[red_holder])
        for entity in (*self.blues, *self.reds, self.ball):
            entity.sync_from_pybullet()

        self.game_steps = 0
        self.winner = Team.NONE
        self.last_goal = None
        self.score = {Team.BLUE: 0, Team.RED: 0}
        # Every change of ball holder during the last step() call, as (physics step, old holder,
        # new holder), like KeepAwayEnv.possession_events
        self.possession_events = []
        self._last_holder = self.ball.possession_player
        self.kicks_applied = [False] * len(self.blues)

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)  # hide the side panels
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=c)
            # Looking straight down with +x to the right and +y up the screen, matching the joystick
            p.resetDebugVisualizerCamera(cameraDistance=self._camera_distance(), cameraYaw=0,
                                         cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=c)
        return self.get_observation(), self._info()

    def step(self, action, kicks=None):
        """
        action: one (vx, vy) per blue player in m/s, in index order (a single (vx, vy) for a team
        of one). Each is capped at blue_max_speed and scaled by possession_speed_multiplier while
        that player has the ball (free_speed_multiplier while it doesn't).

        kicks: None (nobody kicks), or one entry per blue player: None for no kick, or the ball's
        velocity (kx, ky) in m/s, capped at max_kick_speed. A kick only happens if that player
        holds the ball at the start of the step. info["kicks"] says which players kicked.

        Returns (observation, reward, terminated, truncated, info):
        - reward is +1 when blue scores, -1 when red scores (an own goal by blue included), else 0.
        - terminated is True on a goal. truncated is True when time ran out first (a draw).
        """
        if self.winner != Team.NONE or self.game_steps >= self.max_game_steps:
            raise RuntimeError("The game is over, call reset() to start a new one")
        cfg = self.config
        commands = np.asarray(action, dtype=float)
        if commands.ndim == 1:
            commands = commands[None, :]
        if commands.ndim != 2 or commands.shape[0] != len(self.blues) or commands.shape[1] < 2:
            raise ValueError(f"expected one (vx, vy) per blue player ({len(self.blues)}), got shape {commands.shape}")
        blue_commands = [KeepAwayEnv._clip_speed(command[:2], cfg.blue_max_speed) for command in commands]
        if kicks is None:
            kicks = [None] * len(self.blues)
        if len(kicks) != len(self.blues):
            raise ValueError(f"expected one kick (or None) per blue player ({len(self.blues)}), "
                             f"got {len(kicks)}")
        self.possession_events = []

        # Kicks happen before the physics steps, so "holds the ball" means at the start of the step
        self.kicks_applied = [False] * len(self.blues)
        for i, (player, kick) in enumerate(zip(self.blues, kicks)):
            if kick is not None and self.ball.possession_player is player:
                kick_velocity = KeepAwayEnv._clip_speed(np.asarray(kick, dtype=float)[:2], cfg.max_kick_speed)
                self.ball.kick_ball(kick_velocity[0], kick_velocity[1])
                self.kicks_applied[i] = True

        for _ in range(cfg.action_repeat):
            for player, command in zip(self.blues, blue_commands):
                factor = (cfg.possession_speed_multiplier if self.ball.possession_player is player
                          else cfg.free_speed_multiplier)
                player.apply_velocity(command[0] * factor, command[1] * factor)
            for player in self.reds:
                red_velocity = self.red_policy(player)
                player.apply_velocity(red_velocity[0], red_velocity[1])

            p.stepSimulation(physicsClientId=self.client)
            self.ball.sync_from_pybullet()
            for player in (*self.blues, *self.reds):
                player.sync_from_pybullet()
            # an opponent already touching the carrier takes the ball too (see Ball.check_held_contacts)
            self.ball.check_held_contacts()

            self.game_steps += 1
            self._update_rules()
            if self.winner != Team.NONE:
                break  # stop the moment someone scores, even partway through the action

        terminated = self.winner != Team.NONE
        truncated = not terminated and self.game_steps >= self.max_game_steps
        reward = {Team.BLUE: 1.0, Team.RED: -1.0}.get(self.winner, 0.0)
        return self.get_observation(), reward, terminated, truncated, self._info()

    def close(self):
        """Disconnects this game's simulation, once (pybullet reuses freed connection ids)."""
        if self._connected:
            self._connected = False
            Entity.clear_registry(self.client)
            if p.isConnected(self.client):
                p.disconnect(self.client)

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass  # e.g. a constructor that failed before connecting, or interpreter shutdown

    # ---------- Red's fixed policy ----------

    def red_policy(self, player: Player) -> np.ndarray:
        """Every red chases the ball with keep-away's P-controller. A red holding the ball is on
        top of it, so it keeps still. (The one place a smarter red policy would go.)"""
        return KeepAwayEnv.chase_velocity(player.position, self.ball.position,
                                          self.config.red_gain, self.config.red_max_speed)

    # ---------- Goals ----------

    @property
    def blue_goal(self):
        """The goal blue defends (west); a ball in it is a point for red."""
        return self.field.goal_defended_by(Team.BLUE)[0]

    @property
    def red_goal(self):
        """The goal red defends (east), the one blue attacks."""
        return self.field.goal_defended_by(Team.RED)[0]

    # ---------- Observations ----------

    def get_observation(self) -> dict:
        """Every player's observables (with wall distances), the ball's, the goals, the score and
        the game clock."""
        return {
            "blues": [self._player_observation(player) for player in self.blues],
            "reds": [self._player_observation(player) for player in self.reds],
            "ball": self.ball.get_observables(),
            "goals": [{"name": goal.spec.name, "center": goal.spec.center, "normal": goal.spec.normal,
                       "defending_team": goal.defending_team} for goal in self.field.goals],
            "score": dict(self.score),
            "game": {
                "time": self.game_steps * self.time_step,
                "time_remaining": (self.max_game_steps - self.game_steps) * self.time_step,
                "winner": self.winner,
            },
        }

    # ---------- Internals ----------

    def _update_rules(self):
        """Record changes of ball holder, then check both goals (once per physics step)."""
        holder = self.ball.possession_player
        if holder is not self._last_holder:
            self.possession_events.append((self.game_steps, self._last_holder, holder))
            self._last_holder = holder
        for goal in self.field.goals:
            event = goal.update(self.ball)
            if event is not None and self.winner == Team.NONE:
                self.last_goal = event
                self.winner = event.scoring_team
                self.score[event.scoring_team] += 1

    def _team_spawn(self, options: dict, team: str, x: float, count: int) -> list:
        """Start positions (x, y, 0) for one team, from the reset options or the default line-up."""
        single, many = f"{team}_position", f"{team}_positions"
        if single in options and many in options:
            raise ValueError(f"give {single} or {many}, not both")
        if single in options:
            if count != 1:
                raise ValueError(f"{single} is for a team of one; this team has {count} players, use {many}")
            points = [options[single]]
        elif many in options:
            points = list(options[many])
            if len(points) != count:
                raise ValueError(f"{many} needs {count} positions, got {len(points)}")
        else:
            spacing = self.config.spawn_separation
            points = [(x, (i - (count - 1) / 2) * spacing) for i in range(count)]
        return [np.array([*point[:2], 0.0], dtype=float) for point in points]

    def _player_observation(self, player: Player) -> dict:
        observation = player.get_observables()
        walls = self.field.get_wall_vectors(player.position, radius=player.radius)
        observation["wall_distances"] = {name: info["distance"] for name, info in walls.items()}
        return observation

    def _camera_distance(self, margin: float = 1.04) -> float:
        """How far above the field the GUI camera has to be for the field, its walls and the goal
        pockets to just fill the window (as KeepAwayEnv._camera_distance)."""
        cfg = self.config
        half_length = cfg.field_length / 2 + max(self.field.wall_thickness, cfg.goal_depth + PANEL_THICKNESS)
        half_width = cfg.field_width / 2 + self.field.wall_thickness
        proj = p.getDebugVisualizerCamera(physicsClientId=self.client)[3]
        if not proj or proj[0] <= 0 or proj[5] <= 0:
            return max(cfg.field_length, cfg.field_width * 1.6)
        distance = max(half_width * proj[5], half_length * proj[0]) * margin
        return distance + self.field.wall_height

    def _info(self) -> dict:
        return {
            "winner": self.winner,
            "game_time": self.game_steps * self.time_step,
            "kicks": list(self.kicks_applied),
            "goal": self.last_goal,
            "score": dict(self.score),
        }


def main() -> None:
    """
    Play soccer as blue with an on-screen joystick against the chasing red. Space kicks the ball
    the way the joystick points; pushing the knob further kicks harder. Every goal is printed, and
    a new game starts after each goal or time-out.
    """
    import tkinter as tk  # pylint: disable=import-outside-toplevel
    from joystick import Joystick  # pylint: disable=import-outside-toplevel

    env = SoccerEnv(render_mode="human")
    env.reset()
    tally = {Team.BLUE: 0, Team.RED: 0, Team.NONE: 0}

    root = tk.Tk()
    root.title("Blue joystick")
    joystick = Joystick(root, max_speed=env.config.blue_max_speed)
    joystick.pack(padx=10, pady=10)
    status = tk.Label(root, font=("Consolas", 10), justify=tk.LEFT)
    status.pack(padx=10, pady=(0, 10))

    window_open = True
    kick_requested = False

    def on_close():
        nonlocal window_open
        window_open = False

    def on_space(_event=None):
        nonlocal kick_requested
        kick_requested = True

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<space>", on_space)

    while window_open and p.isConnected(env.client):
        frame_start = time.time()
        kicks = None
        if kick_requested:
            kicks = [joystick.velocity / env.config.blue_max_speed * env.config.max_kick_speed]
            kick_requested = False
        try:
            observation, _, terminated, truncated, info = env.step(joystick.velocity, kicks=kicks)
        except p.error:
            break  # the pybullet window was closed mid-step

        if terminated or truncated:
            tally[info["winner"]] += 1
            goal = info["goal"]
            if goal is not None:
                how = "carried in" if goal.carried else "kicked in"
                who = goal.scorer_team.name.title() if goal.scorer is not None else "nobody"
                own = " (own goal)" if goal.own_goal else ""
                print(f"GOAL for {goal.scoring_team.name.title()}: {how} by {who}{own} "
                      f"after {info['game_time']:.1f} s")
            else:
                print(f"Time ran out after {info['game_time']:.1f} s: a draw")
            observation, info = env.reset()

        status.config(text=f"time   {observation['game']['time']:5.1f} s\n"
                           f"games  blue {tally[Team.BLUE]} - {tally[Team.RED]} red "
                           f"({tally[Team.NONE]} drawn)\n"
                           f"space  kick")
        root.update()
        time.sleep(max(0.0, env.control_dt - (time.time() - frame_start)))

    root.destroy()
    env.close()


if __name__ == "__main__":
    main()
### END AI GENERATED
