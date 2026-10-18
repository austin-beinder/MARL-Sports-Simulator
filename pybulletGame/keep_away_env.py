"""
This file was AI generated, see the ### AI GENERATED markers.

The "gym" layer for the keep away game. It owns the pybullet simulation and is the only thing
the RL code needs to talk to:
- reset() builds the field, walls, a red player, a blue player and the ball, and returns the first observation.
- step(action) routes the RL action to the blue player, moves the fixed-policy red player,
  ticks pybullet forward, runs the rules of the game and returns the next observation.
- play_game(policy) plays one whole game with a policy and returns a score.

It follows the gymnasium reset/step interface, so it can be wrapped as a gymnasium.Env later
without changing how it works. Only one environment can run per Python process, because the
entity classes all use pybullet's default connection.

Rules of keep away:
- Blue is the learner, red follows a fixed policy that chases the ball.
- Red wins if it holds the ball for red_win_time (0.3 s) in a row.
- Blue wins if it holds the ball for blue_win_time (20 s) in a row.
- If nobody has won after max_game_time, the game is cut off (truncated) with no winner.
- Blue moves slower while it has the ball (possession_speed_multiplier).
- Both players spawn in the middle of the field, spawn_separation apart. The ball starts
  ball_spawn_fraction of the way from blue to red, so blue reaches it first.
"""

### AI GENERATED
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pybullet as p

from ball import Ball
from entity import Entity
from field import Field
from player import Player
from teams import Team


@dataclass
class KeepAwayConfig:
    """Tunable parameters for the keep away game. Speeds are in m/s, times in seconds, lengths in metres."""

    # Players
    blue_max_speed: float = 3.0
    red_max_speed: float = 2.0  # slower than blue with the ball (3.0 * 0.75 = 2.25), so blue can escape
    red_gain: float = 2.0  # red's P-controller gain: commanded speed per metre of distance to the ball
    possession_speed_multiplier: float = 0.75  # blue's speed is scaled by this while it holds the ball
    ### AI GENERATED
    player_radius: float = 0.3
    ### END AI GENERATED

    # Rules
    red_win_time: float = 0.3
    blue_win_time: float = 20.0
    max_game_time: float = 60.0

    # Spawning
    spawn_separation: float = 2.0
    ball_spawn_fraction: float = 0.3

    # Field
    field_length: float = 21.0
    field_width: float = 13.6

    # Physics steps (1/240 s each) per call to step(). 8 gives the RL layer 30 decisions per second.
    action_repeat: int = 8


class KeepAwayEnv:

    def __init__(self, config: Optional[KeepAwayConfig] = None, render_mode: Optional[str] = None):
        """
        render_mode: "human" opens the pybullet GUI, None runs headless (much faster, for training).
        """
        self.config = config or KeepAwayConfig()
        self.render_mode = render_mode
        ### AI GENERATED
        # Reuse an existing connection: the entity classes all talk to pybullet's default one, so a
        # second connect() would only open a connection nothing uses.
        if not p.isConnected():
            p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        ### END AI GENERATED

        self.time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]
        self.control_dt = self.config.action_repeat * self.time_step

        # Rules are counted in physics steps rather than summed float seconds, so they trigger exactly on time
        self.red_win_steps = max(1, round(self.config.red_win_time / self.time_step))
        self.blue_win_steps = max(1, round(self.config.blue_win_time / self.time_step))
        self.max_game_steps = max(1, round(self.config.max_game_time / self.time_step))

        self.field: Optional[Field] = None
        self.ball: Optional[Ball] = None
        self.blue: Optional[Player] = None
        self.red: Optional[Player] = None

    # ---------- Gym interface ----------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new game. Rebuilds the whole simulation so nothing carries over from the last game.
        seed seeds Python's random module, which decides whether steals succeed.
        ### AI GENERATED
        options can override the spawn (anything left out keeps the default spawn):
        - "blue_position", "red_position", "ball_position": (x, y) in metres
        - "blue_has_ball": True to start the game with blue holding the ball (it is moved onto blue)
        ### END AI GENERATED

        Returns (observation, info).
        """
        if seed is not None:
            random.seed(seed)
        ### AI GENERATED
        options = dict(options or {})
        unknown = set(options) - {"blue_position", "red_position", "ball_position", "blue_has_ball"}
        if unknown:
            raise ValueError(f"unknown reset option(s) {sorted(unknown)}")
        ### END AI GENERATED

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # don't draw while rebuilding

        p.resetSimulation()
        Entity._registry.clear()

        cfg = self.config
        self.field = Field(length=cfg.field_length, width=cfg.field_width)

        blue_start = np.array([-cfg.spawn_separation / 2, 0.0, 0.0])
        red_start = np.array([cfg.spawn_separation / 2, 0.0, 0.0])
        ball_start = blue_start + cfg.ball_spawn_fraction * (red_start - blue_start)
        ### AI GENERATED
        def spawn_xy(key, default):
            return np.array([*options[key][:2], 0.0], dtype=float) if key in options else default

        blue_start = spawn_xy("blue_position", blue_start)
        red_start = spawn_xy("red_position", red_start)
        ball_start = blue_start.copy() if options.get("blue_has_ball") else spawn_xy("ball_position", ball_start)

        self.blue = Player(team=Team.BLUE, radius=cfg.player_radius, initial_position=blue_start,
                           maxJointVelocity=cfg.blue_max_speed)
        self.red = Player(team=Team.RED, radius=cfg.player_radius, initial_position=red_start,
                          maxJointVelocity=cfg.red_max_speed)
        self.ball = Ball(initial_position=ball_start)
        if options.get("blue_has_ball"):
            self.ball._player_takes_possession(self.blue)
        ### END AI GENERATED
        for entity in (self.blue, self.red, self.ball):
            entity.sync_from_pybullet()

        self.game_steps = 0
        self.blue_hold_steps = 0  # how long blue has held the ball without losing it
        self.red_hold_steps = 0
        self.blue_total_hold_steps = 0  # all the time blue has had the ball this game
        self.winner = Team.NONE
        ### AI GENERATED
        # Every change of ball holder during the last step() call, as (physics step, old holder, new holder)
        # with holders being Player objects or None. Checked every physics step, so a possession that is
        # gained and lost within one step() is still recorded.
        self.possession_events = []
        self._last_holder = self.ball.possession_player
        ### END AI GENERATED

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide the side panels
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # Looking straight down with +x to the right and +y up the screen, matching the joystick
            view_distance = max(cfg.field_length, cfg.field_width * 1.6)
            p.resetDebugVisualizerCamera(cameraDistance=view_distance, cameraYaw=0, cameraPitch=-89.9,
                                         cameraTargetPosition=[0, 0, 0])

        return self.get_observation(), self._info()

    def step(self, action):
        """
        action: blue's desired velocity (vx, vy) in m/s. Its length is capped at blue_max_speed, and
        it is scaled by possession_speed_multiplier while blue has the ball.

        Returns (observation, reward, terminated, truncated, info):
        - reward is +1 when blue wins, -1 when red wins, and 0 otherwise.
        - terminated is True when someone won. truncated is True when time ran out first.
        """
        if self.winner != Team.NONE or self.game_steps >= self.max_game_steps:
            raise RuntimeError("The game is over, call reset() to start a new one")

        blue_command = self._clip_speed(np.asarray(action, dtype=float)[:2], self.config.blue_max_speed)
        self.possession_events = []  # AI GENERATED

        for _ in range(self.config.action_repeat):
            # Blue: the RL action, slowed while holding the ball
            blue_velocity = blue_command
            if self.ball.possession_player is self.blue:
                blue_velocity = blue_velocity * self.config.possession_speed_multiplier
            self.blue.apply_velocity(blue_velocity[0], blue_velocity[1])

            # Red: the fixed policy
            red_velocity = self.red_policy()
            self.red.apply_velocity(red_velocity[0], red_velocity[1])

            p.stepSimulation()
            self.ball.sync_from_pybullet()
            self.blue.sync_from_pybullet()
            self.red.sync_from_pybullet()

            self.game_steps += 1
            self._update_rules()
            if self.winner != Team.NONE:
                break  # stop the moment someone wins, even partway through the action

        terminated = self.winner != Team.NONE
        truncated = not terminated and self.game_steps >= self.max_game_steps
        reward = {Team.BLUE: 1.0, Team.RED: -1.0}.get(self.winner, 0.0)
        return self.get_observation(), reward, terminated, truncated, self._info()

    def close(self):
        if p.isConnected():
            p.disconnect()

    # ---------- Playing whole games ----------

    def play_game(self, policy: Callable[[dict], np.ndarray], seed: Optional[int] = None) -> dict:
        """
        Play one full game with policy controlling blue. policy takes an observation (as returned by
        step) and returns blue's velocity action.

        Returns a summary: winner, score (the total reward), how long the game lasted and how long
        blue had the ball in total.
        """
        observation, _ = self.reset(seed=seed)
        score = 0.0
        while True:
            observation, reward, terminated, truncated, info = self.step(policy(observation))
            score += reward
            if terminated or truncated:
                return {
                    "winner": info["winner"],
                    "score": score,
                    "game_time": info["game_time"],
                    "blue_total_possession_time": info["blue_total_possession_time"],
                }

    # ---------- Red's fixed policy ----------

    def red_policy(self) -> np.ndarray:
        """Red chases the ball with a P-controller."""
        return self.chase_velocity(self.red.position, self.ball.position,
                                   self.config.red_gain, self.config.red_max_speed)

    @staticmethod
    def chase_velocity(chaser_position, target_position, gain: float, max_speed: float) -> np.ndarray:
        """Velocity proportional to the x/y offset to the target, capped at max_speed."""
        offset = np.asarray(target_position, dtype=float)[:2] - np.asarray(chaser_position, dtype=float)[:2]
        return KeepAwayEnv._clip_speed(gain * offset, max_speed)

    # ---------- Observations ----------

    def get_observation(self) -> dict:
        """
        Everything the RL layer can see: the observables of every entity, how far each player is
        from the walls, and the state of the game.
        """
        return {
            "blue": self._player_observation(self.blue),
            "red": self._player_observation(self.red),
            "ball": self.ball.get_observables(),
            "game": {
                "time": self.game_steps * self.time_step,
                "time_remaining": (self.max_game_steps - self.game_steps) * self.time_step,
                "blue_possession_time": self.blue_hold_steps * self.time_step,
                "red_possession_time": self.red_hold_steps * self.time_step,
                "blue_possession_time_to_win": self.config.blue_win_time,
                "red_possession_time_to_win": self.config.red_win_time,
                "winner": self.winner,
            },
        }

    def flatten_observation(self, observation: dict) -> np.ndarray:
        """
        The observation as a fixed-length vector of floats, for feeding to a neural network.
        Positions and velocities are in metres and m/s in the static field frame; timers are
        fractions of the time needed to win.

        [blue x, y, vx, vy, red x, y, vx, vy, ball x, y, vx, vy,
         blue has ball, red has ball, blue possession fraction, red possession fraction, game time fraction,
         blue distance to east, west, north, south wall]
        """
        def xy(entity, key):
            return [entity[key]["x"], entity[key]["y"]]

        blue, red, ball, game = observation["blue"], observation["red"], observation["ball"], observation["game"]
        values = (
            xy(blue, "position") + xy(blue, "velocity")
            + xy(red, "position") + xy(red, "velocity")
            + xy(ball, "position") + xy(ball, "velocity")
            + [
                # Read possession from the ball: it's the authoritative record of who holds it
                float(ball["possession_player"] == blue["id"]),
                float(ball["possession_player"] == red["id"]),
                game["blue_possession_time"] / game["blue_possession_time_to_win"],
                game["red_possession_time"] / game["red_possession_time_to_win"],
                game["time"] / self.config.max_game_time,
            ]
            + [blue["wall_distances"][name] for name in ("east", "west", "north", "south")]
        )
        return np.array(values, dtype=np.float32)

    @staticmethod
    def format_observation(observation: dict, decimals: int = 3) -> str:
        """One compact line of JSON, with floats rounded, for printing every frame."""
        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            if isinstance(value, Team):
                return value.name
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (float, np.floating)):
                return round(float(value), decimals) if math.isfinite(value) else str(value)
            return value
        return json.dumps(clean(observation))

    # ---------- Internals ----------

    def _player_observation(self, player: Player) -> dict:
        observation = player.get_observables()
        walls = self.field.get_wall_vectors(player.position, radius=player.radius)
        observation["wall_distances"] = {name: info["distance"] for name, info in walls.items()}
        observation["distance_until_wall"] = self.field.distance_until_wall(
            player.position, player.velocity, radius=player.radius)
        return observation

    def _update_rules(self):
        """Advance the possession timers by one physics step and decide if anyone has won."""
        holder = self.ball.possession_player
        ### AI GENERATED
        if holder is not self._last_holder:
            self.possession_events.append((self.game_steps, self._last_holder, holder))
            self._last_holder = holder
        ### END AI GENERATED
        if holder is self.blue:
            self.blue_hold_steps += 1
            self.blue_total_hold_steps += 1
            self.red_hold_steps = 0
        elif holder is self.red:
            self.red_hold_steps += 1
            self.blue_hold_steps = 0
        else:
            self.blue_hold_steps = 0
            self.red_hold_steps = 0

        if self.red_hold_steps >= self.red_win_steps:
            self.winner = Team.RED
        elif self.blue_hold_steps >= self.blue_win_steps:
            self.winner = Team.BLUE

    def _info(self) -> dict:
        return {
            "winner": self.winner,
            "game_time": self.game_steps * self.time_step,
            "blue_total_possession_time": self.blue_total_hold_steps * self.time_step,
        }

    @staticmethod
    def _clip_speed(velocity: np.ndarray, max_speed: float) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            return velocity * (max_speed / speed)
        return velocity


if __name__ == "__main__":

    # Play keep away as blue with an on-screen joystick. The joystick produces exactly the action
    # the RL layer would: blue's desired velocity. Every observation is printed each frame.
    import tkinter as tk
    from joystick import Joystick

    env = KeepAwayEnv(render_mode="human")
    observation, info = env.reset()

    root = tk.Tk()
    root.title("Blue joystick")
    joystick = Joystick(root, max_speed=env.config.blue_max_speed)
    joystick.pack(padx=10, pady=10)
    status = tk.Label(root, font=("Consolas", 10), justify=tk.LEFT)
    status.pack(padx=10, pady=(0, 10))

    window_open = True

    def on_close():
        global window_open
        window_open = False

    root.protocol("WM_DELETE_WINDOW", on_close)

    while window_open and p.isConnected():
        frame_start = time.time()

        try:
            observation, reward, terminated, truncated, info = env.step(joystick.velocity)
        except p.error:
            break  # the pybullet window was closed mid-step
        print(KeepAwayEnv.format_observation(observation))

        if terminated or truncated:
            result = f"{info['winner'].name} wins" if terminated else "time ran out"
            print(f"Game over: {result} after {info['game_time']:.1f} s (reward {reward:+.0f})")
            observation, info = env.reset()

        game = observation["game"]
        status.config(text=f"time   {game['time']:5.1f} s\n"
                           f"blue   {game['blue_possession_time']:5.2f} / {game['blue_possession_time_to_win']:.1f} s\n"
                           f"red    {game['red_possession_time']:5.2f} / {game['red_possession_time_to_win']:.1f} s")
        root.update()

        # Run in real time: one step covers control_dt seconds of simulation
        time.sleep(max(0.0, env.control_dt - (time.time() - frame_start)))

    root.destroy()
    env.close()
### END AI GENERATED
