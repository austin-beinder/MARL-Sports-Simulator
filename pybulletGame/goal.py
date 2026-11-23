"""
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-23
Author:  Austin Beinder
Usage:
  - 2026-11-23: Generated this entire file from docs/goal_plan.md: GoalSpec, GoalEvent, Goal
    and standard_goals, plus the GUI demo. Design decisions (a mouth in the wall, any ball in
    counts for the team that doesn't defend the goal, players may walk into the pocket, the
    2.0 m default width) by Austin Beinder. The scorer is Ball.last_holder (added to ball.py the
    same day), so a kick before any update still has its kicker.

A goal for the field: a mouth in one of the field walls with a pocket behind it. Pong-like:
whenever the whole ball gets over the goal line between the posts, the goal reports a GoalEvent.

Any ball in a goal is a point for the team that doesn't defend it, however it got there: kicked,
carried in by a player, or an own goal. Players can walk into the pocket.

Three parts:
- GoalSpec: where the goal is and how big, as plain geometry (no pybullet). A goal can sit on any
  wall: `center` is the middle of its mouth on the goal line, and `normal` points from the goal
  into the field.
- Goal: the pocket's pybullet body (a back panel and two side panels) and the scoring check.
  Call goal.update(ball) once per physics tick; it returns a GoalEvent on the tick the ball goes in.
- standard_goals(): one goal at each end of a field, the soccer layout.

Field(goals=[...]) opens the mouths in its walls and builds the Goals (field.goals). A Goal built
on its own, without the field's opening, is just a box behind a solid wall.

The goal is registered in Entity._registry as a wall (is_wall = True, team NONE), like
field.Wall: Ball and Player hand the ball to anything they touch that isn't a wall, so that keeps
the net from "taking" the ball. is_goal = True tells it apart from a plain wall, and
defending_team says whose goal it is.
"""

### AI GENERATED
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from entity import Entity
from teams import Team

# The pocket's panels reach as low as the field walls (field.WALL_BOTTOM_Z), below the entities.
# Repeated here because field.py imports this module.
GOAL_BOTTOM_Z = -0.1

# Thickness of the pocket's back and side panels (m)
PANEL_THICKNESS = 0.1

# The visual panels are this much smaller than the collision panels on every side, so their faces
# never lie in the same plane as a field wall's face (the GUI would flicker between the two)
VISUAL_INSET = 0.003

POST_RADIUS = 0.05
POST_COLOR = [1.0, 1.0, 1.0, 1.0]
# A light, see-through net in the defending team's colour
NET_COLORS = {Team.BLUE: [0.55, 0.65, 1.0, 0.6], Team.RED: [1.0, 0.6, 0.55, 0.6]}


@dataclass(frozen=True)
class GoalSpec:
    """
    Where a goal is and how big it is. Lengths in metres.

    center: (x, y) of the middle of the goal mouth, on the goal line (the field's edge).
    normal: unit (x, y) vector from the goal into the field. The east goal of a field centred on
        the origin has normal (-1, 0).
    width: inner width of the mouth, post to post.
    depth: how far the pocket goes back behind the goal line. It has to be more than the ball's
        diameter for the whole ball to fit over the line.
    height: height of the pocket's panels (the field walls are 0.5).
    defending_team: Team.BLUE or Team.RED. A ball in this goal scores for the other team.
    name: a label for events and debugging, e.g. "east_goal".

    Goal-local coordinates (to_local): `along` is how far a point is behind the goal line (into
    the pocket, negative in the field) and `across` is its sideways offset from the centre.
    """

    center: Tuple[float, float]
    normal: Tuple[float, float]
    defending_team: Team
    width: float = 2.0
    depth: float = 0.5
    height: float = 0.5
    name: str = "goal"

    def __post_init__(self):
        # Plain float tuples, so a spec built from numpy arrays still compares and hashes cleanly
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))
        object.__setattr__(self, "normal", (float(self.normal[0]), float(self.normal[1])))
        if self.defending_team not in (Team.BLUE, Team.RED):
            raise ValueError(f"defending_team must be Team.BLUE or Team.RED, got {self.defending_team}")
        if not math.isclose(math.hypot(*self.normal), 1.0, abs_tol=1e-6):
            raise ValueError(f"normal must be a unit vector, got {self.normal}")
        for name in ("width", "depth", "height"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    @property
    def scoring_team(self) -> Team:
        """The team a ball in this goal scores for: whoever doesn't defend it."""
        return Team.RED if self.defending_team == Team.BLUE else Team.BLUE

    @property
    def into_pocket(self) -> np.ndarray:
        """Unit vector from the goal line back into the pocket (minus the normal)."""
        return -np.array(self.normal)

    @property
    def tangent(self) -> np.ndarray:
        """Unit vector along the goal line: the normal turned 90 degrees anticlockwise."""
        return np.array([-self.normal[1], self.normal[0]])

    def to_local(self, xy) -> Tuple[float, float]:
        """(along, across) of a world (x, y) point. See the class docstring."""
        offset = np.asarray(xy, dtype=float)[:2] - np.array(self.center)
        return float(offset @ self.into_pocket), float(offset @ self.tangent)

    def to_world(self, along: float, across: float) -> np.ndarray:
        """The world (x, y) of a goal-local point."""
        return np.array(self.center) + along * self.into_pocket + across * self.tangent

    def ball_in(self, ball_xy, ball_radius: float) -> bool:
        """Whether the whole ball is over the goal line, between the posts."""
        along, across = self.to_local(ball_xy)
        return along >= ball_radius and abs(across) <= self.width / 2

    def posts(self) -> Tuple[np.ndarray, np.ndarray]:
        """World (x, y) of the two posts (the ends of the mouth)."""
        return self.to_world(0.0, -self.width / 2), self.to_world(0.0, self.width / 2)


@dataclass(frozen=True)
class GoalEvent:
    """
    What Goal.update reports on the tick the ball goes in.

    goal_name: the goal's spec.name.
    scoring_team: the team that gets the point: always the one that doesn't defend the goal.
    scorer: the last Player to hold the ball (ball.last_holder: the kicker, or the carrier), or None
        if nobody had.
    scorer_team: scorer.team, or Team.NONE without a scorer.
    own_goal: the scorer defends this goal.
    carried: a player was holding the ball as it went in (otherwise it was kicked or rolled in).
    sim_time: the ball's simulation clock (seconds) when it went in.
    """

    goal_name: str
    scoring_team: Team
    scorer: Optional[object]  # a Player (not imported here: player.py imports ball.py)
    scorer_team: Team
    own_goal: bool
    carried: bool
    sim_time: float


class Goal:
    """
    A goal's pocket in the simulation, and the scoring check. One static body: a back panel at
    spec.depth behind the goal line and two side panels from the line back to it, open at the
    front. Not an Entity (it never moves), but registered in Entity._registry with the attributes
    the collision code reads (body_id, is_ball, is_wall, team), like field.Wall.
    """

    def __init__(self, spec: GoalSpec, physics_client: int = 0):
        self.spec = spec
        self.client = physics_client
        self.is_ball = False
        self.is_wall = True  # so a ball or player touching the net treats it as a wall
        self.is_goal = True
        self.team = Team.NONE  # the collision code's "team": the net is nobody's player
        self.defending_team = spec.defending_team

        # Whether the ball was in at the last update, so a goal fires once per entry
        self.ball_inside = False

        self.body_id = self._create_body()
        self.posts_id = self._create_posts()
        Entity._registry[(physics_client, self.body_id)] = self

    # Field.Wall attributes, so on_wall_collision handlers can read a goal like a wall
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def normal(self) -> np.ndarray:
        return np.array(self.spec.normal)

    # ---------- Building the pocket ----------

    def _panels(self) -> List[Tuple[list, list]]:
        """(centre, half extents) of the back and side panels, in the body frame: +x runs back
        into the pocket, +y along the goal line, z = 0 at the panels' mid-height."""
        s, t = self.spec, PANEL_THICKNESS
        half_height = s.height / 2
        back = ([s.depth + t / 2, 0.0, 0.0], [t / 2, s.width / 2 + t, half_height])
        side_a = ([s.depth / 2, -(s.width / 2 + t / 2), 0.0], [s.depth / 2, t / 2, half_height])
        side_b = ([s.depth / 2, s.width / 2 + t / 2, 0.0], [s.depth / 2, t / 2, half_height])
        return [back, side_a, side_b]

    def _create_body(self) -> int:
        s, c = self.spec, self.client
        panels = self._panels()
        positions = [centre for centre, _ in panels]
        collision = p.createCollisionShapeArray(
            [p.GEOM_BOX] * len(panels), halfExtents=[half for _, half in panels],
            collisionFramePositions=positions, physicsClientId=c)
        visual = p.createVisualShapeArray(
            [p.GEOM_BOX] * len(panels),
            halfExtents=[[max(h - VISUAL_INSET, 1e-3) for h in half] for _, half in panels],
            visualFramePositions=positions,
            rgbaColors=[NET_COLORS[s.defending_team]] * len(panels), physicsClientId=c)
        # Turn the body so its +x points back into the pocket
        yaw = math.atan2(s.into_pocket[1], s.into_pocket[0])
        base = [*s.center, GOAL_BOTTOM_Z + s.height / 2]
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                 basePosition=base,
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=c)

    def _create_posts(self) -> int:
        """Two white posts at the ends of the mouth, a little taller than the panels. Visual only:
        the wall's (or the side panel's) edge is what the ball actually hits."""
        s, c = self.spec, self.client
        length = s.height + 0.02
        visual = p.createVisualShapeArray(
            [p.GEOM_CYLINDER] * 2, radii=[POST_RADIUS] * 2, lengths=[length] * 2,
            visualFramePositions=[[*post, 0.0] for post in s.posts()],
            rgbaColors=[POST_COLOR] * 2, physicsClientId=c)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
                                 basePosition=[0, 0, GOAL_BOTTOM_Z + length / 2], physicsClientId=c)

    # ---------- Scoring ----------

    def update(self, ball) -> Optional[GoalEvent]:
        """
        Check for a goal. Call once per physics tick, after ball.sync_from_pybullet() (the
        ball's position is read from it).

        The scorer is ball.last_holder: in this game touching a loose ball takes it, and only the
        holder can kick, so the last player to hold the ball is whoever put it in.

        Returns a GoalEvent on the tick the whole ball gets over the line, else None. It fires
        once per entry: a ball resting in the net doesn't score again until it has left the goal
        (as far as update can see) and come back.
        """
        inside = self.spec.ball_in(ball.position, ball.radius)
        event = None
        if inside and not self.ball_inside:
            scorer = ball.last_holder
            scorer_team = scorer.team if scorer is not None else Team.NONE
            event = GoalEvent(
                goal_name=self.spec.name,
                scoring_team=self.spec.scoring_team,
                scorer=scorer,
                scorer_team=scorer_team,
                own_goal=scorer_team == self.spec.defending_team,
                carried=ball.possession_player is not None,
                sim_time=ball.sim_time,
            )
        self.ball_inside = inside
        return event

    def reset(self):
        """Forget whether the ball was in, e.g. after putting the ball back for a kickoff. (Who
        last held the ball is the ball's: clear ball.last_holder too if that should be forgotten.)"""
        self.ball_inside = False


def standard_goals(field_length: float, field_width: float, width: float = 2.0, depth: float = 0.5,
                   height: float = 0.5, west_team: Team = Team.BLUE) -> List[GoalSpec]:
    """
    The soccer layout: a goal in the middle of each end wall of a field centred on the origin,
    [west goal (x = -length / 2), east goal (x = +length / 2)]. west_team defends the west goal and
    the other team the east one. Which team attacks which end is the game's choice, so flip
    west_team as needed.
    """
    east_team = Team.RED if west_team == Team.BLUE else Team.BLUE
    return [
        GoalSpec(center=(-field_length / 2, 0.0), normal=(1.0, 0.0), defending_team=west_team,
                 width=width, depth=depth, height=height, name="west_goal"),
        GoalSpec(center=(field_length / 2, 0.0), normal=(-1.0, 0.0), defending_team=east_team,
                 width=width, depth=depth, height=height, name="east_goal"),
    ]


if __name__ == "__main__":

    # Shows a field with a goal at each end, a ball and a few players in a GUI. Drag them around with
    # the mouse. A small tkinter window has a "kick" button per player (only works while they hold
    # the ball; it kicks at the goal they attack) and a button that puts the ball back on the centre
    # spot. Every goal is printed. Blue defends the west goal and attacks east; red the reverse.
    import tkinter as tk

    from ball import Ball
    from field import Field
    from player import Player

    KICK_SPEED = 3.0  # the ball's maxJointVelocity, so the fastest kick there is

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    field = Field(goals=standard_goals(21.0, 13.6))
    p.resetDebugVisualizerCamera(cameraDistance=14, cameraYaw=0, cameraPitch=-65, cameraTargetPosition=[0, 0, 0])

    ball = Ball(initial_position=np.array([0.0, 0.0, 0.0]))  # at the origin, so its joints are world x/y
    players = [
        Player(team=Team.RED, initial_position=np.array([-3.0, 0.0, 0.0])),
        Player(team=Team.RED, initial_position=np.array([-6.0, 3.0, 0.0])),
        Player(team=Team.BLUE, initial_position=np.array([3.0, 0.0, 0.0])),
        Player(team=Team.BLUE, initial_position=np.array([6.0, -3.0, 0.0])),
    ]
    score = {Team.BLUE: 0, Team.RED: 0}

    def kick(player):
        if ball.possession_player is not player:
            print(f"Player: {player.body_id} doesn't have the ball")
            return
        target = next(goal for goal in field.goals if goal.defending_team != player.team)
        direction = np.array(target.spec.center) - player.position[:2]
        direction /= np.linalg.norm(direction)
        ball.kick_ball(*(direction * KICK_SPEED))

    def ball_to_centre():
        ball._release()
        ball.apply_position(0.0, 0.0)
        ball.apply_velocity(0.0, 0.0)
        ball.last_holder = None
        for goal in field.goals:
            goal.reset()

    root = tk.Tk()
    root.title("Goals")
    buttons = {}
    team_counts = {Team.RED: 0, Team.BLUE: 0}
    for player in players:
        team_counts[player.team] += 1
        colour = "red" if player.team == Team.RED else "blue"
        buttons[player] = tk.Button(root, text=f"{player.team.name.title()} {team_counts[player.team]} kick",
                                    fg=colour, width=16, state=tk.DISABLED,
                                    command=lambda player=player: kick(player))
        buttons[player].pack(padx=8, pady=4)
    tk.Button(root, text="Ball to centre", width=16, command=ball_to_centre).pack(padx=8, pady=4)
    score_label = tk.Label(root, text="Blue 0 - 0 Red", font=("Consolas", 12))
    score_label.pack(padx=8, pady=8)

    window_open = True
    while p.isConnected():
        p.stepSimulation()
        ball.sync_from_pybullet()
        for player in players:
            player.sync_from_pybullet()
        ball.check_held_contacts()
        for goal in field.goals:
            event = goal.update(ball)
            if event is not None:
                score[event.scoring_team] += 1
                how = "carried in" if event.carried else "kicked in"
                who = f"{event.scorer_team.name.title()} player {event.scorer.body_id}" if event.scorer else "nobody"
                own = " (own goal)" if event.own_goal else ""
                print(f"GOAL for {event.scoring_team.name.title()} in the {event.goal_name}: {how} by {who}{own}"
                      f" at {event.sim_time:.1f} s. Blue {score[Team.BLUE]} - {score[Team.RED]} Red")

        if window_open:
            try:
                for player, button in buttons.items():
                    state = tk.NORMAL if ball.possession_player is player else tk.DISABLED
                    if button["state"] != state:
                        button.config(state=state)
                score_label.config(text=f"Blue {score[Team.BLUE]} - {score[Team.RED]} Red")
                root.update()  # process clicks without blocking the simulation
            except tk.TclError:
                window_open = False  # the window was closed; keep the simulation running

        time.sleep(1. / 240.)
### END AI GENERATED
