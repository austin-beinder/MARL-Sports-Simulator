"""
Most of this file was AI generated, see the ### AI GENERATED markers.

The field is a rectangle centred on the origin: its length runs along x and its width along y.
It has three parts:
- A floor with a tiling grass texture. It is visual only (no collision shape), because the
  entities are held on the plane by their x/y joints and sit half below z = 0.
- White line markings (touchlines, goal lines, halfway line, centre circle, penalty areas,
  goal areas, penalty spots and arcs), scaled down from a real 105 m x 68 m pitch.
- Four static walls just outside the lines. Walls are registered in Entity._registry so that
  players get a collision event when they touch one.

It also answers "where are the walls relative to me?" for a player at a given position,
in either the static x/y frame or a frame that turns with the player's velocity.

AI edit 2026-10-31 (Claude Code, Austin Beinder): physics_client, so a field can be built in any
pybullet simulation, not just the default connection.

AI edit 2026-11-23 (Claude Code, Austin Beinder): goals. Field(goals=[GoalSpec, ...]) opens a
mouth in the wall each goal sits on and builds the goals (goal.py) behind them, in field.goals.
Without goals the field is built exactly as before.
"""

### AI GENERATED
import math
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import pybullet as p

from entity import Entity
from goal import Goal, GoalSpec  # 2026-11-23
from teams import Team

GRASS_TEXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "textures", "grass.png")

# Real pitch dimensions in metres, used to scale the line markings to the field size
REAL_LENGTH = 105.0
REAL_WIDTH = 68.0
REAL_CENTRE_CIRCLE_RADIUS = 9.15
REAL_PENALTY_AREA_DEPTH = 16.5
REAL_PENALTY_AREA_WIDTH = 40.32
REAL_GOAL_AREA_DEPTH = 5.5
REAL_GOAL_AREA_WIDTH = 18.32
REAL_PENALTY_SPOT_DISTANCE = 11.0

# Entities span z = -0.05 to 0.05, so the floor sits just below them and the lines just above the floor
FLOOR_Z = -0.06
LINE_Z = FLOOR_Z + 0.002

# Walls reach below the entities and well above a carried ball (which tops out around z = 0.16)
WALL_BOTTOM_Z = -0.1

WALL_NAMES = ("east", "west", "north", "south")


class Wall:
    """
    One of the four field walls. Registered in Entity._registry so collisions with it are
    reported through the normal update_collisions / on_new_collision path.

    It isn't an Entity (it has no joints and never moves), but it has the attributes the
    collision code reads: body_id, is_ball, is_wall and team.
    """

    def __init__(self, name: str, body_id: int, normal: np.ndarray, physics_client: int = 0):
        self.name = name
        self.body_id = body_id
        self.client = physics_client  # 2026-10-31: the simulation the wall lives in (see Entity._registry)
        self.normal = normal  # unit vector pointing into the field
        self.is_ball = False
        self.is_wall = True
        self.team = Team.NONE
        Entity._registry[(physics_client, body_id)] = self


class Field:

    def __init__(self,
                 length: float = 21.0,
                 width: float = 13.6,
                 wall_height: float = 0.5,
                 wall_thickness: float = 0.5,
                 line_width: float = 0.08,
                 grass_tile_size: float = 2.0,
                 grass_texture_path: str = GRASS_TEXTURE_PATH,
                 physics_client: int = 0,
                 goals: Sequence[GoalSpec] = ()):
        """
        length: size along x, in metres. The default is a fifth of a real pitch.
        width: size along y, in metres.
        grass_tile_size: metres covered by one repeat of the grass texture (one light and one dark stripe).
        physics_client: which pybullet simulation to build the field in (0 = the default connection).
        goals (2026-11-23): GoalSpecs, e.g. goal.standard_goals(length, width). Each one's mouth has to
            lie on one wall's inner face, inside the field's edge and clear of the other goals on that
            wall. The wall gets an opening there and the goal is built behind it (self.goals).
        """
        self.client = physics_client
        self.length = length
        self.width = width
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.line_width = line_width

        self.x_max = length / 2
        self.x_min = -length / 2
        self.y_max = width / 2
        self.y_min = -width / 2

        # One scale for all markings so circles stay circular on non-standard field shapes
        self.marking_scale = min(length / REAL_LENGTH, width / REAL_WIDTH)

        openings = self._goal_openings(goals)  # 2026-11-23: checked before anything is built

        self.floor_id = self._create_floor(grass_tile_size, grass_texture_path)
        self.lines_id = self._create_lines()
        self.walls: Dict[str, Wall] = self._create_walls(openings)
        # 2026-11-23: built after the walls, so a field without goals has the same body ids as before
        self.goals: List[Goal] = [Goal(spec, physics_client=physics_client) for spec in goals]

    # ---------- Building the field ----------

    def _create_floor(self, tile_size, texture_path) -> int:
        corners = [(self.x_min, self.y_min), (self.x_max, self.y_min),
                   (self.x_max, self.y_max), (self.x_min, self.y_max)]
        vertices = [[x, y, FLOOR_Z] for x, y in corners]

        # UVs larger than 1 make the texture repeat. Offset u so a stripe boundary lands on the
        # halfway line, and keep all UVs positive.
        offset = math.ceil(self.length / tile_size) + 1
        uvs = [[x / tile_size + offset, y / tile_size + offset] for x, y in corners]

        c = self.client
        visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=[0, 1, 2, 0, 2, 3],
                                     uvs=uvs, normals=[[0, 0, 1]] * 4, rgbaColor=[1, 1, 1, 1], physicsClientId=c)
        floor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
                                     physicsClientId=c)
        p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(texture_path, physicsClientId=c),
                            physicsClientId=c)
        return floor_id

    def _create_lines(self) -> int:
        vertices, indices = [], []

        def segment(x0, y0, x1, y1):
            """Add a flat white strip from (x0, y0) to (x1, y1), extended by half a line width at each end
            so that corners and curve segments join without gaps."""
            direction = np.array([x1 - x0, y1 - y0])
            direction /= np.linalg.norm(direction)
            along = direction * self.line_width / 2
            across = np.array([-direction[1], direction[0]]) * self.line_width / 2
            start, end = np.array([x0, y0]) - along, np.array([x1, y1]) + along
            first = len(vertices)
            for corner in (start - across, end - across, end + across, start + across):
                vertices.append([corner[0], corner[1], LINE_Z])
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3])

        def rectangle(x0, y0, x1, y1):
            segment(x0, y0, x1, y0)
            segment(x1, y0, x1, y1)
            segment(x1, y1, x0, y1)
            segment(x0, y1, x0, y0)

        def arc(cx, cy, radius, start_angle, end_angle, pieces):
            angles = np.linspace(start_angle, end_angle, pieces + 1)
            for a0, a1 in zip(angles[:-1], angles[1:]):
                segment(cx + radius * math.cos(a0), cy + radius * math.sin(a0),
                        cx + radius * math.cos(a1), cy + radius * math.sin(a1))

        def spot(cx, cy):
            """A small square, two line widths across."""
            half = self.line_width
            first = len(vertices)
            for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
                vertices.append([cx + dx, cy + dy, LINE_Z])
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3])

        s = self.marking_scale
        inset = self.line_width / 2  # lines are drawn inside the field edge, like a real pitch

        # Boundary, halfway line, centre circle and centre spot
        rectangle(self.x_min + inset, self.y_min + inset, self.x_max - inset, self.y_max - inset)
        segment(0, self.y_min + inset, 0, self.y_max - inset)
        arc(0, 0, REAL_CENTRE_CIRCLE_RADIUS * s, 0, 2 * math.pi, 48)
        spot(0, 0)

        # Penalty area, goal area, penalty spot and penalty arc at each end.
        # side = -1 is the west (x_min) end, side = 1 is the east (x_max) end.
        for side in (-1, 1):
            goal_line_x = side * (self.x_max - inset)
            box_front_x = goal_line_x - side * REAL_PENALTY_AREA_DEPTH * s
            rectangle(goal_line_x, -REAL_PENALTY_AREA_WIDTH * s / 2, box_front_x, REAL_PENALTY_AREA_WIDTH * s / 2)
            rectangle(goal_line_x, -REAL_GOAL_AREA_WIDTH * s / 2,
                      goal_line_x - side * REAL_GOAL_AREA_DEPTH * s, REAL_GOAL_AREA_WIDTH * s / 2)

            spot_x = goal_line_x - side * REAL_PENALTY_SPOT_DISTANCE * s
            spot(spot_x, 0)

            # The arc is the part of the centre-circle-sized circle around the spot that lies outside the box
            half_angle = math.acos((REAL_PENALTY_AREA_DEPTH - REAL_PENALTY_SPOT_DISTANCE) / REAL_CENTRE_CIRCLE_RADIUS)
            facing = 0.0 if side == -1 else math.pi
            arc(spot_x, 0, REAL_CENTRE_CIRCLE_RADIUS * s, facing - half_angle, facing + half_angle, 12)

        visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices,
                                     normals=[[0, 0, 1]] * len(vertices), rgbaColor=[1, 1, 1, 1],
                                     physicsClientId=self.client)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
                                 physicsClientId=self.client)

    ### AI GENERATED (2026-11-23)
    # Each wall's inner face, for placing goals: (normal into the field, which coordinate is fixed
    # on the face (0 = x, 1 = y) and its value, which coordinate runs along the wall)
    def _wall_faces(self) -> Dict[str, tuple]:
        return {
            "east": (np.array([-1.0, 0.0]), 0, self.x_max, 1),
            "west": (np.array([1.0, 0.0]), 0, self.x_min, 1),
            "north": (np.array([0.0, -1.0]), 1, self.y_max, 0),
            "south": (np.array([0.0, 1.0]), 1, self.y_min, 0),
        }

    def _goal_openings(self, goals: Sequence[GoalSpec]) -> Dict[str, List[tuple]]:
        """For each wall, the (low, high) stretches of its long coordinate (y for east and west, x
        for north and south) that goal mouths take up. Raises ValueError for a goal that isn't on a
        wall's inner face, sticks out past the field's edge, or overlaps another goal."""
        faces = self._wall_faces()
        openings: Dict[str, List[tuple]] = {name: [] for name in faces}
        for spec in goals:
            on_wall = [name for name, (normal, fixed, value, _) in faces.items()
                       if np.allclose(spec.normal, normal, atol=1e-6)
                       and math.isclose(spec.center[fixed], value, abs_tol=1e-6)]
            if not on_wall:
                raise ValueError(f"goal {spec.name!r} at {spec.center} facing {spec.normal} isn't on a field "
                                 f"wall: its center has to be on a wall's inner face and its normal has to "
                                 f"point into the field")
            name = on_wall[0]
            along = faces[name][3]
            low, high = spec.center[along] - spec.width / 2, spec.center[along] + spec.width / 2
            edge = (self.x_min, self.x_max) if along == 0 else (self.y_min, self.y_max)
            if low < edge[0] - 1e-9 or high > edge[1] + 1e-9:
                raise ValueError(f"goal {spec.name!r} is {spec.width} m wide and doesn't fit on the {name} wall")
            for other_low, other_high in openings[name]:
                if low < other_high and other_low < high:
                    raise ValueError(f"goal {spec.name!r} overlaps another goal on the {name} wall")
            openings[name].append((low, high))
        return openings
    ### END AI GENERATED

    def _create_walls(self, openings: Dict[str, List[tuple]] = None) -> Dict[str, Wall]:
        t = self.wall_thickness
        h = self.wall_height
        z = WALL_BOTTOM_Z + h / 2

        # East and west walls run the full width plus both wall thicknesses, so the corners are closed
        specs = {
            "east": ([self.x_max + t / 2, 0, z], [t / 2, self.width / 2 + t, h / 2], np.array([-1.0, 0.0])),
            "west": ([self.x_min - t / 2, 0, z], [t / 2, self.width / 2 + t, h / 2], np.array([1.0, 0.0])),
            "north": ([0, self.y_max + t / 2, z], [self.length / 2, t / 2, h / 2], np.array([0.0, -1.0])),
            "south": ([0, self.y_min - t / 2, z], [self.length / 2, t / 2, h / 2], np.array([0.0, 1.0])),
        }

        walls = {}
        for name, (position, half_extents, normal) in specs.items():
            c = self.client
            # AI GENERATED (2026-11-23): a wall with goal mouths is one body made of the pieces between them
            if openings and openings.get(name):  # AI GENERATED
                collision, visual = self._wall_with_openings(name, half_extents, openings[name])  # AI GENERATED
            else:  # AI GENERATED
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=c)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.85, 0.85, 0.82, 1],
                                             physicsClientId=c)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                        baseVisualShapeIndex=visual, basePosition=position, physicsClientId=c)
            walls[name] = Wall(name, body_id, normal, physics_client=c)
        return walls

    ### AI GENERATED (2026-11-23)
    def _wall_with_openings(self, name: str, half_extents: list, wall_openings: List[tuple]) -> tuple:
        """Collision and visual shapes for a wall with goal mouths in it: one box per piece of wall
        between the mouths, placed relative to the wall's centre. The wall stays one body (one
        body_id) and its outer bounds don't change."""
        along = self._wall_faces()[name][3]
        half_length = half_extents[along]
        pieces, start = [], -half_length
        for low, high in sorted(wall_openings) + [(half_length, half_length)]:
            if low - start > 1e-9:
                pieces.append((start, low))
            start = high
        if not pieces:  # can't happen for a goal that fits on the field's edge; guards the shape arrays
            raise ValueError(f"the goals leave nothing of the {name} wall")

        piece_halves, piece_positions = [], []
        for low, high in pieces:
            half = list(half_extents)
            half[along] = (high - low) / 2
            position = [0.0, 0.0, 0.0]
            position[along] = (low + high) / 2
            piece_halves.append(half)
            piece_positions.append(position)

        c = self.client
        collision = p.createCollisionShapeArray([p.GEOM_BOX] * len(pieces), halfExtents=piece_halves,
                                                collisionFramePositions=piece_positions, physicsClientId=c)
        visual = p.createVisualShapeArray([p.GEOM_BOX] * len(pieces), halfExtents=piece_halves,
                                          visualFramePositions=piece_positions,
                                          rgbaColors=[[0.85, 0.85, 0.82, 1]] * len(pieces), physicsClientId=c)
        return collision, visual

    def goal_defended_by(self, team: Team) -> List[Goal]:
        """The goals this team defends (a ball in one is a point for the other team)."""
        return [goal for goal in self.goals if goal.defending_team == team]
    ### END AI GENERATED

    # ---------- Wall distances for players ----------

    def get_wall_vectors(self, position, radius: float = 0.0) -> Dict[str, dict]:
        """
        For each wall, the shortest vector from the player to that wall, in the static x/y frame.

        position: the player's position (only x and y are used).
        radius: measure from the player's edge instead of their centre (pass the player's radius).

        Returns {wall name: {'distance': float, 'vector': np.array([dx, dy]), 'normal': np.array([nx, ny])}}.
        'vector' points from the player to the wall and has length 'distance'. 'normal' is the
        direction the wall faces (into the field). 'distance' goes negative if the player is
        overlapping or past the wall.
        """
        x, y = position[0], position[1]
        distances = {
            "east": self.x_max - x - radius,
            "west": x - self.x_min - radius,
            "north": self.y_max - y - radius,
            "south": y - self.y_min - radius,
        }
        return {
            name: {
                "distance": distance,
                "vector": -self.walls[name].normal * distance,
                "normal": self.walls[name].normal.copy(),
            }
            for name, distance in distances.items()
        }

    def get_wall_vectors_in_velocity_frame(self, position, velocity, radius: float = 0.0) -> Dict[str, dict]:
        """
        Same as get_wall_vectors, but rotated so +x points along the player's velocity and +y points
        to their left. If the player isn't moving, the heading is undefined and the static x/y frame
        is returned, so the output always has the same shape.
        """
        heading = self._heading(velocity)
        if heading is None:
            return self.get_wall_vectors(position, radius)

        c, s = math.cos(heading), math.sin(heading)
        to_velocity_frame = np.array([[c, s], [-s, c]])  # rotation by -heading

        walls = self.get_wall_vectors(position, radius)
        for info in walls.values():
            info["vector"] = to_velocity_frame @ info["vector"]
            info["normal"] = to_velocity_frame @ info["normal"]
        return walls

    def distance_until_wall(self, position, velocity, radius: float = 0.0) -> float:
        """
        How far the player can keep moving in their current direction before reaching a wall.
        Returns math.inf if the player isn't moving.
        """
        heading = self._heading(velocity)
        if heading is None:
            return math.inf

        x, y = position[0], position[1]
        direction = (math.cos(heading), math.sin(heading))
        limits = ((self.x_min + radius, self.x_max - radius), (self.y_min + radius, self.y_max - radius))

        travel = math.inf
        for coordinate, d, (low, high) in zip((x, y), direction, limits):
            if d > 1e-12:
                travel = min(travel, (high - coordinate) / d)
            elif d < -1e-12:
                travel = min(travel, (low - coordinate) / d)
        return max(travel, 0.0)

    @staticmethod
    def _heading(velocity):
        """Angle of the velocity in the x/y plane, or None if it's (almost) zero."""
        vx, vy = velocity[0], velocity[1]
        if math.hypot(vx, vy) < 1e-9:
            return None
        return math.atan2(vy, vx)


if __name__ == "__main__":

    # Shows the field with a ball and a few players in a GUI. You can drag them around with the mouse.
    # A small tkinter window has a "kick" button per player. A player's button only works while
    # they have the ball, and kicks it towards the goal they attack (red attacks east, blue attacks west).
    import tkinter as tk
    from ball import Ball
    from player import Player

    KICK_SPEED = 3.0  # the ball's maxJointVelocity, so the fastest kick there is

    p.connect(p.GUI)
    field = Field()
    p.resetDebugVisualizerCamera(cameraDistance=14, cameraYaw=0, cameraPitch=-65, cameraTargetPosition=[0, 0, 0])

    ball = Ball(initial_position=np.array([0.0, 0.0, 0.0]))
    players = [
        Player(team=Team.RED, initial_position=np.array([-3.0, 0.0, 0.0])),
        Player(team=Team.RED, initial_position=np.array([-6.0, 3.0, 0.0])),
        Player(team=Team.BLUE, initial_position=np.array([3.0, 0.0, 0.0])),
        Player(team=Team.BLUE, initial_position=np.array([6.0, -3.0, 0.0])),
    ]

    def kick(player):
        if ball.possession_player is not player:
            print(f"Player: {player.body_id} doesn't have the ball")
            return
        goal_x = field.x_max if player.team == Team.RED else field.x_min
        direction = np.array([goal_x, 0.0]) - player.position[:2]
        direction /= np.linalg.norm(direction)
        ball.kick_ball(*(direction * KICK_SPEED))

    root = tk.Tk()
    root.title("Kick")
    buttons = {}
    team_counts = {Team.RED: 0, Team.BLUE: 0}
    for player in players:
        team_counts[player.team] += 1
        colour = "red" if player.team == Team.RED else "blue"
        buttons[player] = tk.Button(root, text=f"{player.team.name.title()} {team_counts[player.team]} kick",
                                    fg=colour, width=16, state=tk.DISABLED,
                                    command=lambda player=player: kick(player))
        buttons[player].pack(padx=8, pady=4)

    window_open = True
    while p.isConnected():
        p.stepSimulation()
        ball.sync_from_pybullet()
        for player in players:
            player.sync_from_pybullet()

        if window_open:
            try:
                # Only the player holding the ball can kick
                for player, button in buttons.items():
                    state = tk.NORMAL if ball.possession_player is player else tk.DISABLED
                    if button["state"] != state:
                        button.config(state=state)
                root.update()  # process clicks without blocking the simulation, instead of root.mainloop()
            except tk.TclError:
                window_open = False  # the kick window was closed; keep the simulation running

        time.sleep(1. / 240.)
### END AI GENERATED
