"""
This test file was AI generated, see the ### AI GENERATED markers.
"""

### AI GENERATED
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ball import Ball
from entity import Entity
from field import Field, Wall
from player import Player

from teams import Team


def step(entities, steps=1):
    for _ in range(steps):
        p.stepSimulation()
        for entity in entities:
            entity.sync_from_pybullet()


class TestFieldConstruction(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()

    def tearDown(self):
        p.disconnect()

    def test_default_size_and_bounds(self):
        field = Field()

        self.assertEqual(field.length, 21.0)
        self.assertEqual(field.width, 13.6)
        self.assertEqual((field.x_min, field.x_max), (-10.5, 10.5))
        self.assertEqual((field.y_min, field.y_max), (-6.8, 6.8))

    def test_custom_size(self):
        field = Field(length=10.0, width=6.0)

        self.assertEqual((field.x_min, field.x_max), (-5.0, 5.0))
        self.assertEqual((field.y_min, field.y_max), (-3.0, 3.0))
        self.assertAlmostEqual(field.marking_scale, min(10.0 / 105.0, 6.0 / 68.0))

    def test_walls_line_the_field_edges(self):
        field = Field(length=10.0, width=6.0)

        self.assertEqual(set(field.walls), {"east", "west", "north", "south"})

        # Each wall's inner face sits exactly on the field edge
        east_min, _ = p.getAABB(field.walls["east"].body_id)
        _, west_max = p.getAABB(field.walls["west"].body_id)
        north_min, _ = p.getAABB(field.walls["north"].body_id)
        _, south_max = p.getAABB(field.walls["south"].body_id)
        self.assertAlmostEqual(east_min[0], 5.0, places=2)
        self.assertAlmostEqual(west_max[0], -5.0, places=2)
        self.assertAlmostEqual(north_min[1], 3.0, places=2)
        self.assertAlmostEqual(south_max[1], -3.0, places=2)

    def test_walls_are_tall_enough_for_a_carried_ball(self):
        field = Field()
        ball = Ball()

        for wall in field.walls.values():
            wall_min, wall_max = p.getAABB(wall.body_id)
            self.assertLess(wall_min[2], -0.05)  # below the bottom of the entities
            self.assertGreater(wall_max[2], ball.CARRY_HEIGHT + 0.05)  # above the top of a carried ball

    def test_walls_are_registered_for_collision_events(self):
        field = Field()

        for name, wall in field.walls.items():
            self.assertIsInstance(wall, Wall)
            self.assertIs(Entity._registry[wall.body_id], wall)
            self.assertTrue(wall.is_wall)
            self.assertFalse(wall.is_ball)
            self.assertEqual(wall.team, Team.NONE)
            self.assertEqual(wall.name, name)

    def test_wall_normals_point_into_the_field(self):
        field = Field()

        self.assertTrue(np.array_equal(field.walls["east"].normal, [-1, 0]))
        self.assertTrue(np.array_equal(field.walls["west"].normal, [1, 0]))
        self.assertTrue(np.array_equal(field.walls["north"].normal, [0, -1]))
        self.assertTrue(np.array_equal(field.walls["south"].normal, [0, 1]))

    def test_floor_and_lines_have_no_collision(self):
        # Entities sit half below z = 0, so a solid floor would collide with all of them
        field = Field()

        self.assertEqual(len(p.getCollisionShapeData(field.floor_id, -1)), 0)
        self.assertEqual(len(p.getCollisionShapeData(field.lines_id, -1)), 0)

    def test_entities_on_the_floor_touch_nothing(self):
        field = Field()
        player = Player(initial_position=np.array([2.0, 2.0, 0.0]))

        step([player])

        self.assertEqual(p.getContactPoints(bodyA=player.body_id), ())
        self.assertEqual(player.collided_entities, [])


class TestWallCollisions(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()
        self.field = Field(length=10.0, width=6.0)

    def tearDown(self):
        p.disconnect()

    def test_player_touching_wall_fires_wall_collision_event(self):
        # Overlaps the east wall (inner face at x = 5) from the start
        player = Player(initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))

        with mock.patch.object(Player, "on_wall_collision") as on_wall_collision:
            step([player])

        on_wall_collision.assert_called_once_with(self.field.walls["east"])
        self.assertIn(self.field.walls["east"], player.collided_entities)

    def test_wall_collision_event_fires_once_per_contact(self):
        player = Player(initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))

        with mock.patch.object(Player, "on_wall_collision") as on_wall_collision:
            step([player], steps=5)  # stays against the wall

        self.assertEqual(on_wall_collision.call_count, 1)

    def test_player_running_into_wall_is_stopped_and_fires_event(self):
        player = Player(initial_position=np.array([3.0, 0.0, 0.0]))

        with mock.patch.object(Player, "on_wall_collision") as on_wall_collision:
            for _ in range(480):  # 2 s, plenty of time to cover 2 m at up to 3 m/s
                player.apply_velocity(x=3.0)
                step([player])

        on_wall_collision.assert_called_with(self.field.walls["east"])
        # Stopped at the wall: centre no further than one radius (plus a little solver slack) from the face
        self.assertLess(player.position[0], 5.0 - player.radius + 0.05)

    def test_player_hits_each_wall(self):
        cases = {
            "east": [5.0 - 0.2, 0.0],
            "west": [-5.0 + 0.2, 0.0],
            "north": [0.0, 3.0 - 0.2],
            "south": [0.0, -3.0 + 0.2],
        }
        for name, (x, y) in cases.items():
            with self.subTest(wall=name):
                player = Player(initial_position=np.array([x, y, 0.0]))
                with mock.patch.object(Player, "on_wall_collision") as on_wall_collision:
                    step([player])
                on_wall_collision.assert_called_once_with(self.field.walls[name])
                p.removeBody(player.body_id)

    def test_wall_does_not_take_ball_from_player(self):
        player = Player(team=Team.RED, initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))
        ball = Ball(initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))
        ball.possession_flip_percentage = 100  # a wall would always "steal" if it were treated as an opponent
        ball.on_new_collision(player)

        step([ball, player], steps=5)

        self.assertIs(ball.possession_player, player)
        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(player.ball, ball)

    def test_wall_collision_prints_message(self):
        player = Player(initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))

        ### AI GENERATED (the print is a debug message now, off unless Entity.print_collisions is set)
        with mock.patch.object(Entity, "print_collisions", True), mock.patch("builtins.print") as printed:
            step([player])
        ### END AI GENERATED

        printed.assert_any_call("Entity collided with wall")

    ### AI GENERATED
    def test_wall_collision_is_quiet_by_default(self):
        player = Player(initial_position=np.array([5.0 - 0.2, 0.0, 0.0]))

        with mock.patch("builtins.print") as printed:
            step([player])

        printed.assert_not_called()
    ### END AI GENERATED

    def test_ball_touching_wall_fires_wall_collision_event(self):
        ball = Ball(initial_position=np.array([5.0 - 0.1, 0.0, 0.0]))

        with mock.patch.object(Ball, "on_wall_collision") as on_wall_collision:
            step([ball])

        on_wall_collision.assert_called_once_with(self.field.walls["east"])

    def test_plain_entity_touching_wall_fires_wall_collision_event(self):
        entity = Entity(initial_position=np.array([0.0, 3.0 - 0.2, 0.0]))

        with mock.patch.object(Entity, "on_wall_collision") as on_wall_collision:
            step([entity])

        on_wall_collision.assert_called_once_with(self.field.walls["north"])

    def test_loose_ball_touching_wall_stays_loose(self):
        ball = Ball(initial_position=np.array([5.0 - 0.1, 0.0, 0.0]))

        step([ball])

        self.assertIn(self.field.walls["east"], ball.collided_entities)
        self.assertEqual(ball.possession, Team.NONE)
        self.assertIsNone(ball.possession_player)
        self.assertIsNone(ball.carry_constraint)


class TestWallVectors(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()
        self.field = Field(length=10.0, width=6.0)  # x in [-5, 5], y in [-3, 3]

    def tearDown(self):
        p.disconnect()

    def assertVector(self, actual, expected):
        self.assertTrue(np.allclose(actual, expected), f"{actual} != {expected}")

    def test_wall_vectors_from_centre(self):
        walls = self.field.get_wall_vectors(np.array([0.0, 0.0, 0.0]))

        self.assertAlmostEqual(walls["east"]["distance"], 5.0)
        self.assertAlmostEqual(walls["west"]["distance"], 5.0)
        self.assertAlmostEqual(walls["north"]["distance"], 3.0)
        self.assertAlmostEqual(walls["south"]["distance"], 3.0)
        self.assertVector(walls["east"]["vector"], [5, 0])
        self.assertVector(walls["west"]["vector"], [-5, 0])
        self.assertVector(walls["north"]["vector"], [0, 3])
        self.assertVector(walls["south"]["vector"], [0, -3])

    def test_wall_vectors_off_centre(self):
        walls = self.field.get_wall_vectors(np.array([2.0, -1.0, 0.0]))

        self.assertVector(walls["east"]["vector"], [3, 0])
        self.assertVector(walls["west"]["vector"], [-7, 0])
        self.assertVector(walls["north"]["vector"], [0, 4])
        self.assertVector(walls["south"]["vector"], [0, -2])

    def test_wall_vectors_include_normals(self):
        walls = self.field.get_wall_vectors(np.array([0.0, 0.0, 0.0]))

        for name, info in walls.items():
            self.assertVector(info["normal"], self.field.walls[name].normal)
            # The vector to a wall points against the way the wall faces
            self.assertVector(info["vector"], -info["normal"] * info["distance"])

    def test_wall_vectors_measured_from_player_edge(self):
        walls = self.field.get_wall_vectors(np.array([0.0, 0.0, 0.0]), radius=0.3)

        self.assertAlmostEqual(walls["east"]["distance"], 4.7)
        self.assertAlmostEqual(walls["north"]["distance"], 2.7)
        self.assertVector(walls["east"]["vector"], [4.7, 0])

    def test_wall_distance_negative_when_overlapping(self):
        walls = self.field.get_wall_vectors(np.array([4.9, 0.0, 0.0]), radius=0.3)

        self.assertAlmostEqual(walls["east"]["distance"], -0.2)

    def test_returned_normals_are_copies(self):
        walls = self.field.get_wall_vectors(np.array([0.0, 0.0, 0.0]))
        walls["east"]["normal"][0] = 99

        self.assertVector(self.field.walls["east"].normal, [-1, 0])

    def test_velocity_frame_moving_along_x_matches_static_frame(self):
        position = np.array([1.0, 1.0, 0.0])
        static = self.field.get_wall_vectors(position)
        moving = self.field.get_wall_vectors_in_velocity_frame(position, np.array([2.0, 0.0, 0.0]))

        for name in static:
            self.assertVector(moving[name]["vector"], static[name]["vector"])
            self.assertAlmostEqual(moving[name]["distance"], static[name]["distance"])

    def test_velocity_frame_moving_along_y(self):
        # Heading north: north wall is straight ahead (+x), east wall is to the right (-y)
        walls = self.field.get_wall_vectors_in_velocity_frame(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

        self.assertVector(walls["north"]["vector"], [3, 0])
        self.assertVector(walls["south"]["vector"], [-3, 0])
        self.assertVector(walls["east"]["vector"], [0, -5])
        self.assertVector(walls["west"]["vector"], [0, 5])
        self.assertVector(walls["north"]["normal"], [-1, 0])  # facing back towards the player

    def test_velocity_frame_diagonal(self):
        walls = self.field.get_wall_vectors_in_velocity_frame(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]))

        # East wall vector (5, 0) rotated by -45 degrees
        self.assertVector(walls["east"]["vector"], [5 / math.sqrt(2), -5 / math.sqrt(2)])
        # Distances don't depend on the frame
        self.assertAlmostEqual(walls["east"]["distance"], 5.0)

    def test_velocity_frame_with_zero_velocity_falls_back_to_static(self):
        position = np.array([1.0, -2.0, 0.0])
        static = self.field.get_wall_vectors(position)
        still = self.field.get_wall_vectors_in_velocity_frame(position, np.array([0.0, 0.0, 0.0]))

        for name in static:
            self.assertVector(still[name]["vector"], static[name]["vector"])

    def test_distance_until_wall_straight(self):
        origin = np.array([0.0, 0.0, 0.0])

        self.assertAlmostEqual(self.field.distance_until_wall(origin, np.array([1.0, 0.0, 0.0])), 5.0)
        self.assertAlmostEqual(self.field.distance_until_wall(origin, np.array([-3.0, 0.0, 0.0])), 5.0)
        self.assertAlmostEqual(self.field.distance_until_wall(origin, np.array([0.0, 2.0, 0.0])), 3.0)
        self.assertAlmostEqual(self.field.distance_until_wall(origin, np.array([0.0, -2.0, 0.0])), 3.0)

    def test_distance_until_wall_diagonal_hits_nearest_wall_first(self):
        # Moving at 45 degrees from the centre reaches y = 3 after 3 * sqrt(2) m, before x = 5
        distance = self.field.distance_until_wall(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]))

        self.assertAlmostEqual(distance, 3 * math.sqrt(2))

    def test_distance_until_wall_ignores_speed(self):
        position = np.array([1.0, 0.5, 0.0])

        slow = self.field.distance_until_wall(position, np.array([0.1, 0.2, 0.0]))
        fast = self.field.distance_until_wall(position, np.array([3.0, 6.0, 0.0]))

        self.assertAlmostEqual(slow, fast)

    def test_distance_until_wall_with_radius(self):
        distance = self.field.distance_until_wall(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), radius=0.3)

        self.assertAlmostEqual(distance, 4.7)

    def test_distance_until_wall_when_already_touching(self):
        distance = self.field.distance_until_wall(np.array([4.8, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), radius=0.3)

        self.assertEqual(distance, 0.0)

    def test_distance_until_wall_when_standing_still(self):
        distance = self.field.distance_until_wall(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

        self.assertEqual(distance, math.inf)


if __name__ == "__main__":
    unittest.main()
### END AI GENERATED
