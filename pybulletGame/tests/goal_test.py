"""
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-23
Author:  Austin Beinder
Usage:
  - 2026-11-23: Generated this entire file: tests for goal.py (GoalSpec geometry, the goal's
    body, the mouths Field opens in its walls, and the goal events).
"""

### AI GENERATED
import math
import os
import sys
import unittest

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ball import Ball
from entity import Entity
from field import Field
from goal import Goal, GoalEvent, GoalSpec, standard_goals
from player import Player
from teams import Team

# A 10 x 6 m field: x in [-5, 5], y in [-3, 3]. The goals are 2 m wide and 0.5 m deep.
LENGTH, WIDTH = 10.0, 6.0


def tick(entities, goals, steps=1):
    """Step the simulation, sync everything, and collect the goal events."""
    events = []
    ball = next(entity for entity in entities if entity.is_ball)
    for _ in range(steps):
        p.stepSimulation()
        for entity in entities:
            entity.sync_from_pybullet()
        ball.check_held_contacts()
        for goal in goals:
            event = goal.update(ball)
            if event is not None:
                events.append(event)
    return events


def ball_at(x, y, **kwargs):
    """A ball built at the origin (so its joints are world x/y) and moved to (x, y)."""
    ball = Ball(initial_position=np.array([0.0, 0.0, 0.0]), **kwargs)
    ball.apply_position(x, y)
    return ball


def player_at(team, x, y, radius=0.25):
    player = Player(team=team, radius=radius, initial_position=np.array([0.0, 0.0, 0.0]))
    player.apply_position(x, y)
    return player


class TestGoalSpec(unittest.TestCase):
    """Geometry only, no simulation."""

    def setUp(self):
        self.west, self.east = standard_goals(LENGTH, WIDTH)

    def test_standard_goals(self):
        self.assertEqual(self.west.center, (-5.0, 0.0))
        self.assertEqual(self.west.normal, (1.0, 0.0))
        self.assertEqual(self.west.defending_team, Team.BLUE)
        self.assertEqual(self.east.center, (5.0, 0.0))
        self.assertEqual(self.east.normal, (-1.0, 0.0))
        self.assertEqual(self.east.defending_team, Team.RED)
        self.assertEqual((self.west.width, self.west.depth), (2.0, 0.5))

    def test_standard_goals_can_swap_ends(self):
        west, east = standard_goals(LENGTH, WIDTH, west_team=Team.RED)
        self.assertEqual((west.defending_team, east.defending_team), (Team.RED, Team.BLUE))

    def test_scoring_team_is_the_other_team(self):
        self.assertEqual(self.west.scoring_team, Team.RED)
        self.assertEqual(self.east.scoring_team, Team.BLUE)

    def test_to_local(self):
        # along = behind the line, across = along the line (the normal turned anticlockwise)
        along, across = self.east.to_local([5.3, 0.4])
        self.assertAlmostEqual(along, 0.3)
        self.assertAlmostEqual(across, -0.4)
        along, across = self.west.to_local([-5.3, 0.4])
        self.assertAlmostEqual(along, 0.3)
        self.assertAlmostEqual(across, 0.4)
        along, _ = self.east.to_local([4.0, 0.0])
        self.assertAlmostEqual(along, -1.0)  # in the field

    def test_to_world_inverts_to_local(self):
        for spec in (self.west, self.east):
            point = spec.to_world(0.3, -0.7)
            along, across = spec.to_local(point)
            self.assertAlmostEqual(along, 0.3)
            self.assertAlmostEqual(across, -0.7)

    def test_ball_in_needs_the_whole_ball_over_the_line(self):
        r = 0.15
        self.assertFalse(self.east.ball_in([5.0, 0.0], r))  # centre on the line
        self.assertFalse(self.east.ball_in([5.0 + r - 0.01, 0.0], r))  # still touching the line
        self.assertTrue(self.east.ball_in([5.0 + r, 0.0], r))
        self.assertFalse(self.east.ball_in([4.0, 0.0], r))

    def test_ball_in_needs_the_ball_between_the_posts(self):
        r = 0.15
        self.assertTrue(self.east.ball_in([5.3, 0.99], r))
        self.assertFalse(self.east.ball_in([5.3, 1.01], r))
        self.assertFalse(self.east.ball_in([5.3, -1.01], r))

    def test_goal_on_a_side_wall(self):
        north = GoalSpec(center=(1.0, 3.0), normal=(0.0, -1.0), defending_team=Team.BLUE)
        self.assertTrue(north.ball_in([1.5, 3.2], 0.15))
        self.assertFalse(north.ball_in([2.5, 3.2], 0.15))
        posts = sorted(tuple(post) for post in north.posts())
        np.testing.assert_allclose(posts, [(0.0, 3.0), (2.0, 3.0)], atol=1e-12)

    def test_posts(self):
        posts = sorted(tuple(post) for post in self.east.posts())
        np.testing.assert_allclose(posts, [(5.0, -1.0), (5.0, 1.0)], atol=1e-12)

    def test_numpy_inputs_become_float_tuples(self):
        spec = GoalSpec(center=np.array([5, 0]), normal=np.array([-1, 0]), defending_team=Team.RED)
        self.assertEqual(spec.center, (5.0, 0.0))
        self.assertEqual(spec.normal, (-1.0, 0.0))

    def test_rejects_team_none(self):
        with self.assertRaises(ValueError):
            GoalSpec(center=(5.0, 0.0), normal=(-1.0, 0.0), defending_team=Team.NONE)

    def test_rejects_a_normal_that_isnt_a_unit_vector(self):
        with self.assertRaises(ValueError):
            GoalSpec(center=(5.0, 0.0), normal=(-2.0, 0.0), defending_team=Team.RED)

    def test_rejects_sizes_that_arent_positive(self):
        for name in ("width", "depth", "height"):
            with self.subTest(size=name), self.assertRaises(ValueError):
                GoalSpec(center=(5.0, 0.0), normal=(-1.0, 0.0), defending_team=Team.RED, **{name: 0.0})


class TestFieldWithGoals(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()

    def tearDown(self):
        p.disconnect()

    def test_no_goals_builds_the_same_field_as_before(self):
        field = Field(length=LENGTH, width=WIDTH)

        self.assertEqual(field.goals, [])
        for wall in field.walls.values():
            self.assertEqual(len(p.getCollisionShapeData(wall.body_id, -1)), 1)
        # floor, lines, four walls: no extra bodies
        self.assertEqual(p.getNumBodies(), 6)

    def test_goal_walls_have_a_mouth_and_keep_their_bounds(self):
        field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))

        for name in ("east", "west"):
            self.assertEqual(len(p.getCollisionShapeData(field.walls[name].body_id, -1)), 2)
        for name in ("north", "south"):
            self.assertEqual(len(p.getCollisionShapeData(field.walls[name].body_id, -1)), 1)
        east_min, east_max = p.getAABB(field.walls["east"].body_id)
        self.assertAlmostEqual(east_min[0], 5.0, places=2)
        self.assertAlmostEqual(east_min[1], -3.5, places=2)
        self.assertAlmostEqual(east_max[1], 3.5, places=2)

    def test_the_mouth_is_open_and_the_rest_of_the_wall_isnt(self):
        field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))
        east_goal, east_wall = field.goals[1], field.walls["east"]

        def first_hit(y):
            return p.rayTest([4.0, y, 0.0], [7.0, y, 0.0])[0][0]

        self.assertEqual(first_hit(0.0), east_goal.body_id)  # through the mouth to the back of the net
        self.assertEqual(first_hit(0.9), east_goal.body_id)
        self.assertEqual(first_hit(1.1), east_wall.body_id)  # just past the post
        self.assertEqual(first_hit(-2.0), east_wall.body_id)

    def test_goals_are_registered_as_walls(self):
        field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))

        for goal, team in zip(field.goals, (Team.BLUE, Team.RED)):
            self.assertIsInstance(goal, Goal)
            self.assertIs(Entity._registry[(goal.client, goal.body_id)], goal)
            self.assertTrue(goal.is_wall)
            self.assertTrue(goal.is_goal)
            self.assertFalse(goal.is_ball)
            self.assertEqual(goal.team, Team.NONE)
            self.assertEqual(goal.defending_team, team)
        self.assertEqual(field.goal_defended_by(Team.BLUE), [field.goals[0]])
        self.assertEqual(field.goal_defended_by(Team.RED), [field.goals[1]])

    def test_pocket_is_behind_the_goal_line(self):
        field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))

        (low, high) = p.getAABB(field.goals[1].body_id)
        self.assertAlmostEqual(low[0], 5.0, places=2)
        self.assertGreater(high[0], 5.5)  # back panel behind the 0.5 m pocket
        self.assertAlmostEqual(low[2], -0.1, places=2)
        self.assertGreater(high[2], Ball().CARRY_HEIGHT + 0.05)  # taller than a carried ball

    def test_goal_on_a_side_wall(self):
        spec = GoalSpec(center=(1.0, 3.0), normal=(0.0, -1.0), defending_team=Team.BLUE, name="north_goal")
        field = Field(length=LENGTH, width=WIDTH, goals=[spec])

        self.assertEqual(len(p.getCollisionShapeData(field.walls["north"].body_id, -1)), 2)
        self.assertEqual(p.rayTest([1.0, 2.0, 0.0], [1.0, 5.0, 0.0])[0][0], field.goals[0].body_id)
        self.assertEqual(p.rayTest([-1.0, 2.0, 0.0], [-1.0, 5.0, 0.0])[0][0], field.walls["north"].body_id)

    def test_rejects_a_goal_off_the_walls(self):
        for center, normal in (((0.0, 0.0), (-1.0, 0.0)),  # in the middle of the field
                               ((5.0, 0.0), (1.0, 0.0))):  # on the east edge but facing out
            with self.subTest(center=center, normal=normal), self.assertRaises(ValueError):
                Field(length=LENGTH, width=WIDTH,
                      goals=[GoalSpec(center=center, normal=normal, defending_team=Team.RED)])

    def test_rejects_a_goal_past_the_field_edge(self):
        with self.assertRaises(ValueError):
            Field(length=LENGTH, width=WIDTH,
                  goals=[GoalSpec(center=(5.0, 2.5), normal=(-1.0, 0.0), defending_team=Team.RED)])

    def test_rejects_overlapping_goals(self):
        goals = [GoalSpec(center=(5.0, 0.0), normal=(-1.0, 0.0), defending_team=Team.RED),
                 GoalSpec(center=(5.0, 1.5), normal=(-1.0, 0.0), defending_team=Team.BLUE)]
        with self.assertRaises(ValueError):
            Field(length=LENGTH, width=WIDTH, goals=goals)

    def test_two_goals_on_one_wall(self):
        goals = [GoalSpec(center=(5.0, -1.5), normal=(-1.0, 0.0), defending_team=Team.RED, width=1.0),
                 GoalSpec(center=(5.0, 1.5), normal=(-1.0, 0.0), defending_team=Team.BLUE, width=1.0)]
        field = Field(length=LENGTH, width=WIDTH, goals=goals)

        self.assertEqual(len(p.getCollisionShapeData(field.walls["east"].body_id, -1)), 3)
        self.assertEqual(p.rayTest([4.0, 0.0, 0.0], [7.0, 0.0, 0.0])[0][0], field.walls["east"].body_id)


class TestGoalPhysics(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()
        self.field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))
        self.west_goal, self.east_goal = self.field.goals

    def tearDown(self):
        p.disconnect()

    def test_ball_rolled_at_the_mouth_ends_up_in_the_net(self):
        ball = ball_at(4.5, 0.0)
        ball.apply_velocity(3.0, 0.0)

        tick([ball], self.field.goals, steps=480)

        self.assertTrue(self.east_goal.spec.ball_in(ball.position, ball.radius))
        self.assertLess(ball.position[0], 5.5)  # stopped by the back of the net
        self.assertEqual(ball.possession, Team.NONE)  # the net never takes the ball
        self.assertIsNone(ball.possession_player)

    def test_ball_rolled_beside_the_mouth_stops_at_the_wall(self):
        ball = ball_at(4.5, 2.0)
        ball.apply_velocity(3.0, 0.0)

        events = tick([ball], self.field.goals, steps=480)

        self.assertEqual(events, [])
        self.assertLess(ball.position[0], 5.0 - ball.radius + 0.05)

    def test_player_can_walk_into_the_pocket(self):
        player = player_at(Team.RED, 4.0, 0.0)

        for _ in range(480):
            player.apply_velocity(3.0, 0.0)
            p.stepSimulation()
            player.sync_from_pybullet()

        self.assertGreater(player.position[0], 5.0 + 0.15)
        self.assertLess(player.position[0], 5.5 - player.radius + 0.05)  # stopped by the back of the net


class TestGoalEvents(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()
        self.field = Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH))
        self.west_goal, self.east_goal = self.field.goals

    def tearDown(self):
        p.disconnect()

    def kick_from(self, player, velocity):
        ball = ball_at(*player.position[:2])
        ball.on_new_collision(player)  # the player picks it up
        self.assertIs(ball.possession_player, player)
        ball.kick_ball(*velocity)
        return ball

    def test_kick_into_the_east_goal_scores_for_blue(self):
        blue = player_at(Team.BLUE, 4.0, 0.0)
        ball = self.kick_from(blue, (3.0, 0.0))

        events = tick([ball, blue], self.field.goals, steps=480)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertIsInstance(event, GoalEvent)
        self.assertEqual(event.goal_name, "east_goal")
        self.assertEqual(event.scoring_team, Team.BLUE)
        self.assertIs(event.scorer, blue)
        self.assertEqual(event.scorer_team, Team.BLUE)
        self.assertFalse(event.own_goal)
        self.assertFalse(event.carried)
        self.assertGreater(event.sim_time, 0.0)

    def test_kick_into_the_west_goal_scores_for_red(self):
        red = player_at(Team.RED, -4.0, 0.0)
        ball = self.kick_from(red, (-3.0, 0.0))

        events = tick([ball, red], self.field.goals, steps=480)

        self.assertEqual([(e.goal_name, e.scoring_team, e.own_goal) for e in events],
                         [("west_goal", Team.RED, False)])

    def test_own_goal_scores_for_the_other_team(self):
        red = player_at(Team.RED, 4.0, 0.0)  # red defends the east goal and kicks into it
        ball = self.kick_from(red, (3.0, 0.0))

        events = tick([ball, red], self.field.goals, steps=480)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].scoring_team, Team.BLUE)
        self.assertIs(events[0].scorer, red)
        self.assertTrue(events[0].own_goal)

    def test_carried_ball_counts(self):
        blue = player_at(Team.BLUE, 4.0, 0.0)
        ball = ball_at(4.0, 0.0)
        ball.on_new_collision(blue)

        events = []
        for _ in range(480):
            blue.apply_velocity(3.0, 0.0)
            events += tick([ball, blue], self.field.goals)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].scoring_team, Team.BLUE)
        self.assertIs(events[0].scorer, blue)
        self.assertTrue(events[0].carried)
        self.assertIs(ball.possession_player, blue)  # touching the net doesn't take the ball away

    def test_untouched_ball_has_no_scorer(self):
        ball = ball_at(5.3, 0.0)

        events = tick([ball], self.field.goals)

        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].scorer)
        self.assertEqual(events[0].scorer_team, Team.NONE)
        self.assertFalse(events[0].own_goal)
        self.assertEqual(events[0].scoring_team, Team.BLUE)

    def test_ball_resting_in_the_net_scores_once(self):
        ball = ball_at(5.3, 0.0)

        events = tick([ball], self.field.goals, steps=240)

        self.assertEqual(len(events), 1)

    def test_ball_that_leaves_and_comes_back_scores_again(self):
        ball = ball_at(5.3, 0.0)
        events = tick([ball], self.field.goals)
        ball.apply_position(3.0, 0.0)
        events += tick([ball], self.field.goals)
        ball.apply_position(5.3, 0.0)
        events += tick([ball], self.field.goals)

        self.assertEqual(len(events), 2)

    def test_kick_before_any_update_still_has_its_kicker(self):
        # A player who starts holding the ball and kicks on the first step: no goal.update has
        # seen them hold it, but the ball remembers (Ball.last_holder)
        blue = player_at(Team.BLUE, 4.0, 0.0)
        ball = self.kick_from(blue, (3.0, 0.0))
        self.assertIs(ball.last_holder, blue)

        events = tick([ball, blue], self.field.goals, steps=480)

        self.assertIs(events[0].scorer, blue)

    def test_last_holder_is_the_latest_one(self):
        blue = player_at(Team.BLUE, 0.0, 0.0)
        red = player_at(Team.RED, 2.0, 0.0)
        ball = self.kick_from(blue, (0.0, 0.0))  # dropped where blue stands
        ball.on_new_collision(red)

        self.assertIs(ball.last_holder, red)

    def test_reset_forgets_that_the_ball_was_in(self):
        ball = ball_at(5.3, 0.0)
        tick([ball], self.field.goals)
        self.assertTrue(self.east_goal.ball_inside)

        self.east_goal.reset()

        self.assertFalse(self.east_goal.ball_inside)
        self.assertEqual(len(tick([ball], self.field.goals)), 1)  # counts again

    def test_net_collision_is_a_wall_collision(self):
        ball = ball_at(4.5, 0.0)
        ball.apply_velocity(3.0, 0.0)
        hits = []
        ball.on_wall_collision = hits.append

        tick([ball], self.field.goals, steps=480)

        self.assertIn(self.east_goal, hits)  # it hit the back of the net
        self.assertEqual(ball.possession, Team.NONE)


class TestGoalsInTwoSimulations(unittest.TestCase):
    def setUp(self):
        Entity._registry.clear()
        self.clients = [p.connect(p.DIRECT), p.connect(p.DIRECT)]

    def tearDown(self):
        for client in self.clients:
            p.disconnect(client)
        Entity._registry.clear()  # the registry is shared: don't leave this test's entries for others

    def test_each_simulation_scores_on_its_own(self):
        fields, balls = [], []
        for client in self.clients:
            fields.append(Field(length=LENGTH, width=WIDTH, goals=standard_goals(LENGTH, WIDTH),
                                physics_client=client))
            balls.append(Ball(initial_position=np.array([0.0, 0.0, 0.0]), physics_client=client))
        balls[1].apply_position(5.3, 0.0)  # only the second simulation has a ball in the net

        events = [[], []]
        for field, ball, client, out in zip(fields, balls, self.clients, events):
            p.stepSimulation(physicsClientId=client)
            ball.sync_from_pybullet()
            for goal in field.goals:
                event = goal.update(ball)
                if event is not None:
                    out.append(event)

        self.assertEqual(events[0], [])
        self.assertEqual([e.goal_name for e in events[1]], ["east_goal"])
        for field, client in zip(fields, self.clients):
            for goal in field.goals:
                self.assertEqual(goal.client, client)
                self.assertIs(Entity._registry[(client, goal.body_id)], goal)


if __name__ == "__main__":
    unittest.main()
### END AI GENERATED
