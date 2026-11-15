"""
This test file was AI generated, see the ### AI GENERATED markers.
AI edit 2026-11-15 (Claude Code, Austin Beinder): pylint cleanup (wrapped lines over 100
characters, unused unpacked values on the kick tests' lines replaced with _); no behaviour change.
"""

### AI GENERATED
import json
import math
import os
import sys
import tkinter as tk
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity import Entity
from joystick import Joystick, knob_to_velocity
from keep_away_env import KeepAwayConfig, KeepAwayEnv

from teams import Team


def xy(observation, key):
    return np.array([observation[key]["x"], observation[key]["y"]])


def stand_still(observation):
    return np.zeros(2)


def grab_and_run(observation):
    """Simple scripted blue: get the ball, then run away from red, turning along walls when close to one."""
    blue = xy(observation["blue"], "position")
    if observation["ball"]["possession_player"] != observation["blue"]["id"]:
        return (xy(observation["ball"], "position") - blue) * 5
    away = blue - xy(observation["red"], "position")
    away /= np.linalg.norm(away)
    if min(observation["blue"]["wall_distances"].values()) < 1.5:
        away = np.array([-away[1], away[0]])
    return away * 3


class KeepAwayEnvTestCase(unittest.TestCase):
    config = KeepAwayConfig()

    def setUp(self):
        self.env = KeepAwayEnv(self.config)
        self.observation, self.info = self.env.reset()

    def tearDown(self):
        self.env.close()

    def give_ball_to(self, player):
        self.env.ball.on_new_collision(player)

    def step_until_done(self, action=(0.0, 0.0), max_steps=10000):
        for _ in range(max_steps):
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                return observation, reward, terminated, truncated, info
        self.fail("game never ended")


class TestReset(KeepAwayEnvTestCase):

    def test_spawns_field_players_and_ball(self):
        self.assertIsNotNone(self.env.field)
        self.assertEqual(self.env.blue.team, Team.BLUE)
        self.assertEqual(self.env.red.team, Team.RED)
        self.assertTrue(self.env.ball.is_ball)
        self.assertEqual(len(self.env.field.walls), 4)

    def test_players_spawn_in_the_middle_two_metres_apart(self):
        blue = xy(self.observation["blue"], "position")
        red = xy(self.observation["red"], "position")

        self.assertAlmostEqual(np.linalg.norm(red - blue), 2.0)
        self.assertTrue(np.allclose((blue + red) / 2, [0, 0]))

    def test_ball_spawns_30_percent_of_the_way_from_blue_to_red(self):
        blue = xy(self.observation["blue"], "position")
        red = xy(self.observation["red"], "position")
        ball = xy(self.observation["ball"], "position")

        self.assertTrue(np.allclose(ball, blue + 0.3 * (red - blue)))
        self.assertLess(np.linalg.norm(ball - blue), np.linalg.norm(ball - red))

    def test_nobody_has_the_ball_at_the_start(self):
        self.assertEqual(self.observation["ball"]["possession_team"], Team.NONE)
        self.assertIsNone(self.observation["ball"]["possession_player"])
        self.assertEqual(self.info["winner"], Team.NONE)
        self.assertEqual(self.observation["game"]["time"], 0.0)

    def test_reset_starts_a_fresh_game(self):
        self.step_until_done()  # red wins
        observation, info = self.env.reset()

        self.assertEqual(info["winner"], Team.NONE)
        self.assertEqual(observation["game"]["time"], 0.0)
        self.assertEqual(observation["game"]["red_possession_time"], 0.0)
        self.assertIsNone(observation["ball"]["possession_player"])
        # Only this game's 3 entities and 4 walls are registered
        self.assertEqual(len(Entity._registry), 7)

    def test_step_after_game_over_raises(self):
        self.step_until_done()

        with self.assertRaises(RuntimeError):
            self.env.step([0, 0])


class TestObservations(KeepAwayEnvTestCase):

    def test_includes_all_entity_observables(self):
        for name, entity in (("blue", self.env.blue), ("red", self.env.red), ("ball", self.env.ball)):
            with self.subTest(entity=name):
                for key in entity.get_observables():
                    self.assertIn(key, self.observation[name])

    def test_includes_wall_distances_for_players(self):
        for name in ("blue", "red"):
            walls = self.observation[name]["wall_distances"]
            self.assertEqual(set(walls), {"east", "west", "north", "south"})
            self.assertIn("distance_until_wall", self.observation[name])

        # Blue spawns at x = -1 on a 21 m field and is 0.3 m wide
        self.assertAlmostEqual(self.observation["blue"]["wall_distances"]["west"], 10.5 - 1.0 - 0.3)

    def test_includes_game_state(self):
        game = self.observation["game"]

        self.assertEqual(game["blue_possession_time_to_win"], 20.0)
        self.assertEqual(game["red_possession_time_to_win"], 0.3)
        self.assertAlmostEqual(game["time_remaining"], 60.0)
        self.assertEqual(game["winner"], Team.NONE)

    def test_flatten_observation(self):
        self.give_ball_to(self.env.blue)
        observation, *_ = self.env.step([0, 0])
        flat = self.env.flatten_observation(observation)

        self.assertEqual(flat.shape, (21,))
        self.assertTrue(np.all(np.isfinite(flat)))
        self.assertTrue(np.allclose(flat[0:2], xy(observation["blue"], "position")))
        self.assertTrue(np.allclose(flat[4:6], xy(observation["red"], "position")))
        self.assertTrue(np.allclose(flat[8:10], xy(observation["ball"], "position")))
        self.assertEqual(flat[12], 1.0)  # blue has the ball
        self.assertEqual(flat[13], 0.0)  # red doesn't

    def test_format_observation_is_one_line_of_json(self):
        text = KeepAwayEnv.format_observation(self.observation)

        self.assertNotIn("\n", text)
        parsed = json.loads(text)
        self.assertEqual(parsed["blue"]["team"], "BLUE")
        self.assertEqual(parsed["blue"]["position"]["x"], -1.0)


class TestCommands(KeepAwayEnvTestCase):

    def test_step_advances_time_by_action_repeat_physics_steps(self):
        observation, *_ = self.env.step([0, 0])

        self.assertAlmostEqual(observation["game"]["time"], self.env.control_dt)
        self.assertAlmostEqual(self.env.control_dt, 8 / 240)

    def test_action_moves_blue(self):
        observation, *_ = self.env.step([0.0, 2.0])  # north, away from the ball

        self.assertAlmostEqual(observation["blue"]["velocity"]["x"], 0.0, delta=0.05)
        self.assertAlmostEqual(observation["blue"]["velocity"]["y"], 2.0, delta=0.1)
        self.assertGreater(observation["blue"]["position"]["y"], 0.0)

    def test_action_does_not_move_red(self):
        observation, *_ = self.env.step([0.0, 2.0])

        # Red only chases the ball, which is on y = 0
        self.assertAlmostEqual(observation["red"]["velocity"]["y"], 0.0, delta=0.05)

    def test_action_is_capped_at_blue_max_speed(self):
        observation, *_ = self.env.step([-10.0, 10.0])

        speed = np.linalg.norm(xy(observation["blue"], "velocity"))
        self.assertAlmostEqual(speed, 3.0, delta=0.1)

    def test_blue_is_slower_with_the_ball(self):
        self.give_ball_to(self.env.blue)
        observation, *_ = self.env.step([0.0, 3.0])

        self.assertAlmostEqual(observation["blue"]["velocity"]["y"], 3.0 * 0.75, delta=0.1)


class TestKicks(KeepAwayEnvTestCase):
    # 2026-11-07: kicking. Red stands still far away, and blue starts with the ball on the west
    # side.
    config = KeepAwayConfig(red_gain=0.0)

    def setUp(self):
        super().setUp()
        self.observation, self.info = self.env.reset(
            options={"blue_position": (-8, 0), "red_position": (8, 5), "blue_has_ball": True})

    def test_nobody_has_kicked_at_the_start(self):
        self.assertEqual(self.info["kicks"], [False])

    def test_kick_releases_the_ball_in_the_kick_direction(self):
        observation, _, _, _, info = self.env.step([0, 0], kicks=[(4.0, 0.0)])

        self.assertEqual(info["kicks"], [True])
        self.assertIsNone(self.env.ball.possession_player)
        self.assertGreater(observation["ball"]["velocity"]["x"], 3.5)
        self.assertAlmostEqual(observation["ball"]["velocity"]["y"], 0.0, delta=0.05)
        self.assertGreater(observation["ball"]["position"]["x"],
                           observation["blue"]["position"]["x"] + 0.45)

    def test_no_kick_without_the_ball(self):
        self.env.reset(options={"blue_position": (-8, 0), "red_position": (8, 5),
                                "ball_position": (0, -3)})
        ball = np.array([0.0, -3.0])
        observation, _, _, _, info = self.env.step([0, 3.0], kicks=[(4.0, 0.0)])

        self.assertEqual(info["kicks"], [False])
        self.assertTrue(np.allclose(xy(observation["ball"], "position"), ball, atol=0.01))

    def test_kick_speed_is_capped(self):
        observation, *_ = self.env.step([0, 0], kicks=[(20.0, 0.0)])

        speed = np.linalg.norm(xy(observation["ball"], "velocity"))
        self.assertLessEqual(speed, 6.0)
        self.assertGreater(speed, 5.0)  # it has slowed a little during the step

    def test_ball_slows_down_the_way_ball_damping_says(self):
        # dv/dt = -d v (1 + v), so after t seconds a ball kicked at v0 has rolled
        # ln((A - e^(-d t)) / (A - 1)) / d metres, where A = 1 + 1 / v0
        start = -8 + 0.3 + 0.15 + 0.02  # where kick_ball puts it: just in front of blue
        observation, *_ = self.env.step([0, 0], kicks=[(3.0, 0.0)])
        for _ in range(59):  # 60 steps of 8 / 240 s = 2 s
            observation, *_ = self.env.step([0, 0])

        d, v0, t = self.config.ball_damping, 3.0, 2.0
        a = 1 + 1 / v0
        # about 2.6 m with ball_damping 0.4
        expected = math.log((a - math.exp(-d * t)) / (a - 1)) / d
        self.assertAlmostEqual(observation["ball"]["position"]["x"] - start, expected, delta=0.15)

    def test_kicker_can_pick_the_ball_straight_back_up(self):
        # No cooldown: a soft kick while running the same way is caught again at once
        self.env.step([3.0, 0.0], kicks=[(1.0, 0.0)])
        for _ in range(3):
            self.env.step([3.0, 0.0])

        self.assertIs(self.env.ball.possession_player, self.env.blue)

    # 2026-11-07: kicks next to a wall used to put the ball inside the wall, and pybullet pushed it
    # out of the arena. The default field's inner wall faces are at x = +-10.5 and y = +-6.8.
    def kick_from(self, blue_position, kick, steps=30):
        self.env.reset(options={"blue_position": blue_position, "red_position": (-8, -5),
                                "blue_has_ball": True})
        observation, _, _, _, info = self.env.step([0, 0], kicks=[kick])
        kicked = info["kicks"]
        for _ in range(steps):
            observation, *_ = self.env.step([0, 0])
        return observation, kicked

    def assert_ball_inside(self, observation):
        ball = xy(observation["ball"], "position")
        self.assertLess(abs(ball[0]), 10.5)
        self.assertLess(abs(ball[1]), 6.8)

    def test_kick_into_the_wall_from_against_it_rebounds_to_the_kicker(self):
        observation, kicked = self.kick_from((10.19, 0), (6.0, 0.0))

        self.assertEqual(kicked, [True])  # the kick happened (ICD: trigger + possession)...
        # ...but the ball came straight back
        self.assertIs(self.env.ball.possession_player, self.env.blue)
        self.assert_ball_inside(observation)

    def test_kick_into_a_corner_rebounds_too(self):
        observation, _ = self.kick_from((10.19, 6.49), (4.0, 4.0))

        self.assertIs(self.env.ball.possession_player, self.env.blue)
        self.assert_ball_inside(observation)

    def test_kick_along_the_wall_still_works(self):
        observation, _ = self.kick_from((10.19, 0), (0.0, 4.0))

        self.assertIsNone(self.env.ball.possession_player)
        self.assertGreater(observation["ball"]["position"]["y"], 1.0)
        self.assert_ball_inside(observation)

    def test_kick_toward_a_nearby_wall_stays_inside(self):
        # Enough room for the ball, but it hits the wall almost at once
        observation, _ = self.kick_from((9.5, 0), (6.0, 0.0))

        self.assertIsNone(self.env.ball.possession_player)
        self.assert_ball_inside(observation)

    def test_one_kick_per_blue_player(self):
        with self.assertRaises(ValueError):
            self.env.step([0, 0], kicks=[None, None])


class TestRedPolicy(KeepAwayEnvTestCase):

    def test_chase_velocity_points_at_target(self):
        velocity = KeepAwayEnv.chase_velocity([0, 0, 0], [0.5, 0, 0], gain=2.0, max_speed=10.0)

        self.assertTrue(np.allclose(velocity, [1.0, 0.0]))

    def test_chase_velocity_is_capped(self):
        velocity = KeepAwayEnv.chase_velocity([0, 0, 0], [10, 10, 0], gain=2.0, max_speed=2.0)

        self.assertAlmostEqual(np.linalg.norm(velocity), 2.0)
        self.assertAlmostEqual(velocity[0], velocity[1])

    def test_red_runs_towards_the_ball(self):
        observation, *_ = self.env.step([0.0, 2.0])  # blue leaves the ball alone

        self.assertLess(observation["red"]["position"]["x"], 1.0)  # ball is at x = -0.4
        self.assertLess(observation["red"]["velocity"]["x"], 0.0)

    def test_red_never_exceeds_its_max_speed(self):
        for _ in range(10):
            observation, *_ = self.env.step([0.0, 2.0])
            self.assertLessEqual(np.linalg.norm(xy(observation["red"], "velocity")), 2.0 + 0.05)


class TestRules(KeepAwayEnvTestCase):

    def test_red_wins_after_holding_the_ball_for_0_3_seconds(self):
        self.give_ball_to(self.env.red)
        observation, reward, terminated, truncated, info = self.step_until_done()

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["winner"], Team.RED)
        self.assertEqual(reward, -1.0)
        self.assertAlmostEqual(info["game_time"], 0.3)  # the game stops on the exact physics step

    def test_red_wins_if_blue_does_nothing(self):
        observation, reward, terminated, truncated, info = self.step_until_done()

        self.assertEqual(info["winner"], Team.RED)

    def test_possession_timers_count_up(self):
        self.give_ball_to(self.env.blue)
        for _ in range(3):
            observation, *_ = self.env.step([0.0, -3.0])  # away from red

        self.assertAlmostEqual(observation["game"]["blue_possession_time"], 3 * self.env.control_dt)
        self.assertEqual(observation["game"]["red_possession_time"], 0.0)

    def test_blue_timer_resets_when_red_steals(self):
        self.give_ball_to(self.env.blue)
        self.env.step([0.0, -3.0])
        self.give_ball_to(self.env.red)
        observation, *_ = self.env.step([0.0, -3.0])

        self.assertEqual(observation["game"]["blue_possession_time"], 0.0)
        self.assertGreater(observation["game"]["red_possession_time"], 0.0)

    def test_timers_reset_when_ball_is_loose(self):
        self.give_ball_to(self.env.blue)
        self.env.step([0.0, -3.0])
        self.env.ball.kick_ball(-3.0, 0.0)  # away from red
        observation, *_ = self.env.step([0.0, -3.0])

        self.assertEqual(observation["game"]["blue_possession_time"], 0.0)
        self.assertEqual(observation["game"]["red_possession_time"], 0.0)


class TestStealsDuringContact(KeepAwayEnvTestCase):
    # 2026-11-07: red used to take the ball only on the first tick it touched blue, so a blue that
    # got the ball while red was already touching it kept the ball for as long as they touched.
    # Red stands still here (red_gain = 0) so the contacts stay put.
    config = KeepAwayConfig(red_gain=0.0)

    def press_into_red(self, blue_xy, red_xy, steps=10):
        self.env.reset(options={"blue_position": blue_xy, "red_position": red_xy,
                                "ball_position": (5.0, 5.0)})
        push = np.subtract(red_xy, blue_xy)
        for _ in range(steps):
            self.env.step(push)
        self.assertIn(self.env.red, self.env.blue.collided_entities)
        return push

    def test_blue_picking_up_the_ball_while_touching_red_loses_it(self):
        push = self.press_into_red((0.0, 0.0), (0.7, 0.0))
        self.give_ball_to(self.env.blue)  # a loose-ball pickup mid-contact

        self.env.step(push)

        self.assertIs(self.env.ball.possession_player, self.env.red)

    def test_same_in_a_corner(self):
        x, y = self.config.field_length / 2 - 0.35, self.config.field_width / 2 - 0.35
        push = self.press_into_red((x - 0.6, y - 0.6), (x, y))  # red pinned in the corner by blue
        self.give_ball_to(self.env.blue)
        start = self.env.game_steps * self.env.time_step

        _, _, _, _, info = self.step_until_done(push)

        # red took it at once and held it for red_win_time
        self.assertEqual(info["winner"], Team.RED)
        # from the first tick
        self.assertAlmostEqual(info["game_time"] - start, self.config.red_win_time)

    def test_old_carrier_cant_take_it_back_without_letting_go(self):
        push = self.press_into_red((0.0, 0.0), (0.7, 0.0))
        self.give_ball_to(self.env.red)
        self.env.step(push)
        # blue was touching red: it takes the ball
        self.assertIs(self.env.ball.possession_player, self.env.blue)

        for _ in range(20):  # still pressed together well past the steal cooldown
            self.env.step(push)
        self.assertIn(self.env.red, self.env.blue.collided_entities)
        self.assertIs(self.env.ball.possession_player, self.env.blue)


class TestRulesWithShortGames(KeepAwayEnvTestCase):
    # Red can't move, and blue only needs half a second, to keep these tests quick
    config = KeepAwayConfig(red_gain=0.0, blue_win_time=0.5, max_game_time=1.0)

    def test_blue_wins_after_holding_the_ball(self):
        self.give_ball_to(self.env.blue)
        observation, reward, terminated, truncated, info = self.step_until_done()

        self.assertTrue(terminated)
        self.assertEqual(info["winner"], Team.BLUE)
        self.assertEqual(reward, 1.0)
        self.assertAlmostEqual(info["game_time"], 0.5)
        self.assertAlmostEqual(info["blue_total_possession_time"], 0.5)

    def test_game_is_truncated_when_time_runs_out(self):
        observation, reward, terminated, truncated, info = self.step_until_done(action=(-3.0, 0.0))  # avoid the ball

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["winner"], Team.NONE)
        self.assertEqual(reward, 0.0)
        self.assertGreaterEqual(info["game_time"], 1.0)


class TestTeams(KeepAwayEnvTestCase):
    # 2026-10-31: teams of any size
    config = KeepAwayConfig(n_blue=2, n_red=3)

    def test_default_spawn_lines_each_team_up(self):
        self.assertEqual(len(self.env.blues), 2)
        self.assertEqual(len(self.env.reds), 3)
        self.assertIs(self.env.blue, self.env.blues[0])
        self.assertIs(self.env.red, self.env.reds[0])
        reds = np.array([xy(r, "position") for r in self.observation["reds"]])
        self.assertTrue(np.allclose(reds[:, 0], 1.0))
        self.assertTrue(np.allclose(reds[:, 1], [-2.0, 0.0, 2.0]))
        self.assertEqual(len(Entity._registry), 2 + 3 + 1 + 4)

    def test_spawn_positions_and_holder_from_options(self):
        observation, _ = self.env.reset(options={
            "blue_positions": [(-3, 0), (-3, 2)], "red_positions": [(3, 0), (3, 2), (3, -2)], "ball_holder": 1})

        self.assertTrue(np.allclose(xy(observation["blues"][1], "position"), [-3, 2]))
        self.assertIs(self.env.ball.possession_player, self.env.blues[1])
        with self.assertRaises(ValueError):
            self.env.reset(options={"blue_position": (0, 0)})  # a single position for a team of two
        with self.assertRaises(ValueError):
            self.env.reset(options={"blue_positions": [(0, 0)]})  # one position short

    def test_one_action_per_blue_player(self):
        observation, *_ = self.env.step([[0.0, 2.0], [0.0, -2.0]])

        self.assertGreater(observation["blues"][0]["velocity"]["y"], 1.5)
        self.assertLess(observation["blues"][1]["velocity"]["y"], -1.5)
        with self.assertRaises(ValueError):
            self.env.step([0.0, 2.0])

    def test_every_red_chases_the_ball(self):
        observation, *_ = self.env.step([[0.0, 2.0], [0.0, 2.0]])  # blue leaves the ball alone

        for red in observation["reds"]:
            self.assertLess(red["velocity"]["x"], 0.0)


class TestTargetGame(KeepAwayEnvTestCase):
    # 2026-10-31: goto_target. Red count 0, and the ball is an immobile target nobody can touch.
    config = KeepAwayConfig(n_red=0, ball_is_target=True, max_game_time=1.0)

    def test_blue_passes_through_the_target(self):
        ball = xy(self.observation["ball"], "position")
        for _ in range(20):  # 20 * 8 / 240 s at ~3 m/s: about 2 m, from x = -1 to past the ball at -0.4
            observation, *_ = self.env.step([3.0, 0.0])  # straight at the ball and past it

        self.assertTrue(np.allclose(xy(observation["ball"], "position"), ball))
        self.assertGreater(observation["blue"]["position"]["x"], ball[0] + 0.5)
        self.assertIsNone(observation["ball"]["possession_player"])
        self.assertIsNone(observation["red"])

    def test_only_time_ends_the_game(self):
        observation, reward, terminated, truncated, info = self.step_until_done()

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_bad_team_sizes_raise(self):
        with self.assertRaises(ValueError):
            KeepAwayEnv(KeepAwayConfig(n_red=1, ball_is_target=True))
        with self.assertRaises(ValueError):
            self.env.reset(options={"blue_has_ball": True})

    def test_a_real_ball_needs_no_red(self):
        # 2026-11-09: pass practice plays with a real ball and no red at all
        game = KeepAwayEnv(KeepAwayConfig(n_red=0, n_blue=2))
        try:
            game.reset(options={"ball_holder": 0})
            observation, _, terminated, _, _ = game.step([[0.0, 0.0], [0.0, 0.0]])
            self.assertFalse(terminated)
            self.assertIsNone(observation["red"])
        finally:
            game.close()


class TestSideBySide(KeepAwayEnvTestCase):
    # 2026-10-31: every game has its own pybullet simulation

    def test_two_games_run_independently(self):
        other = KeepAwayEnv(self.config)
        try:
            other.reset(options={"blue_position": (-4, 3), "red_position": (4, -3), "ball_position": (0, 3)})
            self.assertNotEqual(self.env.client, other.client)
            self.assertEqual(len(Entity._registry), 2 * 7)  # both games' entities and walls

            for _ in range(5):
                mine, *_ = self.env.step([0.0, 2.0])
                theirs, *_ = other.step([0.0, -2.0])

            self.assertGreater(mine["blue"]["position"]["y"], 0.3)
            self.assertLess(theirs["blue"]["position"]["y"], 3.0 - 0.3)
            self.assertAlmostEqual(mine["game"]["time"], theirs["game"]["time"])
        finally:
            other.close()

        self.assertEqual(len(Entity._registry), 7)  # closing a game forgets only its entities
        observation, *_ = self.env.step([0.0, 2.0])  # and this game keeps running
        self.assertGreater(observation["blue"]["position"]["y"], 0.3)

    def test_reset_does_not_touch_the_global_random_generator(self):
        import random
        random.seed(123)
        expected = random.random()
        random.seed(123)
        self.env.reset(seed=0)
        self.assertEqual(random.random(), expected)


class TestPlayGame(KeepAwayEnvTestCase):
    config = KeepAwayConfig(blue_win_time=3.0)

    def test_doing_nothing_loses(self):
        result = self.env.play_game(stand_still)

        self.assertEqual(result["winner"], Team.RED)
        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["blue_total_possession_time"], 0.0)

    def test_grabbing_the_ball_and_running_wins(self):
        result = self.env.play_game(grab_and_run)

        self.assertEqual(result["winner"], Team.BLUE)
        self.assertEqual(result["score"], 1.0)
        self.assertGreaterEqual(result["blue_total_possession_time"], 3.0)


class TestJoystick(unittest.TestCase):

    def test_centre_is_zero_velocity(self):
        self.assertTrue(np.allclose(knob_to_velocity(0, 0, radius=80, max_speed=3), [0, 0]))

    def test_right_is_plus_x(self):
        self.assertTrue(np.allclose(knob_to_velocity(80, 0, radius=80, max_speed=3), [3, 0]))

    def test_up_on_screen_is_plus_y(self):
        # Screen y grows downwards
        self.assertTrue(np.allclose(knob_to_velocity(0, -80, radius=80, max_speed=3), [0, 3]))

    def test_partial_push_gives_partial_speed(self):
        self.assertTrue(np.allclose(knob_to_velocity(40, 0, radius=80, max_speed=3), [1.5, 0]))

    def test_past_the_edge_is_capped_at_max_speed(self):
        velocity = knob_to_velocity(300, -300, radius=80, max_speed=3)

        self.assertAlmostEqual(np.linalg.norm(velocity), 3.0)
        self.assertTrue(np.allclose(velocity, [3 / math.sqrt(2), 3 / math.sqrt(2)]))

    def test_widget_drag_and_release(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display available for tkinter")
        try:
            joystick = Joystick(root, max_speed=3.0)

            class Event:
                x = joystick.centre + joystick.radius  # dragged fully to the right
                y = joystick.centre

            joystick._on_drag(Event())
            self.assertTrue(np.allclose(joystick.velocity, [3, 0]))

            joystick._on_release()
            self.assertTrue(np.allclose(joystick.velocity, [0, 0]))
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
### END AI GENERATED
