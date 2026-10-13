"""
This test file was AI generated, see the ### AI GENERATED markers.
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
