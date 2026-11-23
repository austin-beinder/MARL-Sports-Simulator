"""
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-11-23
Author:  Austin Beinder
Usage:
  - 2026-11-23: Generated this entire file: tests for soccer_env.py (goals end the game, red
    possession doesn't, the chasing red, time-outs, team sizes, separate simulations) and for
    Ball.goal_lanes.
  - 2026-11-23: Tests for red_policy = "dribble" (red carries the ball into blue's goal, from the
    middle and from near a sideline; other reds chase their own carrier; a loose ball is still
    chased; unknown policies raise). (Austin Beinder)
"""

### AI GENERATED
import os
import sys
import dataclasses
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keep_away_env import KeepAwayConfig, KeepAwayEnv
from soccer_env import SoccerConfig, SoccerEnv
from teams import Team

# The training sizes: a 16 x 10 m field (x in [-8, 8]), 0.25 m players, 2 m/s blues, 12 physics
# steps per step(). Blue attacks the east goal (x = 8).
CONFIG = SoccerConfig(field_length=16.0, field_width=10.0, player_radius=0.25, blue_max_speed=2.0,
                      action_repeat=12)
FAR_RED = (-6.0, 4.0)  # out of the way of anything happening near the east goal
FAR_BLUE = (7.0, -4.5)  # out of the way of anything happening near the west goal


def play(env, action=(0.0, 0.0), kicks=None, max_steps=500):
    """Step with the same action (kicks on the first step only) until the game ends."""
    for _ in range(max_steps):
        result = env.step(action, kicks=kicks)
        kicks = None
        if result[2] or result[3]:
            return result
    raise AssertionError("game never ended")


class SoccerEnvTestCase(unittest.TestCase):
    config = CONFIG

    def setUp(self):
        self.env = SoccerEnv(self.config)

    def tearDown(self):
        self.env.close()


class TestReset(SoccerEnvTestCase):

    def test_builds_both_goals_with_blue_defending_west(self):
        self.env.reset(seed=0)
        self.assertEqual(len(self.env.field.goals), 2)
        self.assertEqual(self.env.blue_goal.spec.center, (-8.0, 0.0))
        self.assertEqual(self.env.red_goal.spec.center, (8.0, 0.0))

    def test_default_kickoff_layout(self):
        observation, info = self.env.reset(seed=0)
        self.assertAlmostEqual(self.env.blues[0].position[0], -2.0)
        self.assertAlmostEqual(self.env.reds[0].position[0], 2.0)
        self.assertTrue(np.allclose(self.env.ball.position[:2], [0.0, 0.0]))
        self.assertIsNone(self.env.ball.possession_player)
        self.assertEqual(info["winner"], Team.NONE)
        self.assertIsNone(info["goal"])
        self.assertEqual(len(observation["goals"]), 2)

    def test_red_ball_holder_starts_red_with_the_ball(self):
        self.env.reset(seed=0, options={"red_ball_holder": 0})
        self.assertIs(self.env.ball.possession_player, self.env.reds[0])

    def test_rejects_two_holders_and_unknown_options(self):
        with self.assertRaises(ValueError):
            self.env.reset(options={"ball_holder": 0, "red_ball_holder": 0})
        with self.assertRaises(ValueError):
            self.env.reset(options={"kickoff": True})

    def test_step_after_game_over_raises(self):
        self.env.reset(seed=0, options={"blue_position": (7.0, 0.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        play(self.env, action=(2.0, 0.0))
        with self.assertRaises(RuntimeError):
            self.env.step((0.0, 0.0))


class TestGoals(SoccerEnvTestCase):

    def test_blue_kick_into_reds_goal_wins(self):
        self.env.reset(seed=0, options={"blue_position": (4.0, 0.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        _, reward, terminated, truncated, info = play(self.env, kicks=[(6.0, 0.0)])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["winner"], Team.BLUE)
        self.assertIs(info["goal"].scorer, self.env.blues[0])
        self.assertFalse(info["goal"].carried)
        self.assertEqual(info["score"][Team.BLUE], 1)

    def test_blue_carrying_the_ball_in_wins(self):
        self.env.reset(seed=0, options={"blue_position": (6.0, 0.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        _, reward, terminated, _, info = play(self.env, action=(2.0, 0.0))
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["goal"].carried)

    def test_own_goal_loses(self):
        self.env.reset(seed=0, options={"blue_position": (-4.0, 0.0), "red_position": (6.0, 4.0),
                                        "ball_holder": 0})
        _, reward, terminated, _, info = play(self.env, kicks=[(-6.0, 0.0)])
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["winner"], Team.RED)
        self.assertTrue(info["goal"].own_goal)

    def test_the_ball_is_in_the_goal_when_the_game_ends(self):
        self.env.reset(seed=0, options={"blue_position": (7.0, 0.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        play(self.env, action=(2.0, 0.0))
        self.assertTrue(self.env.red_goal.spec.ball_in(self.env.ball.position, self.env.ball.radius))

    def test_shot_from_the_mouth_goes_in_instead_of_rebounding(self):
        # Blue stands right at the goal line, so play_area alone would clamp the kick back onto it
        self.env.reset(seed=0, options={"blue_position": (7.7, 0.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        _, reward, terminated, _, info = self.env.step((0.0, 0.0), kicks=[(2.0, 0.0)])
        self.assertTrue(info["kicks"][0])
        self.assertTrue(terminated)
        self.assertEqual(reward, 1.0)

    def test_shot_beside_the_mouth_still_rebounds(self):
        self.env.reset(seed=0, options={"blue_position": (7.7, 3.0), "red_position": FAR_RED,
                                        "ball_holder": 0})
        _, _, terminated, _, _ = self.env.step((0.0, 0.0), kicks=[(2.0, 0.0)])
        self.assertFalse(terminated)
        self.assertIs(self.env.ball.possession_player, self.env.blues[0])


class TestPossessionAndRed(SoccerEnvTestCase):

    def test_red_taking_the_ball_doesnt_end_the_game(self):
        self.env.reset(seed=0, options={"blue_position": (0.0, 0.0), "red_position": (2.0, 0.0),
                                        "ball_holder": 0})
        for _ in range(60):
            _, reward, terminated, truncated, _ = self.env.step((0.0, 0.0))
            self.assertFalse(terminated or truncated)
            self.assertEqual(reward, 0.0)
        self.assertIs(self.env.ball.possession_player, self.env.reds[0])

    def test_red_with_the_ball_keeps_still(self):
        self.env.reset(seed=0, options={"blue_position": (-6.0, 4.0), "red_position": (2.0, 0.0),
                                        "red_ball_holder": 0})
        start = self.env.reds[0].position.copy()
        for _ in range(30):
            self.env.step((0.0, 0.0))
        self.assertIs(self.env.ball.possession_player, self.env.reds[0])
        self.assertLess(np.linalg.norm(self.env.reds[0].position[:2] - start[:2]), 0.05)

    def test_blue_can_take_the_ball_back_from_red(self):
        self.env.reset(seed=0, options={"blue_position": (0.0, 0.0), "red_position": (2.0, 0.0),
                                        "red_ball_holder": 0})
        for _ in range(60):
            self.env.step((2.0, 0.0))
            if self.env.ball.possession_player is self.env.blues[0]:
                break
        self.assertIs(self.env.ball.possession_player, self.env.blues[0])

    def test_red_chases_a_loose_ball(self):
        self.env.reset(seed=0, options={"blue_position": (-6.0, 4.0), "red_position": (4.0, 0.0),
                                        "ball_position": (0.0, 0.0)})
        self.env.step((0.0, 0.0))
        self.assertLess(self.env.reds[0].velocity[0], -1.0)

    def test_possession_changes_are_recorded(self):
        self.env.reset(seed=0, options={"blue_position": (0.0, 0.0), "red_position": (2.0, 0.0),
                                        "ball_holder": 0})
        events = []
        for _ in range(60):
            self.env.step((0.0, 0.0))
            events += self.env.possession_events
        self.assertTrue(any(new is self.env.reds[0] for _, _, new in events))


class TestTimeAndTeams(SoccerEnvTestCase):
    config = SoccerConfig(field_length=16.0, field_width=10.0, player_radius=0.25,
                          blue_max_speed=2.0, action_repeat=12, max_game_time=1.0)

    def test_time_out_is_a_draw(self):
        self.env.reset(seed=0, options={"blue_position": (-6.0, 4.0), "red_position": (6.0, 4.0),
                                        "ball_position": (0.0, -4.0)})
        _, reward, terminated, truncated, info = play(self.env)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["winner"], Team.NONE)

    def test_any_team_sizes(self):
        for n_blue, n_red in ((2, 2), (3, 1), (1, 0)):
            env = SoccerEnv(SoccerConfig(field_length=16.0, field_width=10.0, n_blue=n_blue, n_red=n_red))
            try:
                observation, _ = env.reset(seed=1)
                self.assertEqual(len(observation["blues"]), n_blue)
                self.assertEqual(len(observation["reds"]), n_red)
                env.step(np.zeros((n_blue, 2)))
            finally:
                env.close()

    def test_two_games_dont_share_state(self):
        other = SoccerEnv(self.config)
        try:
            self.env.reset(seed=0, options={"blue_position": (7.0, 0.0), "red_position": FAR_RED,
                                            "ball_holder": 0})
            other.reset(seed=0, options={"blue_position": (-6.0, 4.0), "red_position": (6.0, 4.0),
                                         "ball_position": (0.0, -4.0)})
            play(self.env, action=(2.0, 0.0))
            self.assertEqual(self.env.winner, Team.BLUE)
            self.assertEqual(other.winner, Team.NONE)
            self.assertFalse(other.red_goal.ball_inside)
            other.step((0.0, 0.0))
        finally:
            other.close()


class TestDribblingRed(SoccerEnvTestCase):
    """red_policy = "dribble" (2026-11-23, docs/red_dribble_plan.md)."""
    config = dataclasses.replace(CONFIG, red_policy="dribble")

    def assert_red_carried_it_in(self, info):
        self.assertEqual(info["winner"], Team.RED)
        self.assertTrue(info["goal"].carried)
        self.assertFalse(info["goal"].own_goal)
        self.assertTrue(self.env.blue_goal.ball_inside)
        self.assertFalse(self.env.red_goal.ball_inside)

    def test_red_carries_the_ball_into_blues_goal(self):
        self.env.reset(seed=0, options={"blue_position": FAR_BLUE, "red_position": (2.0, 0.0),
                                        "red_ball_holder": 0})
        _, reward, terminated, _, info = play(self.env, max_steps=200)  # 10 m at 1.8 m/s: ~110 steps
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assert_red_carried_it_in(info)

    def test_red_lines_up_from_near_a_sideline(self):
        self.env.reset(seed=0, options={"blue_position": FAR_BLUE, "red_position": (-5.0, 4.0),
                                        "red_ball_holder": 0})
        _, _, _, _, info = play(self.env, max_steps=200)
        self.assert_red_carried_it_in(info)

    def test_red_chases_a_loose_ball(self):
        self.env.reset(seed=0, options={"blue_position": FAR_BLUE, "red_position": (4.0, 0.0),
                                        "ball_position": (0.0, 0.0)})
        self.env.step((0.0, 0.0))
        self.assertLess(self.env.reds[0].velocity[0], -1.0)

    def test_the_other_reds_chase_their_own_carrier(self):
        env = SoccerEnv(dataclasses.replace(self.config, n_red=2))
        try:
            env.reset(seed=0, options={"blue_position": FAR_BLUE,
                                       "red_positions": [(0.0, 0.0), (3.0, 3.0)], "red_ball_holder": 0})
            env.step((0.0, 0.0))
            to_carrier = env.reds[0].position[:2] - env.reds[1].position[:2]
            self.assertGreater(float(np.dot(env.reds[1].velocity[:2], to_carrier)), 0.0)
            for _ in range(300):
                _, _, terminated, _, info = env.step((0.0, 0.0))
                if terminated:
                    break
                self.assertIs(env.ball.possession_player, env.reds[0])  # teammates can't take it
            self.assertEqual(info["winner"], Team.RED)
        finally:
            env.close()

    def test_chase_is_the_default(self):
        self.assertEqual(SoccerConfig().red_policy, "chase")  # TestPossessionAndRed covers chase

    def test_unknown_red_policy_raises(self):
        with self.assertRaises(ValueError):
            SoccerEnv(dataclasses.replace(CONFIG, red_policy="shoot"))


class TestKeepAwayKicksUnchanged(unittest.TestCase):

    def test_keep_away_still_clamps_kicks_to_the_field(self):
        env = KeepAwayEnv(KeepAwayConfig(field_length=16.0, field_width=10.0, player_radius=0.25))
        try:
            env.reset(seed=0, options={"blue_position": (7.7, 0.0), "red_position": FAR_RED,
                                       "blue_has_ball": True})
            self.assertEqual(env.ball.goal_lanes, [])
            env.step((0.0, 0.0), kicks=[(2.0, 0.0)])
            self.assertIs(env.ball.possession_player, env.blue)  # the kick rebounded
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
### END AI GENERATED
