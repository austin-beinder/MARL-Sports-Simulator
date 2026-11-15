"""
Many cited parts entity_test.py were AI generated. This test file was largely copy pasted from entity_test.py.
AI edit 2026-11-15 (Claude Code, Austin Beinder): pylint cleanup (wrapped lines over 100
characters, a protected-access disable on the test's _release call); no behaviour change.
"""

import os
import sys
import unittest

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player import Player
from ball import Ball

from teams import Team

class TestAdd(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Ball._registry.clear()

    def tearDown(self):
        p.disconnect()

    def test_constructor(self):
        ball = Ball()
        self.assertTrue(np.array_equal(ball.position, np.array([0,0,0])))
        self.assertTrue(np.array_equal(ball.velocity, np.array([0,0,0])))
        self.assertTrue(np.array_equal(ball.collided_entities, []))
        self.assertEqual(ball.team, Team.NONE)

    def test_sync_from_pybullet(self):
        ball = Ball()
        p.resetBasePositionAndOrientation(ball.body_id, [1, 2, 3], [0, 0, 0, 1])
        p.resetBaseVelocity(ball.body_id, linearVelocity=[4, 5, 6])

        ball.sync_from_pybullet()

        self.assertTrue(np.allclose(ball.position, np.array([1, 2, 3])))
        self.assertTrue(np.allclose(ball.velocity, np.array([4, 5, 6])))

    def test_update_collisions_detects_overlapping_ball(self):
        ball_a = Ball()
        ball_b = Ball()  # spawns at the same default position, so it overlaps ball_a

        p.performCollisionDetection()
        ball_a.update_collisions()
        ball_b.update_collisions()

        self.assertIn(ball_b, ball_a.collided_entities)
        self.assertIn(ball_a, ball_b.collided_entities)

    def test_update_collisions_ignores_distant_ball(self):
        ball_a = Ball()
        ball_b = Ball()
        p.resetBasePositionAndOrientation(ball_b.body_id, [100, 100, 100], [0, 0, 0, 1])

        p.performCollisionDetection()
        ball_a.update_collisions()

        self.assertEqual(ball_a.collided_entities, [])

    def test_apply_force(self):
        ball = Ball()

        ball.apply_force(x=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertGreater(ball.velocity[0], 0.0)

        # Testing that we are restricted to a 2D plane
        ball.apply_force(z=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertEqual(ball.velocity[2], 0.0)

    def test_apply_velocity(self):
        ball = Ball()

        ball.apply_velocity(x=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.velocity[0], 3.0, places=1)
        self.assertAlmostEqual(ball.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(ball.velocity[2], 0.0, places=1)

        ball.apply_velocity(y=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(ball.velocity[1], 3.0, places=1)
        self.assertAlmostEqual(ball.velocity[2], 0.0, places=1)

        # Restrict in Z axis
        ball.apply_velocity(z=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(ball.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(ball.velocity[2], 0.0, places=1)

    def test_apply_position(self):
        ball = Ball()

        ball.apply_position(x=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.position[0], 10.0)
        self.assertAlmostEqual(ball.position[1], 0.0)
        self.assertAlmostEqual(ball.position[2], 0.0)

        ball.apply_position(y=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.position[0], 0.0)
        self.assertAlmostEqual(ball.position[1], 10.0)
        self.assertAlmostEqual(ball.position[2], 0.0)

        ball.apply_position(z=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.position[0], 0.0)
        self.assertAlmostEqual(ball.position[1], 0.0)
        self.assertAlmostEqual(ball.position[2], 0.0) # testing this does nothing

        ball.apply_position(x=-1.0, y=17.5, z=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.position[0], -1.0)
        self.assertAlmostEqual(ball.position[1], 17.5)
        self.assertAlmostEqual(ball.position[2], 0.0) # testing this does nothing

    # I don't test the observables yet
    # Test the max velocity stuff
    def test_get_observables(self):
        ball = Ball()

        # Intentionally larger than max velocity of 3
        ball.apply_velocity(x=10.0, y=10.0)
        p.stepSimulation()
        ball.sync_from_pybullet()
        p.stepSimulation()
        ball.sync_from_pybullet()
        p.stepSimulation()
        ball.sync_from_pybullet()

        observables = ball.get_observables()

        self.assertEqual(observables['id'], 0)
        self.assertGreater(observables['position']['x'], 0)
        self.assertGreater(observables['position']['y'], 0)
        self.assertEqual(observables['position']['z'], 0)
        self.assertAlmostEqual(observables['velocity']['x'], 2.84, places=2)
        self.assertAlmostEqual(observables['velocity']['y'], 2.84, places=2)
        self.assertEqual(observables['velocity']['z'], 0)
        self.assertEqual(len(observables['collided_entities']), 0)
        self.assertEqual(observables['possession_team'], Team.NONE)
        self.assertEqual(observables['possession_player'], None)

    ### AI GENERATED
    def test_player_takes_possession(self):
        # End to end: a loose ball overlapping a player gets picked up through the
        # real physics collision path (stepSimulation -> sync_from_pybullet -> on_new_collision).
        ball = Ball()
        player = Player(team=Team.RED)  # spawns at the same default position, so it overlaps the ball

        p.stepSimulation()
        ball.sync_from_pybullet()
        player.sync_from_pybullet()

        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(ball.possession_player, player)
        self.assertIs(player.ball, ball)
        self.assertIsNotNone(ball.carry_constraint)

        observables = ball.get_observables()
        self.assertEqual(observables['possession_team'], Team.RED)
        self.assertEqual(observables['possession_player'], player.body_id)

    # The tests below call on_new_collision directly instead of stepping physics,
    # so each possession rule can be checked deterministically in isolation.

    def _ball_held_by(self, player):
        """Helper: create a ball and give possession to player as a loose-ball pickup."""
        ball = Ball(initial_position=np.array([5.0, 5.0, 0.0]))  # away from the players
        ball.possession_flip_percentage = 100  # steals always succeed unless a test says otherwise
        ball.on_new_collision(player)
        # AI GENERATED (2026-11-07): move on a physics step, since the ball changes hands at most
        # once per step
        ball.sim_time += p.getPhysicsEngineParameters()["fixedTimeStep"]
        return ball

    def test_player_takes_possession_from_no_one(self):
        red = Player(team=Team.RED)
        ball = Ball(initial_position=np.array([5.0, 5.0, 0.0]))

        changed = ball.on_new_collision(red)

        self.assertTrue(changed)
        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(ball.possession_player, red)
        self.assertIs(red.ball, ball)
        self.assertIsNotNone(ball.carry_constraint)

    def test_player_takes_possession_after_kick(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.kick_ball(1.0, 0.0)
        self.assertEqual(ball.possession, Team.NONE)
        self.assertIsNone(ball.possession_player)
        self.assertIsNone(ball.carry_constraint)
        self.assertIsNone(red.ball)

        # A loose ball can be picked up by anyone, and is not limited by the steal cooldown
        changed = ball.on_new_collision(blue)

        self.assertTrue(changed)
        self.assertEqual(ball.possession, Team.BLUE)
        self.assertIs(ball.possession_player, blue)
        self.assertIs(blue.ball, ball)

    def test_player_takes_possession_from_opponent(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)
        old_constraint = ball.carry_constraint

        changed = ball.on_new_collision(blue)

        self.assertTrue(changed)
        self.assertEqual(ball.possession, Team.BLUE)
        self.assertIs(ball.possession_player, blue)
        self.assertIs(blue.ball, ball)
        self.assertIsNone(red.ball)
        self.assertIsNotNone(ball.carry_constraint)
        self.assertNotEqual(ball.carry_constraint, old_constraint)
        self.assertEqual(ball.last_steal_time, ball.sim_time)

    def test_steal_fails_when_flip_roll_fails(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)
        ball.possession_flip_percentage = 0  # steals never succeed

        changed = ball.on_new_collision(blue)

        self.assertFalse(changed)
        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(ball.possession_player, red)
        self.assertIs(red.ball, ball)
        self.assertIsNone(blue.ball)
        self.assertEqual(ball.last_steal_time, -float("inf"))

    def test_collision_with_teammate_keeps_team_possession(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        red2 = Player(team=Team.RED, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)
        old_constraint = ball.carry_constraint

        changed = ball.on_new_collision(red2)

        # Treated as a pass: the team keeps the ball, but the player in possession changes
        self.assertFalse(changed)
        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(ball.possession_player, red2)
        self.assertIs(red2.ball, ball)
        self.assertIsNone(red.ball)
        # The ball is physically re-pinned to the teammate, not left on the old player
        self.assertIsNotNone(ball.carry_constraint)
        self.assertNotEqual(ball.carry_constraint, old_constraint)
        self.assertEqual(p.getConstraintInfo(ball.carry_constraint)[0], red2.body_id)
        # A pass is not a steal, so it doesn't start the steal cooldown
        self.assertEqual(ball.last_steal_time, -float("inf"))

    def test_steal_back_blocked_during_cooldown(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.sim_time = 1.0
        self.assertTrue(ball.on_new_collision(blue))

        # Red tries to take it straight back, still inside the cooldown window
        ball.sim_time = 1.0 + ball.STEAL_COOLDOWN / 2
        changed = ball.on_new_collision(red)

        self.assertFalse(changed)
        self.assertEqual(ball.possession, Team.BLUE)
        self.assertIs(ball.possession_player, blue)
        self.assertIs(blue.ball, ball)
        self.assertEqual(ball.last_steal_time, 1.0)

    def test_steal_back_allowed_after_cooldown(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.sim_time = 1.0
        self.assertTrue(ball.on_new_collision(blue))

        ball.sim_time = 1.0 + ball.STEAL_COOLDOWN * 2
        changed = ball.on_new_collision(red)

        self.assertTrue(changed)
        self.assertEqual(ball.possession, Team.RED)
        self.assertIs(ball.possession_player, red)
        self.assertIs(red.ball, ball)
        self.assertIsNone(blue.ball)
        self.assertEqual(ball.last_steal_time, ball.sim_time)

    def test_cooldown_does_not_block_teammate(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        blue2 = Player(team=Team.BLUE, initial_position=np.array([-2.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.sim_time = 1.0
        self.assertTrue(ball.on_new_collision(blue))

        # Still inside the cooldown, but the cooldown only applies to steals
        ball.on_new_collision(blue2)

        self.assertEqual(ball.possession, Team.BLUE)
        self.assertIs(ball.possession_player, blue2)

    def test_sync_from_pybullet_advances_sim_time(self):
        ball = Ball()
        time_step = p.getPhysicsEngineParameters()["fixedTimeStep"]

        for _ in range(10):
            p.stepSimulation()
            ball.sync_from_pybullet()

        self.assertAlmostEqual(ball.sim_time, 10 * time_step)

    ### AI GENERATED (2026-10-31)
    def test_teammate_handoff_can_be_switched_off(self):
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        blue2 = Player(team=Team.BLUE, initial_position=np.array([-2.0, 0.0, 0.0]))
        ball = self._ball_held_by(blue)
        ball.allow_teammate_handoff = False

        changed = ball.on_new_collision(blue2)

        self.assertFalse(changed)
        self.assertIs(ball.possession_player, blue)
        self.assertIs(blue.ball, ball)
        self.assertIsNone(blue2.ball)
    ### END AI GENERATED

    ### AI GENERATED (2026-11-07)
    # check_held_contacts: contacts that were already there when the ball changed hands. These set
    # collided_entities by hand instead of stepping physics, like the on_new_collision tests above.

    def test_held_contact_takes_the_ball_from_a_new_carrier(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        blue.collided_entities = [red]  # red was touching blue before blue picked up the ball
        ball = self._ball_held_by(blue)

        self.assertTrue(ball.check_held_contacts())

        self.assertIs(ball.possession_player, red)
        self.assertIs(red.ball, ball)
        self.assertIsNone(blue.ball)
        self.assertEqual(ball.contested, {blue})

    def test_held_contact_with_the_ball_counts_too(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(blue)
        ball.collided_entities = [red]

        self.assertTrue(ball.check_held_contacts())
        self.assertIs(ball.possession_player, red)

    def test_old_carrier_must_let_go_before_stealing_back(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(blue)
        ball.sim_time = 1.0
        self.assertTrue(ball.on_new_collision(red))  # red's first touch takes it
        red.collided_entities = [blue]

        ball.sim_time = 2.0  # long past the cooldown, but blue never let go
        self.assertFalse(ball.check_held_contacts())
        self.assertIs(ball.possession_player, red)

        red.collided_entities = []  # they separate...
        self.assertFalse(ball.check_held_contacts())
        self.assertEqual(ball.contested, set())
        red.collided_entities = [blue]  # ...and touch again
        self.assertTrue(ball.check_held_contacts())
        self.assertIs(ball.possession_player, blue)

    def test_contact_during_the_cooldown_gets_no_roll(self):
        # As before check_held_contacts: a contact that starts inside the cooldown is ignored, so a
        # scrum can't bounce the ball back and forth every STEAL_COOLDOWN
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        blue2 = Player(team=Team.BLUE, initial_position=np.array([0.0, -2.0, 0.0]))
        ball = self._ball_held_by(blue)
        ball.sim_time = 1.0
        self.assertTrue(ball.on_new_collision(red))
        red.collided_entities = [blue, blue2]  # a second blue touches red inside the cooldown

        ball.sim_time = 1.0 + ball.STEAL_COOLDOWN / 2
        self.assertFalse(ball.check_held_contacts())
        self.assertEqual(ball.contested, {blue, blue2})
        ball.sim_time = 1.0 + ball.STEAL_COOLDOWN * 2
        self.assertFalse(ball.check_held_contacts())  # still the same contact
        red.collided_entities = [blue]  # blue2 lets go...
        ball.check_held_contacts()
        red.collided_entities = [blue, blue2]  # ...and touches again, after the cooldown
        self.assertTrue(ball.check_held_contacts())
        self.assertIs(ball.possession_player, blue2)

    def test_one_roll_per_contact(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = self._ball_held_by(blue)
        ball.possession_flip_percentage = 0  # the roll always fails
        blue.collided_entities = [red]

        self.assertFalse(ball.check_held_contacts())
        self.assertEqual(ball.contested, {red})
        ball.possession_flip_percentage = 100
        self.assertFalse(ball.check_held_contacts())  # same contact: no second roll
        self.assertIs(ball.possession_player, blue)

    def test_ball_changes_hands_at_most_once_per_step(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        ball = Ball(initial_position=np.array([5.0, 5.0, 0.0]))
        ball.sim_time = 1.0
        ball.collided_entities = [red, blue]  # both touch the loose ball on the same step

        self.assertTrue(ball.on_new_collision(red))  # red is first in the contact list
        self.assertFalse(ball.on_new_collision(blue))  # blue can't take it on that same step...
        self.assertFalse(ball.check_held_contacts())
        self.assertEqual(ball.contested, set())  # ...and hasn't used up its roll

        ball.sim_time += p.getPhysicsEngineParameters()["fixedTimeStep"]
        self.assertTrue(ball.check_held_contacts())  # it gets it on the next step
        self.assertIs(ball.possession_player, blue)

    def test_teammates_walls_and_the_loose_ball_are_ignored(self):
        blue = Player(team=Team.BLUE, initial_position=np.array([2.0, 0.0, 0.0]))
        blue2 = Player(team=Team.BLUE, initial_position=np.array([-2.0, 0.0, 0.0]))
        ball = self._ball_held_by(blue)
        blue.collided_entities = [blue2, ball]

        self.assertFalse(ball.check_held_contacts())
        self.assertIs(ball.possession_player, blue)
        ball._release()  # pylint: disable=protected-access
        self.assertFalse(ball.check_held_contacts())  # nobody holds it: nothing to steal
    ### END AI GENERATED

    def test_kick_moves_ball_clear_of_kicker(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.kick_ball(2.0, 0.0)

        # Ball sits just outside the kicker's edge, in the kick direction, on the ground
        expected_x = red.radius + ball.radius + ball.KICK_GAP
        self.assertAlmostEqual(ball.position[0], expected_x)
        self.assertAlmostEqual(ball.position[1], 0.0)
        self.assertAlmostEqual(ball.position[2], 0.0)
        self.assertAlmostEqual(ball.velocity[0], 2.0)
        self.assertAlmostEqual(ball.velocity[1], 0.0)

    def test_kick_offset_follows_kick_direction(self):
        red = Player(team=Team.RED, initial_position=np.array([1.0, -1.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.kick_ball(-1.0, 1.0)  # diagonally up and to the left

        distance = red.radius + ball.radius + ball.KICK_GAP
        self.assertAlmostEqual(ball.position[0], 1.0 - distance / np.sqrt(2))
        self.assertAlmostEqual(ball.position[1], -1.0 + distance / np.sqrt(2))

    def test_kicker_does_not_regain_ball_after_kick(self):
        # End to end through physics: before the fix, the released ball landed on the
        # kicker and was picked straight back up on the next step.
        red = Player(team=Team.RED)
        ball = Ball()  # overlaps red, so red picks it up on the first step

        p.stepSimulation()
        ball.sync_from_pybullet()
        red.sync_from_pybullet()
        self.assertIs(ball.possession_player, red)

        ball.kick_ball(2.0, 0.0)
        for _ in range(10):
            p.stepSimulation()
            ball.sync_from_pybullet()
            red.sync_from_pybullet()

        self.assertEqual(ball.possession, Team.NONE)
        self.assertIsNone(ball.possession_player)
        self.assertIsNone(red.ball)
        self.assertNotIn(red, ball.collided_entities)
        self.assertGreater(ball.position[0], red.position[0] + red.radius + ball.radius)

    def test_kick_loose_ball_does_not_teleport(self):
        ball = Ball(initial_position=np.array([5.0, 5.0, 0.0]))
        ball.sync_from_pybullet()
        start = ball.position.copy()

        ball.kick_ball(2.0, 0.0)

        # Nobody was holding it, so there's no one to move it away from
        self.assertTrue(np.allclose(ball.position, start))
        self.assertAlmostEqual(ball.velocity[0], 2.0)

    def test_zero_kick_drops_ball_in_place(self):
        red = Player(team=Team.RED, initial_position=np.array([0.0, 0.0, 0.0]))
        ball = self._ball_held_by(red)

        ball.kick_ball(0.0, 0.0)
        ball.sync_from_pybullet()

        # No direction to kick in, so the ball is released where it was held: above the player
        self.assertEqual(ball.possession, Team.NONE)
        self.assertIsNone(red.ball)
        self.assertAlmostEqual(ball.position[0], 0.0)
        self.assertAlmostEqual(ball.position[1], 0.0)
    ### END AI GENERATED


if __name__ == "__main__":
    unittest.main()