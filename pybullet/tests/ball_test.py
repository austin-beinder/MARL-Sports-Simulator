"""
Many cited parts entity_test.py were AI generated. This test file was largely copy pasted from entity_test.py.
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


    def test_player_takes_possession(self):
        ball = Ball()
        player = Player()

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
        self.assertEqual(observables['possession_team'], Team.RED)
        self.assertEqual(observables['possession_player'], player.body_id)


if __name__ == "__main__":
    unittest.main()