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

from teams import Team

class TestAdd(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        Player._registry.clear()

    def tearDown(self):
        p.disconnect()

    def test_constructor(self):
        player = Player()
        self.assertTrue(np.array_equal(player.position, np.array([0,0,0])))
        self.assertTrue(np.array_equal(player.velocity, np.array([0,0,0])))
        self.assertTrue(np.array_equal(player.collided_entities, []))
        self.assertEqual(player.team, Team.RED)

    def test_sync_from_pybullet(self):
        player = Player()
        p.resetBasePositionAndOrientation(player.body_id, [1, 2, 3], [0, 0, 0, 1])
        p.resetBaseVelocity(player.body_id, linearVelocity=[4, 5, 6])

        player.sync_from_pybullet()

        self.assertTrue(np.allclose(player.position, np.array([1, 2, 3])))
        self.assertTrue(np.allclose(player.velocity, np.array([4, 5, 6])))

    def test_update_collisions_detects_overlapping_player(self):
        player_a = Player()
        player_b = Player()  # spawns at the same default position, so it overlaps player_a

        p.performCollisionDetection()
        player_a.update_collisions()
        player_b.update_collisions()

        self.assertIn(player_b, player_a.collided_entities)
        self.assertIn(player_a, player_b.collided_entities)

    def test_update_collisions_ignores_distant_player(self):
        player_a = Player()
        player_b = Player()
        p.resetBasePositionAndOrientation(player_b.body_id, [100, 100, 100], [0, 0, 0, 1])

        p.performCollisionDetection()
        player_a.update_collisions()

        self.assertEqual(player_a.collided_entities, [])

    def test_apply_force(self):
        player = Player()

        player.apply_force(x=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertGreater(player.velocity[0], 0.0)

        # Testing that we are restricted to a 2D plane
        player.apply_force(z=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertEqual(player.velocity[2], 0.0)

    def test_apply_velocity(self):
        player = Player()

        player.apply_velocity(x=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.velocity[0], 3.0, places=1)
        self.assertAlmostEqual(player.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(player.velocity[2], 0.0, places=1)

        player.apply_velocity(y=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(player.velocity[1], 3.0, places=1)
        self.assertAlmostEqual(player.velocity[2], 0.0, places=1)

        # Restrict in Z axis
        player.apply_velocity(z=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(player.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(player.velocity[2], 0.0, places=1)

    def test_apply_position(self):
        player = Player()

        player.apply_position(x=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.position[0], 10.0)
        self.assertAlmostEqual(player.position[1], 0.0)
        self.assertAlmostEqual(player.position[2], 0.0)

        player.apply_position(y=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.position[0], 0.0)
        self.assertAlmostEqual(player.position[1], 10.0)
        self.assertAlmostEqual(player.position[2], 0.0)

        player.apply_position(z=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.position[0], 0.0)
        self.assertAlmostEqual(player.position[1], 0.0)
        self.assertAlmostEqual(player.position[2], 0.0) # testing this does nothing

        player.apply_position(x=-1.0, y=17.5, z=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()

        self.assertAlmostEqual(player.position[0], -1.0)
        self.assertAlmostEqual(player.position[1], 17.5)
        self.assertAlmostEqual(player.position[2], 0.0) # testing this does nothing

    # I don't test the observables yet
    # Test the max velocity stuff
    def test_get_observables(self):
        player = Player()
        player2 = Player()

        # Intentionally larger than max velocity of 3
        player.apply_velocity(x=10.0, y=10.0)
        p.stepSimulation()
        player.sync_from_pybullet()
        p.stepSimulation()
        player.sync_from_pybullet()
        p.stepSimulation()
        player.sync_from_pybullet()

        observables = player.get_observables()

        self.assertEqual(observables['id'], 0)
        self.assertGreater(observables['position']['x'], 0)
        self.assertGreater(observables['position']['y'], 0)
        self.assertEqual(observables['position']['z'], 0)
        self.assertAlmostEqual(observables['velocity']['x'], 2.84, places=2)
        self.assertAlmostEqual(observables['velocity']['y'], 2.84, places=2)
        self.assertEqual(observables['velocity']['z'], 0)
        self.assertEqual(observables['collided_entities'][0], 1)
        self.assertEqual(observables['team'], Team.RED)


if __name__ == "__main__":
    unittest.main()