import os ### AI GENERATED
import sys ### AI GENERATED
import unittest ### AI GENERATED

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ### AI GENERATED
from entity import Entity

class TestAdd(unittest.TestCase):
    ### AI GENERATED
    def setUp(self):
        p.connect(p.DIRECT)
        Entity._registry.clear()

    def tearDown(self):
        p.disconnect()
    ### END AI GENERATED

    def test_constructor(self):
        entity = Entity()
        self.assertTrue(np.array_equal(entity.position, np.array([0,0,0])))
        self.assertTrue(np.array_equal(entity.velocity, np.array([0,0,0])))
        self.assertTrue(np.array_equal(entity.collided_entities, []))

    ### AI GENERATED
    def test_sync_from_pybullet(self):
        entity = Entity()
        p.resetBasePositionAndOrientation(entity.body_id, [1, 2, 3], [0, 0, 0, 1])
        p.resetBaseVelocity(entity.body_id, linearVelocity=[4, 5, 6])

        entity.sync_from_pybullet()

        self.assertTrue(np.allclose(entity.position, np.array([1, 2, 3])))
        self.assertTrue(np.allclose(entity.velocity, np.array([4, 5, 6])))

    def test_update_collisions_detects_overlapping_entity(self):
        entity_a = Entity()
        entity_b = Entity()  # spawns at the same default position, so it overlaps entity_a

        p.performCollisionDetection()
        entity_a.update_collisions()
        entity_b.update_collisions()

        self.assertIn(entity_b, entity_a.collided_entities)
        self.assertIn(entity_a, entity_b.collided_entities)

    def test_update_collisions_ignores_distant_entity(self):
        entity_a = Entity()
        entity_b = Entity()
        p.resetBasePositionAndOrientation(entity_b.body_id, [100, 100, 100], [0, 0, 0, 1])

        p.performCollisionDetection()
        entity_a.update_collisions()

        self.assertEqual(entity_a.collided_entities, [])

    def test_apply_force(self):
        entity = Entity()

        entity.apply_force(x=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertGreater(entity.velocity[0], 0.0)

        ### Austin generated :D
        # Testing that we are restricted to a 2D plane
        entity.apply_force(z=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertEqual(entity.velocity[2], 0.0)
        ### End Human Generated
    ### END AI GENERATED

    def test_apply_velocity(self):
        entity = Entity()

        entity.apply_velocity(x=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.velocity[0], 3.0, places=1)
        self.assertAlmostEqual(entity.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(entity.velocity[2], 0.0, places=1)

        entity.apply_velocity(y=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(entity.velocity[1], 3.0, places=1)
        self.assertAlmostEqual(entity.velocity[2], 0.0, places=1)

        # Restrict in Z axis
        entity.apply_velocity(z=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.velocity[0], 0.0, places=1)
        self.assertAlmostEqual(entity.velocity[1], 0.0, places=1)
        self.assertAlmostEqual(entity.velocity[2], 0.0, places=1)

    def test_apply_position(self):
        entity = Entity()

        entity.apply_position(x=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.position[0], 10.0)
        self.assertAlmostEqual(entity.position[1], 0.0)
        self.assertAlmostEqual(entity.position[2], 0.0)

        entity.apply_position(y=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.position[0], 0.0)
        self.assertAlmostEqual(entity.position[1], 10.0)
        self.assertAlmostEqual(entity.position[2], 0.0)

        entity.apply_position(z=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.position[0], 0.0)
        self.assertAlmostEqual(entity.position[1], 0.0)
        self.assertAlmostEqual(entity.position[2], 0.0) # testing this does nothing

        entity.apply_position(x=-1.0, y=17.5, z=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()

        self.assertAlmostEqual(entity.position[0], -1.0)
        self.assertAlmostEqual(entity.position[1], 17.5)
        self.assertAlmostEqual(entity.position[2], 0.0) # testing this does nothing

    # I don't test the observables yet
    # Test the max velocity stuff
    def test_get_observables(self):
        entity = Entity()
        entity2 = Entity()

        # Intentionally larger than max velocity of 3
        entity.apply_velocity(x=10.0, y=10.0)
        p.stepSimulation()
        entity.sync_from_pybullet()
        p.stepSimulation()
        entity.sync_from_pybullet()
        p.stepSimulation()
        entity.sync_from_pybullet()

        observables = entity.get_observables()
        print(observables)

        self.assertEqual(observables['id'], 0)
        self.assertGreater(observables['position']['x'], 0)
        self.assertGreater(observables['position']['y'], 0)
        self.assertEqual(observables['position']['z'], 0)
        self.assertAlmostEqual(observables['velocity']['x'], 3.0, places=1)
        self.assertAlmostEqual(observables['velocity']['y'], 3.0, places=1)
        self.assertEqual(observables['velocity']['z'], 0)
        self.assertEqual(observables['collided_entities'][0], 1)



### AI GENERATED
if __name__ == "__main__":
    unittest.main()
### END AI GENERATED