"""
Create skeleton code for player
i. Implementable
1) Position
2) Velocity
3) Acceleration
4) Team
5) Collision with other player handling
6) Sending out observable information
7) Brain
8) Taking in observable information
This should all be unit testable

Over the course of this issue, a player class shall be created. It will be observable and usable within 
pybullet. There shall be unit test for each of the sub functions for the player. The player will be capable 
of outputing its useful observable information into a dictionary or json format, and intaking other 
information in a similar dictionary or json format. There shall be some basic acceptance of a "policy" 
or "brain" or "neural network" but implementing that is outside the scope of this story.

It is outside the scope of this issue to implement the actual messaging layer, inside the scope is how 
to handle or send messages period.

There shall be a seperate controlling interface such as a gui or game controller hooked up to control the 
player when selected with a flag.

There shall be a function that can change observable reference frame from absolute position of other 
entities to relative position of other entities.

This player shall inherit from a generic entity class. This class shall be implemented as a part of this issue.
Generic entities shall have:

- Position
- Velocity
- Acceleration
- Collision handling
- Sending observables
- Ingesting observables

Acceptance Criteria:
- Passes through peer review of 2 other group members
- Short demo
"""

import numpy as np
import pybullet as p
from typing import Dict, List
import time
from teams import Team

class Entity:

    ### AI GENERATED
    _registry: Dict[int, "Entity"] = {}
    ### END AI GENERATED

    def __init__(self, 
                 radius=0.3, 
                 height=0.1, 
                 mass=1.0,
                 initial_position=np.array([0.0, 0.0, 0.0]),
                 rgbaColor=[0, 0, 0, 1],
                 maxJointVelocity=3.0):

        self.position = initial_position
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.is_ball = False
        self.team = Team.NONE

        # Orientation?

        self.collided_entities: List[Entity] = []

        self.MAX_FORCE = 10.0

        ### AI GENERATED
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgbaColor)

        # Useful https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
        ### AI Assisted
        self.body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            basePosition=self.position.tolist(),
            linkMasses=[0, 0, mass],
            linkCollisionShapeIndices=[-1, -1, collision_shape],
            linkVisualShapeIndices=[-1, -1, visual_shape],
            linkPositions=[[0, 0, 0]] * 3,
            linkOrientations=[[0, 0, 0, 1]] * 3,
            linkInertialFramePositions=[[0, 0, 0]] * 3,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 3,
            linkParentIndices=[0, 1, 2],     # chain: base -> x -> y -> yaw
            linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC, p.JOINT_REVOLUTE],
            linkJointAxis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        # Claude told me how to do this
        p.changeDynamics(self.body_id, 0, maxJointVelocity=maxJointVelocity)
        p.changeDynamics(self.body_id, 1, maxJointVelocity=maxJointVelocity)
        ### AI Assisted

        # Joints have velocity motors switched on by default, and they act like brakes. Turn them off:
        p.setJointMotorControlArray(self.body_id, [0, 1, 2], p.VELOCITY_CONTROL, forces=[0, 0, 0])


        Entity._registry[self.body_id] = self
        ### END AI GENERATED

    ### AI GENERATED
    def sync_from_pybullet(self):
        pos = p.getLinkState(self.body_id, 2, computeLinkVelocity=1)[0]
        vel = p.getLinkState(self.body_id, 2, computeLinkVelocity=1)[6]
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.update_collisions()
    ### END AI GENERATED

    ### AI GENERATED
    def update_collisions(self):
        """Refresh collided_entities with whatever this entity is touching as of the last stepSimulation."""
        previous_collided_entities = self.collided_entities
        self.collided_entities = []
        seen_body_ids = set()
        for contact in p.getContactPoints(bodyA=self.body_id):
            other_body_id = contact[2]
            if other_body_id in seen_body_ids:
                continue
            seen_body_ids.add(other_body_id)
            other_entity = Entity._registry.get(other_body_id)
            if other_entity is not None:
                self.collided_entities.append(other_entity)
                if not (other_entity in previous_collided_entities):
                    self.on_new_collision(other_entity)

    ### END AI GENERATED

    def on_new_collision(self, collided_entity):
        """
        This function will get called when a new collision occurs.
        """
        print(f"Entity: {self.body_id} collided with Entity: {collided_entity.body_id}")

    def get_observables(self) -> dict:
        """
        Returns: A dict with entity observerables 
        """

        observables: dict = {
            'id': self.body_id,
            'position': {
                'x': self.position[0],
                'y': self.position[1],
                'z': self.position[2]
            },
            'velocity': {
                'x': self.velocity[0],
                'y': self.velocity[1],
                'z': self.velocity[2]
            },
            'collided_entities': [entity.body_id for entity in self.collided_entities]
        }

        return observables

    def apply_velocity(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        This function applies instantaneous velocity to the entity.
        """
        ### AI GENERATED (had to change body_id to self.body_id, vx/vy to x and y)
        # Joint 0 = x slider, 1 = y slider, 2 = yaw
        x_pos, _ = p.getJointState(self.body_id, 0)[:2]
        y_pos, _ = p.getJointState(self.body_id, 1)[:2]
        p.resetJointState(self.body_id, 0, targetValue=x_pos, targetVelocity=x)
        p.resetJointState(self.body_id, 1, targetValue=y_pos, targetVelocity=y)
        self.sync_from_pybullet()
        ### END AI GENERATED

    def apply_force(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        This function applies a force vector to the entity.

        Bounds by self.MAX_FORCE which is set to 10.
        """

        x = max(-self.MAX_FORCE, min(x, self.MAX_FORCE))
        y = max(-self.MAX_FORCE, min(y, self.MAX_FORCE))

        self.sync_from_pybullet()
        p.applyExternalForce(self.body_id, 2, forceObj=[x, y, z], posObj=[0,0,0], flags=p.LINK_FRAME)

    def apply_position(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        This function applies an instantaneous position to the entity
        """
        p.resetJointState(self.body_id, 0, targetValue=x)
        p.resetJointState(self.body_id, 1, targetValue=y)
        self.sync_from_pybullet()


if __name__ == "__main__":

    # When running this script as the main script, 3 entities are created
    #   each with different colors in a GUI. You can swing them around like
    #   hockey pucks
    p.connect(p.GUI)
    entity = Entity(
        initial_position=np.array([0.0, 0.0, 0.0]), 
        rgbaColor=[1, 0, 0, 1]
    )
    entity2 = Entity(
        initial_position=np.array([0.8, 0.0, 0.0]), 
        rgbaColor=[0, 0, 1, 1]
    )
    entity3 = Entity(
        initial_position=np.array([0.0, 0.8, 0.0]), 
        rgbaColor=[1, 1, 1, 1]
    )

    for i in range (10000):
        p.stepSimulation()
        entity.sync_from_pybullet()
        entity2.sync_from_pybullet()
        entity3.sync_from_pybullet()
        time.sleep(1./240.)

