"""
Many cited parts of entity.py were AI generated. 
This file was largely copy pasted from entity.py.
All modifications were human generated or found via Google.
"""

import numpy as np
import pybullet as p
from typing import Dict, List
import time
from entity import Entity
from teams import Team
from ball import Ball

class Player(Entity):

    _registry: Dict[int, "Entity"] = {}

    def __init__(self,
                 team: Team = Team.RED,
                 radius=0.3, 
                 height=0.1, 
                 mass=1.0,
                 initial_position=np.array([0.0, 0.0, 0.0]),
                 maxJointVelocity=3.0):

        self.ball = None  # If you are in possession of the ball, you get a self.ball

        self.team: Team = team
        if self.team == Team.RED:
            rgbaColor=[1, 0, 0, 1]
        else:
            #elif self == Team.BLUE:
            rgbaColor=[0, 0, 1, 1]

        super().__init__(
            radius=radius,
            height=height,
            mass=mass,
            initial_position=initial_position,
            rgbaColor=rgbaColor,
            maxJointVelocity=maxJointVelocity,
        )
        
        self.team: Team = team


        # This creates some friction so it doesn't slide or spin forever or accelerate super quickly
        # Found by googling how to add drag in pybullet.
        p.changeDynamics(self.body_id, 0, linearDamping=1.25, angularDamping=10.05, maxJointVelocity=maxJointVelocity)
        p.changeDynamics(self.body_id, 1, linearDamping=1.25, angularDamping=10.05, maxJointVelocity=maxJointVelocity)

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

    def on_new_collision(self, collided_entity):
        """
        This function will get called when a new collision occurs.
        """
        print(f"Entity: {self.body_id} collided with Entity: {collided_entity.body_id}")

        if collided_entity.is_ball:
            self.ball = collided_entity

        if not collided_entity.is_ball:
            if collided_entity.team != self.team:

                if self.ball is not None:
                    if self.ball.on_new_collision(collided_entity):
                        self.ball = None

    def get_observables(self) -> dict:
        """
        Returns: A dict with entity observerables 
        """

        observables = super().get_observables()
        observables['team'] = self.team
        observables['has_ball'] = self.ball is not None
        return observables


if __name__ == "__main__":

    # When running this script as the main script, 3 entities are created
    #   each with different colors in a GUI. You can swing them around like
    #   hockey pucks
    p.connect(p.GUI)
    player = Player(
        team=Team.RED,
        initial_position=np.array([0.0, 0.0, 0.0])
    )
    player2 = Player(
        team=Team.BLUE,
        initial_position=np.array([0.8, 0.0, 0.0])
    )
    player3 = Player(
        team=Team.BLUE,
        initial_position=np.array([0.0, 0.8, 0.0])
    )

    for i in range (10000):
        p.stepSimulation()
        player.sync_from_pybullet()
        player2.sync_from_pybullet()
        player3.sync_from_pybullet()
        time.sleep(1./240.)

