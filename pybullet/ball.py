"""
Many cited parts of entity.py were AI generated. 
This file was largely copy pasted from entity.py and player.py
"""

import numpy as np
import pybullet as p
from typing import Dict, List
import time
from entity import Entity
from teams import Team
import random

class Ball(Entity):

    _registry: Dict[int, "Entity"] = {}

    def __init__(self,
                 radius=0.15, 
                 height=0.1, 
                 mass=1.0,
                 initial_position=np.array([0.0, 0.0, 0.0]),
                 maxJointVelocity=3.0):

        rgbaColor=[1, 1, 1, 1]

        self.possession = Team.NONE
        self.possession_player: Player = None
        self.carry_constraint = None

        self.possession_flip_percentage: int = 50  # 1-100%
        self.CARRY_HEIGHT = 0.11  # player height (0.1) plus a small gap, so they don't touch

        super().__init__(
            radius=radius,
            height=height,
            mass=mass,
            initial_position=initial_position,
            rgbaColor=rgbaColor,
            maxJointVelocity=maxJointVelocity,
        )
        self.is_ball = True

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

    def on_new_collision(self, collided_entity) -> bool:
        """
        This function will get called when a new collision occurs.

        It might also be called by a player in possession of the ball because they would need to decide possession.
        """
        print(f"Entity: {self.body_id} collided with Entity: {collided_entity.body_id}")

        changed_possession = False
        # TODO include logic for if we have collided with a member or another team

        if self.possession_player == None or self.possession == Team.NONE:
            self.possession = collided_entity.team
            self.possession_player = collided_entity
            self._player_takes_possession(collided_entity)
            changed_possession = True
        # if self.possession_player.body_id == collided_entity.body_id:
        #     pass # Nothing to do, they're in possession anyways
        elif self.possession == collided_entity.team:
            # A pass has occured mysteriously
            self.possession_player = collided_entity
        elif self.possession != Team.NONE and self.possession != collided_entity.team:
            # We must have collided with a player of an opposing team
            num = random.randint(1, 100)

            # 50% chance of possession flip on collision with another team
            if num > self.possession_flip_percentage:
                self.possession = collided_entity.team
                self.possession_player = collided_entity
                self._player_takes_possession(collided_entity)
                changed_possession = True
                collided_entity.ball = self

        return changed_possession

    def get_observables(self) -> dict:
        """
        Returns: A dict with entity observerables 
        """

        observables = super().get_observables()
        observables['possession_team'] = self.possession
        observables['possession_player'] = self.possession_player
        return observables

    def _player_takes_possession(self, player):
        ### AI GENERATED
        self._release()  # drop any existing hold

        # Lift the whole ball (base, and with it the joint chain) to carry height
        bx, by, _ = p.getBasePositionAndOrientation(self.body_id)[0]
        p.resetBasePositionAndOrientation(self.body_id, [bx, by, self.CARRY_HEIGHT], [0, 0, 0, 1])

        # Snap the ball's x/y joints so it starts directly above the player.
        # Joint positions are relative to the base, which is why bx/by are subtracted.
        px, py = player.position[0], player.position[1]
        p.resetJointState(self.body_id, 0, targetValue=px - bx, targetVelocity=0)
        p.resetJointState(self.body_id, 1, targetValue=py - by, targetVelocity=0)

        # Pin ball link 2 to a point above player link 2
        self.carry_constraint = p.createConstraint(
            player.body_id, 2, self.body_id, 2, p.JOINT_POINT2POINT, [0, 0, 0],
            parentFramePosition=[0, 0, self.CARRY_HEIGHT], childFramePosition=[0, 0, 0])
        # self.carry_steps_left = 240  # "for a while": 1 s at 240 Hz
        ### END AI GENERATED

    ### AI GENERATED
    def _release(self):
        if getattr(self, "carry_constraint", None) is not None:
            p.removeConstraint(self.carry_constraint)
            self.carry_constraint = None
            bx, by, _ = p.getBasePositionAndOrientation(self.body_id)[0]
            p.resetBasePositionAndOrientation(self.body_id, [bx, by, 0], [0, 0, 0, 1])
        ### END AI GENERATED
        
        if self.possession_player:
            self.possession_player.ball = None
        self.possession_player = None
        self.possession = Team.NONE

    def kick_ball(self, x: float, y: float):
        """
        """
        self._release()
        # Teleport to side of player
        self.apply_velocity(x, y)


if __name__ == "__main__":

    # When running this script as the main script, 3 entities are created
    #   each with different colors in a GUI. You can swing them around like
    #   hockey pucks
    from player import Player

    p.connect(p.GUI)
    player = Ball(
        initial_position=np.array([0.0, 0.0, 0.0])
    )
    player2 = Player(
        team=Team.RED,
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

