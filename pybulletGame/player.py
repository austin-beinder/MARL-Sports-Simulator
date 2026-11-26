"""
Many cited parts of entity.py were AI generated. 
This file was largely copy pasted from entity.py.
All modifications were human generated or found via Google.
Except: AI edits 2026-10-31 (Claude Code, Austin Beinder): physics_client on every pybullet call,
and only the actual carrier can lose the ball to an opponent's touch. See the AI GENERATED markers.
AI edit 2026-11-14 (Claude Code, Austin Beinder): show_calling, a brighter blue while a blue player
calls "got it" (watch only).
AI edit 2026-11-15 (Claude Code, Austin Beinder): pylint cleanup (wrapped lines over 100
characters); no behaviour change.
AI edit 2026-11-25 (Claude Code (Claude Opus 5.5), JC Stinson III): pylint cleanup (import order,
unused List and Ball imports, whitespace, parens); initial_position defaults to None (Entity
builds a fresh array), so two players never share one default array. No behaviour change.
"""

import time
from typing import Dict
import numpy as np
import pybullet as p
from entity import Entity
from teams import Team

class Player(Entity):

    _registry: Dict[int, "Entity"] = {}

    def __init__(self,
                 team: Team = Team.RED,
                 radius=0.3,
                 height=0.1,
                 mass=1.0,
                 initial_position=None,  # AI GENERATED (None = the origin, see Entity)
                 maxJointVelocity=3.0,
                 physics_client=0):  # AI GENERATED (which pybullet simulation, see Entity)

        self.ball = None  # If you are in possession of the ball, you get a self.ball

        self.team: Team = team
        if self.team == Team.RED:
            rgbaColor=[1, 0, 0, 1]
        else:
            #elif self == Team.BLUE:
            rgbaColor=[0, 0, 1, 1]
        self.base_color = rgbaColor  # AI GENERATED (2026-11-14: restored by show_calling)
        self.calling = False  # AI GENERATED (2026-11-14)

        super().__init__(
            radius=radius,
            height=height,
            mass=mass,
            initial_position=initial_position,
            rgbaColor=rgbaColor,
            maxJointVelocity=maxJointVelocity,
            physics_client=physics_client,  # AI GENERATED
        )

        self.team: Team = team


        # This creates some friction so it doesn't slide or spin forever or accelerate super quickly
        # Found by googling how to add drag in pybullet.
        p.changeDynamics(self.body_id, 0, linearDamping=1.25, angularDamping=10.05, maxJointVelocity=maxJointVelocity,
                         physicsClientId=self.client)  # AI GENERATED (physicsClientId)
        p.changeDynamics(self.body_id, 1, linearDamping=1.25, angularDamping=10.05, maxJointVelocity=maxJointVelocity,
                         physicsClientId=self.client)  # AI GENERATED (physicsClientId)

    ### AI GENERATED (2026-11-14)
    CALLING_COLOR = [0.3, 0.65, 1.0, 1.0]  # a brighter blue than the team's [0, 0, 1]

    def show_calling(self, calling: bool) -> None:
        """Recolor the player while it calls "got it" (ICD §5 calling), back to its team color
        after. Only the GUI shows it, so headless training doesn't need to call this."""
        calling = bool(calling)
        if calling == self.calling:
            return
        self.calling = calling
        p.changeVisualShape(self.body_id, 2,
                            rgbaColor=self.CALLING_COLOR if calling else self.base_color,
                            physicsClientId=self.client)
    ### END AI GENERATED

    def update_collisions(self):
        """Refresh collided_entities with whatever this entity is touching as of the last stepSimulation."""
        previous_collided_entities = self.collided_entities
        self.collided_entities = []
        seen_body_ids = set()
        for contact in p.getContactPoints(bodyA=self.body_id, physicsClientId=self.client):  # AI GENERATED (physicsClientId)
            other_body_id = contact[2]
            if other_body_id in seen_body_ids:
                continue
            seen_body_ids.add(other_body_id)
            other_entity = Entity._registry.get((self.client, other_body_id))  # AI GENERATED (keyed by client too)
            if other_entity is not None:
                self.collided_entities.append(other_entity)
                if other_entity not in previous_collided_entities:  # AI GENERATED (lint: no parens after not)
                    self.on_new_collision(other_entity)

    def on_new_collision(self, collided_entity):
        """
        This function will get called when a new collision occurs.
        """
        ### AI GENERATED (debug switch)
        if Entity.print_collisions:
            print(f"Entity: {self.body_id} collided with Entity: {collided_entity.body_id}")
        ### END AI GENERATED

        ### AI GENERATED
        # Walls have team NONE, so without this check they'd be treated as an opponent below
        if collided_entity.is_wall:
            self.on_wall_collision(collided_entity)
            return
        ### END AI GENERATED

        if collided_entity.is_ball:
            self.ball = collided_entity

        if not collided_entity.is_ball:
            if collided_entity.team != self.team:

                ### AI GENERATED (2026-10-31)
                # self.ball is also set by just touching the ball, so check this player really holds it.
                # Otherwise, with teammates, an opponent touching a non-carrier would steal from the carrier.
                if self.ball is not None and self.ball.possession_player is self:
                ### END AI GENERATED
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
