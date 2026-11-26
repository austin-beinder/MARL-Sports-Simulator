"""
Many cited parts of entity.py were AI generated.
This file was largely copy pasted from entity.py and player.py
AI edits 2026-10-31 (Claude Code, Austin Beinder): physics_client on every pybullet call, the
teammate handoff switch, and a per-ball random generator for steals. See the markers.
AI edit 2026-11-07 (Claude Code, Austin Beinder): the linear_damping parameter (was a fixed 1.25),
so KeepAwayEnv can tune how far a kick rolls.
AI edit 2026-11-07 (Claude Code, Austin Beinder): play_area, so a kick next to a wall can't place
the ball inside the wall (pybullet then pushed it out of the arena).
AI edit 2026-11-07 (Claude Code, Austin Beinder): check_held_contacts and `contested`, so an
opponent who was already touching when the carrier got the ball still takes it (steals used to
fire only on the first tick of a contact), and last_change_time: the ball changes hands at most
once per physics step.
AI edit 2026-11-15 (Claude Code, Austin Beinder): pylint cleanup (wrapped lines over 100
characters); no behaviour change.
"""

### AI GENERATED (2026-11-25, JC Stinson III: pylint import order; unused List dropped)
import os
import random
import time
from typing import Dict
import numpy as np
import pybullet as p
from entity import Entity
from teams import Team
### END AI GENERATED

class Ball(Entity):

    _registry: Dict[int, "Entity"] = {}

    def __init__(self,
                 radius=0.15,
                 height=0.1,
                 mass=1.0,
                 initial_position=None,  # AI GENERATED (2026-11-25: None = the origin, see Entity)
                 maxJointVelocity=3.0,
                 # AI GENERATED (how quickly a loose ball slows down, see below)
                 linear_damping=1.25,
                 physics_client=0):  # AI GENERATED (which pybullet simulation, see Entity)

        rgbaColor=[1, 1, 1, 1]

        ### AI GENERATED
        # Apply a soccer ball texture to the visible link (link 2, the yaw link that holds the shape)
        texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "textures", "soccer_ball.png")
        texture_id = p.loadTexture(texture_path, physicsClientId=physics_client)
        ### AI GENERATED

        self.possession = Team.NONE
        self.possession_player: Player = None
        self.carry_constraint = None

        self.possession_flip_percentage: int = 100  # 1-100%
        self.CARRY_HEIGHT = 0.11  # player height (0.1) plus a small gap, so they don't touch

        ### AI GENERATED
        # Steal cooldown, measured in simulation time (not wall-clock) so it behaves
        # the same whether the sim runs in real time in the GUI or sped up for training.
        self.STEAL_COOLDOWN = 0.1  # seconds
        self.sim_time = 0.0
        self.last_steal_time = -float("inf")

        # Opponents whose contact with the current carrier has already been decided (2026-11-07):
        # they rolled for the ball, or the carrier just took it from them. check_held_contacts skips
        # them until they stop touching. Cleared whenever the ball changes hands.
        self.contested = set()
        # sim_time of the last change of hands. The ball changes hands at most once per physics
        # step, so when both teams touch it on the same step, the second touch waits for the next
        # step (check_held_contacts) instead of instantly stealing it and becoming untouchable.
        self.last_change_time = None

        # Space left between the kicker's edge and the ball's edge when the ball is kicked
        self.KICK_GAP = 0.02

        # Whether a teammate touching the ball takes it from the carrier (2026-10-31). KeepAwayEnv sets
        # this from KeepAwayConfig.teammate_handoff, which is off: the ICD only lets a kick release the ball.
        self.allow_teammate_handoff = True

        # What rolls for steals (2026-10-31). Python's random module by default; KeepAwayEnv gives
        # each game its own seeded random.Random, so games in one process don't share a sequence.
        self.rng = random

        # Half the inner size of the walled field, (half_length, half_width), centred on (0, 0), or
        # None for no walls (2026-11-07). kick_ball keeps a kicked ball inside it. KeepAwayEnv sets
        # it.
        self.play_area = None
        ### END AI GENERATED

        super().__init__(
            radius=radius,
            height=height,
            mass=mass,
            initial_position=initial_position,
            rgbaColor=rgbaColor,
            maxJointVelocity=maxJointVelocity,
            physics_client=physics_client,  # AI GENERATED
        )
        self.is_ball = True

        # This creates some friction so it doesn't slide or spin forever or accelerate super quickly
        # Found by googling how to add drag in pybullet.
        ### AI GENERATED (linear_damping, was 1.25)
        # pybullet slows the ball by dv/dt = -linear_damping * v * (1 + |v|) (measured 2026-11-07),
        # so a ball sent off at v m/s rolls about ln(1 + v) / linear_damping metres in total.
        p.changeDynamics(self.body_id, 0, linearDamping=linear_damping, angularDamping=10.05,
                         maxJointVelocity=maxJointVelocity,
                         physicsClientId=self.client)  # AI GENERATED (physicsClientId)
        p.changeDynamics(self.body_id, 1, linearDamping=linear_damping, angularDamping=10.05,
                         maxJointVelocity=maxJointVelocity,
                         physicsClientId=self.client)  # AI GENERATED (physicsClientId)
        ### END AI GENERATED
        p.changeVisualShape(self.body_id, 2, textureUniqueId=texture_id, physicsClientId=self.client)  # AI GENERATED

    ### AI GENERATED
    def sync_from_pybullet(self):
        # Called once per stepSimulation, so advance the sim clock by one physics step
        # before collisions (and therefore on_new_collision) are processed.
        self.sim_time += p.getPhysicsEngineParameters(physicsClientId=self.client)["fixedTimeStep"]
        super().sync_from_pybullet()
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

    def on_new_collision(self, collided_entity) -> bool:
        """
        This function will get called when a new collision occurs.

        It might also be called by a player in possession of the ball because they would need to decide possession.
        """
        ### AI GENERATED (debug switch)
        if Entity.print_collisions:
            print(f"Entity: {self.body_id} collided with Entity: {collided_entity.body_id}")
        ### END AI GENERATED

        changed_possession = False

        ### AI GENERATED
        # The ball bounces off walls physically, but a wall can never take possession
        if collided_entity.is_wall:
            self.on_wall_collision(collided_entity)
            return changed_possession
        ### END AI GENERATED

        if self.possession_player is None or self.possession == Team.NONE:  # AI GENERATED (lint: is None, not == None)
            self._player_takes_possession(collided_entity)
            changed_possession = True
        # if self.possession_player.body_id == collided_entity.body_id:
        #     pass # Nothing to do, they're in possession anyways
        elif self.possession == collided_entity.team:
            # A pass has occured mysteriously
            ### AI GENERATED
            # Physically hand the ball to the teammate. The team keeps possession,
            # so changed_possession stays False.
            if self.allow_teammate_handoff:  # 2026-10-31: the handoff can be switched off
                self._player_takes_possession(collided_entity)
            ### END AI GENERATED
        elif self.possession not in (Team.NONE, collided_entity.team):  # AI GENERATED (lint: one "not in" test)
            # We must have collided with a player of an opposing team

            ### AI GENERATED
            # Don't allow a steal within STEAL_COOLDOWN of the last one, so the ball
            # doesn't ping pong between players during a complex collision.
            if self.sim_time - self.last_steal_time < self.STEAL_COOLDOWN:
                # 2026-11-07: this contact had its chance; no retry later
                self.contested.add(collided_entity)
                return changed_possession
            # 2026-11-07: nor on the step the ball last changed hands (see last_change_time). Unlike
            # the cooldown, this doesn't use up the contact: check_held_contacts tries it again next
            # step.
            if self.sim_time == self.last_change_time:
                return changed_possession
            ### END AI GENERATED

            num = self.rng.randint(1, 100)  # AI GENERATED (self.rng, was the random module)
            self.contested.add(collided_entity)  # AI GENERATED (this contact has had its roll)

            # 50% chance of possession flip on collision with another team
            if num > (100 - self.possession_flip_percentage):
                victim = self.possession_player  # AI GENERATED
                self._player_takes_possession(collided_entity)
                changed_possession = True
                self.last_steal_time = self.sim_time  # AI GENERATED
                # AI GENERATED: the old carrier can't take it straight back while they're still
                # touching
                self.contested = {victim}

        return changed_possession

    ### AI GENERATED (2026-11-07)
    def check_held_contacts(self) -> bool:
        """
        Give opponents who are touching the carrier, or the ball, their steal roll. Call this every
        physics step, after every entity's sync_from_pybullet.

        on_new_collision only runs on the first tick of a contact. So an opponent who was already
        touching when the carrier got the ball (a loose ball picked up mid-scrum, often pinned in a
        corner) would never get a roll, and the carrier kept the ball for as long as they stayed in
        contact. Each contact still gets one roll (see contested): the player the carrier took the
        ball from has to separate and touch again, so the ball doesn't ping-pong between them.
        Returns True if the ball changed hands.
        """
        holder = self.possession_player
        if holder is None:
            return False
        touching = []
        for entity in (*holder.collided_entities, *self.collided_entities):
            if (not entity.is_wall and not entity.is_ball
                    and entity.team not in (Team.NONE, holder.team)
                    and entity not in touching):
                touching.append(entity)
        self.contested &= set(touching)  # whoever has let go gets a fresh roll next time they touch
        for opponent in touching:
            if opponent not in self.contested and self.on_new_collision(opponent):
                return True
        return False
    ### END AI GENERATED

    def get_observables(self) -> dict:
        """
        Returns: A dict with entity observerables 
        """

        observables = super().get_observables()
        observables['possession_team'] = self.possession
        ### AI GENERATED
        # Report the player's id (like collided_entities) rather than the Player object
        observables['possession_player'] = self.possession_player.body_id if self.possession_player is not None else None
        ### END AI GENERATED
        return observables

    def _player_takes_possession(self, player):
        ### AI GENERATED
        self._release()  # drop any existing hold

        # Lift the whole ball (base, and with it the joint chain) to carry height
        c = self.client
        bx, by, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=c)[0]
        p.resetBasePositionAndOrientation(self.body_id, [bx, by, self.CARRY_HEIGHT], [0, 0, 0, 1], physicsClientId=c)

        # Snap the ball's x/y joints so it starts directly above the player.
        # Joint positions are relative to the base, which is why bx/by are subtracted.
        px, py = player.position[0], player.position[1]
        p.resetJointState(self.body_id, 0, targetValue=px - bx, targetVelocity=0, physicsClientId=c)
        p.resetJointState(self.body_id, 1, targetValue=py - by, targetVelocity=0, physicsClientId=c)

        # Pin ball link 2 to a point above player link 2
        self.carry_constraint = p.createConstraint(
            player.body_id, 2, self.body_id, 2, p.JOINT_POINT2POINT, [0, 0, 0],
            parentFramePosition=[0, 0, self.CARRY_HEIGHT], childFramePosition=[0, 0, 0], physicsClientId=c)
        # self.carry_steps_left = 240  # "for a while": 1 s at 240 Hz

        # Record possession on both the ball and the player, for every way a player can get the ball
        self.possession = player.team
        self.possession_player = player
        player.ball = self
        self.last_change_time = self.sim_time  # 2026-11-07
        ### END AI GENERATED

    ### AI GENERATED
    def _release(self):
        if getattr(self, "carry_constraint", None) is not None:
            p.removeConstraint(self.carry_constraint, physicsClientId=self.client)
            self.carry_constraint = None
            bx, by, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)[0]
            p.resetBasePositionAndOrientation(self.body_id, [bx, by, 0], [0, 0, 0, 1], physicsClientId=self.client)
        ### END AI GENERATED

        if self.possession_player:
            self.possession_player.ball = None
        self.possession_player = None
        self.possession = Team.NONE
        self.contested = set()  # AI GENERATED (a new carrier's contacts are all undecided)

    def kick_ball(self, x: float, y: float) -> bool:
        ### AI GENERATED
        """
        Release the ball and send it off with velocity (x, y).

        If a player was holding the ball, it is first moved to the edge of that player
        in the kick direction, so it doesn't land on top of them and get picked straight back up.
        A zero (x, y) kick has no direction, so the ball is just dropped where it is.

        With play_area set, that spot is kept inside the walls (2026-11-07): placing the ball inside
        a wall made pybullet push it out the far side. If the kicker is so close to the wall that
        the ball doesn't fit in front of them, the kick rebounds: the kicker keeps the ball.
        Returns whether the ball left the kicker.
        """
        kicker = self.possession_player
        speed = np.hypot(x, y)
        target = None
        if kicker is not None and speed > 0:
            direction = np.array([x, y]) / speed
            # Read the kicker's live position; kicker.position is only as fresh as its last sync
            c = self.client
            kicker_xy = np.array(p.getLinkState(kicker.body_id, 2, physicsClientId=c)[0][:2])
            target = kicker_xy + direction * (kicker.radius + self.radius + self.KICK_GAP)
            if self.play_area is not None:
                limit = np.asarray(self.play_area, dtype=float) - self.radius - self.KICK_GAP
                target = np.clip(target, -limit, limit)
                if np.linalg.norm(target - kicker_xy) < kicker.radius + self.radius:
                    # no room between the kicker and the wall: the ball bounces straight back
                    return False
        ### END AI GENERATED
        self._release()

        # Teleport to side of player
        ### AI GENERATED
        if target is not None:
            target_x, target_y = target

            # Joint positions are relative to the base, which is why bx/by are subtracted
            bx, by, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=c)[0]
            p.resetJointState(self.body_id, 0, targetValue=target_x - bx, targetVelocity=0, physicsClientId=c)
            p.resetJointState(self.body_id, 1, targetValue=target_y - by, targetVelocity=0, physicsClientId=c)
        ### END AI GENERATED

        self.apply_velocity(x, y)
        return True  # AI GENERATED


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
    player4 = Player(
        team=Team.RED,
        initial_position=np.array([0.8, 0.8, 0.0])
    )
    player5 = Player(
        team=Team.BLUE,
        initial_position=np.array([0.0, -0.8, 0.0])
    )

    for i in range (10000):
        p.stepSimulation()
        player.sync_from_pybullet()
        player2.sync_from_pybullet()
        player3.sync_from_pybullet()
        player4.sync_from_pybullet()
        player5.sync_from_pybullet()
        time.sleep(1./240.)
