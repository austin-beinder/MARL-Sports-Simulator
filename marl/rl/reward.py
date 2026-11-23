# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the RL-side reward module with a
#     term registry, (1 - gamma) scaling, the survival and goto_target base
#     terms, the interface-only PBRS term (design #14, #15), and the
#     effective_terminal helper (#12).
#   - 2026-10-14: Added the pbrs_ball_acquisition shaping term and its potential
#     (the student's acquisition-ablation spec), RewardModule.parts (base vs
#     shaping, for logging), and passed the full Config to the terms (PBRS needs
#     the learner's gamma).
#   - 2026-11-08: The passing drill: the passing_drill base term, the
#     pass_progress and pass_completed shaping terms, and PassMemory, the
#     per-episode pass bookkeeping they read. (Austin Beinder)
#   - 2026-11-08: passing_drill's hold reward grows with each catch: it pays
#     the number of passes completed so far, not a flat 1. (Austin Beinder)
#   - 2026-11-09: The pass_practice base term (one kick per episode: 1 for a
#     catch, else up to 0.5 for how close the ball came to a teammate), and
#     PassMemory's closest approach. (Austin Beinder)
#   - 2026-11-10: A minimum pass distance (MIN_PASS_DISTANCE, 1 m from the
#     kick point for a catch; the closest approach skips teammates nearer than
#     1.5 m), and pass_practice pays
#     a catch 0.75 to 1 by its length. (Austin Beinder)
#   - 2026-11-10: Both are options, off by default (rl.reward.min_pass_distance,
#     rl.reward.pass_length_bonus), so a stage can train on the old reward
#     first. (Austin Beinder)
#   - 2026-11-11: The kicker_clearance shaping term: the kicker pays for
#     staying near its own kick, and more for taking it back. (Austin Beinder)
#   - 2026-11-11: The pass_catch base term (pass practice to the time limit:
#     each catch pays when it happens), the receiver_spacing shaping term, and
#     PassMemory counts kicks. (Austin Beinder)
#   - 2026-11-11: receiver_spacing is a tent, not a Gaussian: blues bunched
#     0.6 m apart sat on the Gaussian's flat tail and felt no pull apart.
#     (Austin Beinder)
#   - 2026-11-11: With a minimum pass distance d, the receiver must also have
#     been d from the kick point when the ball was kicked (the blues ran as a
#     pack, kicked the ball ahead and chased it down). (Austin Beinder)
#   - 2026-11-13: rl.reward.return_pass_factor: pass_catch pays a pass straight
#     back to the blue who passed to the kicker (A -> B -> A) this fraction;
#     PassMemory counts those returns. (Austin Beinder)
#   - 2026-11-13: kicker_clearance stops once the kicked ball has stopped
#     loose (the kicker may fetch its own miss). (Austin Beinder)
#   - 2026-11-13: The call_accuracy shaping term (rl.call_enabled): a "got it"
#     call pays when the caller gets the ball. PassMemory tracks calls and
#     pickups. (Austin Beinder)
#   - 2026-11-14: call_accuracy is no longer part of the team reward: the call
#     head gets its own per-blue reward (call_values, blue_pickup), weighted by
#     the term's coef (RewardModule.call_coef). The claim_ball scenario's base
#     reward (claim) and the call_yield shaping term. (Austin Beinder)
#   - 2026-11-14: The dead_ball shaping term (claim's urgency for the catch
#     stages), and receiver_spacing leaves out the blue nearest a dead ball,
#     so fetching it doesn't cost the team its spacing. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters and fixed
#     the new warnings (a module-level disable of unused-argument and
#     redefined-outer-name for the shared term signature); no behaviour
#     change. (Austin Beinder)
#   - 2026-11-23: Soccer (docs/soccer_env_plan.md §5): the goal base term,
#     the pbrs_ball_to_goal shaping term, pbrs_ball_acquisition treating a
#     red-held ball as one to win back, and goal counts in PassMemory.
#     (Austin Beinder)
# ==============================================================================
"""RL-side reward module (design #14, #15). The env never computes rewards.

    r = (1 - gamma) * (base + sum of enabled shaping terms * coef)

shared by all blue agents. Every term has the signature
``term(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float``,
fixed from day one so shaping terms that need any of these plug in unchanged.
``cfg`` is the full Config (PBRS needs the learner's gamma).

Base terms:

* ``survival``    -- +1 every step, INCLUDING the step where red captures the
  ball. With the (1 - gamma) scale, an episode of L steps has return
  1 - gamma^L, in [0, 1) (#15).
* ``goto_target`` -- minus the blue-to-ball distance in meters, measured at
  ``s_next`` (the state the action led to).

The ``pbrs_*`` shaping terms are potential-based: F = gamma * Phi(s') - Phi(s) (Ng, Harada &
Russell 1999), with Phi(s') = 0 only when ``effective_terminal`` and the real
Phi(s') at a truncation or a rollout cut (#15). Then every policy's values shift
by exactly -Phi(s), so no advantage changes: shaping can speed up learning
(through the critic's errors) but never changes what is optimal.

* ``pbrs_ball_acquisition`` -- Phi = minus the distance (m) from the closest
  blue to a LOOSE ball; 0 while anyone holds it, so possession starts are
  unshaped. Built 2026-10-14 for the acquisition ablation.
* ``pbrs_red_distance`` -- the #15 potential (ball-to-nearest-red distance);
  interface-only, raises ``NotImplementedError`` if enabled.

The passing drill (2026-11-08; frozen blues, so passing is the only way to keep the ball):

* ``passing_drill`` (base) -- every step a blue holds the ball at ``s_next``, the number of
  passes the team has completed this episode: 1 per step after the first catch, 2 after the
  second, and so on, so every pass makes holding worth more. Holding the opening ball (no pass
  yet) earns nothing, and neither
  does a loose ball (kicking it away from a slow red would otherwise buy free time).
* ``pass_progress`` (shaping, NOT potential-based) -- while a blue's kick is in play, how much
  closer the ball got this step to the nearest OTHER blue (m). Kicking toward a teammate earns a
  little, away from them costs a little. It sums to (start distance - end distance) over a kick.
* ``pass_completed`` (shaping, NOT potential-based) -- 1 on the step a different blue picks up a
  blue's kick.

Pass practice (2026-11-09; one kick per episode, no red, frozen blues):

* ``pass_practice`` (base) -- paid once, on the episode's last step: 1 for a catch (with
  ``rl.reward.pass_length_bonus``, 2026-11-10: ``0.75 + 0.25 * min(1, length / 4 m)``, so longer
  passes pay a bit more), otherwise ``0.5 * exp(-gap / 1 m)``, where gap is how close the ball's
  path came to the edge of a teammate (closest approach, so a kick in the right direction scores
  even if it is too soft or too hard). 0 if the holder never kicked. A catch always beats a near
  miss.

Pass practice to the time limit (2026-11-11; pass_practice_moving, no red, blues free to move):

* ``pass_catch`` (base) -- each completed pass pays on the step it is caught: 1, or 0.75 to 1 by
  its length with ``rl.reward.pass_length_bonus``. Nothing else; a miss just costs the time to
  get the ball back. 2026-11-13: a return pass (straight back to the blue who passed to the
  kicker, ``s["ball"]["passer"]`` at the kick: A -> B -> A) pays ``rl.reward.return_pass_factor``
  times that (1 = no difference, the default). The blues passed back and forth 88% of the time,
  the third blue idle; with three blues a full pay for every pass means rotating A -> B -> C -> A.
  ``rl.passer_obs`` shows the blues the same passer, so they can tell which pass is which.
* ``call_accuracy`` (2026-11-13; needs ``rl.call_enabled``) -- a "got it" call. Two blues that
  both hung back from a stopped loose ball (median 8 s) each had nothing telling them the other
  would go. 2026-11-14: NOT part of the team reward any more. As a team term (a pickup paid the
  getter's call lead less every other blue's) the three blues' calls cancelled out in the one
  shared number, and after 4M steps calling had fallen to 3% of steps, the getter calling less
  than its teammates. Now the call head has its own reward, per blue, scored in hindsight
  (``call_values``): a call at step t is worth +1 to blue i if the next blue pickup, within
  CALL_LEAD steps, is i's; -0.5 if it is a teammate's; 0 if nobody picks the ball up in time;
  less 0.02 for calling at all (CALL_RIGHT, CALL_WRONG, CALL_STEP_COST). The PPO update scores
  the call head with that alone (``Actor.evaluate_split``), weighted by this term's coef
  (``RewardModule.call_coef``); the other heads keep the team advantage. The term function below
  is what the logs (``called_share``, ``miscalled_per_pickup``) measure; it never adds to the
  team reward.
* ``receiver_spacing`` (shaping, NOT potential-based) -- every step, how close each pair of blues
  is to ``rl.reward.spacing_distance`` apart (3 m by default): the mean over pairs of the tent
  ``max(0, 1 - |d - target| / target)``, 0 for blues on top of each other or twice the target
  apart, 1 when every pair is exactly the target apart, and straight lines between. So the blues
  spread out into passing range of each other, ready for the next pass. 2026-11-14: while the
  ball is dead (loose and slower than DEAD_BALL_SPEED), the blue nearest it is left out (with three
  or more blues), so going to fetch it doesn't cost the team its spacing; the others still spread.

Claiming a loose ball (2026-11-14; claim_ball: a loose ball, no red, ends when a blue gets it):

* ``claim`` (base) -- -1 every step until a blue picks the ball up (0 on that step). With the
  (1 - gamma) scale an episode of L steps returns -(1 - gamma^L): the sooner the better.
* ``call_yield`` (shaping, NOT potential-based; needs ``rl.call_enabled``) -- "if a teammate
  called it, stay away". Each step, every blue NOT calling in ``s`` while a teammate was, and the
  ball still loose in ``s_next``, pays its closeness to the ball, 1 - distance / YIELD_DISTANCE
  (0 from YIELD_DISTANCE out); picking up a ball a teammate called costs YIELD_STEAL. Calls are
  read from ``s``: what the blue could see when it moved.
* ``dead_ball`` (shaping, NOT potential-based; 2026-11-14) -- claim's urgency for the stages that
  aren't claim_ball: -1 every step the ball is dead in ``s_next`` (nobody holds it and it moves
  slower than DEAD_BALL_SPEED, so a pass in flight is free). In the catch stages the blues hung
  about 1.5 m off a stopped ball (half of every episode, a third of misses never fetched), since
  nothing there charged for it. Every dead step also makes a miss cost more, so keep the coef well
  below (catch value) / (dead steps per miss), or holding the ball beats kicking it.

Soccer (2026-11-23; the game ends on the first goal, red keeps playing when it has the ball):

* ``goal`` (base) -- +1 / (1 - gamma) on the step blue scores, -1 / (1 - gamma) on the step red
  scores (own goals included: a ball in blue's goal is red's point), 0 otherwise. Read from the
  step's ``goal`` events (ICD §9). The 1 / (1 - gamma) cancels the (1 - gamma) scale, so the
  learner sees +-1 per goal, the size of keep-away's best return; a plain 1 would reach it as
  ~0.01, tiny next to any shaping.
* ``pbrs_ball_to_goal`` (shaping, potential-based) -- Phi = minus the ball's distance to the
  centre of red's goal (east, ``x = Lx / 2``), over the field length. Moving the ball toward the
  goal earns, losing ground costs.
* ``pbrs_ball_acquisition`` in soccer: Phi is 0 only while a BLUE holds the ball; a red-held ball
  counts like a loose one (a ball to win back). In keep-away red possession is terminal, so this
  changes nothing there.

A pass is a blue's kick picked up by a different blue with nobody else touching it first. With
``rl.reward.min_pass_distance`` d > 0 (2026-11-10), the catch must also be at least d from where
the ball was kicked, the receiver must have been at least d from the kick point when the ball was
kicked (2026-11-11: so a teammate running beside the kicker can't chase down a ball kicked ahead),
and the closest approach ignores teammates within d + 0.5 m of the kick point, so a tap into a
teammate next to the kicker earns nothing. That
needs memory across steps (who kicked the ball in play, how many passes so far), so these terms
read ``info["pass"]`` (a ``PassStep``), which ``RewardModule.parts`` adds from a ``PassMemory``.
Every loop that plays episodes owns one memory per episode (``RewardModule.new_episode``): the
training collector and the evaluator share a RewardModule and can be mid-episode at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass  # AI GENERATED
from typing import Any, Callable, NamedTuple

import numpy as np  # AI GENERATED (2026-11-14: call_values)

from marl.common.config import (
    BASE_REWARD_TERMS,
    SHAPING_TERMS_BUILT,
    SHAPING_TERMS_DEFERRED,
    Config,
)

# Every term shares one fixed signature (above), so most ignore some of their arguments, and
# its effective_terminal parameter shares the name of the helper below on purpose.
# pylint: disable=unused-argument,redefined-outer-name

Term = Callable[[dict, dict, dict, bool, bool, dict, Config], float]


def effective_terminal(terminated: bool, truncated: bool, horizon_mode: str) -> bool:
    """True when the future value is zero (#12): a real game over, or, in
    ``finite`` mode only, a time-out. Feeds the GAE bootstrap mask and PBRS."""
    return terminated or (horizon_mode == "finite" and truncated)


def survival(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return 1.0


def goto_target(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    ball = s_next["ball"]["pos"]
    dists = [math.dist(b["pos"], ball) for b in s_next["blue"]]
    return -sum(dists) / len(dists)


def ball_potential(s: dict, red_held_counts: bool = False) -> float:
    """Phi(s) for ``pbrs_ball_acquisition``: minus the distance (m) from the CLOSEST
    blue to the ball while it is loose (ICD §5: no ``has_possession`` set), 0 while
    anyone holds it. Closest only: a team potential, so at N > 1 it doesn't pull
    every teammate onto the ball. With ``red_held_counts`` (soccer, 2026-11-23) a ball a red
    holds counts like a loose one: only a blue holder makes Phi 0."""
    if any(p["has_possession"] for p in s["blue"]) or (
            not red_held_counts and any(p["has_possession"] for p in s["red"])):  # AI GENERATED
        return 0.0
    ball = s["ball"]["pos"]
    return -min(math.dist(b["pos"], ball) for b in s["blue"])


def pbrs_ball_acquisition(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    """F = gamma * Phi(s') - Phi(s), with Phi(s') = 0 at an effective terminal (#15).

    Per unit coef, while the ball is loose and blue is d meters from it: a full-speed
    step toward it earns about gamma * max_speed * dt_control (0.099 at the defaults)
    on top of a drift of (1 - gamma) * d that every action gets (Phi < 0); a pickup
    earns about possession_radius; red capturing the ball refunds +d. These cancel
    over an episode: the discounted sum is gamma^L * Phi(s_L) - Phi(s_0) for EVERY
    policy (a test checks it)."""
    red_held = cfg.shared.scenario == "soccer"  # AI GENERATED (2026-11-23)
    phi_next = 0.0 if effective_terminal else ball_potential(s_next, red_held)
    return cfg.gamma * phi_next - ball_potential(s, red_held)


### AI GENERATED (2026-11-23: soccer)
def goal(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    """+-1 / (1 - gamma) per goal this step: + for blue's, - for red's (see the docstring)."""
    score = sum(1.0 if e["team"] == "blue" else -1.0 for e in info["events"] if e["type"] == "goal")
    return score / (1.0 - cfg.gamma)


def goal_potential(s: dict, cfg) -> float:
    """Phi(s) for ``pbrs_ball_to_goal``: minus the ball's distance (m) to the centre of red's
    goal (blue attacks east, x = Lx / 2), over the field length Lx: about -1 at blue's own goal,
    0 in red's."""
    length = cfg.shared.field_size[0]
    return -math.dist(s["ball"]["pos"][:2], (length / 2, 0.0)) / length


def pbrs_ball_to_goal(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    """F = gamma * Phi(s') - Phi(s), Phi(s') = 0 at an effective terminal (#15). Per unit coef, a
    carry at full speed toward the goal earns about carry speed * dt_control / Lx (0.005 at the
    defaults) a step. At a goal Phi(s') = 0, so a red goal refunds the ball's distance (~1): small
    next to the goal's own -1 / (1 - gamma). Over an episode it sums to gamma^L Phi(s_L) - Phi(s_0)
    for every policy, so it can't change which policy is best."""
    phi_next = 0.0 if effective_terminal else goal_potential(s_next, cfg)
    return cfg.gamma * phi_next - goal_potential(s, cfg)
### END AI GENERATED


### AI GENERATED (2026-11-08: the passing drill)
class PassStep(NamedTuple):
    """One step's pass bookkeeping, as the pass terms see it in ``info["pass"]``."""

    # the blue whose kick was in play during this step (None: no blue kick in play)
    kicker: int | None
    completed: bool  # a different blue picked that kick up this step
    passes: int  # passes completed this episode, this step's included
    # 2026-11-09: the closest the ball's path has come to another blue's centre (m) since the
    # latest kick; inf before any kick
    closest: float = math.inf
    # 2026-11-10: how far the latest completed pass went, kick point to catch point (m)
    length: float = 0.0
    # 2026-11-13: this step's pass went straight back to the blue who passed to the kicker
    returned: bool = False
    # 2026-11-13: the kick in play has come to a stop loose (it can't reach anyone by itself)
    stopped: bool = False
    # 2026-11-13: this step's pickups: the getter's call less the others' (see CALL_LEAD)
    call_credit: float = 0.0
    callers: int = 0  # 2026-11-13: blues calling in s_next


### AI GENERATED (2026-11-10: a minimum pass distance, rl.reward.min_pass_distance)
# With a minimum pass distance d > 0, a catch counts as a pass only if the ball is at least d
# from where it was kicked, and the closest approach ignores teammates within d + PASS_REACH of the
# kick point (they could only catch a shorter one; PASS_REACH: player 0.25 + ball 0.15 m, plus a
# margin). So a tap into a teammate next to the kicker counts for nothing (no pass, no near miss).
# 2026-11-11: the receiver must also have stood at least d from the kick point (centre to the
# ball's position) at the kick. Otherwise a pack of blues could kick the ball a few metres ahead
# and let a packmate chase it down: a "pass" nobody aimed. d = 0 (the default) counts every catch
# and every teammate, as before.
PASS_REACH = 0.5
### END AI GENERATED


@dataclass
class PassMemory:
    """Per-episode memory for the pass terms; ``update`` it once per step, in order."""

    # the blue whose kick is in play (the ball is loose, or rebounded to them)
    kicker: int | None = None
    passes: int = 0
    closest: float = math.inf  # 2026-11-09: see PassStep.closest
    # 2026-11-10: where the ball in play was kicked from
    kick_point: tuple[float, float] | None = None
    length: float = 0.0  # 2026-11-10: see PassStep.length
    min_distance: float = 0.0  # 2026-11-10: rl.reward.min_pass_distance (m); 0 = no minimum
    # 2026-11-11: blue kicks this episode (the pass completion rate is passes / kicks)
    kicks: int = 0
    # 2026-11-11: the blues who stood min_distance or more from the kick point at the latest kick
    # (the only ones who can complete it); None = anyone (no minimum, or no state to tell from)
    receivers: frozenset[int] | None = None
    # 2026-11-13: who passed the ball to the blue whose kick is in play (s["ball"]["passer"] at the
    # kick; None if nobody did, or the env doesn't report it), and the passes back to that blue
    return_to: int | None = None
    returns: int = 0
    stopped: bool = False  # 2026-11-13: see PassStep.stopped
    # 2026-11-13: calls. call_streaks[i] = the steps blue i had called in a row up to the latest
    # state (None until the first update); pickups = blue pickups of the ball; called / miscalled =
    # the getters' call lead and the other blues' at those pickups, summed (0 to 1 each).
    call_streaks: list[int] | None = None
    pickups: int = 0
    called: float = 0.0
    miscalled: float = 0.0
    # 2026-11-23, soccer: goals by blue (goals_for) and by red (goals_against) this episode
    goals_for: int = 0
    goals_against: int = 0
    keeps_score: bool = False  # 2026-11-23: a soccer episode (outcome says who won)

    @property
    def outcome(self) -> str | None:  # AI GENERATED (2026-11-23)
        """Soccer: "win", "loss" or "draw" (no goal, or level); None in other scenarios."""
        if not self.keeps_score:
            return None
        if self.goals_for != self.goals_against:
            return "win" if self.goals_for > self.goals_against else "loss"
        return "draw"

    def _far_enough(self, ball) -> bool:  # AI GENERATED (2026-11-10)
        """The ball is min_distance from the kick point (always, with no minimum or no kick
        point)."""
        return (self.min_distance <= 0 or self.kick_point is None
                or math.dist(ball, self.kick_point) >= self.min_distance)

    def _a_target(self, pos) -> bool:  # AI GENERATED (2026-11-10)
        """A teammate this far from the kick point could catch a pass of min_distance or more."""
        return (self.min_distance <= 0 or self.kick_point is None
                or math.dist(pos[:2], self.kick_point) >= self.min_distance + PASS_REACH)

    def update(self, info: dict, s: dict | None = None, s_next: dict | None = None) -> PassStep:
        """``s`` and ``s_next`` (2026-11-09) feed the closest approach; without them it stays inf.
        They also give the kick point (2026-11-10); without them every catch counts, however
        short."""
        # AI GENERATED (2026-11-13: returned)
        in_play, completed, returned = self.kicker, False, False
        for event in info["events"]:
            ### AI GENERATED (2026-11-23: soccer)
            if event["type"] == "goal":
                if event["team"] == "blue":
                    self.goals_for += 1
                else:
                    self.goals_against += 1
            ### END AI GENERATED
            if event["type"] == "kick":
                self.kicker = in_play = event["agent"][1]  # blue kicks are the only kicks (ICD §6)
                self.kicks += 1  # AI GENERATED (2026-11-11)
                self.closest = math.inf  # a new kick: measure its path from scratch
                # AI GENERATED (2026-11-10)
                self.kick_point = tuple(s["ball"]["pos"][:2]) if s is not None else None
                ### AI GENERATED (2026-11-11: who was far enough away to receive it)
                self.receivers = None if self.min_distance <= 0 or s is None else frozenset(
                    b["id"] for b in s["blue"]
                    if b["id"] != self.kicker
                    and math.dist(b["pos"][:2], self.kick_point) >= self.min_distance)
                ### END AI GENERATED
                # AI GENERATED (2026-11-13)
                self.return_to = s["ball"].get("passer") if s is not None else None
                self.stopped = False  # AI GENERATED (2026-11-13): a new kick is moving
            elif event["type"] == "possession_change" and event["to"] is not None:
                to = event["to"]
                ### AI GENERATED (2026-11-10: a catch has to be min_distance from the kick point)
                caught_at = s_next["ball"]["pos"][:2] if s_next is not None else None
                if (to[0] == "blue" and self.kicker is not None and to[1] != self.kicker
                        and (caught_at is None or self._far_enough(caught_at))
                        # AI GENERATED (2026-11-11)
                        and (self.receivers is None or to[1] in self.receivers)):
                    completed = True
                    self.passes += 1
                    self.length = (math.dist(caught_at, self.kick_point)
                                   if caught_at and self.kick_point else 0.0)
                    ### AI GENERATED (2026-11-13: a pass straight back)
                    returned = to[1] == self.return_to
                    self.returns += returned
                    ### END AI GENERATED
                ### END AI GENERATED
                self.kicker = None  # someone has the ball: no kick in play any more
        if in_play is not None and s is not None and s_next is not None:
            a, b = s["ball"]["pos"], s_next["ball"]["pos"]
            for blue in s_next["blue"]:
                # AI GENERATED (2026-11-10)
                if blue["id"] != in_play and self._a_target(blue["pos"]):
                    self.closest = min(self.closest, _segment_distance(blue["pos"], a, b))
        ### AI GENERATED (2026-11-13: a kick that has stopped loose is over for kicker_clearance)
        # hand-built test states may lack it
        vel = s_next["ball"].get("vel") if s_next is not None else None
        if (self.kicker is not None and vel is not None and not _anyone_holds(s_next)
                and math.hypot(*vel[:2]) < DEAD_BALL_SPEED):
            self.stopped = True
        credit, callers = self._calls(info, s, s_next)  # AI GENERATED (2026-11-13)
        return PassStep(in_play, completed, self.passes, self.closest, self.length, returned,
                        self.stopped and in_play is not None, credit, callers)
        ### END AI GENERATED

    ### AI GENERATED (2026-11-13: calls)
    def _calls(self, info: dict, s: dict | None, s_next: dict | None) -> tuple[float, int]:
        """This step's call credit (each blue pickup: the getter's call lead less the other blues')
        and the blues calling in s_next. A call's lead is min(1, steps called in a row / CALL_LEAD),
        counted up to s: what the teammates could see when they acted."""
        if s is None or s_next is None:
            return 0.0, 0
        if self.call_streaks is None:
            self.call_streaks = [int(bool(b.get("calling"))) for b in s["blue"]]
        credit = 0.0
        for event in info["events"]:
            to = event.get("to") if event["type"] == "possession_change" else None
            if to is not None and to[0] == "blue":
                lead = [min(1.0, k / CALL_LEAD) for k in self.call_streaks]
                right, wrong = lead[to[1]], sum(lead) - lead[to[1]]
                credit += right - wrong
                self.pickups += 1
                self.called += right
                self.miscalled += wrong
        calling = [bool(b.get("calling")) for b in s_next["blue"]]
        self.call_streaks = [k + 1 if c else 0 for k, c in zip(self.call_streaks, calling)]
        return credit, sum(calling)
    ### END AI GENERATED


def _segment_distance(p, a, b) -> float:
    """Distance from point p to the segment a-b (the ball's straight path over one step)."""
    ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


def _blue_holds(s: dict) -> bool:
    return any(b["has_possession"] for b in s["blue"])


def _anyone_holds(s: dict) -> bool:  # AI GENERATED (2026-11-13)
    return any(e["has_possession"] for e in s["blue"] + s.get("red", []))


def _ball_dead(s: dict) -> bool:  # AI GENERATED (2026-11-14): loose and stopped (DEAD_BALL_SPEED)
    vel = s.get("ball", {}).get("vel")  # hand-built test states may lack it: never dead
    return vel is not None and not _anyone_holds(s) and math.hypot(*vel[:2]) < DEAD_BALL_SPEED


def passing_drill(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    # AI GENERATED (2026-11-08): grows per catch
    return float(info["pass"].passes) if _blue_holds(s_next) else 0.0


def _teammate_distance(s: dict, kicker: int) -> float:
    """Distance (m) from the ball to the nearest blue other than the kicker."""
    ball = s["ball"]["pos"]
    return min(math.dist(b["pos"], ball) for b in s["blue"] if b["id"] != kicker)


def pass_progress(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    kicker = info["pass"].kicker
    if kicker is None or any(r["has_possession"] for r in s_next["red"]):
        return 0.0  # no kick in play, or red got it (the episode is over)
    return _teammate_distance(s, kicker) - _teammate_distance(s_next, kicker)


def pass_completed(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return 1.0 if info["pass"].completed else 0.0


### AI GENERATED (2026-11-11: the kicker keeps clear of its own kick)
# kicker_clearance: while a kick is in play, each step costs the kicker's closeness to the ball,
# 1 - distance / KICKER_CLEARANCE (0 from KICKER_CLEARANCE out, nearly 1 on top of it), and the
# kicker taking the ball back costs KICKER_REGRAB at once (in pass_practice it ends the episode, so
# the steps it saves would otherwise make a re-grab the cheap way out; in pass_practice_moving it
# ends the pass, 2026-11-11). So the kicker learns to kick firmly and
# leave the ball to its teammates instead of chasing it. A negative term with a positive coef,
# like every shaping coef; with coef c an episode spent on the ball costs about c per step.
# The kicker isn't marked in o_i: it has to tell from the ball running away from it.
KICKER_CLEARANCE = 1.5  # m, centre to centre
KICKER_REGRAB = 10.0  # a re-grab costs as much as 10 steps right on the ball
# 2026-11-13: once the kicked ball has slowed below DEAD_BALL_SPEED loose (it rolls ln(1.3) / 0.4
# = 0.66 m more at most), the kick is over for this term: no charge for standing near it or picking
# it back up. Before, a miss that stopped kept charging the kicker until somebody else fetched it,
# and the ball sat loose 40% of the time (a median 8 s, half the time to the end of the episode).
DEAD_BALL_SPEED = 0.3  # m/s


def kicker_clearance(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    kicker = info["pass"].kicker
    if kicker is None or info["pass"].stopped:  # AI GENERATED (2026-11-13: stopped)
        return 0.0  # no kick in play, or it has stopped: the kicker may fetch it
    me = next(b for b in s_next["blue"] if b["id"] == kicker)
    if me["has_possession"]:
        return -KICKER_REGRAB  # took its own kick back (or it rebounded straight to it)
    gap = math.dist(me["pos"][:2], s_next["ball"]["pos"][:2])
    return -max(0.0, 1.0 - gap / KICKER_CLEARANCE)
### END AI GENERATED


### AI GENERATED (2026-11-09: pass practice)
PASS_MISS_MAX = 0.5  # the most a missed kick can earn (a catch earns 1, or 0.75 to 1 by length)
PASS_MISS_SCALE = 1.0  # m: a miss's reward halves every ~0.7 m it passes wide of a teammate
### AI GENERATED (2026-11-10: longer catches pay a bit more, rl.reward.pass_length_bonus)
# With the bonus on, a catch pays PASS_CATCH_MIN + (1 - PASS_CATCH_MIN) * min(1, length /
# PASS_FULL_LENGTH): 0.81 for a 1 m pass, 1 from 4 m on. Off, every catch pays 1. Either way a
# catch beats any miss (PASS_MISS_MAX).
PASS_CATCH_MIN = 0.75
PASS_FULL_LENGTH = 4.0  # m
assert PASS_CATCH_MIN > PASS_MISS_MAX
### END AI GENERATED


def _catch_value(p: PassStep, cfg) -> float:
    """What the latest completed pass pays: 1, or 0.75 to 1 by its length with the bonus on."""
    ### AI GENERATED (2026-11-10: the length bonus)
    if not cfg.rl.reward.pass_length_bonus:
        return 1.0
    return PASS_CATCH_MIN + (1.0 - PASS_CATCH_MIN) * min(1.0, p.length / PASS_FULL_LENGTH)
    ### END AI GENERATED


def pass_practice(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    if not (effective_terminal or truncated):
        return 0.0  # paid once, when the kick has played out (or time ran out)
    p = info["pass"]
    if p.passes:
        return _catch_value(p, cfg)
    if math.isinf(p.closest):
        return 0.0  # never kicked
    gap = max(0.0, p.closest - cfg.shared.player_radius)
    return PASS_MISS_MAX * math.exp(-gap / PASS_MISS_SCALE)
### END AI GENERATED


### AI GENERATED (2026-11-11: pass practice to the time limit)
def pass_catch(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    p = info["pass"]
    if not p.completed:
        return 0.0
    # AI GENERATED (2026-11-13)
    return _catch_value(p, cfg) * (cfg.rl.reward.return_pass_factor if p.returned else 1.0)


# receiver_spacing: how close each pair of blues is to rl.reward.spacing_distance apart, averaged
# over pairs: each blue is paid for being in passing range of every teammate, holder included. A
# tent: 0 at 0 m, rising in a straight line to 1 at the target, back to 0 at twice the target, so
# every metre apart pays the same wherever the blues start (a Gaussian bump, until 2026-11-11, was
# flat for blues bunched 0.6 m apart, and they never spread out). Measured centre to centre; a
# pass between blues d apart travels about d - 0.8 m (kick point to catch point), so with a
# minimum pass distance the target has to be about 0.8 m more than the minimum.


def receiver_spacing(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    target = cfg.rl.reward.spacing_distance
    spots = [b["pos"][:2] for b in s_next["blue"]]
    ### AI GENERATED (2026-11-14: the blue fetching a dead ball isn't held to the spacing)
    if len(spots) >= 3 and _ball_dead(s_next):
        ball = s_next["ball"]["pos"][:2]
        del spots[min(range(len(spots)), key=lambda i: math.dist(spots[i], ball))]
    ### END AI GENERATED
    gaps = [math.dist(a, b) for i, a in enumerate(spots) for b in spots[i + 1:]]
    if not gaps:
        return 0.0  # a lone blue has nobody to space from
    return sum(max(0.0, 1.0 - abs(d - target) / target) for d in gaps) / len(gaps)
### END AI GENERATED


### AI GENERATED (2026-11-13: a "got it" call)
# call_accuracy: see the module docstring. With coef c, a pickup the getter called for at least
# CALL_LEAD steps earns c, each other blue's full-length call costs c, and calling costs
# c * CALL_COST a step (a whole episode of one blue calling: 3c, three right calls' worth).
CALL_LEAD = 10  # steps (0.5 s) of calling in a row for a call's full credit
CALL_COST = 0.01  # per blue calling, per step


def call_accuracy(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    p = info["pass"]
    return p.call_credit - CALL_COST * p.callers
### END AI GENERATED


### AI GENERATED (2026-11-14: the call head's own per-blue reward)
# A call at step t is scored by the next blue pickup in the CALL_LEAD steps after it (transitions
# t+1 .. t+CALL_LEAD: a call shows in s_{t+1}, so teammates see it from step t+1 on): +1 for the
# blue who got the ball, CALL_WRONG for each other blue, 0 when nobody got it in time, less
# CALL_STEP_COST for calling at all. So a full CALL_LEAD-step call by the getter sums to CALL_LEAD
# (like the team term's full credit, 1, in units of 1 / CALL_LEAD), and a blue calling the whole
# time pays CALL_STEP_COST every step it isn't about to get the ball.
# The penalties are small on purpose. With -1 and 0.1 (the team term's ratio) calling lost on
# average: most steps have no pickup coming, and near one two of three blues are wrong. The head
# learned "never call" (one shared bias) long before "call when I'm about to get it" (a function
# of the state), and once its chance of calling was near 0 the gradient vanished: claim_ball warm
# from go_to_target fell silent in 20k steps (call entropy 0.28 -> 0.02, the getter calling ~1%).
# With -0.5 and 0.02 the getter had called for 80-90% of pickups by 85k steps (mock, 2026-11-14).
CALL_RIGHT, CALL_WRONG = 1.0, -0.5
CALL_STEP_COST = 0.02


def blue_pickup(info: dict) -> int | None:
    """The blue who got the ball during this step (the first, if several did), or None."""
    for event in info["events"]:
        to = event.get("to") if event["type"] == "possession_change" else None
        if to is not None and to[0] == "blue":
            return to[1]
    return None


def call_values(pickups, episode_end, n_agents: int, lead: int = CALL_LEAD,
                cost: float = CALL_STEP_COST) -> tuple[np.ndarray, np.ndarray]:
    """What calling at each step would have been worth to each blue, in hindsight.

    ``pickups`` [T]: the blue who got the ball in transition t (-1: nobody). ``episode_end`` [T]:
    transition t ended its episode. Returns ``(q, valid)``, both [T, N]: q[t, i] is what a call by
    blue i at step t earns (see CALL_RIGHT; the same whether or not i called), and valid is False
    where the window runs past the end of the rollout with the episode still going (the next
    rollout holds the answer: those steps are left out of the call loss).
    """
    pickups = np.asarray(pickups, dtype=int)
    episode_end = np.asarray(episode_end, dtype=bool)
    T = len(pickups)
    q = np.zeros((T, n_agents))
    valid = np.ones((T, n_agents), dtype=bool)
    for t in range(T):
        for k in range(1, lead + 1):
            if episode_end[t + k - 1]:
                break  # the episode ended first: nobody picked it up in time
            if t + k >= T:
                valid[t] = False  # still going at the end of the rollout: unknown
                break
            getter = pickups[t + k]
            if getter >= 0:
                q[t] = CALL_WRONG
                q[t, getter] = CALL_RIGHT
                break
    return q - cost, valid


# claim (claim_ball's base): -1 a step until a blue has the ball.
def claim(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return 0.0 if _blue_holds(s_next) else -1.0


# call_yield: see the module docstring. With coef c, a blue standing on a ball a teammate called
# pays about c a step, and taking it costs YIELD_STEAL * c.
YIELD_DISTANCE = 1.5  # m, centre to centre (like KICKER_CLEARANCE)
YIELD_STEAL = 10.0  # like KICKER_REGRAB: as much as 10 steps right on the ball


def call_yield(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    calling = [bool(b.get("calling")) for b in s["blue"]]
    if not any(calling) or _anyone_holds(s):
        return 0.0  # nobody called, or the ball wasn't loose
    getter = blue_pickup(info)
    if getter is not None and not calling[getter]:
        return -YIELD_STEAL  # took a ball a teammate had called
    if _anyone_holds(s_next):
        return 0.0
    ball = s_next["ball"]["pos"][:2]
    return -sum(max(0.0, 1.0 - math.dist(b["pos"][:2], ball) / YIELD_DISTANCE)
                for b, called in zip(s_next["blue"], calling) if not called)


# dead_ball: see the module docstring. With coef c, a dead ball costs c a step until somebody
# picks it up (or 30c a second). In catch_moving_longer (~85 dead steps per miss, a catch worth
# ~0.9), c = 0.003 makes a miss cost ~0.25, so a kick is still worth taking at a 30% catch rate.
def dead_ball(s, actions, s_next, effective_terminal, truncated, info, cfg) -> float:
    return -1.0 if _ball_dead(s_next) else 0.0
### END AI GENERATED


# the terms that read info["pass"]
# AI GENERATED (2026-11-11; call_accuracy 2026-11-13)
PASS_TERMS = frozenset({"passing_drill", "pass_progress", "pass_completed", "pass_practice",
                        "kicker_clearance", "pass_catch", "call_accuracy"})
### END AI GENERATED


class RewardParts(NamedTuple):
    """One transition's team reward, split for logging."""

    base: float  # unscaled base term: the task's own reward
    shaping: float  # unscaled sum of coef * term over the enabled shaping terms
    total: float  # what the learner receives: (1 - gamma) * (base + shaping)


def _deferred(name: str, ref: str) -> Term:
    def term(*args: Any) -> float:
        raise NotImplementedError(f"reward term {name!r} is interface-only ({ref})")

    return term


BASE_TERMS: dict[str, Term] = {
    "survival": survival,
    "goto_target": goto_target,
    "passing_drill": passing_drill,  # AI GENERATED (2026-11-08)
    "pass_practice": pass_practice,  # AI GENERATED (2026-11-09)
    "pass_catch": pass_catch,  # AI GENERATED (2026-11-11)
    "claim": claim,  # AI GENERATED (2026-11-14)
    "goal": goal,  # AI GENERATED (2026-11-23)
}
SHAPING_TERMS: dict[str, Term] = {
    "pbrs_ball_acquisition": pbrs_ball_acquisition,
    "pass_progress": pass_progress,  # AI GENERATED (2026-11-08)
    "pass_completed": pass_completed,  # AI GENERATED (2026-11-08)
    "kicker_clearance": kicker_clearance,  # AI GENERATED (2026-11-11)
    "receiver_spacing": receiver_spacing,  # AI GENERATED (2026-11-11)
    "call_accuracy": call_accuracy,  # AI GENERATED (2026-11-13)
    "call_yield": call_yield,  # AI GENERATED (2026-11-14)
    "dead_ball": dead_ball,  # AI GENERATED (2026-11-14)
    "pbrs_ball_to_goal": pbrs_ball_to_goal,  # AI GENERATED (2026-11-23)
    "pbrs_red_distance": _deferred("pbrs_red_distance", "#15"),
}


class RewardModule:
    """Computes the scaled team reward for one transition."""

    def __init__(self, cfg: Config):
        rw = cfg.rl.reward
        if rw.base not in BASE_TERMS:
            raise ValueError(f"unknown base reward term {rw.base!r}; choose from {sorted(BASE_TERMS)}")
        self._base = BASE_TERMS[rw.base]
        self._shaping: list[tuple[Term, float]] = []
        self.needs_memory = rw.base in PASS_TERMS  # AI GENERATED (2026-11-08)
        self.call_coef = 0.0  # AI GENERATED (2026-11-14): > 0 when call_accuracy is on
        for t in rw.shaping:
            if t.name not in SHAPING_TERMS:
                raise ValueError(f"unknown shaping term {t.name!r}; choose from {sorted(SHAPING_TERMS)}")
            if not t.enabled:
                continue
            if t.name in SHAPING_TERMS_DEFERRED:
                raise NotImplementedError(f"shaping term {t.name!r} is interface-only (#15)")
            self.needs_memory |= t.name in PASS_TERMS  # AI GENERATED (2026-11-08)
            ### AI GENERATED (2026-11-14: the call's own reward, not the team's)
            if t.name == "call_accuracy":
                self.call_coef = t.coef  # weights the call head's own loss (see call_values)
                continue
            ### END AI GENERATED
            self._shaping.append((SHAPING_TERMS[t.name], t.coef))
        self.scale = 1.0 - cfg.gamma
        self._cfg = cfg

    ### AI GENERATED (2026-11-08)
    def new_episode(self) -> PassMemory:
        """Fresh per-episode memory, to pass to every ``parts`` call of that episode."""
        return PassMemory(min_distance=self._cfg.rl.reward.min_pass_distance,  # 2026-11-10
                          keeps_score=self._cfg.shared.scenario == "soccer")  # 2026-11-23
    ### END AI GENERATED

    def parts(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
        memory: PassMemory | None = None,  # AI GENERATED (2026-11-08)
    ) -> RewardParts:
        ### AI GENERATED (2026-11-08: the pass terms read info["pass"], built from the episode's
        ### memory)
        if memory is not None:
            # s, s_next: closest approach (2026-11-09)
            info = {**info, "pass": memory.update(info, s, s_next)}
        elif self.needs_memory:
            raise ValueError("this reward has pass terms: pass the episode's memory "
                             "(RewardModule.new_episode())")
        ### END AI GENERATED
        args = (s, actions, s_next, effective_terminal, truncated, info, self._cfg)
        base = self._base(*args)
        shaping = 0.0
        for term, coef in self._shaping:
            shaping += coef * term(*args)
        # Without shaping, base + 0.0 == base exactly: survival-only runs reproduce bit for bit.
        return RewardParts(base, shaping, self.scale * (base + shaping))

    def __call__(
        self,
        s: dict,
        actions: dict,
        s_next: dict,
        effective_terminal: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        return self.parts(s, actions, s_next, effective_terminal, truncated, info).total


# The config validates names against its own constants; keep the two in sync.
assert set(BASE_TERMS) == set(BASE_REWARD_TERMS)
assert set(SHAPING_TERMS) == set(SHAPING_TERMS_BUILT | SHAPING_TERMS_DEFERRED)
