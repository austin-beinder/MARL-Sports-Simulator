# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the featurizer (s -> actor obs +
#     critic inputs), start-type inference, and the rollout collector that
#     fills the buffer, including next_value = V(true s_{t+1}) computed before
#     any reset and episode records (walk-through step 2; design #12, #17,
#     #20, #23).
#   - 2026-10-11: Added mean_reward (unscaled reward per step) to episode
#     records, needed for goto_target, where every episode has the same length.
#   - 2026-10-14: The rollout now reports the mean shaping reward per step
#     (RolloutStats.mean_shaping), and an episode's mean_reward counts the base
#     (task) reward only, so shaping never leaks into it (PBRS ablation).
#   - 2026-11-07: Actions come from all of the actor's heads (kick heads too),
#     and the rollout counts kicks per step. (Austin Beinder)
#   - 2026-11-07: set_reset_options, so a curriculum can change the start
#     distribution between rollouts. (Austin Beinder)
#   - 2026-11-08: Each episode gets its own reward memory (the passing
#     drill's pass bookkeeping). (Austin Beinder)
#   - 2026-11-09: EpisodeRecord.passes, the passes completed in the episode.
#     (Austin Beinder)
#   - 2026-11-11: EpisodeRecord.kicks, the blue kicks in the episode.
#     (Austin Beinder)
#   - 2026-11-13: The featurizer's rl.passer_obs flags; EpisodeRecord.returns,
#     the passes straight back to the blue who passed to the kicker. (Austin Beinder)
#   - 2026-11-13: The call flags (rl.call_enabled); EpisodeRecord.pickups and
#     called / miscalled, for the call accuracy. (Austin Beinder)
#   - 2026-11-14: Each buffer row also gets the step's blue pickup and the call
#     head's log-prob, for the call head's own advantage. (Austin Beinder)
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE); the rollout collector moved here from step 8 because it owns the
#   next_value contract that GAE depends on.
# Human review:   Pending review
# ==============================================================================
"""Rollout collection: step the env with the current policy and fill the buffer.

Per control step t (walk-through step 2):

1. ``o_t``, critic input  <- featurize(s_t)               (RL-side, #11, #13)
2. samples, log-prob      <- actor.act(o_t)                (clamped Beta samples, #18)
   V(s_t)                 <- critic                        (reused from step t-1 when valid)
3. s_{t+1}, flags         <- env.step(actions)             (ICD §4; no auto-reset)
4. effective_terminal, team reward                         (#12, #14)
5. next_value = V(s_{t+1}), computed BEFORE any reset: the true final state at
   an episode end, the live state at the rollout end (#12).
6. Store the row. At an episode end, record the episode; the next step resets.

An episode can span two rollouts: the collector keeps the live state between
``collect`` calls. Rollout cuts are NOT episode ends and are never logged as
truncations (#12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from marl.common.config import Config
from marl.common.spawn import START_FREE_BALL, START_POSSESSION, START_TARGET
from marl.rl.actions import actions_from_samples, commanded_speed_fraction  # AI GENERATED (actions_from_samples)
from marl.rl.buffer import RolloutBuffer
from marl.rl.networks import CALL, MOVE, Actor, Critic, critic_input, critic_input_dim  # AI GENERATED (CALL)
from marl.rl.obs import ObsBuilder
from marl.rl.reward import RewardModule, blue_pickup, effective_terminal  # AI GENERATED (blue_pickup)
from marl.rl.state import StateFlattener, parse_state


class Featurizer:
    """s -> (actor observations [N, obs_dim], critic inputs [N, critic_dim])."""

    def __init__(self, cfg: Config):
        sc = cfg.shared
        self.n_blue, self.n_red = sc.n_blue, sc.n_red
        self.obs_builder = ObsBuilder(sc, cfg.rl.obs_ordering, cfg.rl.obs_mode, cfg.rl.passer_obs,  # AI GENERATED (passer_obs,
                                      cfg.rl.call_enabled)  # call flags, 2026-11-13)
        self.flattener = StateFlattener(sc, cfg.rl.horizon_mode, cfg.rl.passer_obs, cfg.rl.call_enabled)  # AI GENERATED (flags)
        self.critic_mode = cfg.rl.critic_input
        self.obs_dim = self.obs_builder.dim
        self.state_dim = self.flattener.dim
        self.critic_dim = critic_input_dim(self.critic_mode, self.obs_dim, self.state_dim)

    def __call__(self, s: dict) -> tuple[np.ndarray, np.ndarray]:
        st = parse_state(s, self.n_blue, self.n_red)
        obs = self.obs_builder(st)
        return obs, critic_input(self.critic_mode, obs, self.flattener(st))


def start_type_of(s: dict, scenario: str) -> str:
    """Infer the start type from the reset state (no ICD change needed, decision F)."""
    if scenario == "goto_target":
        return START_TARGET
    return START_POSSESSION if any(b["has_possession"] for b in s["blue"]) else START_FREE_BALL


@dataclass(frozen=True)
class EpisodeRecord:
    length: int  # control steps (s["step"] at the end): the evaluation metric (#23)
    terminated: bool
    truncated: bool
    start_type: str
    # BASE (task) reward per step, without the (1 - gamma) scale and without any
    # shaping: 1 for survival, minus the mean blue-to-target distance (m) for
    # goto_target, where length is always the max.
    mean_reward: float
    passes: int = 0  # AI GENERATED (2026-11-09): passes completed (a blue's kick caught by another blue)
    kicks: int = 0  # AI GENERATED (2026-11-11): blue kicks (passes / kicks = the pass completion rate)
    returns: int = 0  # AI GENERATED (2026-11-13): passes straight back to the blue who passed to the kicker
    ### AI GENERATED (2026-11-13: calls)
    pickups: int = 0  # blue pickups of the ball (catches, fetching a loose ball, re-grabs)
    called: float = 0.0  # how well the blue who picked it up had called for it, summed (0 to 1 each)
    miscalled: float = 0.0  # how loudly the other blues had called for those pickups, summed
    ### END AI GENERATED


@dataclass(frozen=True)
class RolloutStats:
    episodes: list[EpisodeRecord]  # episodes that ENDED during this rollout
    mean_commanded_speed: float  # KL-1 symptom metric (#23)
    mean_shaping: float  # unscaled shaping reward per step (0.0 without shaping terms)
    kicks_per_step: float = 0.0  # kicks the env applied, per step, summed over agents  # AI GENERATED


def make_buffer(cfg: Config, featurizer: Featurizer, actor: Actor) -> RolloutBuffer:
    head_dims = {name: head.action_dim for name, head in actor.heads.items()}
    return RolloutBuffer(cfg.rl.ppo.rollout_steps, cfg.shared.n_blue, featurizer.obs_dim, featurizer.critic_dim, head_dims)


class RolloutCollector:
    def __init__(
        self,
        env: Any,
        cfg: Config,
        actor: Actor,
        critic: Critic,
        reward_fn: RewardModule,
        seed_rng: np.random.Generator,
    ):
        self.env, self.cfg = env, cfg
        self.actor, self.critic, self.reward_fn = actor, critic, reward_fn
        self.featurize = Featurizer(cfg)
        self._seed_rng = seed_rng  # draws a fresh env seed for every reset (training stream)
        self._reset_options = cfg.shared.start.to_reset_options()
        self._s: dict | None = None  # live state; None -> reset at the next step
        self._value: torch.Tensor | None = None  # cached V(s_t)

    ### AI GENERATED
    def set_reset_options(self, options: dict[str, Any]) -> None:
        """Start options for episodes that begin from now on (a curriculum changes them between
        rollouts). An episode already under way keeps the start it had."""
        self._reset_options = dict(options)
    ### END AI GENERATED

    def collect(self, buf: RolloutBuffer) -> RolloutStats:
        buf.reset()
        # The critic changed during the last update: a V(s_t) cached with the old
        # weights must not leak into this rollout's first row.
        self._value = None
        episodes: list[EpisodeRecord] = []
        speed = shaping = 0.0
        kicks = 0  # AI GENERATED
        for _ in range(buf.n_steps):
            if self._s is None:
                self._begin_episode()
            s, obs, cin = self._s, self._obs, self._cin
            with torch.no_grad():
                samples, log_prob = self.actor.act(torch.from_numpy(obs))
                value = self._value if self._value is not None else self.critic(torch.from_numpy(cin))
            ### AI GENERATED (2026-11-14: the call's part of the log-prob, for its own advantage)
            call_lp = (self.actor.head_log_prob(torch.from_numpy(obs), samples, CALL).numpy()
                       if CALL in samples else None)
            ### END AI GENERATED
            move = samples[MOVE].numpy()
            actions = actions_from_samples(samples)  # AI GENERATED (kick heads too)
            speed += float(commanded_speed_fraction(move).mean())

            s_next, terminated, truncated, info = self.env.step(actions)
            kicks += sum(info["kick_applied"])  # AI GENERATED
            eff = effective_terminal(terminated, truncated, self.cfg.rl.horizon_mode)
            parts = self.reward_fn.parts(s, actions, s_next, eff, truncated, info, self._reward_memory)  # AI GENERATED (memory)
            shaping += parts.shaping
            self._ep_base += parts.base
            obs_next, cin_next = self.featurize(s_next)
            with torch.no_grad():
                # V(true s_{t+1}), BEFORE any reset (#12): at an episode end this is the
                # final state; at the rollout end it is the live state.
                next_value = self.critic(torch.from_numpy(cin_next))

            buf.add(
                obs=obs,
                critic_in=cin,
                samples={h: x.numpy() for h, x in samples.items()},
                log_prob=log_prob.numpy(),
                value=value.numpy(),
                next_value=next_value.numpy(),
                reward=parts.total,
                terminated=terminated,
                truncated=truncated,
                effective_terminal=eff,
                pickup=blue_pickup(info),  # AI GENERATED (2026-11-14)
                call_log_prob=call_lp,  # AI GENERATED (2026-11-14)
            )
            if terminated or truncated:
                episodes.append(
                    EpisodeRecord(
                        length=s_next["step"],
                        terminated=terminated,
                        truncated=truncated,
                        start_type=self._start_type,
                        mean_reward=self._ep_base / s_next["step"],
                        passes=self._reward_memory.passes,  # AI GENERATED (2026-11-09)
                        kicks=self._reward_memory.kicks,  # AI GENERATED (2026-11-11)
                        returns=self._reward_memory.returns,  # AI GENERATED (2026-11-13)
                        pickups=self._reward_memory.pickups,  # AI GENERATED (2026-11-13)
                        called=self._reward_memory.called,  # AI GENERATED (2026-11-13)
                        miscalled=self._reward_memory.miscalled,  # AI GENERATED (2026-11-13)
                    )
                )
                self._s = None  # no auto-reset (ICD §4): the next step starts a new episode
            else:
                self._s, self._obs, self._cin = s_next, obs_next, cin_next
                self._value = next_value  # V(s_{t+1}) is the next step's V(s_t)
        return RolloutStats(episodes, speed / buf.n_steps, shaping / buf.n_steps, kicks / buf.n_steps)  # AI GENERATED (kicks)

    def _begin_episode(self) -> None:
        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        self._s, _ = self.env.reset(seed=seed, options=self._reset_options)
        self._start_type = start_type_of(self._s, self.cfg.shared.scenario)
        self._ep_base = 0.0  # unscaled base rewards summed over the episode (it may span rollouts)
        self._reward_memory = self.reward_fn.new_episode()  # AI GENERATED (2026-11-08): pass bookkeeping
        self._obs, self._cin = self.featurize(self._s)
        self._value = None
