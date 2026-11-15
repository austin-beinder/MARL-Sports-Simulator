# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the fixed-size rollout buffer
#     (per-agent [T, N] + shared [T] layout, design #20), advantage
#     computation via compute_gae, and shuffled minibatches over the
#     flattened T*N samples.
#   - 2026-11-14: The call head's own per-blue advantage: each step's blue
#     pickup and call log-prob, compute_call_advantages, and the call fields
#     of a Minibatch. (Austin Beinder)
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE) with hand-built boundary tests.
# Human review:   Pending review
# ==============================================================================
"""Rollout buffer (design #20).

Layout for one rollout of T control steps and N blue agents:

* per agent, [T, N, ...]: observation o_i, critic input, clamped samples per
  head, log-prob, V(s_t), next_value = V(true s_{t+1}); advantage and return
  after ``compute_advantages``.
* shared, [T]: team reward, terminated, truncated, effective_terminal.

The critic input is stored exactly as the critic saw it during the rollout (in
``global`` mode, the flattened s repeated per agent; #20 notes the duplicated
per-(t, i) V is equivalent and cheap).

The call head (2026-11-14, when call_accuracy is on) has its own advantage instead of the team's:
``compute_call_advantages`` scores each blue's call in hindsight from the pickups
(``reward.call_values``: q = what calling was worth) as A = q * (a - p), a = 1 if the blue called
and p = the rollout policy's chance of calling. That is Q(a) - V exactly, if the call doesn't change
who gets the ball: calling is worth q and not calling 0, and the policy expects p * q. So every step
pushes the call probability the right way, called or not.

For the update, the T x N samples are flattened row-major (index t * N + i) and
shuffled: every (t, i) is one independent sample with its OWN PPO ratio (#20).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from marl.rl.gae import compute_gae
from marl.rl.reward import call_values  # AI GENERATED (2026-11-14)


@dataclass(frozen=True)
class Minibatch:
    obs: torch.Tensor  # [B, obs_dim]
    critic_in: torch.Tensor  # [B, critic_dim]
    samples: dict[str, torch.Tensor]  # head -> [B, k], clamped [0, 1] samples
    log_probs: torch.Tensor  # [B], from the rollout policy (constants in the ratio)
    values: torch.Tensor  # [B], from the rollout critic
    advantages: torch.Tensor  # [B], RAW (normalized per minibatch in the loss, #20)
    returns: torch.Tensor  # [B], critic targets
    ### AI GENERATED (2026-11-14: the call head's own advantage; zeros without one)
    call_log_probs: torch.Tensor  # [B], the call head's part of log_probs
    call_advantages: torch.Tensor  # [B], RAW, not normalized (see compute_call_advantages)
    call_valid: torch.Tensor  # [B], 1.0 where the call advantage is known
    ### END AI GENERATED


class RolloutBuffer:
    def __init__(self, n_steps: int, n_agents: int, obs_dim: int, critic_dim: int, head_dims: dict[str, int]):
        T, N = n_steps, n_agents
        self.n_steps, self.n_agents = T, N
        self.obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        self.critic_in = np.zeros((T, N, critic_dim), dtype=np.float32)
        self.samples = {h: np.zeros((T, N, k), dtype=np.float32) for h, k in head_dims.items()}
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.next_values = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float64)
        self.terminated = np.zeros(T, dtype=bool)
        self.truncated = np.zeros(T, dtype=bool)
        self.effective_terminal = np.zeros(T, dtype=bool)
        ### AI GENERATED (2026-11-14: the call head's own advantage)
        self.pickups = np.full(T, -1, dtype=np.int64)  # the blue who got the ball in step t (-1: nobody)
        self.call_log_probs = np.zeros((T, N), dtype=np.float32)
        self.call_advantages = np.zeros((T, N), dtype=np.float32)
        self.call_valid = np.zeros((T, N), dtype=np.float32)
        ### END AI GENERATED
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._advantages_ready = False

    @property
    def full(self) -> bool:
        return self.t == self.n_steps

    def add(
        self,
        *,
        obs: Any,
        critic_in: Any,
        samples: dict[str, Any],
        log_prob: Any,
        value: Any,
        next_value: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        effective_terminal: bool,
        pickup: int | None = None,  # AI GENERATED (2026-11-14): the blue who got the ball this step
        call_log_prob: Any = None,  # AI GENERATED (2026-11-14): [N], the call head's part of log_prob
    ) -> None:
        """Store one control step (all N agents)."""
        if self.full:
            raise RuntimeError("the buffer is full; call reset() before adding more steps")
        if set(samples) != set(self.samples):
            raise ValueError(f"samples have heads {sorted(samples)}, the buffer has {sorted(self.samples)}")
        if effective_terminal and not (terminated or truncated):
            raise ValueError("effective_terminal without an episode end")
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        t = self.t
        self.obs[t] = _row(obs, self.obs.shape[1:], "obs")
        self.critic_in[t] = _row(critic_in, self.critic_in.shape[1:], "critic_in")
        for h, x in samples.items():
            self.samples[h][t] = _row(x, self.samples[h].shape[1:], f"samples[{h!r}]")
        self.log_probs[t] = _row(log_prob, (self.n_agents,), "log_prob")
        self.values[t] = _row(value, (self.n_agents,), "value")
        self.next_values[t] = _row(next_value, (self.n_agents,), "next_value")
        self.rewards[t] = reward
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.effective_terminal[t] = effective_terminal
        ### AI GENERATED (2026-11-14)
        self.pickups[t] = -1 if pickup is None else pickup
        self.call_log_probs[t] = 0.0 if call_log_prob is None else _row(call_log_prob, (self.n_agents,), "call_log_prob")
        ### END AI GENERATED
        self.t += 1

    def compute_advantages(self, gamma: float, lam: float) -> None:
        if not self.full:
            raise RuntimeError(f"the buffer holds {self.t} of {self.n_steps} steps; fill it first")
        adv, ret = compute_gae(
            self.rewards,
            self.values,
            self.next_values,
            self.effective_terminal,
            self.terminated | self.truncated,  # chain cut at ANY episode end
            gamma,
            lam,
        )
        self.advantages[:] = adv
        self.returns[:] = ret
        self.call_advantages[:] = 0.0  # AI GENERATED (2026-11-14): until compute_call_advantages
        self.call_valid[:] = 0.0  # AI GENERATED
        self._advantages_ready = True

    ### AI GENERATED (2026-11-14)
    def compute_call_advantages(self) -> None:
        """The call head's own per-blue advantage, A = q * (a - p) (see the module docstring).
        Call after ``compute_advantages``."""
        if not self._advantages_ready:
            raise RuntimeError("call compute_advantages() first")
        if "call" not in self.samples:
            raise RuntimeError("the buffer has no call head")
        q, valid = call_values(self.pickups, self.terminated | self.truncated, self.n_agents)
        called = self.samples["call"][..., 0]
        p_taken = np.exp(self.call_log_probs.astype(np.float64))  # the chance of the call actually made
        p_call = np.where(called > 0.5, p_taken, 1.0 - p_taken)
        self.call_advantages[:] = q * (called - p_call)
        self.call_valid[:] = valid
    ### END AI GENERATED

    def minibatches(self, num_minibatches: int, rng: np.random.Generator) -> Iterator[Minibatch]:
        """One epoch: a fresh shuffle of all T*N samples, split into equal minibatches."""
        if not self._advantages_ready:
            raise RuntimeError("call compute_advantages() before iterating minibatches")
        n = self.n_steps * self.n_agents
        if num_minibatches < 1 or n % num_minibatches:
            raise ValueError(f"T * N = {n} is not divisible by num_minibatches = {num_minibatches}")
        size = n // num_minibatches

        def flat(a: np.ndarray) -> torch.Tensor:  # [T, N, ...] -> [T*N, ...], row t*N + i
            return torch.from_numpy(a.reshape(n, *a.shape[2:]))

        obs, critic_in = flat(self.obs), flat(self.critic_in)
        samples = {h: flat(x) for h, x in self.samples.items()}
        log_probs, values = flat(self.log_probs), flat(self.values)
        advantages, returns = flat(self.advantages), flat(self.returns)
        call_lp, call_adv, call_valid = flat(self.call_log_probs), flat(self.call_advantages), flat(self.call_valid)  # AI GENERATED
        perm = torch.from_numpy(rng.permutation(n))
        for k in range(num_minibatches):
            j = perm[k * size : (k + 1) * size]
            yield Minibatch(
                obs=obs[j],
                critic_in=critic_in[j],
                samples={h: x[j] for h, x in samples.items()},
                log_probs=log_probs[j],
                values=values[j],
                advantages=advantages[j],
                returns=returns[j],
                call_log_probs=call_lp[j],  # AI GENERATED (2026-11-14)
                call_advantages=call_adv[j],  # AI GENERATED
                call_valid=call_valid[j],  # AI GENERATED
            )


def _row(x: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a
