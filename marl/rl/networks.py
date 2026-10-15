# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the shared actor (tanh MLP trunk
#     + dict of heads; Beta movement head with clamped samples, log-probs from
#     the stored sample, analytic entropy) and the critic with global/local
#     inputs (design #1, #4, #5, #9, #10, #18, #21; PPO defaults D).
#   - 2026-10-11: Added seeded_networks, the one seeded init shared by training
#     and by the evaluation script's "untrained policy" baseline.
# Prompt summary: Implement Phase 3 step 6 of the issue #7 plan (actor/critic
#   + Beta movement head + action adapter).
# Human review:   Pending review
# ==============================================================================
"""Actor and critic networks.

Actor (#4, #9, #21): ONE actor shared by all blue agents (parameter sharing),
built as a tanh MLP trunk + a dict of heads. Milestone 1 builds only the
movement head. The kick-vector and trigger heads join later behind
``kick_enabled``; a movement checkpoint warm-starts them via
``load_state_dict(strict=False)``, so the new heads keep their fresh init.

Movement head (#18): an independent Beta per component (mx, my), with
alpha, beta = softplus(.) + 1 >= 1, so every density is unimodal (no U-shape
that piles probability at the edges of the action range).

* Samples are clamped to [eps, 1 - eps]. The CLAMPED [0, 1] sample is what the
  buffer stores, and log-probs are always computed from it, the same way at
  rollout and at update, so the PPO ratio starts at exactly 1. (A sample at
  exactly 0 or 1 gives log_prob = -inf, and the ratio becomes NaN.)
* The env-facing map 2x - 1 lives in ``marl/rl/actions.py``. Its Jacobian adds a
  constant -log 2 per dimension to the log-prob; that cancels in the ratio, so
  log-probs are taken in [0, 1] space.
* Entropy is ANALYTIC (``Beta.entropy()``, from the parameters), never
  -log pi(stored sample): the latter's expected gradient is zero (#10).

Critic: V of the state only. It never sees the action, so it is V(s), not Q(s, a).
``critic_input = global`` feeds the flattened global state (MAPPO); ``local``
feeds each agent's own o_i (IPPO-style). Actor and critic share NO parameters
(decision D), so each gets its own optimizer and gradient clipping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from marl.common.config import Config

MOVE = "move"  # head name (matches the ICD §6 action key)
CRITIC_INPUTS = ("global", "local")


def layer_init(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    """Orthogonal weights and zero bias (the CleanRL / MAPPO convention)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden_sizes:
        layers += [layer_init(nn.Linear(in_dim, h)), nn.Tanh()]
        in_dim = h
    return nn.Sequential(*layers)


class BetaHead(nn.Module):
    """An independent Beta(alpha, beta) per action component; alpha, beta = softplus(.) + 1."""

    def __init__(self, in_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        # Gain 0.01 -> outputs ~0 -> alpha = beta ~ softplus(0) + 1 = 1 + ln 2 ~ 1.69:
        # a wide distribution centered on 0.5, i.e. "stand still on average".
        self.linear = layer_init(nn.Linear(in_dim, 2 * action_dim), gain=0.01)

    def forward(self, h: torch.Tensor) -> Beta:
        alpha_raw, beta_raw = self.linear(h).chunk(2, dim=-1)
        # Beta validates its parameters by default, so a NaN fails loudly here.
        return Beta(F.softplus(alpha_raw) + 1.0, F.softplus(beta_raw) + 1.0)


class Actor(nn.Module):
    """Shared policy for all blue agents: obs [B, obs_dim] -> per-head distributions."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (64, 64), beta_eps: float = 1e-6):
        super().__init__()
        if not 0.0 < beta_eps < 0.5:
            raise ValueError(f"beta_eps must be in (0, 0.5), got {beta_eps}")
        self.beta_eps = beta_eps
        self.trunk = mlp(obs_dim, hidden_sizes)
        self.heads = nn.ModuleDict({MOVE: BetaHead(hidden_sizes[-1], action_dim=2)})

    def distributions(self, obs: torch.Tensor) -> dict[str, Beta]:
        h = self.trunk(obs)
        return {name: head(h) for name, head in self.heads.items()}

    # no_grad, not inference_mode: inference tensors can't be saved for backward,
    # and the stored samples are reused in evaluate() during the update.
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Choose actions for a batch of observations (rollout; no gradients).

        Returns the clamped [0, 1] samples per head (exactly what the buffer
        stores) and their joint log-prob [B]. ``deterministic`` uses the Beta
        mean (the secondary eval diagnostic, #23).
        """
        dists = self.distributions(obs)
        samples = {}
        for name, d in dists.items():
            x = d.mean if deterministic else d.sample()
            samples[name] = x.clamp(self.beta_eps, 1.0 - self.beta_eps)
        return samples, _joint_log_prob(dists, samples)

    def evaluate(
        self, obs: torch.Tensor, samples: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Re-score stored samples under the current policy (PPO update; with gradients).

        Returns the joint log-prob [B], the analytic entropy [B] (per state; it
        doesn't depend on the samples) and the per-head entropies (for logging).
        """
        dists = self.distributions(obs)
        head_entropy = {name: d.entropy().sum(-1) for name, d in dists.items()}
        entropy = torch.stack(list(head_entropy.values())).sum(0)
        return _joint_log_prob(dists, samples), entropy, head_entropy


def _joint_log_prob(dists: dict[str, Beta], samples: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of log-probs over every head and component (independent factors).

    One joint log-prob per (t, i) -> one PPO ratio per (t, i) (#20). Milestone 1:
    the movement head is always counted; the kick-head masks (#10) join here.
    """
    if set(samples) != set(dists):
        raise ValueError(f"samples have heads {sorted(samples)}, the actor has {sorted(dists)}")
    return torch.stack([dists[name].log_prob(samples[name]).sum(-1) for name in dists]).sum(0)


class Critic(nn.Module):
    """State-value function: input rows [..., D] -> V [...]. No action input."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (64, 64)):
        super().__init__()
        self.body = mlp(input_dim, hidden_sizes)
        self.v = layer_init(nn.Linear(hidden_sizes[-1], 1), gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.body(x)).squeeze(-1)


def critic_input_dim(mode: str, obs_dim: int, state_dim: int) -> int:
    if mode == "global":
        return state_dim
    if mode == "local":
        return obs_dim
    raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}, got {mode!r}")


def critic_input(mode: str, obs: np.ndarray, state_flat: np.ndarray) -> np.ndarray:
    """Per-agent critic input rows [N, D] (#1).

    ``global``: the flattened s, repeated for every agent (so all agents get the
    same V, #5); ``local``: each agent's own o_i.
    """
    if mode == "global":
        return np.repeat(state_flat[None, :], len(obs), axis=0)
    if mode == "local":
        return obs
    raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}, got {mode!r}")


def build_networks(cfg: Config, obs_dim: int, state_dim: int) -> tuple[Actor, Critic]:
    """Actor + critic sized from the config. Seed torch first for a reproducible init."""
    hidden = cfg.rl.ppo.hidden_sizes
    actor = Actor(obs_dim, hidden, cfg.rl.beta_eps)
    critic = Critic(critic_input_dim(cfg.rl.critic_input, obs_dim, state_dim), hidden)
    return actor, critic


def seeded_networks(cfg: Config, obs_dim: int, state_dim: int) -> tuple[Actor, Critic]:
    """THE initialization of a run: seed torch with ``run.seed``, then build. Training
    uses it, and the evaluation script uses it to rebuild the untrained policy exactly."""
    torch.manual_seed(cfg.run.seed)
    return build_networks(cfg, obs_dim, state_dim)
