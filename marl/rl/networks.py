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
#   - 2026-11-07: The kick heads behind kick_enabled: a Bernoulli kick_trigger
#     and a Beta kick_vec, masked by possession (read from o_i) and by the
#     trigger. (Austin Beinder)
#   - 2026-11-08: move_enabled=False (the passing drill, where the blues
#     can't move) masks the movement head out of the log-prob and entropy.
#     (Austin Beinder)
#   - 2026-11-09: The same for pass_practice (config.FROZEN_SCENARIOS).
#     (Austin Beinder)
#   - 2026-11-13: The call head behind rl.call_enabled: a Bernoulli, never
#     masked (teammates always see the call). (Austin Beinder)
#   - 2026-11-14: Actor.evaluate_split and Actor.head_log_prob, so PPO can
#     give the call head its own per-blue advantage. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
# ==============================================================================
"""Actor and critic networks.

Actor (#4, #9, #21): ONE actor shared by all blue agents (parameter sharing),
built as a tanh MLP trunk + a dict of heads. The movement head is always there.
``kick_enabled`` adds the kick heads (2026-11-07); a movement checkpoint
warm-starts them via ``load_state_dict(strict=False)``, so the new heads keep
their fresh init.

Kick heads (ICD §6):

* ``kick_trigger``: a Bernoulli. Its bias starts at logit(KICK_PRIOR), so an
  untrained holder kicks about once a second (20 Hz control) instead of every
  other step.
* ``kick_vec``: an independent Beta per component, like the movement head,
  mapped the same way (2x - 1).
* Masks (#10): the trigger only matters when the agent holds the ball, and the
  kick vector only when it also kicks. A head that can't affect the env adds
  nothing to the joint log-prob (and its entropy isn't rewarded), so it adds no
  noise to the PPO ratio. Possession is read from o_i (``OWN_POSSESSION``),
  which is the possession at the START of the step, the same one the env uses
  to decide whether a kick happens (ICD §3).

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
# AI GENERATED (Bernoulli, Distribution)
from torch.distributions import Bernoulli, Beta, Distribution
from torch.nn import functional as F

from marl.common.config import FROZEN_SCENARIOS, Config  # AI GENERATED (FROZEN_SCENARIOS)
from marl.rl.obs import OWN_POSSESSION  # AI GENERATED

MOVE = "move"  # head name (matches the ICD §6 action key)
### AI GENERATED
KICK_TRIGGER = "kick_trigger"  # head names = the ICD §6 action keys
KICK_VEC = "kick_vec"
KICK_PRIOR = 0.05  # the untrained policy's chance of kicking on a step where it holds the ball
# 2026-11-13: a "got it" call (rl.call_enabled). The env echoes it to the teammates' next
# observations, so it always matters: no mask, and its entropy always counts.
CALL = "call"
CALL_PRIOR = 0.1  # the untrained policy's chance of calling on a step
### END AI GENERATED
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


### AI GENERATED
class BernoulliHead(nn.Module):
    """One yes/no decision (the kick trigger). Samples are 0.0 or 1.0, shape [B, 1]."""

    def __init__(self, in_dim: int, prior: float):
        super().__init__()
        self.action_dim = 1
        # Small weights + bias = logit(prior): the untrained head says "yes" with probability ~prior
        self.linear = layer_init(nn.Linear(in_dim, 1), gain=0.01)
        nn.init.constant_(self.linear.bias, math.log(prior / (1.0 - prior)))

    def forward(self, h: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.linear(h))
### END AI GENERATED


class Actor(nn.Module):
    """Shared policy for all blue agents: obs [B, obs_dim] -> per-head distributions."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        beta_eps: float = 1e-6,
        kick_enabled: bool = False,  # AI GENERATED
        move_enabled: bool = True,  # AI GENERATED (2026-11-08)
        call_enabled: bool = False,  # AI GENERATED (2026-11-13)
    ):
        super().__init__()
        if not 0.0 < beta_eps < 0.5:
            raise ValueError(f"beta_eps must be in (0, 0.5), got {beta_eps}")
        self.beta_eps = beta_eps
        self.trunk = mlp(obs_dim, hidden_sizes)
        self.heads = nn.ModuleDict({MOVE: BetaHead(hidden_sizes[-1], action_dim=2)})
        ### AI GENERATED
        self.kick_enabled = kick_enabled
        # False when the env ignores move (passing_drill, pass_practice): the move head still
        # samples, but adds nothing to the log-prob or entropy, so it neither learns noise nor
        # drifts (2026-11-08).
        # A flag, not a weight: checkpoints load the same either way.
        self.move_enabled = move_enabled
        if kick_enabled:
            self.heads[KICK_TRIGGER] = BernoulliHead(hidden_sizes[-1], KICK_PRIOR)
            self.heads[KICK_VEC] = BetaHead(hidden_sizes[-1], action_dim=2)
        self.call_enabled = call_enabled  # 2026-11-13
        if call_enabled:
            self.heads[CALL] = BernoulliHead(hidden_sizes[-1], CALL_PRIOR)
        ### END AI GENERATED

    # AI GENERATED (Distribution)
    def distributions(self, obs: torch.Tensor) -> dict[str, Distribution]:
        h = self.trunk(obs)
        return {name: head(h) for name, head in self.heads.items()}

    # no_grad, not inference_mode: inference tensors can't be saved for backward,
    # and the stored samples are reused in evaluate() during the update.
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Choose actions for a batch of observations (rollout; no gradients).

        Returns the clamped [0, 1] samples per head (exactly what the buffer
        stores) and their joint log-prob [B]. ``deterministic`` uses the Beta
        mean (the secondary eval diagnostic, #23) and the more likely trigger value.
        """
        dists = self.distributions(obs)
        samples = {}
        for name, d in dists.items():
            ### AI GENERATED (the trigger is 0/1, not a clamped Beta sample)
            if isinstance(d, Bernoulli):
                samples[name] = (d.probs > 0.5).float() if deterministic else d.sample()
                continue
            ### END AI GENERATED
            x = d.mean if deterministic else d.sample()
            samples[name] = x.clamp(self.beta_eps, 1.0 - self.beta_eps)
        # AI GENERATED (masks)
        return samples, _joint_log_prob(dists, samples,
                                        _head_masks(obs, samples, self.move_enabled))

    def evaluate(
        self, obs: torch.Tensor, samples: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Re-score stored samples under the current policy (PPO update; with gradients).

        Returns the joint log-prob [B], the analytic entropy [B] (per state; it
        doesn't depend on the samples) and the per-head entropies (for logging).
        The kick heads' entropies count only in states where the agent holds the ball.
        """
        # AI GENERATED (2026-11-14: shared with evaluate_split)
        return self._score(obs, samples, self.distributions(obs))

    def _score(
        self, obs: torch.Tensor, samples: dict[str, torch.Tensor], dists: dict[str, Distribution]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        ### AI GENERATED (kick-head entropy gated by possession, #10)
        holds = (obs[..., OWN_POSSESSION] > 0.5).float()
        # 2026-11-08: a frozen blue's move head has no entropy bonus
        moves = 1.0 if self.move_enabled else 0.0
        # 2026-11-13: the call always counts; the kick heads only when holding
        weight = {MOVE: moves, CALL: 1.0}
        head_entropy = {name: d.entropy().sum(-1) * weight.get(name, holds)
                        for name, d in dists.items()}
        ### END AI GENERATED
        entropy = torch.stack(list(head_entropy.values())).sum(0)
        # AI GENERATED (masks)
        return (_joint_log_prob(dists, samples, _head_masks(obs, samples, self.move_enabled)),
                entropy, head_entropy)

    ### AI GENERATED (2026-11-14: the call head's own advantage)
    def evaluate_split(
        self, obs: torch.Tensor, samples: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Like ``evaluate``, but the log-prob comes in two parts: every head except the call [B],
        and the call head alone [B] (zeros without one). Their sum is ``evaluate``'s log-prob."""
        dists = self.distributions(obs)
        logp, entropy, head_entropy = self._score(obs, samples, dists)
        if CALL not in samples:
            return logp, torch.zeros_like(logp), entropy, head_entropy
        call = dists[CALL].log_prob(samples[CALL]).sum(-1)
        return logp - call, call, entropy, head_entropy

    @torch.no_grad()
    def head_log_prob(
        self, obs: torch.Tensor, samples: dict[str, torch.Tensor], head: str
    ) -> torch.Tensor:
        """One head's log-prob [B] of stored samples, unmasked (the rollout stores the call's)."""
        return self.distributions(obs)[head].log_prob(samples[head]).sum(-1)
    ### END AI GENERATED


### AI GENERATED
def _head_masks(
    obs: torch.Tensor, samples: dict[str, torch.Tensor], move_enabled: bool = True
) -> dict[str, torch.Tensor]:
    """Per-head [B] masks: 1 where the head's sample can affect the env (#10).

    The trigger matters only if the agent holds the ball at the start of the step, and the kick
    vector only if it also kicks. The movement head always matters, unless the env ignores it
    (``move_enabled`` False, the passing drill, 2026-11-08).
    """
    masks = {} if move_enabled else {MOVE: torch.zeros(obs.shape[:-1])}
    if KICK_TRIGGER not in samples:
        return masks
    holds = (obs[..., OWN_POSSESSION] > 0.5).float()
    return {**masks, KICK_TRIGGER: holds, KICK_VEC: holds * samples[KICK_TRIGGER].squeeze(-1)}


def _joint_log_prob(
    dists: dict[str, Distribution], samples: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Sum of log-probs over every head and component (independent factors).

    One joint log-prob per (t, i) -> one PPO ratio per (t, i) (#20). A masked head (#10) adds 0
    where it can't affect the env: the kick heads without the ball, the movement head in the
    passing drill (2026-11-08).
    """
    if set(samples) != set(dists):
        raise ValueError(f"samples have heads {sorted(samples)}, the actor has {sorted(dists)}")
    terms = []
    for name in dists:
        logp = dists[name].log_prob(samples[name]).sum(-1)
        terms.append(logp * masks[name] if name in masks else logp)
    return torch.stack(terms).sum(0)
### END AI GENERATED


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
    actor = Actor(obs_dim, hidden, cfg.rl.beta_eps,
                  kick_enabled=cfg.rl.kick_enabled,  # AI GENERATED (kick_enabled)
                  # AI GENERATED (2026-11-08: frozen blues)
                  move_enabled=cfg.shared.scenario not in FROZEN_SCENARIOS,
                  call_enabled=cfg.rl.call_enabled)  # AI GENERATED (2026-11-13)
    critic = Critic(critic_input_dim(cfg.rl.critic_input, obs_dim, state_dim), hidden)
    return actor, critic


def seeded_networks(cfg: Config, obs_dim: int, state_dim: int) -> tuple[Actor, Critic]:
    """THE initialization of a run: seed torch with ``run.seed``, then build. Training
    uses it, and the evaluation script uses it to rebuild the untrained policy exactly."""
    torch.manual_seed(cfg.run.seed)
    return build_networks(cfg, obs_dim, state_dim)
