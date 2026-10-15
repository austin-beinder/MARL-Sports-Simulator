# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the PPO update (clipped
#     surrogate with per-minibatch advantage normalization, analytic entropy
#     bonus, MSE value loss, separate Adam optimizers with per-network
#     gradient clipping, the first-minibatch ratio guard and diagnostics),
#     following design #10, #20, #23 and PPO defaults D.
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
"""The PPO update (walk-through step 4).

For every minibatch of (t, i) samples, K epochs over the whole buffer:

    ratio      = exp(log pi_new(a|o) - log pi_old(a|o))            one per (t, i) (#20)
    A_hat      = (A - mean(A)) / (std(A) + 1e-8)                    per minibatch (#20)
    L_policy   = -mean( min(ratio * A_hat, clip(ratio, 1-eps, 1+eps) * A_hat) )
    L_actor    = L_policy - ent_coef * sum(entropy) / B             analytic, per state (#10)
    L_critic   = mean( (V(critic_in) - return)^2 )                  MSE, no value clipping (D)

The actor and critic share no parameters, so each has its own Adam optimizer
and its own gradient-norm clip (decision D): a large value-loss gradient can
never shrink the policy step through a shared clip.

Ratio guard (walk-through step 4): on the first minibatch of the first epoch the
policy hasn't changed yet, so every ratio must be 1 up to float noise. If not,
the stored log-probs and the update's log-probs disagree (a wiring bug), and
training stops loudly instead of learning from garbage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from marl.common.config import PPOConfig
from marl.rl.buffer import RolloutBuffer
from marl.rl.networks import Actor, Critic

RATIO_GUARD_ATOL = 1e-4  # float32 noise is ~1e-6; anything near 1e-4 is a real mismatch


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Population std (correction=0): finite even for a 1-sample minibatch.
    return (adv - adv.mean()) / (adv.std(correction=0) + eps)


def clipped_surrogate_loss(ratio: torch.Tensor, adv: torch.Tensor, clip_eps: float) -> torch.Tensor:
    """-mean(min(r * A, clip(r) * A)): the pessimistic (lower) bound of the two."""
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean()


class PPOUpdater:
    def __init__(self, actor: Actor, critic: Critic, ppo: PPOConfig):
        self.actor, self.critic, self.ppo = actor, critic, ppo
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=ppo.lr, eps=ppo.adam_eps)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=ppo.lr, eps=ppo.adam_eps)

    def update(self, buf: RolloutBuffer, rng: np.random.Generator) -> dict[str, float]:
        """K epochs of minibatch updates on a full buffer (advantages computed). Returns
        diagnostics averaged over every gradient step."""
        p = self.ppo
        sums: dict[str, float] = {}
        n_updates = 0
        ratio_check = float("nan")
        for epoch in range(p.update_epochs):
            for k, mb in enumerate(buf.minibatches(p.num_minibatches, rng)):
                logp, entropy, head_entropy = self.actor.evaluate(mb.obs, mb.samples)
                log_ratio = logp - mb.log_probs
                ratio = log_ratio.exp()
                if epoch == 0 and k == 0:
                    ratio_check = (ratio.detach() - 1.0).abs().max().item()
                    if not ratio_check < RATIO_GUARD_ATOL:  # also catches NaN
                        raise RuntimeError(
                            f"PPO ratio guard: max |ratio - 1| = {ratio_check:.3g} before any update; "
                            "the rollout and update log-probs disagree (check what the buffer stores)"
                        )

                policy_loss = clipped_surrogate_loss(ratio, normalize_advantages(mb.advantages), p.clip_eps)
                entropy_mean = entropy.mean()  # = sum of the gated per-sample entropies / B (#10)
                actor_loss = policy_loss - p.ent_coef * entropy_mean
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad_actor = clip_grad_norm_(self.actor.parameters(), p.max_grad_norm)
                self.actor_opt.step()

                value_loss = F.mse_loss(self.critic(mb.critic_in), mb.returns)
                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                grad_critic = clip_grad_norm_(self.critic.parameters(), p.max_grad_norm)
                self.critic_opt.step()

                with torch.no_grad():
                    step = {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": entropy_mean.item(),
                        # k3 estimator of KL(old || new): mean((r - 1) - log r) >= 0
                        "approx_kl": ((ratio - 1.0) - log_ratio).mean().item(),
                        "clip_frac": ((ratio - 1.0).abs() > p.clip_eps).float().mean().item(),
                        "grad_norm_actor": grad_actor.item(),
                        "grad_norm_critic": grad_critic.item(),
                    }
                    step |= {f"entropy_{h}": e.mean().item() for h, e in head_entropy.items()}
                for key, value in step.items():
                    sums[key] = sums.get(key, 0.0) + value
                n_updates += 1
        stats = {key: total / n_updates for key, total in sums.items()}
        stats["ratio_check"] = ratio_check
        stats["updates"] = n_updates
        return stats


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var(return - V) / Var(return): 1 = perfect critic, 0 = no better than a constant."""
    var = float(np.var(returns))
    return float("nan") if var == 0.0 else 1.0 - float(np.var(returns - values)) / var
