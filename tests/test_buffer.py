# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: rollout buffer tests (shapes at
#     N=1 and N=2, input validation, GAE wiring, minibatch coverage and
#     field alignment via id-coded rows, deterministic shuffles).
# Prompt summary: Implement Phase 3 step 7 of the issue #7 plan (buffer +
#   GAE) with hand-built boundary tests.
# Human review:   Pending review
# ==============================================================================
import numpy as np
import pytest
import torch

from marl.rl.buffer import RolloutBuffer
from marl.rl.gae import compute_gae

OBS, CRIT = 5, 4


def make(T=8, N=2):
    return RolloutBuffer(T, N, OBS, CRIT, {"move": 2})


def row(N, t, **over):
    """A valid row whose every field encodes its (t, i) as id = t * N + i."""
    ids = np.arange(N, dtype=np.float32) + t * N
    kw = dict(
        obs=np.repeat(ids[:, None], OBS, axis=1),
        critic_in=np.repeat(ids[:, None], CRIT, axis=1),
        samples={"move": np.repeat(ids[:, None], 2, axis=1)},
        log_prob=ids,
        value=ids,
        next_value=ids + 0.5,
        reward=1.0,
        terminated=False,
        truncated=False,
        effective_terminal=False,
    )
    return kw | over


def fill(buf, flags=None):
    for t in range(buf.n_steps):
        buf.add(**row(buf.n_agents, t, **(flags(t) if flags else {})))
    return buf


@pytest.mark.parametrize("N", [1, 2])
def test_shapes(N):
    buf = fill(make(T=8, N=N))
    buf.compute_advantages(0.99, 0.95)
    assert buf.obs.shape == (8, N, OBS) and buf.critic_in.shape == (8, N, CRIT)
    assert buf.samples["move"].shape == (8, N, 2) and buf.advantages.shape == (8, N)
    assert buf.rewards.shape == (8,)  # shared team reward
    mbs = list(buf.minibatches(4, np.random.default_rng(0)))
    B = 8 * N // 4
    assert len(mbs) == 4
    for mb in mbs:
        assert mb.obs.shape == (B, OBS) and mb.critic_in.shape == (B, CRIT)
        assert mb.samples["move"].shape == (B, 2)
        assert mb.log_probs.shape == mb.values.shape == mb.advantages.shape == mb.returns.shape == (B,)
        assert mb.obs.dtype == torch.float32


def test_every_sample_once_per_epoch_and_fields_stay_aligned():
    T, N = 8, 2
    buf = fill(make(T, N))
    buf.compute_advantages(0.99, 0.95)
    seen = []
    for mb in buf.minibatches(4, np.random.default_rng(0)):
        ids = mb.log_probs
        # Every field of a minibatch row must come from the SAME (t, i).
        assert torch.equal(mb.obs[:, 0], ids) and torch.equal(mb.critic_in[:, 0], ids)
        assert torch.equal(mb.samples["move"][:, 0], ids) and torch.equal(mb.values, ids)
        torch.testing.assert_close(mb.returns - mb.advantages, ids)  # return = A + V, same row
        seen += ids.tolist()
    assert sorted(seen) == list(range(T * N))  # each (t, i) exactly once


def test_advantages_match_compute_gae_with_both_masks():
    def flags(t):
        if t == 2:
            return {"terminated": True, "effective_terminal": True}
        if t == 5:
            return {"truncated": True}
        return {}

    buf = fill(make(T=8, N=2), flags)
    buf.compute_advantages(0.9, 0.8)
    end = buf.terminated | buf.truncated
    adv, ret = compute_gae(buf.rewards, buf.values, buf.next_values, buf.effective_terminal, end, 0.9, 0.8)
    np.testing.assert_allclose(buf.advantages, adv, rtol=1e-6)
    np.testing.assert_allclose(buf.returns, ret, rtol=1e-6)


def test_shuffles_are_seeded_and_differ_between_epochs():
    buf = fill(make())
    buf.compute_advantages(0.99, 0.95)
    first = [mb.log_probs.tolist() for mb in buf.minibatches(4, np.random.default_rng(7))]
    again = [mb.log_probs.tolist() for mb in buf.minibatches(4, np.random.default_rng(7))]
    rng = np.random.default_rng(7)
    epoch1 = [mb.log_probs.tolist() for mb in buf.minibatches(4, rng)]
    epoch2 = [mb.log_probs.tolist() for mb in buf.minibatches(4, rng)]
    assert first == again == epoch1 and epoch1 != epoch2


def test_lifecycle_errors():
    buf = make(T=2)
    with pytest.raises(RuntimeError, match="fill it first"):
        buf.compute_advantages(0.99, 0.95)
    fill(buf)
    with pytest.raises(RuntimeError, match="compute_advantages"):
        next(buf.minibatches(2, np.random.default_rng(0)))
    with pytest.raises(RuntimeError, match="full"):
        buf.add(**row(2, 0))
    buf.compute_advantages(0.99, 0.95)
    with pytest.raises(ValueError, match="divisible"):
        next(buf.minibatches(3, np.random.default_rng(0)))
    buf.reset()
    assert buf.t == 0 and not buf.full
    with pytest.raises(RuntimeError, match="compute_advantages"):
        next(buf.minibatches(2, np.random.default_rng(0)))  # advantages invalidated by reset


@pytest.mark.parametrize(
    "over, match",
    [
        ({"obs": np.zeros((2, OBS + 1))}, "obs"),
        ({"log_prob": np.zeros(3)}, "log_prob"),
        ({"value": np.array([0.0, np.nan])}, "finite"),
        ({"next_value": np.array([np.inf, 0.0])}, "finite"),
        ({"samples": {"kick_vec": np.zeros((2, 2))}}, "heads"),
        ({"reward": float("nan")}, "reward"),
        ({"effective_terminal": True}, "effective_terminal"),  # without terminated/truncated
    ],
)
def test_add_rejects_bad_rows(over, match):
    buf = make()
    with pytest.raises(ValueError, match=match):
        buf.add(**row(2, 0, **over))


def test_accepts_torch_tensors_from_the_networks():
    buf = make(T=1, N=2)
    buf.add(**row(2, 0, log_prob=torch.tensor([0.1, 0.2]), value=torch.zeros(2), next_value=torch.ones(2)))
    np.testing.assert_allclose(buf.log_probs[0], [0.1, 0.2])
