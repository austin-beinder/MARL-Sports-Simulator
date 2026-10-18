# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Generated this entire file: the training loop's evaluator,
#     run in-process for the mock env and in its own process for envs that
#     share one simulation per process (PyBullet).
#   - 2026-10-18: Pass print_collisions through to the worker's env.
# Prompt summary: Run evaluation in its own process so the shared pybullet
#   simulation doesn't break training.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
"""The training loop's evaluator.

Evaluation must never reset the training env mid-episode (see ``marl/rl/evaluate.py``).
The mock gets that from a second env instance. PyBullet can't: every instance in a
process shares pybullet's one default simulation. So for those envs
(``marl.envs.SHARED_SIM_ENVS``) evaluation runs in a worker process with its own env
and its own copy of the actor:

* each call sends the actor's current weights to the worker and waits for the episode
  records, so evaluation happens at the same points in training and blocks it the same way;
* the worker runs the same ``evaluate_actor`` on the same seeds, so results match an
  in-process evaluation exactly (torch sampling is reseeded with ``eval_seed`` either way);
* the worker is started with "spawn" (the only start method on Windows), so it builds its
  own pybullet connection from scratch.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any

import torch

from marl.common.config import Config, config_from_dict, config_to_dict
from marl.envs import SHARED_SIM_ENVS, make_env
from marl.rl.evaluate import evaluate_actor
from marl.rl.networks import Actor, build_networks
from marl.rl.reward import RewardModule
from marl.rl.rollout import EpisodeRecord, Featurizer


class InProcessEvaluator:
    """Evaluates on a dedicated env instance in this process."""

    def __init__(self, cfg: Config, featurizer: Featurizer, reward_fn: RewardModule):
        self.cfg, self.featurizer, self.reward_fn = cfg, featurizer, reward_fn
        self.env = make_env(cfg)  # never the training env: that one is mid-episode between rollouts

    def __call__(self, actor: Actor, deterministic: bool) -> list[EpisodeRecord]:
        return evaluate_actor(self.env, self.cfg, actor, self.featurizer, self.reward_fn, deterministic)

    def close(self) -> None:
        pass


class SubprocessEvaluator:
    """Evaluates in a worker process that owns its own env (and simulation)."""

    def __init__(self, cfg: Config, print_collisions: bool = False):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker, args=(child, config_to_dict(cfg), print_collisions), daemon=True)
        self._proc.start()
        child.close()
        self._call("ping")  # surface a failed env/actor build now, not at the first evaluation

    def __call__(self, actor: Actor, deterministic: bool) -> list[EpisodeRecord]:
        weights = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
        return self._call("eval", weights, deterministic)

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._conn.send(("close",))
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
        self._conn.close()

    def _call(self, *msg: Any) -> Any:
        self._conn.send(msg)
        try:
            status, payload = self._conn.recv()
        except EOFError:
            raise RuntimeError(f"the evaluation worker died (exit code {self._proc.exitcode})") from None
        if status == "error":
            raise RuntimeError(f"the evaluation worker failed:\n{payload}")
        return payload


def make_evaluator(
    cfg: Config, featurizer: Featurizer, reward_fn: RewardModule, print_collisions: bool = False
) -> InProcessEvaluator | SubprocessEvaluator:
    """The worker's env is always headless: only the training env is ever rendered."""
    if cfg.run.env in SHARED_SIM_ENVS:
        return SubprocessEvaluator(cfg, print_collisions)
    return InProcessEvaluator(cfg, featurizer, reward_fn)


def _worker(conn: Any, cfg_dict: dict[str, Any], print_collisions: bool) -> None:
    try:
        torch.set_num_threads(1)
        cfg = config_from_dict(cfg_dict)
        featurizer, reward_fn = Featurizer(cfg), RewardModule(cfg)
        actor, _ = build_networks(cfg, featurizer.obs_dim, featurizer.state_dim)  # weights arrive per call
        env = make_env(cfg, print_collisions=print_collisions)
    except Exception:
        _reply_errors_forever(conn, traceback.format_exc())
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return  # the trainer went away
        if msg[0] == "close":
            return
        try:
            if msg[0] == "ping":
                conn.send(("ok", None))
            elif msg[0] == "eval":
                _, weights, deterministic = msg
                actor.load_state_dict(weights)
                conn.send(("ok", evaluate_actor(env, cfg, actor, featurizer, reward_fn, deterministic)))
            else:
                raise ValueError(f"unknown message {msg[0]!r}")
        except Exception:
            conn.send(("error", traceback.format_exc()))


def _reply_errors_forever(conn: Any, error: str) -> None:
    """Setup failed: answer every request with the error until the trainer closes the pipe."""
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "close":
            return
        conn.send(("error", error))
### END AI GENERATED
