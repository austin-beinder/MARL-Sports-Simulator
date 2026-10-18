# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the training loop (seeding with
#     separate train/shuffle/eval streams, collect -> GAE -> PPO update,
#     per-iteration and evaluation CSV logs, checkpoints, safe output
#     directories) and the command-line entry point with --set overrides
#     (walk-through steps 0-6; design #19, #23; decision E).
#   - 2026-10-11: Network init now goes through seeded_networks (shared with
#     the evaluation script); the behavior is unchanged.
#   - 2026-10-18: The env comes from run.env (make_env), and evaluation goes
#     through make_evaluator, which runs it in its own process for PyBullet.
#     (Austin Beinder)
#   - 2026-10-18: Added --render (PyBullet GUI on the training env) and
#     --debug (the sim's collision prints). (Austin Beinder)
#   - 2026-10-18: --overwrite empties checkpoints/ instead of deleting it
#     (OneDrive makes folders read-only). (Austin Beinder)
#   - 2026-10-11: Progress lines now flush immediately, so they appear even when
#     output is redirected to a file (e.g. seeds run in the background).
# Prompt summary: Implement Phase 3 step 8 of the issue #7 plan (PPO update +
#   training loop).
# Human review:   Pending review
# ==============================================================================
"""Training loop (walk-through steps 0-6) and its command-line entry point.

Seeding (walk-through step 0):

* ``run.seed`` -> one numpy SeedSequence, split into two child streams: training
  env reset seeds, and minibatch shuffles.
* torch is seeded globally with ``run.seed`` (network init + training-time action
  sampling). torch sampling can't take a Generator, so evaluation runs inside
  ``fork_rng`` (see ``marl/rl/evaluate.py``).
* Evaluation seeds come from ``run.eval_seed`` only: the same episodes for every
  training seed.

Each iteration: collect T steps -> GAE -> K epochs of PPO -> log a train.csv row.
Every ``run.eval_every`` iterations (and at the end): evaluate on the fixed seeds
(eval.csv) and save a checkpoint. The untrained policy is evaluated first, at
iteration 0 (it's one of the #23 baselines).
"""

from __future__ import annotations

import argparse
import json
import stat  # AI GENERATED
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl.common.config import Config, config_to_dict, load_config, with_overrides
from marl.envs import make_env  # AI GENERATED
from marl.rl.checkpoint import save_checkpoint
from marl.rl.eval_worker import make_evaluator  # AI GENERATED
from marl.rl.evaluate import summarize
from marl.rl.logger import CSVLogger
from marl.rl.networks import Actor, Critic, seeded_networks
from marl.rl.ppo import PPOUpdater, explained_variance
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer, RolloutCollector, make_buffer

RUN_FILES = frozenset({"config.json", "train.csv", "eval.csv", "checkpoints"})


@dataclass
class TrainResult:
    out_dir: Path
    actor: Actor
    critic: Critic
    final_eval: dict[str, Any]  # stochastic policy: the primary metric (#23)
    final_eval_deterministic: dict[str, Any]  # Beta mean: the secondary diagnostic


def _print_flush(msg: str) -> None:
    print(msg, flush=True)  # without flush, redirected output only appears when the run ends


def train(
    cfg: Config,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    log: Callable[[str], None] = _print_flush,
    render: bool = False,  # AI GENERATED: show the training env in the PyBullet GUI
    print_collisions: bool = False,  # AI GENERATED: the sim's collision debug prints (PyBullet)
) -> TrainResult:
    out_dir = _prepare_out_dir(Path(out_dir), overwrite)
    (out_dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2) + "\n", encoding="utf-8")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)  # AI GENERATED (exist_ok: --overwrite empties it but keeps it)

    # Tiny networks: one thread is faster, and parallel seed runs don't fight over cores.
    torch.set_num_threads(1)
    env_seq, shuffle_seq = np.random.SeedSequence(cfg.run.seed).spawn(2)

    featurizer = Featurizer(cfg)
    # Seeds torch with run.seed: network init, then training-time sampling continues from there.
    actor, critic = seeded_networks(cfg, featurizer.obs_dim, featurizer.state_dim)
    reward_fn = RewardModule(cfg)
    ### AI GENERATED
    train_env = make_env(cfg, render=render, print_collisions=print_collisions)
    collector = RolloutCollector(train_env, cfg, actor, critic, reward_fn, np.random.default_rng(env_seq))
    # Never the training env: it is mid-episode between rollouts. For PyBullet this is a worker
    # process with its own simulation (every env in one process shares pybullet's).
    evaluator = make_evaluator(cfg, featurizer, reward_fn, print_collisions)
    ### END AI GENERATED
    buf = make_buffer(cfg, featurizer, actor)
    updater = PPOUpdater(actor, critic, cfg.rl.ppo)
    shuffle_rng = np.random.default_rng(shuffle_seq)

    T = cfg.rl.ppo.rollout_steps
    n_iters = cfg.run.total_steps // T
    print_every = max(1, n_iters // 50)
    log(
        f"[train] env={cfg.run.env} scenario={cfg.shared.scenario} {cfg.shared.n_blue}v{cfg.shared.n_red} "  # AI GENERATED (env=)
        f"critic={cfg.rl.critic_input} gamma={cfg.gamma:.4f} seed={cfg.run.seed} "
        f"iterations={n_iters} x {T} steps -> {out_dir}"
    )

    ### AI GENERATED (try/finally so the evaluation worker always shuts down)
    try:
        with CSVLogger(out_dir / "train.csv") as train_log, CSVLogger(out_dir / "eval.csv") as eval_log:

            def evaluate(iteration: int, deterministic: bool) -> dict[str, Any]:
                records = evaluator(actor, deterministic)  # AI GENERATED
                row = {
                    "iteration": iteration,
                    "env_steps": iteration * T,
                    "mode": "deterministic" if deterministic else "stochastic",
                    **summarize(records),
                }
                eval_log.log(row)
                log(_eval_line(row))
                return row

            def checkpoint(name: str, iteration: int) -> None:
                save_checkpoint(
                    ckpt_dir / name,
                    actor=actor,
                    critic=critic,
                    updater=updater,
                    iteration=iteration,
                    env_steps=iteration * T,
                    cfg=cfg,
                )

            evaluate(0, deterministic=False)  # the untrained policy (#23 baseline)
            start = time.perf_counter()
            final_eval: dict[str, Any] = {}
            for it in range(1, n_iters + 1):
                t0 = time.perf_counter()
                stats = collector.collect(buf)
                buf.compute_advantages(cfg.gamma, cfg.rl.ppo.gae_lambda)
                reward_mean = float(buf.rewards.mean())
                pre = {
                    "reward_mean": reward_mean,
                    "reward_mean_unscaled": reward_mean / reward_fn.scale,
                    "commanded_speed": stats.mean_commanded_speed,
                    "adv_std_raw": float(buf.advantages.std()),  # before per-minibatch normalization (#23)
                    "explained_variance": explained_variance(buf.values, buf.returns),
                }
                upd = updater.update(buf, shuffle_rng)
                row = {
                    "iteration": it,
                    "env_steps": it * T,
                    **{f"ep_{k}": v for k, v in summarize(stats.episodes).items()},
                    **pre,
                    **upd,
                    "sps": T / (time.perf_counter() - t0),
                    "wall_time_s": time.perf_counter() - start,
                }
                train_log.log(row)
                if it == 1 or it % print_every == 0 or it == n_iters:
                    log(_train_line(row, n_iters))
                if it % cfg.run.eval_every == 0 or it == n_iters:
                    final_eval = evaluate(it, deterministic=False)
                    checkpoint(f"iter_{it:05d}.pt", it)
            final_det = evaluate(n_iters, deterministic=True)
            checkpoint("final.pt", n_iters)
    finally:
        evaluator.close()
    ### END AI GENERATED
    log(f"[train] done in {time.perf_counter() - start:.0f} s -> {out_dir}")
    return TrainResult(out_dir, actor, critic, final_eval, final_det)


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> Path:
    """Create the run directory. Refuse to mix runs; with ``overwrite``, delete only
    the files a run creates (never anything unrecognized)."""
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already holds a run; choose another --out-dir or pass --overwrite")
        entries = list(out_dir.iterdir())
        unknown = [e.name for e in entries if e.name not in RUN_FILES]
        ckpt = out_dir / "checkpoints"
        if ckpt.is_dir():
            unknown += [f"checkpoints/{e.name}" for e in ckpt.iterdir() if e.suffix != ".pt"]
        if unknown:
            raise FileExistsError(f"{out_dir} holds files a run doesn't create ({unknown}); refusing to overwrite")
        ### AI GENERATED
        # Empty checkpoints/ but keep the folder: OneDrive marks synced folders read-only, and
        # Windows refuses to delete a read-only folder (PermissionError: Access is denied).
        if ckpt.is_dir():
            for e in ckpt.iterdir():
                _unlink(e)
        for e in entries:
            if e.is_file():
                _unlink(e)
        ### END AI GENERATED
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


### AI GENERATED
def _unlink(path: Path) -> None:
    """Delete a file, clearing the Windows read-only attribute first if it's set."""
    try:
        path.unlink()
    except PermissionError:
        path.chmod(stat.S_IWRITE)
        path.unlink()
### END AI GENERATED


def _train_line(r: dict[str, Any], n_iters: int) -> str:
    return (
        f"it {r['iteration']:>4}/{n_iters} | steps {r['env_steps']:>9,} | "
        f"ep_len {r['ep_len_mean']:6.1f} (n={r['ep_episodes']:>3}, trunc {r['ep_frac_truncated']:4.0%}) | "
        f"r/step {r['reward_mean_unscaled']:+7.3f} | ent {r['entropy']:+.3f} | vloss {r['value_loss']:.2e} | "
        f"ev {r['explained_variance']:+.2f} | kl {r['approx_kl']:.4f} | clip {r['clip_frac']:.3f} | "
        f"{r['sps']:,.0f} sps"
    )


def _eval_line(r: dict[str, Any]) -> str:
    return (
        f"  eval@{r['iteration']:>4} [{r['mode'][:5]}] len {r['len_mean']:6.1f} +/- {r['len_std']:5.1f} "
        f"(trunc {r['frac_truncated']:4.0%}) | possession {r['len_mean_possession']:6.1f} | "
        f"free_ball {r['len_mean_free_ball']:6.1f} | r/step {r['mean_reward']:+7.3f}"
    )


# --- Command line ---------------------------------------------------------------------
def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["a.b=1", "a.c=local"]`` -> ``{"a": {"b": 1, "c": "local"}}``. Values are parsed
    as TOML (numbers, booleans, lists); anything else is kept as a string."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key or any(not part for part in key.split(".")):
            raise ValueError(f"--set expects KEY=VALUE with a dotted key, got {item!r}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {item!r} conflicts with another override")
        node[leaf] = value
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train PPO/MAPPO on the mock or PyBullet env (issue #7).")  # AI GENERATED
    ap.add_argument("--config", type=Path, required=True, help="TOML config, e.g. configs/goto_target.toml")
    ap.add_argument("--seed", type=int, help="override run.seed")
    ap.add_argument("--total-steps", type=int, help="override run.total_steps")
    ap.add_argument("--env", choices=("mock", "pybullet"), help="override run.env")  # AI GENERATED
    ### AI GENERATED
    ap.add_argument("--render", action="store_true",
                    help="show the training env in the PyBullet GUI (slow; runs as fast as it can, not real time)")
    ap.add_argument("--debug", action="store_true",
                    help="turn on the PyBullet sim's collision prints ('Entity: 6 collided with Entity: 4')")
    ### END AI GENERATED
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key (repeatable), e.g. --set shared.red_max_speed=1.6")
    ap.add_argument("--out-dir", type=Path, help="default: <run.out_dir>/<config name>[_<env>]_seed<seed>")  # AI GENERATED
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run in --out-dir")
    args = ap.parse_args(argv)

    overrides = parse_overrides(args.set)
    run = overrides.setdefault("run", {})
    if args.seed is not None:
        run["seed"] = args.seed
    if args.total_steps is not None:
        run["total_steps"] = args.total_steps
    ### AI GENERATED
    if args.env is not None:
        run["env"] = args.env
    ### END AI GENERATED
    cfg = with_overrides(load_config(args.config), overrides)
    ### AI GENERATED (non-mock envs get a suffix so their runs don't collide with mock runs)
    suffix = "" if cfg.run.env == "mock" else f"_{cfg.run.env}"
    out_dir = args.out_dir or Path(cfg.run.out_dir) / f"{args.config.stem}{suffix}_seed{cfg.run.seed}"
    ### END AI GENERATED
    ### AI GENERATED
    if args.render and cfg.run.env != "pybullet":
        ap.error("--render needs the PyBullet env (--env pybullet)")
    train(cfg, out_dir, overwrite=args.overwrite, render=args.render, print_collisions=args.debug)
    ### END AI GENERATED
    return 0
