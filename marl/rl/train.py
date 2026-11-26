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
#     (walk-through steps 0-6; design #19, #23; decision E). Network init goes
#     through seeded_networks (shared with the evaluation script; the behavior
#     is unchanged), and progress lines flush immediately, so they appear even
#     when output is redirected to a file (e.g. seeds run in the background).
#   - 2026-10-14: Added the reward_shaping_mean_unscaled column to train.csv
#     (PBRS ablation; the student approved a separate shaping column).
#   - 2026-10-18: The env comes from run.env (make_env), and evaluation goes
#     through make_evaluator, which runs it in its own process for PyBullet.
#     Added --render (PyBullet GUI on the training env) and --debug (the sim's
#     collision prints). --overwrite empties checkpoints/ instead of deleting it
#     (OneDrive makes folders read-only). (Austin Beinder)
#   - 2026-10-31: Evaluation runs in-process for PyBullet too (each env has
#     its own simulation now), and the training env is closed at the end.
#     (Austin Beinder)
#   - 2026-10-31: A progress line prints for every iteration; removed the
#     throttle to ~50 lines per run (backlog task 1, the student's choice).
#   - 2026-11-01: PyBullet is the default env; run dirs always carry the env
#     (<config>_<env>_seed<N>), and the --env/--render help and errors say so
#     (backlog task 3).
#   - 2026-11-07: Two hooks for curriculum/runner.py: init_from (warm start from
#     a checkpoint; also --init-from) and curriculum (sets each rollout's start
#     options, adds its columns to train.csv, can end the run early). init_from
#     goes through checkpoint.warm_start, so it also works when the team sizes
#     grow (e.g. a 2v1 stage after 1v1). Logs kicks_per_step to train.csv. The
#     printed training line shows the curriculum gate when there is one; the
#     training and eval lines show passes per episode, the share of kicks caught
#     (pass_rate), the share of passes straight back (return_share) and how well
#     the blues called for the ball (rl.call_enabled). The warm-start log line
#     says when the critic starts fresh (a different task or reward), whether
#     new inputs start at zero or random (rl.new_player_init), and whether the
#     passer flags were added. The PPO updater gets the call coef, and the
#     buffer the call head's own advantage when call_accuracy is on. Pylint
#     cleanup: wrapped lines over 100 characters, and the nan checks use
#     math.isnan instead of comparing a value with itself; no behaviour change.
#     (Austin Beinder)
#   - 2026-11-25: Citation cleanup (backlog task 9): one Usage entry per PR,
#     oldest first; no behaviour change.
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
import math  # AI GENERATED (2026-11-15)
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
# AI GENERATED (load_checkpoint, warm_start)
from marl.rl.checkpoint import load_checkpoint, save_checkpoint, warm_start
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
    # PPO iterations actually run (fewer than planned if a curriculum ended early)  # AI GENERATED
    iterations: int = 0


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
    ### AI GENERATED (2026-11-07, for curriculum/runner.py)
    init_from: str | Path | None = None,
    curriculum: Any = None,
    ### END AI GENERATED
) -> TrainResult:
    ### AI GENERATED
    """Train one PPO run into ``out_dir``.

    ``init_from``: a checkpoint to start from instead of the fresh init. The actor loads with
    ``strict=False``, so heads the checkpoint lacks (e.g. kick heads) keep their fresh init. If the
    teams grew (e.g. 1v1 -> 2v1), the first layers grow to fit (``checkpoint.warm_start``). The
    critic loads only if the scenario and reward are the same; otherwise it starts fresh.
    Optimizer states start fresh.

    ``curriculum``: an object with ``reset_options() -> dict`` (start options for the next
    rollout's new episodes), ``update(episodes) -> dict`` (sees the episodes that ended in that
    rollout; returns the same columns every time, added to train.csv) and ``finished`` (True ends
    the run early, after a last evaluation and checkpoint). See ``curriculum/runner.py``.
    """
    ### END AI GENERATED
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
    # Never the training env: it is mid-episode between rollouts. A second env, with its own
    # simulation for PyBullet, in this process.
    evaluator = make_evaluator(cfg, featurizer, reward_fn)
    ### END AI GENERATED
    buf = make_buffer(cfg, featurizer, actor)
    # AI GENERATED (2026-11-14: call_coef)
    updater = PPOUpdater(actor, critic, cfg.rl.ppo, call_coef=reward_fn.call_coef)
    shuffle_rng = np.random.default_rng(shuffle_seq)

    T = cfg.rl.ppo.rollout_steps
    n_iters = cfg.run.total_steps // T
    log(
        f"[train] env={cfg.run.env} scenario={cfg.shared.scenario} {cfg.shared.n_blue}v{cfg.shared.n_red} "  # AI GENERATED (env=)
        f"critic={cfg.rl.critic_input} gamma={cfg.gamma:.4f} seed={cfg.run.seed} "
        f"iterations={n_iters} x {T} steps -> {out_dir}"
    )
    ### AI GENERATED (warm start)
    if init_from is not None:
        ckpt = load_checkpoint(init_from)
        old = ckpt["config"]["shared"]
        missing, critic_loaded = warm_start(actor, critic, ckpt, cfg)
        fresh = f" (fresh: {', '.join(missing)})" if missing else ""
        # AI GENERATED (2026-11-12)
        start = "at zero" if cfg.rl.new_player_init == "zero" else "random"
        teams = (f" ({old['n_blue']}v{old['n_red']} -> {cfg.shared.n_blue}v{cfg.shared.n_red}, "
                 f"new inputs start {start})"
                 if (old["n_blue"], old["n_red"]) != (cfg.shared.n_blue, cfg.shared.n_red) else "")
        ### AI GENERATED (2026-11-13)
        for key, what in (("passer_obs", "passer"), ("call_enabled", "call")):  # call: 2026-11-13
            if ckpt["config"]["rl"].get(key, False) != getattr(cfg.rl, key):
                teams += (f" ({what} flags "
                          f"{'added, starting ' + start if getattr(cfg.rl, key) else 'dropped'})")
        ### END AI GENERATED
        critic_note = ("" if critic_loaded
                       else " (actor only: the critic starts fresh for the new task/reward)")
        log(f"[train] warm start from {init_from}{teams}{critic_note}{fresh}")
    ### END AI GENERATED

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
            last_it = 0  # AI GENERATED
            for it in range(1, n_iters + 1):
                ### AI GENERATED (curriculum: the start options for this rollout's new episodes)
                if curriculum is not None:
                    collector.set_reset_options(curriculum.reset_options())
                ### END AI GENERATED
                t0 = time.perf_counter()
                stats = collector.collect(buf)
                # AI GENERATED
                cur = curriculum.update(stats.episodes) if curriculum is not None else {}
                buf.compute_advantages(cfg.gamma, cfg.rl.ppo.gae_lambda)
                # AI GENERATED (2026-11-14): the call head's own advantage
                if reward_fn.call_coef > 0:
                    buf.compute_call_advantages()  # AI GENERATED
                reward_mean = float(buf.rewards.mean())
                pre = {
                    "reward_mean": reward_mean,
                    "reward_mean_unscaled": reward_mean / reward_fn.scale,
                    # The shaping part of reward_mean_unscaled (0 without shaping). The learner's
                    # reward includes it; the evaluation metric (episode length) never does.
                    "reward_shaping_mean_unscaled": stats.mean_shaping,
                    "commanded_speed": stats.mean_commanded_speed,
                    "kicks_per_step": stats.kicks_per_step,  # AI GENERATED (0 without kick heads)
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
                    **cur,  # AI GENERATED (curriculum columns)
                    "sps": T / (time.perf_counter() - t0),
                    "wall_time_s": time.perf_counter() - start,
                }
                train_log.log(row)
                done = curriculum is not None and curriculum.finished  # AI GENERATED
                log(_train_line(row, n_iters))
                if it % cfg.run.eval_every == 0 or it == n_iters or done:  # AI GENERATED (or done)
                    final_eval = evaluate(it, deterministic=False)
                    checkpoint(f"iter_{it:05d}.pt", it)
                ### AI GENERATED
                last_it = it
                if done:
                    log(f"[train] the curriculum ended this run at iteration {it} of {n_iters}")
                    break
            final_det = evaluate(last_it, deterministic=True)
            checkpoint("final.pt", last_it)
            ### END AI GENERATED
    finally:
        evaluator.close()
        if hasattr(train_env, "close"):  # PyBullet: free this run's simulation (the mock has none)
            train_env.close()
    ### END AI GENERATED
    log(f"[train] done in {time.perf_counter() - start:.0f} s -> {out_dir}")
    # AI GENERATED (last_it)
    return TrainResult(out_dir, actor, critic, final_eval, final_det, last_it)


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
        # AI GENERATED (2026-11-09; rate 2026-11-11)
        f"passes {r['ep_passes_mean']:4.2f} ({r['ep_pass_rate']:4.0%} caught, "
        f"{r['ep_return_share']:4.0%} back) | "  # AI GENERATED (2026-11-13)
        + _call_text(r, "ep_") +  # AI GENERATED (2026-11-13)
        f"r/step {r['reward_mean_unscaled']:+7.3f} | ent {r['entropy']:+.3f} | vloss {r['value_loss']:.2e} | "
        f"ev {r['explained_variance']:+.2f} | kl {r['approx_kl']:.4f} | clip {r['clip_frac']:.3f} | "
        f"{r['sps']:,.0f} sps"
    ) + _curriculum_text(r)  # AI GENERATED


### AI GENERATED (2026-11-13)
def _call_text(r: dict[str, Any], prefix: str) -> str:
    """How well the blues called for the ball they got, when there were pickups and calls
    (rl.call_enabled)."""
    called = r.get(f"{prefix}called_share", float("nan"))
    wrong = r.get(f"{prefix}miscalled_per_pickup", float("nan"))
    if math.isnan(called) or called + wrong == 0:  # nan, or nobody ever called
        return ""
    return f"calls {called:4.0%} right, {wrong:.2f} wrong | "
### END AI GENERATED


### AI GENERATED
def _curriculum_text(r: dict[str, Any]) -> str:
    """The curriculum gate, for a curriculum run: survival mean, the lower bound the gate checks,
    and the current level's option value if the stage has levels. Nothing for a plain run."""
    if "gate_lower" not in r:
        return ""
    text = f" | gate {r['gate_mean']:.2f} (low {r['gate_lower']:.2f})"
    value = r.get("cur_value", float("nan"))
    if not math.isnan(value):  # not nan: the stage has levels
        text += f" lvl {r['cur_level']} = {value:g}"
    return text
### END AI GENERATED


def _eval_line(r: dict[str, Any]) -> str:
    return (
        f"  eval@{r['iteration']:>4} [{r['mode'][:5]}] len {r['len_mean']:6.1f} +/- {r['len_std']:5.1f} "
        f"(trunc {r['frac_truncated']:4.0%}) | possession {r['len_mean_possession']:6.1f} | "
        f"free_ball {r['len_mean_free_ball']:6.1f} | r/step {r['mean_reward']:+7.3f} | "
        # AI GENERATED (2026-11-09; rate 2026-11-11)
        f"passes {r['passes_mean']:.2f} ({r['pass_rate']:4.0%} of kicks caught, "
        f"{r['return_share']:4.0%} straight back)"  # AI GENERATED (2026-11-13)
        # AI GENERATED (2026-11-13)
        + (" | " + _call_text(r, "").rstrip(" |") if _call_text(r, "") else "")
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
    ap.add_argument("--env", choices=("mock", "pybullet"),
                    help="override run.env (default: pybullet; mock is the fast stand-in)")  # AI GENERATED
    ### AI GENERATED
    ap.add_argument("--render", action="store_true",
                    help="show the training env in the PyBullet GUI (slow; runs as fast as it can, not real time)")
    ap.add_argument("--debug", action="store_true",
                    help="turn on the PyBullet sim's collision prints ('Entity: 6 collided with Entity: 4')")
    ### END AI GENERATED
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key (repeatable), e.g. --set shared.red_max_speed=1.6")
    ap.add_argument("--out-dir", type=Path, help="default: <run.out_dir>/<config name>_<env>_seed<seed>")  # AI GENERATED
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run in --out-dir")
    ap.add_argument("--init-from", type=Path, metavar="CHECKPOINT",
                    # AI GENERATED
                    help="start from this checkpoint's weights instead of a fresh init")
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
    ### AI GENERATED (every run dir names its env, so mock and PyBullet runs never collide, 2026-11-01)
    out_dir = args.out_dir or Path(cfg.run.out_dir) / f"{args.config.stem}_{cfg.run.env}_seed{cfg.run.seed}"
    ### END AI GENERATED
    ### AI GENERATED
    if args.render and cfg.run.env != "pybullet":
        ap.error("--render needs the PyBullet env (--env pybullet)")
    train(cfg, out_dir, overwrite=args.overwrite, render=args.render, print_collisions=args.debug,
          init_from=args.init_from)  # AI GENERATED (init_from)
    ### END AI GENERATED
    return 0
