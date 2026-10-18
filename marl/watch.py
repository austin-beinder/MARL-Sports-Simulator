# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-18
# Author:  Austin Beinder
# Usage:
#   - 2026-10-18: Generated this entire file: watch a trained policy (or a
#     scripted baseline) play keep-away in the PyBullet GUI in real time.
#   - 2026-10-18: Added --start free|possession to force the start type.
# Prompt summary: Make a watch script that plays games with the latest trained
#   model in the PyBullet GUI.
# Human review:   Pending review
# ==============================================================================
### AI GENERATED
"""Watch a policy play keep-away in the PyBullet GUI, in real time.

    python scripts/watch.py runs/keepaway_1v1_pybullet_seed0             # newest checkpoint
    python scripts/watch.py runs/keepaway_1v1_pybullet_seed0 --episodes 5 --seed 3
    python scripts/watch.py runs/.../checkpoints/iter_00100.pt --deterministic
    python scripts/watch.py --baseline flee                              # a scripted policy
    python scripts/watch.py runs/keepaway_1v1_pybullet_seed0 --start free  # always a loose ball

A run directory plays its newest checkpoint (``final.pt`` if training finished, else the
highest ``iter_*.pt``), so it also works while that run is still training. The game is
always shown in PyBullet, whichever env the run trained on. Every episode prints its seed,
so ``--seed`` can replay it exactly.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from marl.baselines import baselines_for, make_baseline
from marl.common.config import Config, load_config, with_overrides
from marl.envs import make_env
from marl.rl.checkpoint import load_actor
from marl.rl.evaluate import Policy, actor_policy

_ITER = re.compile(r"iter_(\d+)\.pt")


def latest_checkpoint(run_dir: Path) -> Path:
    """``final.pt`` if the run finished, else the highest-numbered ``iter_*.pt``."""
    ckpt_dir = run_dir / "checkpoints"
    if (ckpt_dir / "final.pt").is_file():
        return ckpt_dir / "final.pt"
    iters = [(int(m.group(1)), p) for p in ckpt_dir.glob("iter_*.pt") if (m := _ITER.fullmatch(p.name))]
    if not iters:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir} yet (one is saved every run.eval_every iterations)")
    return max(iters)[1]


def watch_episodes(
    env, policy: Policy, seeds: Sequence[int], cfg: Config, *, speed: float = 1.0,
    log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Play one episode per seed, paced to ``speed`` x real time (``inf`` = no pacing)."""
    dt = cfg.shared.dt_control / speed
    max_steps = cfg.shared.max_episode_steps
    results = []
    for seed in seeds:
        torch.manual_seed(seed)  # a stochastic policy replays exactly with the same --seed
        s, _ = env.reset(seed=int(seed), options=cfg.shared.start.to_reset_options())
        start = "blue has the ball" if any(b["has_possession"] for b in s["blue"]) else "loose ball"
        while True:
            t0 = time.perf_counter()
            with torch.no_grad():
                actions = policy(s)
            s, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break
            sleep(max(0.0, dt - (time.perf_counter() - t0)))
        seconds = s["step"] * cfg.shared.dt_control
        ending = "red got the ball" if terminated else "blue survived the whole episode"
        log(f"[watch] seed {seed}: {start} -> {ending} after {s['step']}/{max_steps} steps ({seconds:.1f} s)")
        results.append({"seed": int(seed), "steps": s["step"], "terminated": terminated, "truncated": truncated})
        sleep(1.0 / speed if np.isfinite(speed) else 0.0)  # a beat before the next kickoff
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Watch a trained policy (or a baseline) play keep-away in the PyBullet GUI.")
    ap.add_argument("run", nargs="?", type=Path, help="run directory (plays its newest checkpoint) or a .pt checkpoint")
    ap.add_argument("--checkpoint", help="checkpoint name inside the run, e.g. iter_00100.pt (default: newest)")
    ap.add_argument("--baseline", help="watch a scripted policy instead of a checkpoint: random, still or flee")
    ap.add_argument("--config", type=Path, default=Path("configs/keepaway_1v1.toml"),
                    help="config for --baseline (a checkpoint carries its own)")
    ap.add_argument("--episodes", type=int, default=1, help="games to play (default 1)")
    ap.add_argument("--seed", type=int, help="seed of the first game; the rest follow (default: random)")
    ap.add_argument("--start", choices=("free", "possession"),
                    help="force the start: a loose ball, or blue holding it (default: either, by seed)")
    ap.add_argument("--deterministic", action="store_true", help="play the Beta-mean action instead of sampling")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time (default 1)")
    ap.add_argument("--debug", action="store_true", help="turn on the sim's collision prints")
    args = ap.parse_args(argv)

    if (args.run is None) == (args.baseline is None):
        ap.error("give a run directory / checkpoint, or --baseline NAME (not both)")
    if args.episodes < 1 or args.speed <= 0:
        ap.error("--episodes must be >= 1 and --speed > 0")

    torch.set_num_threads(1)
    if args.baseline is not None:
        cfg = load_config(args.config)
        if args.baseline not in baselines_for(cfg.shared.scenario):
            ap.error(f"--baseline must be one of {baselines_for(cfg.shared.scenario)}")
        policy = make_baseline(args.baseline, cfg.shared, seed=args.seed or 0)
        what = f"baseline '{args.baseline}'"
    else:
        if args.run.suffix == ".pt":
            ckpt = args.run
        elif args.checkpoint:
            ckpt = args.run / "checkpoints" / args.checkpoint
        else:
            ckpt = latest_checkpoint(args.run)
        cfg, actor, featurizer = load_actor(ckpt)
        policy = actor_policy(actor, featurizer, deterministic=args.deterministic)
        what = f"{ckpt} ({'deterministic' if args.deterministic else 'stochastic'}, trained on {cfg.run.env})"

    if args.start is not None:  # start_mix is the chance of a possession start (ICD §9)
        cfg = with_overrides(cfg, {"shared": {"start": {"start_mix": 0.0 if args.start == "free" else 1.0}}})
    first = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1 - args.episodes))
    seeds = range(first, first + args.episodes)
    env = make_env(with_overrides(cfg, {"run": {"env": "pybullet"}}), render=True, print_collisions=args.debug)
    print(f"[watch] {what}: {args.episodes} game(s) at {args.speed:g}x; close the window or Ctrl+C to stop")

    import pybullet  # only for its error type: closing the GUI window makes the next call raise it

    try:
        watch_episodes(env, policy, seeds, cfg, speed=args.speed)
    except (KeyboardInterrupt, pybullet.error):
        print("[watch] stopped")
    finally:
        env.close()
    return 0
### END AI GENERATED
