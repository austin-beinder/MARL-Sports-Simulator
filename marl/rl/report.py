# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the final evaluation table
#     (trained runs, stochastic + deterministic; the untrained initial
#     policy; scripted baselines), all on the fixed evaluation episodes, with
#     mean +/- spread across training seeds (design #23), and its
#     command-line entry point.
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
"""The evaluation table for the report (design #23).

For a set of training runs of the SAME task (one per training seed):

* ``ppo``: each run's final checkpoint, stochastic policy (the primary metric);
* ``ppo_deterministic``: the same checkpoint, Beta-mean actions (secondary);
* ``untrained``: each run's initial policy, rebuilt exactly with ``seeded_networks``;
* the scripted baselines for the scenario (random, still, flee | oracle).

Every policy plays the same fixed evaluation episodes (seeds from ``run.eval_seed``).
Learned policies are summarized as the mean over training seeds +/- the standard
deviation across seeds; scripted baselines don't depend on the training seed, so
they are evaluated once (seed spread = NaN).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl.baselines import baselines_for, make_baseline
from marl.common.config import Config
from marl.envs.mock_env import MockEnv
from marl.rl.checkpoint import load_actor
from marl.rl.evaluate import eval_env_seeds, evaluate_actor, run_episodes, summarize
from marl.rl.networks import seeded_networks
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer

METRICS = ("len_mean", "len_mean_possession", "len_mean_free_ball", "frac_truncated", "mean_reward")


def _same_task(a: Config, b: Config) -> bool:
    keys = ("eval_seed", "eval_episodes")
    return a.shared == b.shared and a.rl == b.rl and all(getattr(a.run, k) == getattr(b.run, k) for k in keys)


def evaluate_runs(run_dirs: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (per_run_rows, table_rows). Raises if the runs aren't the same task."""
    per_run: list[dict[str, Any]] = []
    cfg0: Config | None = None
    for run_dir in run_dirs:
        cfg, actor, featurizer = load_actor(Path(run_dir) / "checkpoints" / "final.pt")
        if cfg0 is None:
            cfg0 = cfg
        elif not _same_task(cfg0, cfg):
            raise ValueError(f"{run_dir} is a different task (config differs beyond run.seed); compare like with like")
        reward_fn, env = RewardModule(cfg), MockEnv(cfg.shared)
        for policy, deterministic in (("ppo", False), ("ppo_deterministic", True)):
            records = evaluate_actor(env, cfg, actor, featurizer, reward_fn, deterministic)
            per_run.append({"policy": policy, "seed": cfg.run.seed, "run": str(run_dir), **summarize(records)})
        with torch.random.fork_rng(devices=[]):
            untrained, _ = seeded_networks(cfg, featurizer.obs_dim, featurizer.state_dim)
        records = evaluate_actor(env, cfg, untrained, featurizer, reward_fn)
        per_run.append({"policy": "untrained", "seed": cfg.run.seed, "run": str(run_dir), **summarize(records)})
    if cfg0 is None:
        raise ValueError("no runs given")

    reward_fn, seeds = RewardModule(cfg0), eval_env_seeds(cfg0.run.eval_seed, cfg0.run.eval_episodes)
    for name in baselines_for(cfg0.shared.scenario):
        policy = make_baseline(name, cfg0.shared, seed=cfg0.run.eval_seed)
        records = run_episodes(MockEnv(cfg0.shared), policy, cfg0, reward_fn, seeds)
        per_run.append({"policy": name, "seed": float("nan"), "run": "", **summarize(records)})
    return per_run, aggregate(per_run, cfg0.shared.max_episode_steps)


def aggregate(per_run: Sequence[dict[str, Any]], max_steps: int) -> list[dict[str, Any]]:
    table = []
    for policy in dict.fromkeys(r["policy"] for r in per_run):  # first-seen order
        rows = [r for r in per_run if r["policy"] == policy]
        out: dict[str, Any] = {"policy": policy, "n_seeds": len(rows)}
        for m in METRICS:
            vals = np.array([r[m] for r in rows], dtype=float)
            out[m] = float(np.mean(vals))  # NaN if a start type never occurred
            out[f"{m}_seed_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out["survival"] = out["len_mean"] / max_steps
        table.append(out)
    return table


def _fmt(x: float, spec: str) -> str:
    return "   -" if isinstance(x, float) and math.isnan(x) else format(x, spec)


def format_table(table: Sequence[dict[str, Any]]) -> str:
    lines = [f"{'policy':<18} {'seeds':>5} | {'len':>6} {'+/-':>5} | {'poss':>6} {'free':>6} | {'trunc':>5} | {'r/step':>7}"]
    for r in table:
        lines.append(
            f"{r['policy']:<18} {r['n_seeds']:>5} | {_fmt(r['len_mean'], '6.1f')} {_fmt(r['len_mean_seed_std'], '5.1f')} | "
            f"{_fmt(r['len_mean_possession'], '6.1f')} {_fmt(r['len_mean_free_ball'], '6.1f')} | "
            f"{_fmt(r['frac_truncated'], '5.0%')} | {_fmt(r['mean_reward'], '+7.3f')}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate trained runs + baselines on the fixed eval episodes (#23).")
    ap.add_argument("runs", nargs="+", type=Path, help="run directories of the SAME task, one per training seed")
    ap.add_argument("--out", type=Path, help="table CSV (default: <first run's parent>/eval_<first run name>.csv)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    per_run, table = evaluate_runs(args.runs)
    print(format_table(table))
    out = args.out or args.runs[0].parent / f"eval_{args.runs[0].name}.csv"
    for path, rows in ((out, table), (out.with_name(out.stem + "_per_run.csv"), per_run)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[evaluate] table -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
