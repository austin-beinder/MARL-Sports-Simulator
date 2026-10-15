# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-11
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-11: Generated this entire file: the red-speed calibration sweep
#     (scripted baselines on dedicated calibration seeds), the band-center
#     selection rule, and its command-line entry point (design #22, ICD §10).
# Prompt summary: Implement Phase 3 step 10 of the issue #7 plan (baselines,
#   evaluation, calibration).
# Human review:   Pending review
# ==============================================================================
"""Task calibration (design #22, ICD §10).

Tune ``red_max_speed`` so the scripted flee policy survives about 30-70% of
``max_episode_steps``. If red is too fast, capture is quick whatever blue does;
if red is too slow, fleeing always survives. Either way there is nothing to learn.

Selection rule: among the swept speeds whose flee survival (mean episode length /
max_episode_steps) lies inside the band, pick the one closest to the band center
(0.5); ties go to the slower red.

The sweep uses its OWN episode seeds (``CALIBRATION_SEED``), never the
evaluation seeds, so the task isn't tuned on the episodes policies are scored on.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from marl.baselines import make_baseline
from marl.common.config import Config, load_config, with_overrides
from marl.envs.mock_env import MockEnv
from marl.rl.evaluate import eval_env_seeds, run_episodes, summarize
from marl.rl.reward import RewardModule

CALIBRATION_SEED = 7
BAND = (0.3, 0.7)
TARGET = 0.5
POLICIES = ("flee", "random", "still")


def calibration_seeds(n: int) -> list[int]:
    return eval_env_seeds(CALIBRATION_SEED, n)


def survival_table(cfg: Config, speeds: Sequence[float], episodes: int, policies: Sequence[str] = POLICIES) -> list[dict[str, Any]]:
    """One row per (red speed, policy): survival = mean episode length / max_episode_steps."""
    if cfg.shared.scenario != "keepaway":
        raise ValueError("calibration is for keep-away")
    seeds = calibration_seeds(episodes)
    rows = []
    for speed in speeds:
        c = with_overrides(cfg, {"shared": {"red_max_speed": float(speed)}})
        reward_fn = RewardModule(c)
        for name in policies:
            policy = make_baseline(name, c.shared, seed=CALIBRATION_SEED)
            stats = summarize(run_episodes(MockEnv(c.shared), policy, c, reward_fn, seeds))
            rows.append(
                {"red_max_speed": float(speed), "policy": name, "survival": stats["len_mean"] / c.shared.max_episode_steps, **stats}
            )
    return rows


def pick_speed(rows: Sequence[dict[str, Any]], target: float = TARGET, band: tuple[float, float] = BAND) -> float:
    flee = [r for r in rows if r["policy"] == "flee"]
    inside = [r for r in flee if band[0] <= r["survival"] <= band[1]]
    if not inside:
        raise ValueError(f"no swept speed puts flee survival inside {band}; widen the sweep")
    # Round the distance so float noise (|0.55 - 0.5| vs |0.45 - 0.5|) can't break a real tie.
    return min(inside, key=lambda r: (round(abs(r["survival"] - target), 9), r["red_max_speed"]))["red_max_speed"]


def parse_speeds(text: str) -> list[float]:
    """``"1.5:2.0:0.05"`` (inclusive range) or ``"1.6,1.7,1.8"``."""
    if ":" in text:
        start, stop, step = (float(x) for x in text.split(":"))
        if step <= 0 or stop < start:
            raise ValueError(f"bad range {text!r}")
        n = int(round((stop - start) / step)) + 1
        return [round(start + k * step, 10) for k in range(n)]
    speeds = [float(x) for x in text.split(",") if x.strip()]
    if not speeds:
        raise ValueError(f"no speeds in {text!r}")
    return speeds


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate red_max_speed against the scripted flee policy (design #22).")
    ap.add_argument("--config", type=Path, default=Path("configs/keepaway_1v1.toml"))
    ap.add_argument("--speeds", default="1.5:2.0:0.05", help='"start:stop:step" or "a,b,c" (m/s)')
    ap.add_argument("--episodes", type=int, default=100, help="episodes per (speed, policy)")
    ap.add_argument("--out", type=Path, help="CSV path (default: runs/calibration_<config name>.csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    speeds = parse_speeds(args.speeds)
    max_steps = cfg.shared.max_episode_steps
    print(f"[calibrate] {len(speeds)} speeds x {args.episodes} episodes; carrier speed = "
          f"{cfg.shared.max_speed * cfg.shared.carry_speed_factor:.2f} m/s; band {BAND}, target {TARGET}")
    print(f"{'red':>5} | {'flee':>6} {'(poss':>6} {'free)':>6} {'trunc':>6} | {'random':>6} | {'still':>6}")
    rows: list[dict[str, Any]] = []
    for speed in speeds:
        new = survival_table(cfg, [speed], args.episodes)
        rows += new
        by = {r["policy"]: r for r in new}
        f = by["flee"]
        print(f"{speed:5.2f} | {f['survival']:6.3f} {f['len_mean_possession'] / max_steps:6.3f} "
              f"{f['len_mean_free_ball'] / max_steps:6.3f} {f['frac_truncated']:6.0%} | "
              f"{by['random']['survival']:6.3f} | {by['still']['survival']:6.3f}", flush=True)

    out = args.out or Path(cfg.run.out_dir) / f"calibration_{args.config.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = pick_speed(rows)
    flee_best = next(r for r in rows if r["policy"] == "flee" and r["red_max_speed"] == best)
    print(f"[calibrate] recommended red_max_speed = {best:g} (flee survival {flee_best['survival']:.3f}; "
          f"red/max_speed = {best / cfg.shared.max_speed:.3f}) -> table saved to {out}")
    print(f'[calibrate] to apply: set `red_max_speed = {best:g}` under [shared] in {args.config}')
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
