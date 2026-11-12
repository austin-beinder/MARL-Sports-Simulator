# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-07
# Author:  Austin Beinder
# Usage:
#   - 2026-11-07: Generated this entire file: the curriculum runner (TOML
#     loading, the confidence gate, the per-stage difficulty levels, running
#     the stages in order with warm starts, the summary CSV and the CLI).
#   - 2026-11-07: Added --render and --debug (passed to every stage's train()).
#   - 2026-11-07: Config errors (e.g. kicking on the mock) print as CLI errors.
#   - 2026-11-07: threshold is optional (no gate: the stage runs to its cap,
#     e.g. a goto_target warm-up); the stage header names scenario and teams.
#   - 2026-11-09: pass_practice stages gate on the catch rate instead of
#     survival; curriculum.csv adds eval_frac_passed. (Austin Beinder)
#   - 2026-11-09: So do pass_practice_moving stages. (Austin Beinder)
#   - 2026-11-11: --start-at / --init-from: restart a curriculum from a later
#     stage, keeping the earlier stages' output and curriculum.csv rows.
# Prompt summary: Lay out a curriculum for keep-away: a possession stage, then
#   a free-ball stage whose ball starts close and moves further away (CACTUS).
# Human review:   Pending review
# ==============================================================================
"""
Curriculum training: run a list of training stages in order, each one starting from the
weights the stage before it ended with.

A curriculum is a TOML file in this folder (see keepaway_1v1.toml). It names a base experiment
config (configs/*.toml) and lists the stages. Each stage is an ordinary training run
(marl.rl.train.train) with its own config overrides and step cap, written to

    runs/curriculum_<name>[_<env>]_seed<seed>/<NN>_<stage>/

so evaluate.py and watch.py work on any stage. The curriculum folder also gets curriculum.csv
(one row per finished stage) and a copy of the curriculum TOML.

Difficulty levels (the CACTUS idea, Phan et al. 2024, papers/currcilum.pdf):
- A stage can step one reset option through a list of levels, e.g. max_ball_dist = 1, 2, 4, ...
  metres: the free ball starts right next to blue, then further and further away, until 0
  (anywhere on the field, the real task).
- Each level samples the whole easier range too (the ball lands anywhere within max_ball_dist),
  so the easy cases keep showing up and aren't forgotten.
- The stage moves up a level when a confidence gate passes. Each training episode scores its
  survival fraction (episode length / max_episode_steps, so 1.0 = held out until the time
  limit). Over the last `window` episodes, mean - confidence * standard error >= threshold.
  The paper uses the standard deviation; our scores are much noisier per episode than its
  per-episode completion rates, so we use the standard error (std / sqrt(window)).
- A stage ends at its step cap, or, with stop_when_passed, as soon as the gate passes on its
  last level. A stage without a threshold has no gate and always runs to its cap (e.g. a
  goto_target warm-up, whose episodes never end early, so survival says nothing).
- pass_practice and pass_practice_moving stages (2026-11-09) score each episode 1 if a teammate caught the kick, else 0,
  instead of survival (their episodes end as soon as the kick plays out), so the gate is on the
  catch rate.
- The next stage always starts from the actor. It keeps the critic too only if the scenario and
  reward are the same (a critic trained on goto_target's -distance would be wrong for keep-away).

eval.csv always measures the stage's real task (its config's own start options), not the
current level, so it shows how the curriculum is doing on the thing we care about.

Restarting part way (2026-11-11): --start-at N (a 1-based stage number or a stage name) skips
stages 1 to N-1 and starts stage N from stage N-1's final.pt, i.e. from
<NN-1>_<that stage's name in the TOML>/checkpoints/final.pt in the same curriculum folder. So you
can change stage N onward in the TOML and rerun just those, keeping the earlier stages' output.
The earlier stages' curriculum.csv rows are kept; stage N onward replaces the rest (pass
--overwrite if those stage folders already have a run). --init-from CHECKPOINT starts the first
stage that runs from any checkpoint instead.

Run it from the repo root:

    python curriculum/runner.py curriculum/keepaway_1v1.toml --seed 0
    python curriculum/runner.py my_curriculum.toml --env mock             # faster; only without kicking
    python curriculum/runner.py curriculum/keepaway_1v1.toml --render     # watch it train (PyBullet only)
    python curriculum/runner.py curriculum/passing_3v1.toml --seed 0 --start-at 3    # keep stages 1-2
"""

import argparse
import copy
import csv
import math
import shutil
import sys
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a script (python curriculum/runner.py): make `marl` importable
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from marl.common.config import PASS_PRACTICE_SCENARIOS, Config, load_config, with_overrides  # noqa: E402
from marl.common.spawn import RESET_OPTION_KEYS  # noqa: E402
from marl.rl.logger import CSVLogger  # noqa: E402
from marl.rl.train import train  # noqa: E402


def say(message: str):
    print(message, flush=True)  # flush, so redirected output shows up while the run is going


# ---------- What a curriculum file holds ----------

@dataclass
class Stage:
    name: str
    total_steps: int  # the stage's step cap
    threshold: float | None = None  # the gate: survival fraction (catch rate in pass_practice) to reach (0-1); None = no gate
    confidence: float = 2.0  # standard errors below the mean that must still reach the threshold
    window: int = 50  # how many recent training episodes the gate looks at
    option: str | None = None  # the reset option this stage steps through levels, e.g. "max_ball_dist"
    levels: list[float] = field(default_factory=list)
    stop_when_passed: bool = True  # end the stage when the gate passes on the last level
    overrides: dict = field(default_factory=dict)  # config overrides, e.g. {"shared": {"start": {...}}}


@dataclass
class Curriculum:
    name: str
    config: Path  # the base experiment config
    env: str | None = None  # "mock" or "pybullet"; None keeps the base config's run.env
    overrides: dict = field(default_factory=dict)  # config overrides for every stage
    stages: list[Stage] = field(default_factory=list)


def load_curriculum(path) -> Curriculum:
    """Read a curriculum TOML. Unknown keys raise, like the experiment configs."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    _no_unknown_keys(data, {"name", "config", "env", "overrides", "stage"}, str(path))
    if not data.get("stage"):
        raise ValueError(f"{path}: a curriculum needs at least one [[stage]]")

    stages = []
    for i, s in enumerate(data["stage"], 1):
        _no_unknown_keys(s, {f.name for f in Stage.__dataclass_fields__.values()}, f"{path} stage {i}")
        stage = Stage(**s)
        if stage.option is not None and stage.option not in RESET_OPTION_KEYS:
            raise ValueError(f"stage {stage.name!r}: option must be a reset option {sorted(RESET_OPTION_KEYS)}, "
                             f"got {stage.option!r}")
        if (stage.option is None) != (not stage.levels):
            raise ValueError(f"stage {stage.name!r}: give both option and levels, or neither")
        if stage.levels and stage.threshold is None:
            raise ValueError(f"stage {stage.name!r}: levels need a threshold (the gate that moves them up)")
        if stage.window < 2 or stage.confidence < 0:
            raise ValueError(f"stage {stage.name!r}: window must be >= 2 and confidence >= 0")
        stages.append(stage)
    if len({s.name for s in stages}) != len(stages):
        raise ValueError(f"{path}: stage names must be unique")

    config = Path(data["config"])
    return Curriculum(
        name=data["name"],
        config=config if config.is_absolute() else REPO_ROOT / config,  # repo-relative, like the commands
        env=data.get("env"),
        overrides=data.get("overrides", {}),
        stages=stages,
    )


def _no_unknown_keys(table: dict, allowed: set, where: str):
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


# ---------- The gate ----------

class ConfidenceGate:
    """
    Says when the agent is good enough to move on. It keeps the scores of the last `window`
    episodes and passes when mean - confidence * standard error >= threshold.
    """

    def __init__(self, threshold: float, confidence: float, window: int):
        self.threshold = threshold
        self.confidence = confidence
        self.scores = deque(maxlen=window)

    def add(self, scores):
        self.scores.extend(scores)

    def clear(self):
        self.scores.clear()

    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else math.nan

    def lower_bound(self) -> float:
        """mean - confidence * standard error, once the window is full (nan before that)."""
        if len(self.scores) < self.scores.maxlen:
            return math.nan
        standard_error = float(np.std(self.scores, ddof=1)) / math.sqrt(len(self.scores))
        return self.mean() - self.confidence * standard_error

    def passed(self) -> bool:
        if self.threshold is None:  # a stage without a gate never passes; it runs to its step cap
            return False
        return self.lower_bound() >= self.threshold  # nan (window not full yet) never passes


# ---------- One stage while it trains ----------

class StageCurriculum:
    """
    What train() talks to during one stage (see train's docstring): it hands out the start
    options for each rollout, watches the episodes that ended, and moves up a level (or ends
    the stage) when the gate passes.
    """

    def __init__(self, stage: Stage, base_options: dict, max_episode_steps: int, log=say, catch_gate: bool = False):
        self.stage = stage
        self.catch_gate = catch_gate  # 2026-11-09: score episodes by catch (pass_practice), not survival
        self.base_options = dict(base_options)
        self.max_episode_steps = max_episode_steps
        self.log = log
        self.gate = ConfidenceGate(stage.threshold, stage.confidence, stage.window)
        self.level = 0
        self.passed_last_level = False  # the gate passed on the last level at least once
        self.finished = False

    @property
    def last_level(self) -> int:
        return max(0, len(self.stage.levels) - 1)

    def value(self) -> float:
        """The option's value at the current level (nan for a stage without levels)."""
        return float(self.stage.levels[self.level]) if self.stage.levels else math.nan

    def reset_options(self) -> dict:
        options = dict(self.base_options)
        if self.stage.option is not None:
            options[self.stage.option] = self.value()
        return options

    def update(self, episodes) -> dict:
        if self.catch_gate:
            self.gate.add(1.0 if episode.passes else 0.0 for episode in episodes)
        else:
            self.gate.add(episode.length / self.max_episode_steps for episode in episodes)
        # Log the gate as it was when the decision was made, before a level change clears it
        row = {"cur_level": self.level, "cur_value": self.value(),
               "gate_mean": self.gate.mean(), "gate_lower": self.gate.lower_bound()}

        if self.gate.passed():
            if self.level < self.last_level:
                self.level += 1
                self.gate.clear()  # the new level has to earn its own pass
                self.log(f"[curriculum] {self.stage.name}: gate passed (lower bound {row['gate_lower']:.3f} "
                         f">= {self.stage.threshold}), {self.stage.option} -> {self.value():g}")
            elif not self.passed_last_level:
                self.passed_last_level = True
                self.log(f"[curriculum] {self.stage.name}: gate passed on the last level "
                         f"(lower bound {row['gate_lower']:.3f} >= {self.stage.threshold})")
                self.finished = self.stage.stop_when_passed
        return row


# ---------- Running a whole curriculum ----------

def stage_configs(curriculum: Curriculum, seed: int | None = None, env: str | None = None) -> list[Config]:
    """
    The resolved config of every stage, built up front so a typo in stage 3 fails before stage 1
    trains for an hour. Every level's value is checked too (e.g. max_ball_dist must exceed
    possession_radius).
    """
    base = load_config(curriculum.config)
    run = {}
    if curriculum.env is not None or env is not None:
        run["env"] = env or curriculum.env
    if seed is not None:
        run["seed"] = seed
    base = with_overrides(base, {**curriculum.overrides, "run": {**curriculum.overrides.get("run", {}), **run}})

    configs = []
    for stage in curriculum.stages:
        overrides = copy.deepcopy(stage.overrides)
        overrides.setdefault("run", {})["total_steps"] = stage.total_steps
        cfg = with_overrides(base, overrides)
        for value in stage.levels:
            with_overrides(cfg, {"shared": {"start": {stage.option: value}}})  # raises if a level is invalid
        configs.append(cfg)
    return configs


SUMMARY_COLUMNS = ["stage", "name", "iterations", "env_steps", "option", "level_reached", "value_reached",  # AI GENERATED
                   "gate_passed", "eval_len_mean", "eval_frac_truncated", "eval_frac_passed", "out_dir"]  # AI GENERATED


def default_out_dir(curriculum: Curriculum, cfg: Config) -> Path:
    suffix = "" if cfg.run.env == "mock" else f"_{cfg.run.env}"
    return Path(cfg.run.out_dir) / f"curriculum_{curriculum.name}{suffix}_seed{cfg.run.seed}"


### AI GENERATED (2026-11-11: restarting from a later stage)
def stage_number(curriculum: Curriculum, start_at) -> int:
    """The 1-based stage number for --start-at: a number (int or digits) or a stage name."""
    names = [s.name for s in curriculum.stages]
    if isinstance(start_at, int) or str(start_at).isdigit():
        k = int(start_at)
        if not 1 <= k <= len(names):
            raise ValueError(f"--start-at {k}: the curriculum has stages 1 to {len(names)}")
        return k
    if start_at not in names:
        raise ValueError(f"--start-at {start_at!r}: no such stage; the stages are {names}")
    return names.index(start_at) + 1


def _earlier_summary_rows(path: Path, start: int) -> list[dict]:
    """curriculum.csv's rows for the stages before `start` (kept when a curriculum restarts there)."""
    if start <= 1 or not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r["stage"]) < start]
### END AI GENERATED


def run_curriculum(curriculum: Curriculum, *, seed=None, env=None, out_dir=None, overwrite=False,
                   curriculum_path=None, render=False, print_collisions=False, log=say,
                   start_at=1, init_from=None) -> list[dict]:
    """
    Train every stage in order. Returns one summary row per stage it trained (also in
    curriculum.csv). render shows each stage's training env in the PyBullet GUI; print_collisions
    turns on the sim's collision prints. Both need the PyBullet env.

    start_at (2026-11-11): the first stage to train (1-based number or name). Later starts
    warm-start from the previous stage's final.pt in this curriculum folder and keep the earlier
    stages' curriculum.csv rows. init_from: a checkpoint for the first trained stage instead.
    """
    configs = stage_configs(curriculum, seed, env)
    if (render or print_collisions) and configs[0].run.env != "pybullet":
        raise ValueError("render and print_collisions need the PyBullet env")
    root = Path(out_dir) if out_dir is not None else default_out_dir(curriculum, configs[0])
    stage_dirs = [root / f"{k:02d}_{stage.name}" for k, stage in enumerate(curriculum.stages, 1)]
    ### AI GENERATED (2026-11-11: restarting from a later stage)
    start = stage_number(curriculum, start_at)
    if init_from is None and start > 1:
        init_from = stage_dirs[start - 2] / "checkpoints" / "final.pt"
    if init_from is not None and not Path(init_from).exists():
        raise FileNotFoundError(f"{init_from} doesn't exist: stage {start} starts from it. Run the earlier stages "
                                f"first, or pass --init-from with the checkpoint to start from")
    if not overwrite:
        taken = [d for d in stage_dirs[start - 1:] if d.exists() and any(d.iterdir())]
        if taken:
            raise FileExistsError(f"{taken[0]} already has a run; pass --overwrite to replace it")
    root.mkdir(parents=True, exist_ok=True)
    kept = _earlier_summary_rows(root / "curriculum.csv", start)  # read before the summary file is rewritten
    if curriculum_path is not None:
        shutil.copyfile(curriculum_path, root / "curriculum.toml")
    if start > 1 or init_from is not None:
        log(f"[curriculum] starting at stage {start}/{len(configs)} {curriculum.stages[start - 1].name}"
            + (f", from {init_from}" if init_from is not None else ""))
    ### END AI GENERATED

    rows = []
    with CSVLogger(root / "curriculum.csv") as summary:
        for old in kept:  # AI GENERATED (2026-11-11): the stages before start_at, as they were
            summary.log({key: old.get(key, "") for key in SUMMARY_COLUMNS})
        for k, (stage, cfg, stage_dir) in enumerate(zip(curriculum.stages, configs, stage_dirs), 1):
            if k < start:  # AI GENERATED (2026-11-11): already trained; init_from is its final.pt
                continue
            levels = f", {stage.option} levels {stage.levels}" if stage.option else ""
            gate = f"gate {stage.threshold} over {stage.window} episodes" if stage.threshold is not None else "no gate"
            log(f"[curriculum] stage {k}/{len(configs)} {stage.name} ({cfg.shared.scenario}, "
                f"{cfg.shared.n_blue}v{cfg.shared.n_red}): up to {stage.total_steps:,} steps, {gate}{levels}")
            controller = StageCurriculum(stage, cfg.shared.start.to_reset_options(), cfg.shared.max_episode_steps, log,
                                         catch_gate=cfg.shared.scenario in PASS_PRACTICE_SCENARIOS)
            result = train(cfg, stage_dir, overwrite=overwrite, log=log, init_from=init_from, curriculum=controller,
                           render=render, print_collisions=print_collisions)
            init_from = result.out_dir / "checkpoints" / "final.pt"

            row = {
                "stage": k,
                "name": stage.name,
                "iterations": result.iterations,
                "env_steps": result.iterations * cfg.rl.ppo.rollout_steps,
                "option": stage.option or "",
                "level_reached": controller.level,
                "value_reached": controller.value(),
                "gate_passed": controller.passed_last_level,
                "eval_len_mean": result.final_eval.get("len_mean", math.nan),
                "eval_frac_truncated": result.final_eval.get("frac_truncated", math.nan),
                "eval_frac_passed": result.final_eval.get("frac_passed", math.nan),  # 2026-11-09
                "out_dir": str(result.out_dir),
            }
            assert list(row) == SUMMARY_COLUMNS  # AI GENERATED (2026-11-11): kept rows use the same columns
            summary.log(row)
            rows.append(row)
    log(f"[curriculum] done -> {root}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a curriculum: stages in order, each warm-started from the last.")
    ap.add_argument("curriculum", type=Path, help="curriculum TOML, e.g. curriculum/keepaway_1v1.toml")
    ap.add_argument("--seed", type=int, help="override run.seed for every stage")
    ap.add_argument("--env", choices=("mock", "pybullet"), help="override the curriculum's env")
    ap.add_argument("--out-dir", type=Path, help="default: <run.out_dir>/curriculum_<name>[_<env>]_seed<seed>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing curriculum run in --out-dir")
    ap.add_argument("--render", action="store_true",
                    help="show the training env in the PyBullet GUI (slow; runs as fast as it can, not real time)")
    ap.add_argument("--debug", action="store_true", help="turn on the PyBullet sim's collision prints")
    ### AI GENERATED (2026-11-11: restarting from a later stage)
    ap.add_argument("--start-at", default="1", metavar="STAGE",
                    help="first stage to train (number from 1, or name); the earlier stages' output is kept and "
                         "this stage starts from the previous stage's final.pt")
    ap.add_argument("--init-from", type=Path, metavar="CHECKPOINT",
                    help="start the first trained stage from this checkpoint instead")
    ### END AI GENERATED
    args = ap.parse_args(argv)

    curriculum = load_curriculum(args.curriculum)
    try:
        rows = run_curriculum(curriculum, seed=args.seed, env=args.env, out_dir=args.out_dir,
                              overwrite=args.overwrite, curriculum_path=args.curriculum,
                              render=args.render, print_collisions=args.debug,
                              start_at=args.start_at, init_from=args.init_from)  # AI GENERATED (2026-11-11)
    except (ValueError, NotImplementedError, FileNotFoundError, FileExistsError) as e:  # bad curriculum or config, e.g. kicking on the mock
        ap.error(str(e))
    for row in rows:
        reached = f", {row['option']} reached {row['value_reached']:g}" if row["option"] else ""
        say(f"  {row['stage']}. {row['name']}: {row['env_steps']:,} steps{reached}, gate passed: {row['gate_passed']}, "
              f"eval len {row['eval_len_mean']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
