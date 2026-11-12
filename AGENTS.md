<!--
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-10-25
Author:  Austin Beinder
Usage:
  2026-10-25: Rewrote this file for the repo as it stands after PRs #22 and #23
    (marl/ RL layer, pybulletGame/ rename, PyBullet adapter). The issue #10
    version described a YAML/runner infrastructure that was never merged.
  2026-10-31: PyBulletEnv now runs N blue vs M red and goto_target; updated
    the limits, the expected test skips and the dynamics test file.
  2026-10-31: Replaced the one-simulation-per-process constraint: every env
    has its own pybullet connection.
  2026-11-01: (Claude Code (Claude Opus 5.5), prompted by JC Stinson III)
    PyBullet is now the default run.env; updated the Configuration notes, the
    run-name convention and the command examples (backlog task 3).
  2026-11-03: (Claude Code (Claude Opus 5.5), prompted by JC Stinson III)
    Added scripts/regress.py and the regression-record rule (backlog task 4).
Prompt summary: Update AGENTS.md with the current state of the repo.
Human review:   Pending review
-->

# AGENTS.md

## Project orientation

MARL-Sports-Simulator is a WashU CSE 5106 (Multi-Agent Systems) course project:
multi-agent reinforcement learning on a PyBullet soccer-style simulation.

Milestone 1, the current state: **keep-away, 1 blue (learner) vs 1 scripted
red, movement only**, trained with PPO / MAPPO-style centralized critic. Both
envs also run N blue vs M red and the `goto_target` sanity check. Kicking is
built in the PyBullet env only (`rl.kick_enabled = true`, which needs
`run.env = "pybullet"`); the mock has no ball physics.

## Layout

| Path | What it is |
| --- | --- |
| `pybulletGame/` | The PyBullet game: `KeepAwayEnv`, `Entity`/`Player`/`Ball`/`Field`, joystick tool, unittest tests |
| `marl/common/` | Config schema + TOML loading (`config.py`), spawn sampling (`spawn.py`). Torch-free. |
| `marl/envs/` | `make_env` selector, `MockEnv` (pure-numpy ICD env), `PyBulletEnv` (ICD adapter around `KeepAwayEnv`) |
| `marl/rl/` | PPO: networks, rollout, buffer, GAE, rewards, observations, training loop, evaluation, checkpoints, CSV logging |
| `marl/` (top level) | `baselines.py` (random/still/flee/oracle), `calibrate.py`, `plots.py`, `watch.py` |
| `curriculum/` | Curriculum training: `runner.py` (everything) and curriculum TOMLs such as `keepaway_1v1.toml` |
| `scripts/` | Thin CLI wrappers: `train.py`, `evaluate.py`, `watch.py`, `calibrate.py`, `plot.py` |
| `configs/` | Experiment configs: `keepaway_1v1.toml`, `goto_target.toml` |
| `docs/ICD.md` | Env ↔ RL interface control document (DRAFT v0.2) |
| `tests/` | pytest suite for `marl/`, including ICD conformance tests run against both envs |
| `examples/` | Early PyBullet prototypes, not used by the project code |
| `papers/` | Reference papers and the project proposal |
| `_env_setup.py`, `*.ps1`, `Other_OS_Setup_Scripts/` | One-click venv setup scripts |

## Architecture

```
configs/*.toml -> marl.common.config.Config
               -> marl.envs.make_env(cfg)  -- run.env = "mock"     -> MockEnv
                                           -- run.env = "pybullet" -> PyBulletEnv -> pybulletGame.KeepAwayEnv
               -> marl.rl.train (featurize s -> obs, compute rewards, PPO update)
```

- **The ICD (`docs/ICD.md`) is the contract.** The env reports a global state
  `s`, events, and `terminated`/`truncated`. The RL layer builds per-agent
  observations and computes **all rewards**; the env never does.
  `step(actions)` returns `(s, terminated, truncated, info)`, with no reward and
  no auto-reset.
- `KeepAwayEnv` is the raw game and still has its own Gymnasium-style
  `step(action) -> (obs, reward, terminated, truncated, info)` with its own
  reward and observation. `PyBulletEnv` drops those and translates to the ICD:
  ICD actions in [-1, 1] → m/s, `SharedConfig` → `KeepAwayConfig`, spawns from
  `marl.common.spawn.sample_spawn`, red possession terminal after one tick,
  blue's "hold for 20 s" win switched off, teammate handoff off (only a kick
  releases the ball, ICD §8), and for `goto_target` an untouchable, immobile
  ball (`KeepAwayConfig.ball_is_target`). Kicks: `kick_vec` → the ball's
  velocity (`max_kick_speed`-capped), passed as `KeepAwayEnv.step(action,
  kicks=...)`. No re-grab cooldown. `Ball.play_area` keeps a kicked ball
  inside the walls; a point-blank kick into a wall rebounds to the kicker.
- The loose ball slows by `dv/dt = -ball_damping · v · (1 + |v|)` (pybullet's
  link damping), so a kick at `v` rolls `ln(1 + v) / ball_damping` m. Walls
  stop it dead.
- Kick heads (`marl/rl/networks.py`): a Bernoulli `kick_trigger` (starts at
  ~5% per step) and a Beta `kick_vec`. They are masked by the agent's own
  possession flag in `o_i` and by the trigger, so heads that can't affect the
  env add nothing to the PPO ratio.
- Possession in PyBullet is contact-based (player touching the ball, or an
  opponent touching the carrier), not `possession_radius`. So the same policy
  survives longer there than in the mock. That difference is accepted.
  `Ball.check_held_contacts` runs every physics tick so a contact that began
  before blue got the ball still steals (`on_new_collision` only fires on a
  contact's first tick). Each contact gets one roll (`Ball.contested`); the
  ball changes hands at most once per tick. Red's
  P-controller (`KeepAwayConfig.red_gain = 20`) only slows within 9 cm of the
  ball, so in practice it chases at full speed, like the mock's red.
- `MockEnv` implements the same ICD in numpy. It's fast, so most tests and quick
  experiments use it (tests set `run.env = "mock"` explicitly, since the default
  is PyBullet). The ICD conformance tests run against both envs.

## Hard constraints

- **Never name a directory `pybullet`.** It shadows the third-party `pybullet`
  library. That is why the game lives in `pybulletGame/`.
- **`pybulletGame/` imports.** Files inside it use bare sibling imports
  (`from ball import Ball`) so they still run as scripts; its `__init__.py`
  puts the folder on `sys.path`. Code outside the folder must import from the
  package (`from pybulletGame import KeepAwayEnv, KeepAwayConfig, Entity, Team`),
  **not** from submodules like `pybulletGame.entity`, which would load a second
  copy of the module whose classes and settings the game never sees.
- **One PyBullet simulation per env.** Each `KeepAwayEnv` opens its own
  pybullet connection, and every pybullet call in `pybulletGame/` passes
  `physicsClientId` (the `physics_client` / `.client` of `Entity`, `Ball`,
  `Player`, `Field`, `Wall`). `Entity._registry` is keyed by
  `(client, body_id)`, since body ids restart in every simulation. Keep both
  when adding pybullet code: a call without `physicsClientId` silently acts on
  connection 0. Any number of envs can run in one process (training evaluates
  in-process), but pybullet allows only one GUI per process. Close envs you
  create: `KeepAwayEnv.close()` frees the connection, and runs only once
  because pybullet reuses freed connection ids.
- **`marl/common/config.py` stays torch-free** so the env side can import it.

## Curriculum

- `curriculum/` holds only curriculum TOMLs, `__init__.py` and `runner.py`.
  Keep all curriculum logic in `runner.py`, in plain classes; the team
  prefers that over many small modules.
- A curriculum is a list of stages. Each stage is a normal `train()` run with
  its own overrides and step cap, warm-started (`init_from`) from the previous
  stage's `final.pt`.
- A stage can step one reset option through `levels`: CACTUS-style, from
  `papers/currcilum.pdf`. `max_ball_dist` starts the free ball near blue and
  grows to 0, which means anywhere. It moves up a level when a confidence gate
  passes on the training episodes' survival fraction: the mean minus
  `confidence` standard errors must reach `threshold`, over `window` episodes.
- `train()` exposes two hooks for this: `init_from` and `curriculum` (an
  object with `reset_options()`, `update(episodes)` and `finished`).
  `eval.csv` still measures the stage's real task.
- Warm starts work across team sizes (`checkpoint.warm_start`). Each
  input is built from per-player blocks (`obs_blocks`, `state_blocks`). Old
  columns move to their slot in the new layout, and new players' columns get
  zero weight, so a grown 2v1 network first acts exactly like the 1v1 one.
  A team can also shrink (1v1 -> 3v0 for pass practice): the removed slots'
  columns are dropped.
- `curriculum/keepaway_1v1.toml` has these stages:
  1. `go_to_target`: goto_target warm-up, no gate, fixed budget.
  2. `possession`: 1v1. Commented out for now.
  3. `free_ball`: 1v1, `max_ball_dist` 1, 2, then 3 m. Red (1.8 m/s) is
     faster than the carrier (1.5 m/s), so longer distances were nearly
     unwinnable.
  4. `two_v_one`: `n_blue = 2`, mixed starts.
  5. `slow_two_v_one`: the same 2v1 with `free_speed_factor = 0.875`, so
     blue is slower than red even without the ball (1.75 vs 1.8 m/s).

  Kicking is on in every stage, so it needs PyBullet.
- `curriculum/passing_3v1.toml` aims at monkey in the middle: `go_to_target`,
  `free_ball`, `catch_moving` (moving pass practice, teammates 3.9 m away),
  then `catch_moving_long` / `_longer` / `_longest` (minimum pass 1, 2, 3 m,
  length bonus on), all four with `kicker_clearance`, then
  `keepaway_from_catch` (3v1 keep-away whose resets start in the catch
  layout, `shared.start.catch_start_mix` stepped 1, 0.5, 0), then a short
  `go_to_target_again` (1v0), then 3v1 keep-away from a loose ball 1, 2,
  then 3 m away, with `free_speed_factor = 0.875`.
- `shared.start.catch_start_mix` (keepaway only, default 0): the share of
  resets that use the pass practice layout (a blue holding the ball,
  teammates `pass_distance` away) with each red anywhere at least
  `min_red_ball_dist` from the ball. At 0 the spawn draws are unchanged.
- The `pass_practice` scenario (PyBullet only, needs kicking): 3 frozen blues,
  no red, one kick per episode. Each teammate stands
  `shared.start.pass_distance` (±15%) from the holder. `PyBulletEnv` ends the
  episode (`terminated`) once the kick has played out: a blue holds the ball
  again, or the ball can't roll as far as the nearest teammate any more.
  Reward `rl.reward.base = "pass_practice"`, paid once at the end: 1 for a
  catch, else `0.5 · exp(-gap / 1 m)` for the ball's closest approach
  (`PassMemory.closest`), 0 without a kick. Two options, off by default:
  `rl.reward.pass_length_bonus` pays a catch `0.75 + 0.25 · min(1, length /
  4 m)` (kick point to catch point; every catch still beats every miss), and
  `rl.reward.min_pass_distance` d (m, any scenario) makes a catch nearer
  than d to the kick point no pass, and drops teammates within d + 0.5 m of
  it from the closest approach. The `kicker_clearance` shaping term
  (coef c) charges the kicker `c · (1 - d / 1.5 m)` a step while its kick is
  in play (d = its distance to the ball) and `10 c` for taking the ball
  back, so it stops chasing its own pass. The actor isn't told who kicked.
  These stages gate on the catch
  rate (`StageCurriculum(catch_gate=True)`) instead of survival.
  `train.csv` / `eval.csv` log `passes_mean` and `frac_passed` for every
  scenario.
- `pass_practice_moving`: the same, but the blues can move (not in
  `FROZEN_SCENARIOS`; both are in `PASS_PRACTICE_SCENARIOS`). A miss ends
  only once no teammate could run to the ball before the time limit.
  `curriculum/passing_only_3v1.toml` runs it as `catch_moving` (plain
  reward), then `catch_moving_long` (1 m minimum pass, length bonus).
- The `passing_drill` scenario (PyBullet only, needs kicking): the blues
  are frozen on a circle of `shared.start.drill_radius` round red, and one
  starts with the ball. `PyBulletEnv` zeroes their `move`. The actor masks
  the movement head out of the log-prob and entropy (`Actor.move_enabled`),
  so a warm-started movement skill doesn't drift. Its reward is
  `rl.reward.base = "passing_drill"`: each step a blue holds the ball pays
  the number of passes completed so far this episode (0 before the first
  catch, then 1, 2, ...). The `pass_progress` and
  `pass_completed` shaping terms add credit for kicks toward a teammate and
  for completed passes; they are not potential-based. These terms need
  memory across steps (`PassMemory`), so every episode loop passes its own
  `RewardModule.new_episode()` to `parts()`. One `RewardModule` is shared
  by training and the in-process evaluator, which can both be mid-episode.
- A stage without `threshold` has no gate and runs to its step cap. The
  critic carries into the next stage only if the scenario and reward match;
  otherwise only the actor does.
- Output goes to `runs/curriculum_<name>[_<env>]_seed<seed>/<NN>_<stage>/`,
  plus `curriculum.csv`.
- `--start-at N` (a stage number or name) restarts a curriculum at stage N
  after editing the TOML: stages before N are skipped, stage N warm-starts
  from `<N-1>_<name>/checkpoints/final.pt` in the same folder (that stage's
  name in the current TOML), and their `curriculum.csv` rows are kept. Stage
  N onward needs `--overwrite` if it already ran. `--init-from CHECKPOINT`
  starts the first trained stage from any checkpoint instead.

## Configuration

- TOML, three sections: `[shared]` (values env and RL must agree on, ICD §10,
  including `[shared.start]` reset options), `[rl]` (PPO, reward, critic
  input, observation options), `[run]` (seeds, budget, eval schedule,
  `out_dir`, `env`).
- Loading is strict: unknown keys or option values raise `ValueError`; options
  that exist in the interface but aren't built yet raise `NotImplementedError`
  (see `OPTIONS` in `config.py`).
- `run.env` defaults to `"pybullet"`, the real sim. Pass `--env mock` (or set
  `run.env = "mock"`) for the fast stand-in. Override any key with
  `--set section.key=value`.
- Run directories always name their env: `runs/<config>_<env>_seed<N>/`
  (e.g. `keepaway_1v1_pybullet_seed0`, `keepaway_1v1_mock_seed0`), and
  calibration tables are `calibration_<config>_<env>.csv`. Runs from before
  2026-11-01 (e.g. `runs/Old_mock_env/`, `results/golden/`) have the old names:
  no suffix for the mock.
- A saved run whose config has no `run.env` (saved before 2026-10-18, when the
  mock was the only env) loads as the mock: `saved_config_from_dict` in
  `config.py`, used by `load_actor`, so `evaluate.py`/`watch.py` on old runs
  rebuild the env they trained on.
- `KeepAwayConfig`'s own defaults (21 × 13.6 m field, blue 3.0 m/s, etc.)
  are only used when running `keep_away_env.py` directly. Training uses
  `SharedConfig` values (16 × 10 m, 2.0 m/s, `control_k = 12`). Red's
  1.8 m/s is the same in both.
- Blue's speed is `max_speed × carry_speed_factor` with the ball and
  `max_speed × free_speed_factor` without it (default 1.0). Slow blue down
  with `free_speed_factor`, not `max_speed`: `max_speed` is also the
  velocity unit in every observation and the action scale.
- Both envs support `keepaway` and `goto_target` and any team sizes the config
  allows; only `PyBulletEnv` runs `passing_drill`, `pass_practice` and
  `pass_practice_moving`. Only `PyBulletEnv` kicks; `MockEnv` raises on a kick it would have
  to apply. `PyBulletEnv` also needs `physics_dt = 1/240` (pybullet's fixed
  time step).

## Common commands

Run from the repo root with the venv active.

```bash
python scripts/train.py --config configs/keepaway_1v1.toml --env pybullet --seed 0
python scripts/train.py --config configs/goto_target.toml          # mock env (add --env pybullet for the sim)
python curriculum/runner.py curriculum/keepaway_1v1.toml --seed 0  # PyBullet only (it kicks)
python curriculum/runner.py curriculum/passing_3v1.toml --seed 0   # the passing drill, then 3v1
python curriculum/runner.py curriculum/passing_3v1.toml --seed 0 --start-at 3 --overwrite  # keep stages 1-2
python scripts/evaluate.py runs/keepaway_1v1_pybullet_seed0
python scripts/watch.py runs/keepaway_1v1_pybullet_seed0           # newest checkpoint in the GUI
python scripts/watch.py --baseline flee
python scripts/calibrate.py --config configs/keepaway_1v1.toml     # -> runs/calibration_keepaway_1v1_pybullet.csv
python scripts/plot.py
python scripts/regress.py                                          # regression check vs results/silver (~1 min)
python pybulletGame/keep_away_env.py                               # play blue with the on-screen joystick
```

`train.py` also takes `--render` (GUI on the training env), `--debug` (sim
collision prints), `--total-steps`, `--out-dir`, `--overwrite` and
`--init-from CHECKPOINT` (warm start). Each script has `--help`.

## Outputs

Training writes `runs/<config name>[_<env>]_seed<seed>/` with `config.json`,
`train.csv`, `eval.csv` and `checkpoints/` (`iter_*.pt`, `final.pt`).
Figures go to `runs/figures/`. `runs/` is gitignored; never commit run output.

## Tests

Both suites are GUI-free.

```bash
python -m pytest
python -m unittest discover -s pybulletGame/tests -p "*_test.py" -t pybulletGame
```

- **Regression record:** `python scripts/regress.py` retrains the cases in
  `results/silver/suite.toml` (seeds 90-91, short runs) and compares them with the
  frozen outputs bit for bit. A change that shouldn't affect results must come back
  all `IDENTICAL`; commit the rewritten `results/silver/report.md` with it. For an
  intended change, `--update --reason "..."` and commit `results/silver/`. See the
  README's "Regression tests (silver record)".
- `pytest` covers `tests/` only (`pyproject.toml`). It takes about 2 minutes.
  The 1 skip is expected: the kick-with-possession conformance test on the
  mock, which doesn't kick.
- `tests/test_env_dynamics.py` checks the game rules both envs share, through
  each env's `set_state_for_testing` hook; `tests/test_mock_env.py` pins the
  mock's exact kinematics.
- The `pybulletGame` tests are `unittest`-style `*_test.py` files and are not
  collected by pytest.

## Environment

- Python 3.13 (the code needs ≥ 3.11 for `tomllib`). The team venv is
  `.venv-CSE-5106-Project-Beinder-Stinson-Wang/`, created by
  `__DevEnvSetup.ps1` / `_env_setup.py`. A plain `.venv/` also works.
- Dependencies: `requirements.in` → `pip-compile` → `requirements.txt`. See
  `Update_Requirements.md`.
- The repo is often checked out under OneDrive on Windows. OneDrive can make
  folders read-only, which is why `--overwrite` empties `checkpoints/` rather
  than deleting it.

## Code conventions

- **AI attribution is required.** Files generated with an AI tool start with an
  `AI CITATION` header (Tool, Date, Author, Usage log with dated entries,
  Prompt summary, Human review); see any file in `marl/` for the format. AI
  edits inside other files are wrapped in `### AI GENERATED` …
  `### END AI GENERATED`, or marked with a trailing `# AI GENERATED` on single
  lines. When you change a file, keep its markers and add a dated Usage entry.
- Comments cite design decisions by number (`#21`, `#23`) and ICD sections
  (`ICD §7`). Keep those references when editing nearby code.

## Known stale spots

- `README.md` says `python example.py`; the file is now `examples/example.py`.
- `_env_setup.py`'s `DEMO_SCRIPT` still points at a root `example.py` (TODO in
  the file), so the `_SetupAndRunDemo` scripts don't find a demo.
- `requirements.txt` was not regenerated after `requirements.in` gained `torch`,
  `pytest` and `matplotlib`, and `Update_Requirements.md` mentions
  `requirements-dev.*` files that don't exist.
- `docs/ICD.md` is still a draft with **[OPEN]** items (§12).
