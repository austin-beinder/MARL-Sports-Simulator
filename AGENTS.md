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
Prompt summary: Update AGENTS.md with the current state of the repo.
Human review:   Pending review
-->

# AGENTS.md

## Project orientation

MARL-Sports-Simulator is a WashU CSE 5106 (Multi-Agent Systems) course project:
multi-agent reinforcement learning on a PyBullet soccer-style simulation.

Milestone 1, the current state: **keep-away, 1 blue (learner) vs 1 scripted
red, movement only (no kicking)**, trained with PPO / MAPPO-style centralized
critic. Both envs also run N blue vs M red and the `goto_target` sanity check.
Kicking is in the interface but not implemented in either env yet.

## Layout

| Path | What it is |
| --- | --- |
| `pybulletGame/` | The PyBullet game: `KeepAwayEnv`, `Entity`/`Player`/`Ball`/`Field`, joystick tool, unittest tests |
| `marl/common/` | Config schema + TOML loading (`config.py`), spawn sampling (`spawn.py`). Torch-free. |
| `marl/envs/` | `make_env` selector, `MockEnv` (pure-numpy ICD env), `PyBulletEnv` (ICD adapter around `KeepAwayEnv`) |
| `marl/rl/` | PPO: networks, rollout, buffer, GAE, rewards, observations, training loop, evaluation, checkpoints, CSV logging |
| `marl/` (top level) | `baselines.py` (random/still/flee/oracle), `calibrate.py`, `plots.py`, `watch.py` |
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
  ball (`KeepAwayConfig.ball_is_target`).
- Possession in PyBullet is contact-based (player touching the ball, or an
  opponent touching the carrier), not `possession_radius`. So the same policy
  survives longer there than in the mock. That difference is accepted. Red's
  P-controller (`KeepAwayConfig.red_gain = 20`) only slows within 9 cm of the
  ball, so in practice it chases at full speed, like the mock's red.
- `MockEnv` implements the same ICD in numpy. It's fast, so most tests and quick
  experiments use it. The ICD conformance tests run against both envs.

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

## Configuration

- TOML, three sections: `[shared]` (values env and RL must agree on, ICD §10,
  including `[shared.start]` reset options), `[rl]` (PPO, reward, critic
  input, observation options), `[run]` (seeds, budget, eval schedule,
  `out_dir`, `env`).
- Loading is strict: unknown keys or option values raise `ValueError`; options
  that exist in the interface but aren't built yet raise `NotImplementedError`
  (see `OPTIONS` in `config.py`).
- `run.env` defaults to `"mock"`. Pass `--env pybullet` (or set it) to train on
  the real sim. Override any key with `--set section.key=value`.
- `KeepAwayConfig`'s own defaults (21 × 13.6 m field, blue 3.0 m/s, etc.)
  are only used when running `keep_away_env.py` directly. Training uses
  `SharedConfig` values (16 × 10 m, 2.0 m/s, `control_k = 12`). Red's
  1.8 m/s is the same in both.
- Both envs support `keepaway` and `goto_target`, any team sizes the config
  allows, and only `kick_trigger = 0`. `PyBulletEnv` also needs
  `physics_dt = 1/240` (pybullet's fixed time step).

## Common commands

Run from the repo root with the venv active.

```bash
python scripts/train.py --config configs/keepaway_1v1.toml --env pybullet --seed 0
python scripts/train.py --config configs/goto_target.toml          # mock env (add --env pybullet for the sim)
python scripts/evaluate.py runs/keepaway_1v1_pybullet_seed0
python scripts/watch.py runs/keepaway_1v1_pybullet_seed0           # newest checkpoint in the GUI
python scripts/watch.py --baseline flee
python scripts/calibrate.py --config configs/keepaway_1v1.toml
python scripts/plot.py
python pybulletGame/keep_away_env.py                               # play blue with the on-screen joystick
```

`train.py` also takes `--render` (GUI on the training env), `--debug` (sim
collision prints), `--total-steps`, `--out-dir` and `--overwrite`. Each script
has `--help`.

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

- `pytest` covers `tests/` only (`pyproject.toml`). It takes about 2 minutes.
  The 2 skips are expected: the kick-with-possession conformance test, on
  each env, since neither implements kicking yet.
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
