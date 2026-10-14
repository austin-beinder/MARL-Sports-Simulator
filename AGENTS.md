# AGENTS.md

## Project orientation

MARL-Sports-Simulator is a PyBullet-based multi-agent reinforcement learning
course project. The repository currently contains a working 1v1 keep-away
environment that follows a Gymnasium-like reset/step contract. Later issues
will scale this toward multi-agent keep-away / 2v2 soccer.

## Current source layout

The current simulation code lives under `pybullet/`.

Important: this directory has the same name as the third-party `pybullet`
library. Existing source files use sibling imports such as `from ball import
Ball` and use `import pybullet as p` for the external dependency.

Do **not** add `pybullet/__init__.py` or casually convert this directory into a
normal Python package without first renaming it, because doing so can shadow
the third-party `pybullet` module.

## Infrastructure execution path

Run:

```bash
python pybullet/main.py --config configs/default.yaml
```

or for the GUI/debug view:

```bash
python pybullet/main.py --config configs/debug.yaml
```

Flow:

`YAML -> main.py -> runner.py -> KeepAwayEnv -> demo or RL bridge`

## Ownership boundaries

Overall infrastructure owns:

- YAML configuration loading and validation
- command-line entry point
- reproducible seeds
- run/output directories
- configuration snapshots
- metrics file locations
- generic checkpoint helpers
- GUI/debug configuration around the environment
- invoking the RL entrypoint

`keep_away_env.py` owns:

- the PyBullet connection
- reset/step semantics
- field/player/ball construction
- physics stepping
- keep-away rules
- observations and rewards

The RL issue owns:

- policy/network definitions
- PPO/IPPO/MAPPO implementation
- optimization/training
- algorithm-specific metrics

Do not create a second PyBullet connection in infrastructure. `KeepAwayEnv`
already owns the simulation connection.

## RL integration seam

The provisional contract is:

```python
def run(env, config, context):
    ...
```

configured with:

```yaml
run:
  mode: "rl"

training:
  entrypoint: "train:run"
```

The separate Gym/RL API/ICD issue should review this signature before it is
considered permanent.

## Configuration

Current `game.config` YAML keys intentionally match `KeepAwayConfig` in
`keep_away_env.py`.

When the 2v2 environment exists, add a new game/environment branch instead of
pretending the current 1v1 environment accepts player counts or arbitrary
spawn lists that it does not yet support.

## Outputs

Each run writes under `outputs/`:

- copied `config.yaml`
- `run_metadata.json`
- `metrics.csv`
- `checkpoints/`

Generated run output should not be committed.

## Debugging

`simulation.gui: true` uses the GUI connection already created by
`KeepAwayEnv`. `debug.camera` can override the environment's default camera
after reset, and `draw_labels` adds labels to the current players.

For behavior-level manual debugging, the existing joystick tool is useful
because it emits the same 2-D velocity command shape that the RL layer sends.

## Tests

Infrastructure-only tests should avoid requiring a GUI. Unit tests may add the
`pybullet/` source directory to `sys.path` because the current repository uses
sibling imports.
