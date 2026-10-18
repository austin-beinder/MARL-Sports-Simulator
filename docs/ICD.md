<!--
AI CITATION
Tool:    Claude Code (Claude Opus 5.5)
Date:    2026-10-04
Author:  JC Stinson III (LifeHackerIII)
Usage:
  2026-10-04: Drafted this ICD from design decisions JC Stinson III made
    while designing the RL layer for issue #7 (mentor-mode sessions on
    2026-09-27, 2026-10-03 and 2026-10-04). The AI turned those decisions
    into a specification format; the design choices are the student's.
  2026-10-10: Repaired this header after Prettier mangled it (no content change).
  2026-10-10: v0.2. Merged the student's triage of a multi-agent panel review:
    the possession semantics section, mid-block termination, kick release
    formula, red speed, arena walls, max_episode_steps, the go-to-target
    scenario, reset/info details, action types, and conformance tests.
  2026-10-14: Added the returned-state snapshot rule (§5) and its conformance
    test (§11). The PBRS reward term needs them; the student approved the change.
Prompt summary: Draft the Gym/RL interface control document (issue #11) from
  the student's issue #7 design decisions, for teammates to review.
Human review:   Pending review (student added the axis convention, 2026-10-10)
-->

# Gym ↔ RL Layer Interface Control Document (ICD)

**Status:** DRAFT v0.2, proposed for team review (issue #11). Nothing here is final until the team
agrees. Items marked **[OPEN]** need a team decision; items marked **[PROPOSED]** are the RL side's
proposal, which the env owner may change.
**Scope:** milestone 1 (keep-away, 1 blue vs 1 red, movement only), written so it extends to N blue vs
M red, and to kicking, without changing the interface.

## 1. Who does what

| Responsibility                                                      | Env / sim (PyBullet side) | RL layer             |
| ------------------------------------------------------------------- | ------------------------- | -------------------- |
| Physics, collisions, game rules (possession, carry slowdown, kicks) | ✅                        |                      |
| Scripted red behavior                                               | ✅                        |                      |
| Spawning / resets (start distribution, seeding)                     | ✅                        | passes the options   |
| Deciding when an episode ends (`terminated` / `truncated`)          | ✅                        | only reads the flags |
| Reporting the **global state** `s` and events                       | ✅                        |                      |
| Building each agent's observation from `s`                          |                           | ✅                   |
| Rewards (the env never computes rewards)                            |                           | ✅                   |
| Choosing actions (the policy)                                       |                           | ✅                   |
| Scaling action _fractions_ into physical units, applying speed caps | ✅                        |                      |

Principle: the env is a **game simulator** with a clean API. Anything RL-specific (observations,
rewards, how actions are parameterized) stays on the RL side, so either side can change without
breaking the other. The RL side develops against a **mock env** that implements this ICD exactly.

## 2. Conventions

- **2D for milestone 1.** Positions and velocities are `[x, y]`. (A z component may be added later.)
  Bodies spawn at resting height on the ground plane.
- **Units:** SI (meters, m/s, seconds). **World frame**, origin at the field center. +x along the
  field's long side, +y across the short side.
- **Entity ordering is fixed by ID** (`blue[0..N-1]`, `red[0..M-1]`), and IDs stay stable for the
  whole episode. The RL side does its own reordering.
- **Seeding:** the same `seed` + `options` must reproduce the same episode (deterministic resets,
  including which start type is drawn, and a deterministic red script given the same blue actions).
- **Types:** everything crossing the interface is plain Python (`float`, `int`, `bool`, `list`,
  `dict`), not numpy or torch types.

## 3. Timing: physics rate vs. control rate (action repeat)

- The physics step is `physics_dt` (PyBullet default 1/240 s).
- The RL layer acts every `control_k` physics ticks. **`env.step(actions)` applies the given actions,
  holds them constant for `control_k` ticks, then returns the new state.** One `step()` call = one
  **control step**.
- **Defaults:** `physics_dt = 1/240`, `control_k = 12` → 20 Hz control.
- Both values are **shared config** (the RL layer derives its discount factor from them, so the env
  and RL layer must agree on them).
- **Start-of-step rule:** any decision that depends on possession during a step (e.g. whether a
  kick is applied) uses possession **at the start of the step**. If an agent gains the ball partway
  through the `control_k` ticks, it can kick at the next step.
- **Mid-block termination:** the env checks the terminal condition (§7) **after every physics tick**.
  On the first terminal event, it **stops the block early** and returns that state with
  `terminated = True`. (Checking only at the end of the block could miss a possession that was gained
  and then lost within the block.)

## 4. API

Plain Python dicts for milestone 1. [OPEN] adopt the Gymnasium / PettingZoo `ParallelEnv` API instead.

```
s, info                            = env.reset(seed: int, options: dict)
s_next, terminated, truncated, info = env.step(actions: dict)
```

- `terminated` / `truncated` are **episode-level bools**, kept **separate** (never merged into one
  `done`). See §7.
- No per-agent observations are returned; the RL layer builds them from `s`.
- No auto-reset: after `terminated` or `truncated`, the RL layer calls `reset()`. The final `s_next`
  of an episode must be the true final state (the RL layer needs it).

## 5. Global state `s`

A complete snapshot of everything that affects what happens next.

```python
s = {
    "step": int,                 # CONTROL steps elapsed in this episode (0 at reset)
    "blue": [                    # length N, ordered by id
        {"id": 0, "pos": [x, y], "vel": [vx, vy], "has_possession": bool},
        ...
    ],
    "red": [                     # length M (may be 0; see §9 scenario), ordered by id
        {"id": 0, "pos": [x, y], "vel": [vx, vy], "has_possession": bool},
        ...
    ],
    "ball": {"pos": [x, y], "vel": [vx, vy]},
}
```

- At most one entity has `has_possession = True`. All False means the ball is loose.
- **A returned `s` is a snapshot (added 2026-10-14).**
  - Each `reset()` / `step()` call returns a new `s`.
  - The env never modifies a returned `s` afterwards, neither in later `step()` calls nor in `reset()`.
  - Why: the RL side keeps `s` while it steps the env. For example, potential-based reward shaping reads
    Φ(s) after `step()` has returned `s_next`.
  - How: build fresh dicts and lists on every call; never return references to internal buffers that the
    next tick updates.
- **Hidden state rule:** if the env keeps **any** other memory that affects future dynamics, it must
  be added to `s`. Examples: possession hysteresis or contact counters, kick cooldowns, ball
  spin / angular velocity (if it affects rolling), internal state in red's script.
  **The milestone-1 mock has none.** The RL side rejects unknown keys in `s` with an error (so new
  hidden state can't be silently ignored). **[OPEN] env owner: please list any such state**, and we'll
  agree on where it goes.
- Static parameters (field size, max speeds, etc.) aren't repeated in `s`. They live in the shared
  config (§10).

## 6. Actions

One entry per blue agent, as **fractions** in [−1, 1]. The env converts them to physical units.

```python
actions = {
    "blue": [                    # length N, ordered by id
        {
            "move":         [mx, my],   # floats, each in [-1, 1]
            "kick_trigger": 0 or 1,     # int
            "kick_vec":     [kx, ky],   # floats, each in [-1, 1]
        },
        ...
    ]
}
```

**Env semantics:**

- **Move:** the desired velocity is `move × current_max_speed`, where `current_max_speed` includes the
  slowdown while carrying the ball. **Speed cap: [PROPOSED] 2-norm**: if ‖move‖ > 1, rescale it to
  length 1, so diagonal movement isn't faster. (The alternative is a per-component cap.) How it's
  applied in PyBullet (velocity control vs. force) is the env owner's choice.
- **Kick:** applied **only if** `kick_trigger == 1` **and** the agent had possession at the start of
  the step (§3). The ball's release velocity is `max_kick_speed × kick_vec / max(1, ‖kick_vec‖)`, so
  the speed is capped at `max_kick_speed`, and a zero vector simply drops the ball at zero speed.
  Otherwise `kick_vec` is ignored.
- **Milestone 1 is movement-only:** the RL side always sends `kick_trigger = 0`, `kick_vec = [0, 0]`.
  The env needs no special mode; the format is the same before and after kicking is added.
- Red agents aren't in `actions`; red is scripted inside the env.
- Malformed actions (wrong length, NaN, out of range, wrong type) should **raise an error**, not be
  silently fixed. (The RL side guarantees valid ranges and types; an error means a bug.)

## 7. Episode end

| Flag                | When                                                         | Meaning to the RL layer               |
| ------------------- | ------------------------------------------------------------ | ------------------------------------- |
| `terminated = True` | **red gains possession** (§8; checked every tick, §3)        | real game over; future value = 0      |
| `truncated = True`  | `step` reaches `max_episode_steps`                           | time limit only; the game wasn't lost |

- If both happen on the same step, report `terminated = True`.
- **The env always reports a time-out as `truncated`.** (If the RL side runs a "finite horizon" mode, it
  relabels time-outs itself; the env doesn't need to know.)
- **Arena: [PROPOSED] a bounded field with walls** (bodies can't leave it). The mock uses walls.
- Any possession mechanic is fine, but pin one for experiments, since changing it changes the game.

## 8. Possession semantics

**[PROPOSED]**; the mock implements exactly this. The env owner may change details, but please tell the
RL side.

- **Acquisition:** a player gains possession of a **loose** ball when the ball is within
  `possession_radius` of the player's center. (For milestone 1, a red player within
  `possession_radius` of a loose ball **or of the blue carrier's ball** gains possession → `terminated`.)
- **Carrying:** while possessed, the ball **follows the carrier** (it takes the carrier's x, y; the real
  env may put it on top of the carrier in z). The carrier moves at `carry_speed_factor × max_speed`.
- **Release:** only by a kick (§6). The release/cooldown details (can the kicker immediately
  re-possess?) are **deferred until kicking is enabled**; any cooldown would be hidden state (§5).
- **Collisions:** the mock has **no player-player collisions** (players pass through each other). The
  real env's collision rule is [OPEN]; it's an expected mock-vs-sim difference.
- **Spawning:** a "blue possession" start places the ball on a blue player (possessed at step 0). A
  reset **never** starts with red possession: red spawns at least `min_red_ball_dist`
  (> `possession_radius`) from the ball.

## 9. `info`, `reset` options, scenarios

**`info` (returned by `reset` and `step`):**

```python
info = {
    "events": [                  # what happened during this step, in order
        {"type": "possession_change", "from": ("blue", 0) or None, "to": ("red", 0) or None},
        {"type": "kick", "agent": ("blue", 0)},
        ...
    ],
    "kick_applied": [bool, ...], # per blue agent: did a kick actually happen this step
}
```

At `reset`, `events` is empty and `kick_applied` is all False. The episode length (the evaluation
metric) is `s["step"]` at episode end, in control steps.

**`reset(seed, options)`, where `options` controls the start distribution:**

```python
options = {
    "start_mix": 0.5,            # fraction of episodes starting with a blue agent in possession
                                 # (the rest start with a free ball); the type is drawn from the seed
    "require_blue_closer": True, # free-ball starts: the ball is closer to some blue than to any red
    "min_clearance": float,      # minimum distance between any two players at spawn (meters)
    "min_red_ball_dist": float,  # minimum red-to-ball distance at spawn (> possession_radius)
}
```

- All bodies spawn at randomized positions within these constraints, deterministically for a given seed.
- Overlapping spawns must be impossible (PyBullet resolves them violently).
- The RL side provides a reference **spawn sampler** in a small shared module with **no torch
  dependency** (pure: random generator + options → positions), which the env may use.

**Scenarios** (shared config key `scenario`):
- `keepaway` (default): as described above.
- `goto_target`: a sanity-check task. `n_red = 0`, the ball is an **immobile target** (possession
  disabled, the ball never moves), there's no termination, only truncation. The RL side's reward is the
  negative distance from blue to the ball.

## 10. Shared configuration

These values must be identical on both sides. [OPEN] the format (YAML/JSON) and loader are the infra
teammate's call. Both sides validate the config **at load**.

| Key                                 | Default       | Notes                                                    |
| ----------------------------------- | ------------- | -------------------------------------------------------- |
| `scenario`                          | `keepaway`    | or `goto_target` (§9)                                    |
| `n_blue`, `n_red`                   | 1, 1          | milestone 1 is 1v1; `n_red = 0` allowed (goto_target)    |
| `physics_dt`                        | 1/240 s       |                                                          |
| `control_k`                         | 12            | → 20 Hz control                                          |
| `max_episode_steps`                 | 300           | in control steps (15 s at 20 Hz); truncation limit       |
| `field_size`                        | [OPEN]        | `[Lx, Ly]`, FULL extents in meters; the RL side scales by half of it |
| `max_speed`                         | [OPEN]        | blue's top speed (m/s)                                   |
| `red_max_speed`                     | [OPEN]        | red's top speed; a **calibration knob** (see below)      |
| `carry_speed_factor`                | [OPEN]        | carrier speed = factor × `max_speed`                     |
| `max_kick_speed`                    | [OPEN]        | unused at milestone 1                                    |
| `possession_radius`                 | [OPEN]        | §8                                                       |
| start-distribution options          | see §9        |                                                          |

**Calibration:** before training, the RL side tunes `red_max_speed` (and `carry_speed_factor` if needed) so
that a scripted "flee" policy survives roughly 30–70% of `max_episode_steps`. If red is always faster
than the carrier, capture is inevitable and there's little to learn; if red is much slower, fleeing
always survives and there's nothing to learn either.

## 11. Conformance tests

The RL side will write a test suite against this ICD that can run on **both** the mock and the real env
(it doubles as an acceptance test for the real env):
- same seed + options → identical episode;
- malformed actions raise;
- `terminated` ⇒ some red has `has_possession` in `s_next`;
- terminated takes priority over truncated;
- the final `s_next` is the true final state (no auto-reset);
- a returned `s` is never modified afterwards, by later `step()` or `reset()` calls (§5);
- `kick_applied[i]` == (`kick_trigger == 1` and possession at the start of the step);
- spawn constraints hold; no reset starts with red possession;
- `n_red = 0` works (goto_target).

## 12. Open questions for the team

1. Plain dicts vs. the Gymnasium / PettingZoo API (§4).
2. Speed cap: 2-norm (proposed) vs. per-component (§6).
3. Possession details, the collision rule and walls in the real env (§7, §8).
4. Hidden env state that must go into `s` (§5).
5. Field size, speeds, `possession_radius`, kick speed (§10).
6. Config format and ownership (§10).
7. Who owns this document going forward (issue #11 is currently unassigned).
