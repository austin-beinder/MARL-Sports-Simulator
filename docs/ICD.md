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
  2026-10-10: The student added the axis convention.
  2026-10-10: v0.2. Merged the student's triage of a multi-agent panel review:
    the possession semantics section, mid-block termination, kick release
    formula, red speed, arena walls, max_episode_steps, the go-to-target
    scenario, reset/info details, action types, and conformance tests.
  2026-10-14: Added the returned-state snapshot rule (§5) and its conformance
    test (§11). The PBRS reward term needs them; the student approved the change.
  2026-11-07: Kicking is built in the PyBullet env (the mock stays movement-only):
    §6 kick semantics, §8 release rule (no cooldown), §10 max_kick_speed and the
    new ball_damping. Decisions by Austin Beinder.
  2026-11-07: Added the max_ball_dist reset option (§9), which the curriculum
    (curriculum/runner.py) grows. Decision by Austin Beinder.
  2026-11-07: §8: a kick into a wall the kicker is touching rebounds to the
    kicker (it used to push the ball out of the arena).
  2026-11-07: ball_damping 0.2 -> 0.4 (§10): kicks max out near 5 m.
  2026-11-07: §8: PyBullet steals also count contacts that were already there
    when blue got the ball (blue used to keep it while pinned against red).
  2026-11-08: §10: free_speed_factor, blue's speed without the ball.
  2026-11-08: §9: the passing_drill scenario (frozen blues) and the
    drill_radius reset option.
  2026-11-09: §7, §9: the pass_practice scenario (one kick per episode, no
    red), its own terminal, and the pass_distance reset option.
  2026-11-09: §9: pass_practice_moving (pass practice with the blues free to
    move).
  2026-11-10: §9: optional RL-side pass rules: a catch paid by its length,
    and a minimum pass distance.
  2026-11-11: §9: the catch_start_mix reset option (keepaway).
  2026-11-11: §9: pass_practice_moving runs to the time limit (no one-kick
    limit); the RL side pays each catch when it happens.
  2026-11-13: §5: s["ball"]["passer"], the blue who passed the ball to its
    holder (always None in the mock). Decision by Austin Beinder.
  2026-11-13: §5, §6: the optional "call" action, echoed as each blue's
    "calling" in s (both envs). Decision by Austin Beinder.
  2026-11-14: §9, §10: the claim_ball scenario (a loose ball, no red,
    terminated when a blue gets it; both envs). Decision by Austin Beinder.
  2026-11-23: §5, §7, §8, §9, §10, §11: the soccer scenario (PyBullet only):
    goals, red possession not terminal, the goal event, [shared.soccer].
    Decisions by Austin Beinder (docs/soccer_env_plan.md).
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
    "ball": {"pos": [x, y], "vel": [vx, vy], "passer": int | None},  # passer: added 2026-11-13
}
```

- At most one entity has `has_possession = True`. All False means the ball is loose.
- **Soccer (2026-11-23):** a red may hold the ball mid-episode (red possession isn't terminal there),
  and on the final state after a goal the ball's position is past the end line, in the goal's pocket
  (`|x|` up to `Lx / 2 + goal_depth`).
- **`ball.passer` (added 2026-11-13):** the id of the blue who passed the ball to its current (or, while
  it is loose, its latest) blue holder, else None.
  - A blue who picks up a teammate's kick was passed to by that teammate.
  - A blue who picks its own kick back up keeps the passer it had.
  - Any other pickup (a loose ball nobody kicked, a red taking it) clears it.
  - Every kick counts, however short; the RL side's pass rules (§9, minimum distance) don't apply.
  - The RL side uses it to tell a pass on from a pass straight back (`rl.reward.return_pass_factor`),
    and to show each blue who passed (`rl.passer_obs`). The mock can't kick, so it is always None there.
  - Optional for the RL side's parser: a hand-built state without it has no passer.
- **Blue `calling` (added 2026-11-13):** each blue entry also has `"calling": bool`, the `call` it sent
  on the latest step (§6; False at reset). Reds have no `calling`. Optional for the RL side's parser,
  like `passer`.
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
            "call":         0 or 1,     # int, OPTIONAL (added 2026-11-13; missing = 0)
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
- **Who kicks (2026-11-07):** the PyBullet env implements kicks; the mock does not, and raises
  `NotImplementedError` for a kick it would have to apply. Without `rl.kick_enabled` the RL side
  always sends `kick_trigger = 0`, `kick_vec = [0, 0]`. The format is the same either way.
- **Kick order:** the kick happens before the step's first physics tick, so its `kick` event comes
  before the `possession_change` (kicker → `None`) it causes.
- **Call (added 2026-11-13):** a "got it" signal with no physical effect. The env only echoes it: each
  blue's `calling` in the returned `s` (§5) is its `call` from this step (False without the key).
  Both envs do this. The RL side sends it only with `rl.call_enabled`.
- Red agents aren't in `actions`; red is scripted inside the env.
- Malformed actions (wrong length, NaN, out of range, wrong type) should **raise an error**, not be
  silently fixed. (The RL side guarantees valid ranges and types; an error means a bug.)

## 7. Episode end

| Flag                | When                                                         | Meaning to the RL layer               |
| ------------------- | ------------------------------------------------------------ | ------------------------------------- |
| `terminated = True` | **red gains possession** (§8; checked every tick, §3)        | real game over; future value = 0      |
| `truncated = True`  | `step` reaches `max_episode_steps`                           | time limit only; the game wasn't lost |
| `terminated = True` | `pass_practice` only (2026-11-09): the episode's kick has played out (§9) | the one kick is over; future value = 0 |
| `terminated = True` | `soccer` only (2026-11-23): **a goal**, by either team (§9). Red possession is *not* terminal in soccer | the game is over; future value = 0 |

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
- **Release:** only by a kick (§6). The kicked ball starts just in front of the kicker. **There is no
  cooldown (2026-11-07):** anyone, the kicker included, who touches the loose ball takes it, so a
  soft kick ahead and a chase is a legal dribble. (A cooldown would be hidden state, §5.)
- **Kicks at a wall (PyBullet, 2026-11-07):** the kicked ball always starts inside the walls. If the
  kicker is so close to the wall that the ball doesn't fit in front of them, the kick rebounds: the
  kicker keeps the ball. It still counts as a kick (`kick_applied` is True, §6).
- **Loose ball (PyBullet):** it slows down by `dv/dt = -ball_damping · v · (1 + |v|)`, so a ball
  released at `v` m/s rolls about `ln(1 + v) / ball_damping` m. Walls stop it dead (no bounce).
- **Collisions:** the mock has **no player-player collisions** (players pass through each other). The
  real env's collision rule is [OPEN]; it's an expected mock-vs-sim difference.
- **Possession by contact (PyBullet, 2026-11-07):** a red player touching the blue carrier or its ball
  takes the ball, including when they were already touching before blue got it (a loose ball picked
  up mid-scrum). The ball changes hands at most once per physics tick.
- **Soccer (2026-11-23):** the same contact rule works both ways, so a blue touching a red carrier
  takes the ball back. Only blues kick; a red holding the ball keeps still (its fixed policy chases
  the ball, which it is standing on).
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
        {"type": "goal", "team": "blue", "scorer": ("blue", 0) or ("red", 0) or None,
         "own_goal": False, "carried": True},  # soccer only, last (2026-11-23)
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
    "max_ball_dist": float,      # free-ball starts: the ball spawns within this distance of a random
                                 # blue (uniform in that disc); 0 = anywhere (2026-11-07, for the curriculum)
    "drill_radius": float,       # passing_drill only: the blues' circle radius round red (2026-11-08)
    "pass_distance": float,      # pass_practice only: each teammate's distance from the holder (2026-11-09)
    "catch_start_mix": float,    # keepaway only: share of resets in the pass_practice layout (a blue holding
                                 # the ball, teammates pass_distance away) plus red min_red_ball_dist from
                                 # the ball; 0 = the usual starts (2026-11-11, for the curriculum)
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
- `passing_drill` (2026-11-08, PyBullet only): keep-away rules (red possession is terminal), but the
  blues **can't move**: the env ignores `move` (it is still validated). `n_blue >= 2`, and the RL side
  needs kicking on. Every episode starts with a blue holding the ball; the blues spawn spread evenly
  round a circle of about `drill_radius` m (radius ±15%, angles ±15°) with red within 30% of the
  radius from its centre. Only `min_clearance` and `drill_radius` apply to this layout. The RL side
  pays its own drill reward (holding the ball after the team's first pass, plus pass shaping) and
  masks the movement head out of PPO.
- `pass_practice` (2026-11-09, PyBullet only): passing with no opponent. `n_blue >= 2`, `n_red = 0`,
  kicking on, blues frozen as in `passing_drill`. A random blue starts with the ball, and every other
  blue stands `pass_distance` m (±15%) from it in a random direction; only `min_clearance` and
  `pass_distance` apply. One kick per episode: once a blue has kicked, the episode is `terminated` as
  soon as some blue holds the ball again (a catch, or the kicker getting it back) or the loose ball
  can't reach a teammate any more (what it has left to roll, `ln(1 + |v|) / ball_damping`, is less
  than its gap to the nearest teammate other than the kicker). No kick: `truncated` at the time limit.
  The RL side pays 1 for a catch, else up to 0.5 for how close the ball came to a teammate. RL-side
  options (2026-11-10) can pay a catch 0.75 to 1 by its length and set a minimum pass distance.
- `pass_practice_moving` (2026-11-09, PyBullet only): `pass_practice`'s spawn with the blues free to
  move (the holder at carry speed). Since 2026-11-11 there is no one-kick limit and no `terminated`:
  every episode runs to the time limit (`truncated`), and the blues pass as often as they like. After
  a miss somebody has to fetch the loose ball. The RL side pays each catch on the step it happens
  (`pass_catch`), with the same RL-side options.
- `claim_ball` (2026-11-14, both envs): `n_red = 0`, any `n_blue`. Every episode starts with a loose
  ball (the free-ball spawn; `start_mix` doesn't apply, `max_ball_dist` does), and it is `terminated`
  on the step a blue picks the ball up (`truncated` at the time limit if nobody does). Practice for
  the `call` (§6): the RL side pays -1 a step until the pickup, scores each blue's calls by who got
  the ball, and charges blues that crowd a ball a teammate called.
- `soccer` (2026-11-23, PyBullet only): any `n_blue >= 1`, `n_red >= 0` (0 = an empty net). The field
  has a goal in the middle of each end wall (`[shared.soccer]`, §10): blue defends the west goal
  (`x = -Lx / 2`) and attacks east. Spawns are drawn exactly as for `keepaway`. Red possession is
  **not** terminal: play goes on, and either team takes the ball from the other by touching the
  carrier. Every red chases the ball. The episode is `terminated` on the first goal: the whole ball
  over a goal line between the posts, kicked, carried in, or an own goal; it scores for the team that
  doesn't defend that goal. That step's `events` end with one `goal` event (`team` gets the point,
  `scorer` is whoever last held the ball, `own_goal`, `carried`). No goal by the time limit:
  `truncated`, a draw. The RL side pays +1 for a blue goal and -1 for a red one.

## 10. Shared configuration

These values must be identical on both sides. [OPEN] the format (YAML/JSON) and loader are the infra
teammate's call. Both sides validate the config **at load**.

| Key                                 | Default       | Notes                                                    |
| ----------------------------------- | ------------- | -------------------------------------------------------- |
| `scenario`                          | `keepaway`    | or `goto_target`, `passing_drill`, `pass_practice`, `pass_practice_moving`, `claim_ball`, `soccer` (§9) |
| `n_blue`, `n_red`                   | 1, 1          | milestone 1 is 1v1; `n_red = 0` in goto_target, pass_practice |
| `physics_dt`                        | 1/240 s       |                                                          |
| `control_k`                         | 12            | → 20 Hz control                                          |
| `max_episode_steps`                 | 300           | in control steps (15 s at 20 Hz); truncation limit       |
| `field_size`                        | [OPEN]        | `[Lx, Ly]`, FULL extents in meters; the RL side scales by half of it |
| `max_speed`                         | [OPEN]        | blue's top speed (m/s)                                   |
| `red_max_speed`                     | [OPEN]        | red's top speed; a **calibration knob** (see below)      |
| `carry_speed_factor`                | [OPEN]        | carrier speed = factor × `max_speed`                     |
| `free_speed_factor`                 | 1.0           | blue's speed without the ball = factor × `max_speed` (2026-11-08); doesn't stack with `carry_speed_factor` |
| `max_kick_speed`                    | 6.0 m/s       | kick speed cap (§6); also the ball's top speed in the sim |
| `ball_damping`                      | 0.4           | loose-ball slowdown (§8): a 6 m/s kick rolls ~4.9 m, 3 m/s ~3.5 m |
| `possession_radius`                 | [OPEN]        | §8                                                       |
| start-distribution options          | see §9        |                                                          |
| `soccer.goal_width`, `soccer.goal_depth` | 2.0, 0.5 m | soccer only (2026-11-23): each goal's mouth, post to post, and how far its pocket goes back |

**Calibration:** before training, the RL side tunes `red_max_speed` (and `carry_speed_factor` if needed) so
that a scripted "flee" policy survives roughly 30–70% of `max_episode_steps`. If red is always faster
than the carrier, capture is inevitable and there's little to learn; if red is much slower, fleeing
always survives and there's nothing to learn either.

## 11. Conformance tests

The RL side will write a test suite against this ICD that can run on **both** the mock and the real env
(it doubles as an acceptance test for the real env):
- same seed + options → identical episode;
- malformed actions raise;
- `terminated` ⇒ some red has `has_possession` in `s_next` (except in `pass_practice` and `soccer`, §9);
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
5. Field size, speeds, `possession_radius` (§10). Kick speed and ball damping were set 2026-11-07.
6. Config format and ownership (§10).
7. Who owns this document going forward (issue #11 is currently unassigned).
