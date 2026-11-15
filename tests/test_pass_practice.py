# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-09
# Author:  Austin Beinder
# Usage:
#   - 2026-11-09: Generated this entire file: tests for pass practice (the
#     spawn, the config, the env ending each episode once its kick plays out,
#     the reward, the catch gate and training end to end).
#   - 2026-11-09: A miss ends once the ball can't reach a teammate (not when it
#     stops: a slow ball crawls for seconds).
#   - 2026-11-09: pass_practice_moving: blues free to move, a teammate can
#     run onto a short pass, and a hopeless kick still ends early.
#   - 2026-11-10: The minimum pass distance (a tap to a teammate next to the
#     kicker is no pass and no near miss) and catches paid by their length.
#   - 2026-11-10: Both are options now (rl.reward.min_pass_distance,
#     rl.reward.pass_length_bonus), off by default.
#   - 2026-11-11: kicker_clearance: the kicker pays for staying near its own
#     kick and more for taking it back; chasing costs more than standing back.
#   - 2026-11-11: pass_practice_moving runs to the time limit and pays each
#     catch when it happens (pass_catch); receiver_spacing; the gate scores
#     passes / kicks.
#   - 2026-11-11: receiver_spacing is a tent: every metre apart pays the same.
#   - 2026-11-11: With a minimum pass distance the receiver must have stood that
#     far from the kick point at the kick (no chasing down a ball kicked ahead).
#   - 2026-11-13: The env's passer, and rl.reward.return_pass_factor for a
#     pass straight back. (Austin Beinder)
#   - 2026-11-13: A stopped kick no longer charges the kicker; the
#     call_accuracy term. (Austin Beinder)
#   - 2026-11-14: call_accuracy no longer adds to the team reward (the call
#     head has its own, tests/test_calls.py): these tests read the term itself.
#     (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters and fixed
#     the new warnings; no behaviour change. (Austin Beinder)
# ==============================================================================
import math
import types  # AI GENERATED (2026-11-13)

import numpy as np
import pytest

from marl.common.config import config_from_dict, with_overrides
from marl.common.spawn import START_POSSESSION, sample_spawn
from marl.envs.mock_env import MockEnv
from marl.envs.pybullet_env import PyBulletEnv
from marl.rl.networks import build_networks
from marl.rl.reward import (KICKER_CLEARANCE, KICKER_REGRAB, PASS_CATCH_MIN,  # AI GENERATED
                             PASS_FULL_LENGTH,  # AI GENERATED
                             PASS_MISS_MAX, PASS_MISS_SCALE, PassMemory, PassStep,  # AI GENERATED
                             RewardModule, kicker_clearance,  # AI GENERATED (2026-11-11)
                             receiver_spacing)  # AI GENERATED (2026-11-11)
from marl.rl.rollout import EpisodeRecord
from marl.rl.train import train
from curriculum.runner import Stage, StageCurriculum

# pytest passes the fixtures (cfg, env) in as arguments of the same name.
# pylint: disable=redefined-outer-name

PRACTICE = {
    "shared": {"scenario": "pass_practice", "n_blue": 3, "n_red": 0, "max_episode_steps": 80},
    "rl": {"kick_enabled": True, "reward": {"base": "pass_practice"}},
    "run": {"env": "pybullet"},
}


@pytest.fixture
def cfg():
    return config_from_dict(PRACTICE)


@pytest.fixture
def env(cfg):
    env = PyBulletEnv(cfg.shared)
    env.reset(seed=0)
    yield env
    env.close()


def actions(kicker=None, kick_vec=(0.0, 0.0), move=(0.0, 0.0)):
    return {"blue": [{"move": list(move), "kick_trigger": int(i == kicker),
                      "kick_vec": list(kick_vec)}
                     for i in range(3)]}


def play(env, cfg, first, memory=None):
    """Step ``first`` then nothing until the episode ends; returns (s, terminated, truncated,
    base reward sum)."""
    rm = RewardModule(cfg)
    memory = memory or rm.new_episode()
    s, total = env._state(), 0.0  # pylint: disable=protected-access
    act = first
    while True:
        s_next, term, trunc, info = env.step(act)
        total += rm.parts(s, act, s_next, term, trunc, info, memory).base
        s, act = s_next, actions()
        if term or trunc:
            return s, term, trunc, total, memory


# --- Spawn and config -------------------------------------------------------------------
def spawn(cfg, seed, **options):
    sc = cfg.shared
    return sample_spawn(np.random.default_rng(seed), scenario="pass_practice", n_blue=sc.n_blue,
                        n_red=sc.n_red, field_size=sc.field_size, player_radius=sc.player_radius,
                        possession_radius=sc.possession_radius,
                        options={**sc.start.to_reset_options(), **options})


@pytest.mark.parametrize("distance", [1.5, 2.75, 3.9])
def test_teammates_stand_pass_distance_from_the_holder(cfg, distance):
    sc = cfg.shared
    half = np.asarray(sc.field_size) / 2 - sc.player_radius
    for seed in range(100):
        sp = spawn(cfg, seed, pass_distance=distance)
        blue = np.array(sp.blue_pos)
        assert (sp.start_type == START_POSSESSION and sp.red_pos == ()
                and sp.ball_pos == sp.blue_pos[sp.holder])
        assert np.all(np.abs(blue) <= half + 1e-9)
        d = np.linalg.norm(np.delete(blue, sp.holder, axis=0) - blue[sp.holder], axis=-1)
        assert np.all((d >= 0.85 * distance - 1e-9) & (d <= 1.15 * distance + 1e-9))
    assert spawn(cfg, 4) == spawn(cfg, 4) and spawn(cfg, 4) != spawn(cfg, 5)


@pytest.mark.parametrize("change, match", [
    ({"shared": {"n_red": 1}}, "n_red = 0"),
    ({"shared": {"n_blue": 1}}, "n_blue >= 2"),
    ({"rl": {"kick_enabled": False}}, "kick_enabled"),
    ({"shared": {"start": {"pass_distance": 0.0}}}, "pass_distance"),
])
def test_bad_practice_configs_raise(change, match):
    merged = {k: {**PRACTICE[k], **change.get(k, {})} for k in PRACTICE}
    if "start" in change.get("shared", {}):
        merged["shared"]["start"] = change["shared"]["start"]
    with pytest.raises(ValueError, match=match):
        config_from_dict(merged)


def test_mock_refuses_pass_practice(cfg):
    with pytest.raises(NotImplementedError, match="pass_practice"):
        MockEnv(cfg.shared)


def test_practice_networks_are_frozen(cfg):
    actor, _ = build_networks(cfg, 41, 20)
    assert not actor.move_enabled


# --- Env + reward ------------------------------------------------------------------------
def test_a_kick_at_a_teammate_is_caught_ends_the_episode_and_pays_1(env, cfg):
    s0 = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[],
                                   holder=("blue", 0))
    s, term, trunc, total, memory = play(env, cfg, actions(kicker=0, kick_vec=(1.0, 0.0),
                                                           move=(1.0, 1.0)))
    assert term and not trunc and s["step"] < 40
    assert s["blue"][1]["has_possession"] and memory.passes == 1
    assert 2.4 < memory.length < 3.1  # caught by blue 1, 3 m out
    assert total == 1.0  # no length bonus by default
    # nobody moved, though blue 0 asked to (the catch nudges blue 1 ~2 mm)
    for b0, b in zip(s0["blue"], s["blue"]):
        assert b["pos"] == pytest.approx(b0["pos"], abs=0.02)


def test_a_miss_ends_once_the_ball_cant_reach_a_teammate(env, cfg):
    def miss(kick_vec):
        env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[],
                                  holder=("blue", 0), step=0)
        s, term, trunc, total, _ = play(env, cfg, actions(kicker=0, kick_vec=kick_vec))
        assert term and not trunc and not any(b["has_possession"] for b in s["blue"])
        roll_left = math.log1p(math.hypot(*s["ball"]["vel"])) / cfg.shared.ball_damping
        gaps = [math.dist(b["pos"], s["ball"]["pos"]) - cfg.shared.player_radius
                for b in s["blue"][1:]]
        assert roll_left < min(gaps)  # it can't get to blue 1 or blue 2 any more
        return s, total

    _, near = miss((0.4, -0.15))  # slightly wide of blue 1, and soft
    s_far, far = miss((-0.5, -0.5))  # away from both
    assert 0 < far < near < PASS_MISS_MAX
    assert s_far["step"] < 30  # it ends long before the time limit, while the ball still rolls


def test_no_kick_runs_to_the_time_limit_and_pays_nothing(env, cfg):
    s, term, trunc, total, memory = play(env, cfg, actions())
    assert trunc and not term and s["step"] == cfg.shared.max_episode_steps
    assert total == 0.0 and memory.passes == 0


def test_closest_approach_follows_the_balls_path():
    memory = PassMemory()
    blues = [{"id": 0, "pos": [0.0, 0.0]}, {"id": 1, "pos": [2.0, 1.0]},
             {"id": 2, "pos": [-3.0, 0.0]}]

    def st(ball):
        return {"blue": blues, "ball": {"pos": list(ball)}}

    kick = {"events": [{"type": "kick", "agent": ("blue", 0)}]}
    memory.update(kick, st((0, 0)), st((1, 0)))
    assert memory.closest == pytest.approx(math.hypot(1, 1))  # nearest point of this step: (1, 0)
    memory.update({"events": []}, st((1, 0)), st((4, 0)))  # passes under blue 1 at (2, 0)
    assert memory.closest == pytest.approx(1.0)
    memory.update(kick, st((0, 0)), st((0, -1)))  # a new kick starts a new measurement
    # blue 1, from the start of this short path
    assert memory.closest == pytest.approx(math.hypot(2, 1))


### AI GENERATED (2026-11-10: the minimum pass distance and catches paid by length, both options)
LONG = {"rl": {"reward": {"min_pass_distance": 1.0, "pass_length_bonus": True}}}
def practice_state(holder=None, ball=(0.0, 0.0), blues=((0.0, 0.0), (0.6, 0.0), (0.0, 3.0))):
    return {
        "step": 0,
        "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": i == holder}
                 for i, p in enumerate(blues)],
        "red": [],
        "ball": {"pos": list(blues[holder] if holder is not None else ball), "vel": [0.0, 0.0]},
    }


def practice_info(*events):
    return {"events": list(events), "kick_applied": [False] * 3}


KICK_0 = {"type": "kick", "agent": ("blue", 0)}
RELEASE_0 = {"type": "possession_change", "from": ("blue", 0), "to": None}


def caught_by(i):
    return {"type": "possession_change", "from": None, "to": ("blue", i)}


def tap(rm, memory):
    """Blue 0 taps the ball 0.6 m into blue 1; returns the kick step's and the catch step's
    rewards."""
    kick = rm.parts(practice_state(0), {}, practice_state(ball=(0.3, 0.0)), False, False,
                    practice_info(KICK_0, RELEASE_0), memory)
    end = rm.parts(practice_state(ball=(0.3, 0.0)), {}, practice_state(1), True, False,
                   practice_info(caught_by(1)), memory)
    return kick, end


def test_by_default_a_tap_to_a_teammate_is_a_pass(cfg):
    rm = RewardModule(cfg)
    _, end = tap(rm, rm.new_episode())
    assert end.base == 1.0  # the old reward: any catch


def test_with_a_minimum_a_tap_to_a_teammate_next_to_the_kicker_is_no_pass_and_pays_nothing(cfg):
    cfg = with_overrides(cfg, LONG)
    rm = RewardModule(cfg)
    memory = rm.new_episode()  # blue 1 stands 0.6 m from blue 0
    assert memory.min_distance == 1.0
    kick, end = tap(rm, memory)
    assert kick.base == 0 and memory.passes == 0  # no pass
    # and no near miss on blue 1 either: the only credit is for blue 2, 3 m away, like a kick that
    # went nowhere
    assert memory.closest == pytest.approx(3.0)
    assert end.base == pytest.approx(
        PASS_MISS_MAX * math.exp(-(3.0 - cfg.shared.player_radius) / PASS_MISS_SCALE))


def test_the_closest_approach_ignores_teammates_next_to_the_kicker():
    # blue 1 is 1.4 m from the kick point (it could only catch a pass under 1 m)
    memory = PassMemory(min_distance=1.0)
    blues = [{"id": 0, "pos": [0.0, 0.0]}, {"id": 1, "pos": [1.4, 0.0]},
             {"id": 2, "pos": [0.0, 3.0]}]

    def st(ball):
        return {"blue": blues, "ball": {"pos": list(ball)}}

    memory.update({"events": [KICK_0]}, st((0, 0)), st((1.0, 0)))  # straight at blue 1
    assert memory.closest == pytest.approx(3.0)  # only blue 2 counts, from the kick point
    blues[1]["pos"] = [1.6, 0.0]  # blue 1 steps back: now a 1 m pass could reach it
    memory.update({"events": []}, st((1.0, 0)), st((1.2, 0)))
    assert memory.closest == pytest.approx(0.4)


@pytest.mark.parametrize("length", [1.0, 2.0, 3.0, 5.0])
def test_with_the_bonus_a_catch_pays_more_the_further_it_went(cfg, length):
    rm = RewardModule(with_overrides(cfg, LONG))
    memory = rm.new_episode()
    blues = ((0.0, 0.0), (length, 0.0), (0.0, 3.0))
    rm.parts(practice_state(0, blues=blues), {}, practice_state(ball=(0.5, 0.0), blues=blues),
             False, False, practice_info(KICK_0, RELEASE_0), memory)
    end = rm.parts(practice_state(ball=(length - 0.2, 0.0), blues=blues), {},
                   practice_state(1, blues=blues), True, False, practice_info(caught_by(1)),
                   memory)
    assert memory.passes == 1 and memory.length == pytest.approx(length)
    assert end.base == pytest.approx(
        min(1.0, PASS_CATCH_MIN + (1 - PASS_CATCH_MIN) * length / PASS_FULL_LENGTH))
    assert PASS_MISS_MAX < end.base <= 1.0  # every catch beats every miss


def test_min_pass_distance_cant_be_negative():
    with pytest.raises(ValueError, match="min_pass_distance"):
        config_from_dict({**PRACTICE, "rl": {**PRACTICE["rl"], "reward": {
            "base": "pass_practice", "min_pass_distance": -1.0}}})
### END AI GENERATED


# --- Gate and training ------------------------------------------------------------------
# 2026-11-11: one kick per episode here, so passes / kicks
def test_practice_stages_gate_on_the_catch_rate():
    stage = Stage(name="catch", total_steps=1000, threshold=0.8, window=10)
    ctl = StageCurriculum(stage, {}, max_episode_steps=80, log=lambda _m: None, catch_gate=True)

    def episodes(caught, n):  # long episodes would pass a survival gate; caught ones are short
        return [EpisodeRecord(length=20 if caught else 80, terminated=caught, truncated=not caught,
                              start_type=START_POSSESSION, mean_reward=0.0, passes=int(caught),
                              kicks=1)] * n

    row = ctl.update(episodes(False, 10))
    assert row["gate_mean"] == 0.0 and not ctl.finished
    ctl.update(episodes(True, 10))
    assert ctl.gate.mean() == 1.0 and ctl.finished


def test_the_gate_scores_passes_per_kick():  # AI GENERATED (2026-11-11)
    stage = Stage(name="catch", total_steps=1000, threshold=0.8, window=3)
    ctl = StageCurriculum(stage, {}, max_episode_steps=300, log=lambda _m: None, catch_gate=True)
    ctl.update([EpisodeRecord(length=300, terminated=False, truncated=True,
                              start_type=START_POSSESSION, mean_reward=0.0, passes=p, kicks=k)
                for p, k in [(6, 8), (0, 0), (3, 3)]])
    assert list(ctl.gate.scores) == [0.75, 0.0, 1.0]  # no kick scores 0


def test_training_pass_practice_runs_end_to_end(tmp_path, cfg):
    small = with_overrides(cfg, {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(small, tmp_path / "practice", log=lambda *_: None)
    assert result.iterations == 2 and (tmp_path / "practice" / "checkpoints" / "final.pt").is_file()
    assert {"passes_mean", "frac_passed"} <= set(result.final_eval)


# --- Pass practice with the blues free to move (2026-11-09) -------------------------------
MOVING = {**PRACTICE, "shared": {**PRACTICE["shared"], "scenario": "pass_practice_moving",
                                 "max_episode_steps": 100},
          "rl": {**PRACTICE["rl"], "reward": {"base": "pass_catch"}}}  # 2026-11-11: paid per catch


def soft_pass_to_blue_1(env, cfg, blue_1_runs, until_catch=False):
    """Blue 0 kicks softly at blue 1, 3 m away (0.9 m/s: it rolls ~1.6 m, ~0.5 m short); blue 1
    runs at it or stands.
    until_catch (2026-11-11): stop once a teammate has the ball, not at the episode's end."""
    env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0),
                              step=0)
    rm = RewardModule(cfg)
    memory, s, total = rm.new_episode(), env._state(), 0.0  # pylint: disable=protected-access
    move = [(0.0, 0.0), (-1.0, 0.0) if blue_1_runs else (0.0, 0.0), (0.0, 0.0)]
    first = True
    while True:
        act = {"blue": [{"move": list(m), "kick_trigger": int(first and i == 0),
                         "kick_vec": [0.15, 0.0]}
                        for i, m in enumerate(move)]}
        s_next, term, trunc, info = env.step(act)
        total += rm.parts(s, act, s_next, term, trunc, info, memory).base
        s, first = s_next, False
        if term or trunc or (until_catch and any(b["has_possession"] for b in s["blue"][1:])):
            return s, term, trunc, total


def test_moving_practice_config_and_networks():
    cfg = config_from_dict(MOVING)
    actor, _ = build_networks(cfg, 41, 20)
    assert actor.move_enabled  # the movement head trains here
    with pytest.raises(NotImplementedError, match="pass_practice_moving"):
        MockEnv(cfg.shared)
    with pytest.raises(ValueError, match="n_red = 0"):
        config_from_dict({**MOVING, "shared": {**MOVING["shared"], "n_red": 1}})


def test_a_teammate_can_run_onto_a_short_pass():
    cfg = config_from_dict(MOVING)
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        s, term, trunc, total = soft_pass_to_blue_1(env, cfg, blue_1_runs=True, until_catch=True)
        assert not term and not trunc and s["blue"][1]["has_possession"]  # 2026-11-11: play goes on
        assert total == 1.0  # paid on the catch
        assert s["blue"][1]["pos"][0] < 2.9  # it came to meet the ball
    finally:
        env.close()


def test_the_same_short_pass_is_a_miss_for_a_frozen_teammate(env, cfg):
    # frozen: its run is ignored
    s, term, trunc, total = soft_pass_to_blue_1(env, cfg, blue_1_runs=True)
    assert term and not trunc and not any(b["has_possession"] for b in s["blue"])
    assert 0 < total < PASS_MISS_MAX


### AI GENERATED (2026-11-11: pass practice to the time limit)
def test_a_moving_episode_runs_to_the_time_limit_after_a_miss():
    cfg = config_from_dict(MOVING)
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[],
                                  holder=("blue", 0), step=0)
        # full kick away
        s, term, trunc, total, memory = play(env, cfg,
                                             actions(kicker=0, kick_vec=(-0.7071, -0.7071)))
        assert trunc and not term and s["step"] == cfg.shared.max_episode_steps
        assert total == 0.0 and memory.kicks == 1 and memory.passes == 0
    finally:
        env.close()


def test_two_passes_in_one_episode_pay_twice():
    """Blue 0 passes to blue 1, which passes on to blue 2; both catches pay, in the same episode."""
    cfg = config_from_dict(MOVING)
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        s = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [3, 3]], red_pos=[],
                                      holder=("blue", 0), step=0)
        rm = RewardModule(cfg)
        memory, paid = rm.new_episode(), []
        for _ in range(cfg.shared.max_episode_steps):
            holder = next((b["id"] for b in s["blue"] if b["has_possession"]), None)
            act = actions()
            # each holder kicks once, at the next blue
            if holder in (0, 1) and memory.kicks == holder:
                aim = np.subtract(s["blue"][holder + 1]["pos"], s["ball"]["pos"])
                act = actions(kicker=holder, kick_vec=tuple(0.6 * aim / np.linalg.norm(aim)))
            s_next, term, trunc, info = env.step(act)
            base = rm.parts(s, act, s_next, term, trunc, info, memory).base
            if base:
                paid.append(base)
            s = s_next
            if term or trunc:
                break
        assert trunc and not term
        assert paid == [1.0, 1.0] and memory.passes == 2 and memory.kicks == 2
        assert s["blue"][2]["has_possession"]
    finally:
        env.close()


def chase(min_distance, catcher_starts_at):
    """Blue 0 kicks the ball 3 m ahead along x; blue 1, starting at ``catcher_starts_at``, picks it
    up 3 m from the kick point. Returns the memory."""
    memory = PassMemory(min_distance=min_distance)
    start = ((0.0, 0.0), catcher_starts_at, (-3.0, 3.0))
    memory.update({"events": [KICK_0, RELEASE_0]}, practice_state(0, blues=start),
                  practice_state(ball=(0.5, 0.0), blues=start))
    there = ((0.0, 0.0), (3.3, 0.0), (-3.0, 3.0))
    memory.update({"events": [caught_by(1)]}, practice_state(ball=(2.9, 0.0), blues=there),
                  {**practice_state(1, blues=there),
                   "ball": {"pos": [3.0, 0.0], "vel": [0.0, 0.0]}})
    return memory


def test_a_ball_chased_down_by_a_packmate_is_no_pass():
    assert chase(2.0, catcher_starts_at=(0.6, 0.5)).passes == 0  # it ran beside the kicker: no pass
    assert chase(2.0, catcher_starts_at=(2.5, 0.5)).passes == 1  # it stood in range at the kick
    assert chase(0.0, catcher_starts_at=(0.6, 0.5)).passes == 1  # no minimum: any catch counts
    memory = chase(2.0, catcher_starts_at=(0.6, 0.5))
    assert memory.receivers == frozenset({2}) and memory.kicks == 1  # only blue 2 stood 2 m away


def test_pass_catch_pays_on_the_catch_with_the_length_bonus():
    rm = RewardModule(with_overrides(config_from_dict(MOVING), LONG))
    memory = rm.new_episode()
    blues = ((0.0, 0.0), (3.0, 0.0), (0.0, 3.0))
    kick = rm.parts(practice_state(0, blues=blues), {},
                    practice_state(ball=(0.5, 0.0), blues=blues), False, False,
                    practice_info(KICK_0, RELEASE_0), memory)
    catch = rm.parts(practice_state(ball=(2.8, 0.0), blues=blues), {},
                     practice_state(1, blues=blues), False, False,
                     practice_info(caught_by(1)), memory)
    later = rm.parts(practice_state(1, blues=blues), {}, practice_state(1, blues=blues), False,
                     True, practice_info(), memory)
    assert kick.base == 0.0 and later.base == 0.0  # nothing for the kick, nothing at the time limit
    assert catch.base == pytest.approx(
        PASS_CATCH_MIN + (1 - PASS_CATCH_MIN) * 3.0 / PASS_FULL_LENGTH)


def spacing(spots, target=3.0):
    cfg = with_overrides(config_from_dict(MOVING), {"rl": {"reward": {"spacing_distance": target}}})
    s_next = {"blue": [{"id": i, "pos": [*p, 0.0]} for i, p in enumerate(spots)]}
    return receiver_spacing(None, None, s_next, False, False, {}, cfg)


def test_receiver_spacing_pays_most_for_blues_the_target_distance_apart():
    triangle = [(0.0, 0.0), (3.0, 0.0), (1.5, 1.5 * math.sqrt(3))]  # every pair 3 m apart
    assert spacing(triangle) == pytest.approx(1.0)
    # a tent: 1 m off costs a third
    assert spacing([(0.0, 0.0), (4.0, 0.0)]) == pytest.approx(2 / 3)
    assert spacing([(0.0, 0.0), (2.0, 0.0)]) == pytest.approx(2 / 3)
    assert spacing([(0.0, 0.0), (0.0, 0.0)]) == 0.0 and spacing([(0.0, 0.0), (6.0, 0.0)]) == 0.0
    assert spacing([(0.0, 0.0), (9.0, 0.0)]) == 0.0  # never negative
    # 2026-11-11: blues bunched 0.6 m apart still feel the pull: each metre apart pays the same
    assert (spacing([(0.0, 0.0), (1.6, 0.0)]) - spacing([(0.0, 0.0), (0.6, 0.0)])
            == pytest.approx(1 / 3))
    assert spacing([(0.0, 0.0), (4.0, 0.0)], target=4.0) == pytest.approx(1.0)
    assert spacing([(0.0, 0.0)]) == 0.0  # nobody to space from


def test_spacing_distance_must_be_positive():
    with pytest.raises(ValueError, match="spacing_distance"):
        config_from_dict({**MOVING, "rl": {**MOVING["rl"], "reward": {
            "base": "pass_catch", "spacing_distance": 0.0}}})


def test_moving_practice_needs_the_pass_catch_reward():
    with pytest.raises(ValueError, match="pass_catch"):
        config_from_dict({**MOVING, "rl": {**MOVING["rl"], "reward": {"base": "pass_practice"}}})
### END AI GENERATED


def test_moving_practice_trains_end_to_end(tmp_path):
    small = with_overrides(config_from_dict(MOVING), {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(small, tmp_path / "moving", log=lambda *_: None)
    assert result.iterations == 2 and result.actor.move_enabled


### AI GENERATED (2026-11-11: kicker_clearance, the kicker keeps clear of its own kick)
CLEAR = {"rl": {"reward": {"shaping": [
    {"name": "kicker_clearance", "enabled": True, "coef": 0.025}]}}}


def clearance(kicker_pos, ball_pos, kicker=0, held=False):
    """kicker_clearance for blue ``kicker`` at ``kicker_pos`` with the ball at ``ball_pos``."""
    spots = [kicker_pos, (5.0, 5.0), (-5.0, 5.0)]
    s_next = {"blue": [{"id": i, "pos": [*p, 0.0], "has_possession": held and i == kicker}
                       for i, p in enumerate(spots)],
              "ball": {"pos": [*ball_pos, 0.0]}}
    info = {"pass": PassStep(kicker, False, 0)}
    return kicker_clearance(None, None, s_next, False, False, info, None)


def test_kicker_clearance_grows_as_the_kicker_nears_the_ball():
    assert clearance((0, 0), (0, 0), kicker=None) == 0.0  # no kick in play
    assert clearance((0, 0), (KICKER_CLEARANCE, 0)) == 0.0
    assert clearance((0, 0), (3.0, 0)) == 0.0
    assert clearance((0, 0), (KICKER_CLEARANCE / 2, 0)) == pytest.approx(-0.5)
    assert clearance((0, 0), (0.3, 0)) == pytest.approx(-0.8)
    assert clearance((0, 0), (0.3, 0), held=True) == -KICKER_REGRAB  # took its own kick back


def test_kicker_clearance_is_a_built_term_and_needs_pass_memory():
    cfg = with_overrides(config_from_dict(MOVING), CLEAR)
    assert RewardModule(cfg).needs_memory


def kick_and_follow(cfg, chase: bool) -> float:
    """Blue 0 kicks firmly at blue 2 (3 m up) and then stands still or runs after the ball; returns
    the summed kicker_clearance (before its coef)."""
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        s = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[],
                                      holder=("blue", 0), step=0)
        rm, total = RewardModule(cfg), 0.0
        memory = rm.new_episode()
        act = actions(kicker=0, kick_vec=(0.0, 1.0))
        while True:
            s_next, term, trunc, info = env.step(act)
            total += rm.parts(s, act, s_next, term, trunc, info, memory).shaping
            if term or trunc:
                return total / 0.025
            s, act = s_next, actions()
            if chase:
                # the kicker runs after the ball; its teammates stand still
                act["blue"][0]["move"] = [0.0, 1.0]
    finally:
        env.close()


def test_chasing_your_own_kick_costs_more_than_standing_back():
    cfg = with_overrides(config_from_dict(MOVING), CLEAR)
    still, chase = kick_and_follow(cfg, chase=False), kick_and_follow(cfg, chase=True)
    assert -3.0 < still < 0  # a firm kick clears 1.5 m in ~0.3 s: a small, unavoidable cost
    assert chase < still - 1.0
### END AI GENERATED


### AI GENERATED (2026-11-13: who passed to the holder, and passes straight back)
ROTATE = {"rl": {"reward": {"return_pass_factor": 0.5}}}


def relay(receivers):
    """Blue 0 holds the ball; each holder in turn kicks it at the next blue in ``receivers``.
    Returns the passer the env reports after each catch, what each catch paid, and the memory."""
    cfg = with_overrides(config_from_dict(MOVING), ROTATE)
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        s = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [3, 3]], red_pos=[],
                                      holder=("blue", 0), step=0)
        assert s["ball"]["passer"] is None
        rm = RewardModule(cfg)
        memory, passers, paid = rm.new_episode(), [], []
        for _ in range(cfg.shared.max_episode_steps):
            holder = next((b["id"] for b in s["blue"] if b["has_possession"]), None)
            act = actions()
            if holder is not None and memory.kicks == memory.passes < len(receivers):
                aim = np.subtract(s["blue"][receivers[memory.kicks]]["pos"], s["ball"]["pos"])
                act = actions(kicker=holder, kick_vec=tuple(0.6 * aim / np.linalg.norm(aim)))
            s_next, term, trunc, info = env.step(act)
            base = rm.parts(s, act, s_next, term, trunc, info, memory).base
            if base:
                paid.append(base)
                passers.append(s_next["ball"]["passer"])
            s = s_next
            if term or trunc or memory.passes == len(receivers):
                break
        return passers, paid, memory
    finally:
        env.close()


def test_passing_on_to_the_third_blue_pays_in_full():
    passers, paid, memory = relay([1, 2])  # 0 -> 1 -> 2
    assert passers == [0, 1] and paid == [1.0, 1.0] and memory.returns == 0


def test_a_pass_straight_back_pays_the_return_factor():
    passers, paid, memory = relay([1, 0])  # 0 -> 1 -> 0
    assert passers == [0, 1] and paid == [1.0, 0.5] and memory.returns == 1


def test_the_passer_survives_a_regrab_and_clears_on_any_other_pickup():
    env = types.SimpleNamespace(_kicked_by=None, _passer=None)

    def track(*events):
        PyBulletEnv._track_passer(env, list(events))  # pylint: disable=protected-access
        return env._passer  # pylint: disable=protected-access

    assert track(caught_by(0)) is None  # a loose ball nobody kicked
    assert track(KICK_0, RELEASE_0) is None and track(caught_by(1)) == 0  # 0 passed to 1
    # 1's own kick back
    assert track({"type": "kick", "agent": ("blue", 1)}) == 0 and track(caught_by(1)) == 0
    # a steal
    assert track({"type": "possession_change", "from": ("blue", 1), "to": ("red", 0)}) is None


def test_return_pass_factor_must_be_in_0_1():
    with pytest.raises(ValueError, match="return_pass_factor"):
        config_from_dict({"rl": {"reward": {"return_pass_factor": 1.5}}})
### END AI GENERATED


### AI GENERATED (2026-11-13: a stopped kick is over for kicker_clearance; "got it" calls)
def field(holder=None, ball=(1.0, 0.0), vel=(0.0, 0.0), calling=(),
          blues=((0.0, 0.0), (3.0, 0.0), (0.0, 3.0))):
    """A 3v0 state with velocities: the ball held by ``holder`` or loose at ``ball`` moving at
    ``vel``; the blues in ``calling`` are calling."""
    return {"step": 0, "red": [],
            "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": i == holder,
                      "calling": i in calling} for i, p in enumerate(blues)],
            "ball": {"pos": list(blues[holder] if holder is not None else ball), "vel": list(vel)}}


def kick_then(rm, memory, ball_vel):
    """Blue 0 kicks; the ball ends the step 0.6 m away moving at ``ball_vel``. Returns that step's
    shaping."""
    return rm.parts(field(0), {}, field(ball=(0.6, 0.0), vel=ball_vel), False, False,
                    practice_info(KICK_0, RELEASE_0), memory).shaping


def test_a_kick_that_has_stopped_no_longer_charges_the_kicker():
    rm = RewardModule(with_overrides(config_from_dict(MOVING), CLEAR))
    moving = rm.new_episode()
    # rolling away, the kicker 0.6 m behind it: charged
    assert kick_then(rm, moving, (2.0, 0.0)) < 0
    regrab = rm.parts(field(ball=(0.6, 0.0), vel=(2.0, 0.0)), {}, field(0), False, False,
                      practice_info(caught_by(0)), moving).shaping
    # taking a moving kick back: the full charge
    assert regrab == pytest.approx(-0.025 * KICKER_REGRAB)
    stopped = rm.new_episode()
    assert kick_then(rm, stopped, (0.1, 0.0)) == 0.0  # stopped at once (a dead ball): no charge
    near = rm.parts(field(ball=(0.6, 0.0)), {}, field(ball=(0.6, 0.0)), False, False,
                    practice_info(), stopped)
    fetch = rm.parts(field(ball=(0.6, 0.0)), {}, field(0), False, False,
                     practice_info(caught_by(0)), stopped)
    assert near.shaping == 0.0 and fetch.shaping == 0.0  # standing by it and fetching it are free


CALLS = {"rl": {"call_enabled": True,
                "reward": {"shaping": [{"name": "call_accuracy", "enabled": True, "coef": 1.0}]}}}


def calls_then_pickup(callers, steps, getter=1):
    """``callers`` call for ``steps`` steps while the ball rolls loose; then ``getter`` picks it up.
    Returns the pickup step's call_accuracy and the memory."""
    # AI GENERATED (2026-11-14: the term itself, not the team reward)
    from marl.rl.reward import call_accuracy
    cfg = with_overrides(config_from_dict(MOVING), CALLS)
    rm = RewardModule(cfg)
    memory = rm.new_episode()
    loose = field(ball=(1.5, 1.5), vel=(1.0, 0.0), calling=callers)
    for _ in range(steps):
        rm.parts(loose, {}, loose, False, False, practice_info(), memory)
    ### AI GENERATED (2026-11-14)
    end, info = field(getter, calling=callers), practice_info(caught_by(getter))
    info["pass"] = memory.update(info, loose, end)
    return call_accuracy(loose, {}, end, False, False, info, cfg), memory
    ### END AI GENERATED


def test_a_call_pays_when_the_caller_gets_the_ball():
    from marl.rl.reward import CALL_COST, CALL_LEAD
    full, memory = calls_then_pickup({1}, CALL_LEAD)
    # called for 0.5 s+: full credit, less this step's cost
    assert full == pytest.approx(1.0 - CALL_COST)
    assert (memory.pickups, memory.called, memory.miscalled) == (1, 1.0, 0.0)
    late, _ = calls_then_pickup({1}, 2)
    # only just called (3 steps by the pickup): less
    assert late == pytest.approx(3 / CALL_LEAD - CALL_COST)
    wrong, memory = calls_then_pickup({2}, CALL_LEAD)
    # blue 2 called, blue 1 got it
    assert wrong == pytest.approx(-1.0 - CALL_COST) and memory.miscalled == 1.0
    silent, _ = calls_then_pickup(set(), CALL_LEAD)
    assert silent == 0.0


### AI GENERATED (2026-11-14: the call's reward is its own, not the team's)
def test_call_accuracy_is_not_in_the_team_reward():
    rm = RewardModule(with_overrides(config_from_dict(MOVING), CALLS))
    memory = rm.new_episode()
    held = field(0, calling={1, 2})
    assert rm.parts(held, {}, held, False, False, practice_info(), memory).shaping == 0.0
    assert rm.call_coef == 1.0  # it weights the call head's own loss instead
### END AI GENERATED


def test_call_accuracy_needs_calls_enabled():
    with pytest.raises(ValueError, match="call_enabled"):
        config_from_dict({**MOVING, "rl": {**MOVING["rl"], "reward": {
            "base": "pass_catch",
            "shaping": [{"name": "call_accuracy", "enabled": True, "coef": 1.0}]}}})
### END AI GENERATED
