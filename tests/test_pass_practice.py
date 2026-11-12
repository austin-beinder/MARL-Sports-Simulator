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
# Prompt summary: A curriculum stage with three frozen blues and no red, where
#   the holder is rewarded for kicking the ball near a teammate and most for a
#   catch; three stages with the teammates further and further away.
# Human review:   Pending review
# ==============================================================================
import math

import numpy as np
import pytest

from marl.common.config import config_from_dict, with_overrides
from marl.common.spawn import START_POSSESSION, sample_spawn
from marl.envs.mock_env import MockEnv
from marl.envs.pybullet_env import PyBulletEnv
from marl.rl.networks import build_networks
from marl.rl.reward import (KICKER_CLEARANCE, KICKER_REGRAB, PASS_CATCH_MIN, PASS_FULL_LENGTH,  # AI GENERATED
                             PASS_MISS_MAX, PASS_MISS_SCALE, PassMemory, PassStep, RewardModule,  # AI GENERATED
                             kicker_clearance)  # AI GENERATED (2026-11-11)
from marl.rl.rollout import EpisodeRecord
from marl.rl.train import train
from curriculum.runner import Stage, StageCurriculum

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
    return {"blue": [{"move": list(move), "kick_trigger": int(i == kicker), "kick_vec": list(kick_vec)}
                     for i in range(3)]}


def play(env, cfg, first, memory=None):
    """Step ``first`` then nothing until the episode ends; returns (s, terminated, truncated, base reward sum)."""
    rm = RewardModule(cfg)
    memory = memory or rm.new_episode()
    s, total = env._state(), 0.0
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
    return sample_spawn(np.random.default_rng(seed), scenario="pass_practice", n_blue=sc.n_blue, n_red=sc.n_red,
                        field_size=sc.field_size, player_radius=sc.player_radius,
                        possession_radius=sc.possession_radius, options={**sc.start.to_reset_options(), **options})


@pytest.mark.parametrize("distance", [1.5, 2.75, 3.9])
def test_teammates_stand_pass_distance_from_the_holder(cfg, distance):
    sc = cfg.shared
    half = np.asarray(sc.field_size) / 2 - sc.player_radius
    for seed in range(100):
        sp = spawn(cfg, seed, pass_distance=distance)
        blue = np.array(sp.blue_pos)
        assert sp.start_type == START_POSSESSION and sp.red_pos == () and sp.ball_pos == sp.blue_pos[sp.holder]
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
    s0 = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0))
    s, term, trunc, total, memory = play(env, cfg, actions(kicker=0, kick_vec=(1.0, 0.0), move=(1.0, 1.0)))
    assert term and not trunc and s["step"] < 40
    assert s["blue"][1]["has_possession"] and memory.passes == 1
    assert 2.4 < memory.length < 3.1  # caught by blue 1, 3 m out
    assert total == 1.0  # no length bonus by default
    for b0, b in zip(s0["blue"], s["blue"]):  # nobody moved, though blue 0 asked to (the catch nudges blue 1 ~2 mm)
        assert b["pos"] == pytest.approx(b0["pos"], abs=0.02)


def test_a_miss_ends_once_the_ball_cant_reach_a_teammate(env, cfg):
    def miss(kick_vec):
        env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0), step=0)
        s, term, trunc, total, _ = play(env, cfg, actions(kicker=0, kick_vec=kick_vec))
        assert term and not trunc and not any(b["has_possession"] for b in s["blue"])
        roll_left = math.log1p(math.hypot(*s["ball"]["vel"])) / cfg.shared.ball_damping
        gaps = [math.dist(b["pos"], s["ball"]["pos"]) - cfg.shared.player_radius for b in s["blue"][1:]]
        assert roll_left < min(gaps)  # it can't get to blue 1 or blue 2 any more
        return s, total

    s_near, near = miss((0.4, -0.15))  # slightly wide of blue 1, and soft
    s_far, far = miss((-0.5, -0.5))  # away from both
    assert 0 < far < near < PASS_MISS_MAX
    assert s_far["step"] < 30  # it ends long before the time limit, while the ball still rolls


def test_no_kick_runs_to_the_time_limit_and_pays_nothing(env, cfg):
    s, term, trunc, total, memory = play(env, cfg, actions())
    assert trunc and not term and s["step"] == cfg.shared.max_episode_steps
    assert total == 0.0 and memory.passes == 0


def test_closest_approach_follows_the_balls_path():
    memory = PassMemory()
    blues = [{"id": 0, "pos": [0.0, 0.0]}, {"id": 1, "pos": [2.0, 1.0]}, {"id": 2, "pos": [-3.0, 0.0]}]

    def st(ball):
        return {"blue": blues, "ball": {"pos": list(ball)}}

    kick = {"events": [{"type": "kick", "agent": ("blue", 0)}]}
    memory.update(kick, st((0, 0)), st((1, 0)))
    assert memory.closest == pytest.approx(math.hypot(1, 1))  # nearest point of this step: (1, 0)
    memory.update({"events": []}, st((1, 0)), st((4, 0)))  # passes under blue 1 at (2, 0)
    assert memory.closest == pytest.approx(1.0)
    memory.update(kick, st((0, 0)), st((0, -1)))  # a new kick starts a new measurement
    assert memory.closest == pytest.approx(math.hypot(2, 1))  # blue 1, from the start of this short path


### AI GENERATED (2026-11-10: the minimum pass distance and catches paid by length, both options)
LONG = {"rl": {"reward": {"min_pass_distance": 1.0, "pass_length_bonus": True}}}
def practice_state(holder=None, ball=(0.0, 0.0), blues=((0.0, 0.0), (0.6, 0.0), (0.0, 3.0))):
    return {
        "step": 0,
        "blue": [{"id": i, "pos": list(p), "vel": [0.0, 0.0], "has_possession": i == holder} for i, p in enumerate(blues)],
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
    """Blue 0 taps the ball 0.6 m into blue 1; returns the kick step's and the catch step's rewards."""
    kick = rm.parts(practice_state(0), {}, practice_state(ball=(0.3, 0.0)), False, False,
                    practice_info(KICK_0, RELEASE_0), memory)
    end = rm.parts(practice_state(ball=(0.3, 0.0)), {}, practice_state(1), True, False, practice_info(caught_by(1)), memory)
    return kick, end


def test_by_default_a_tap_to_a_teammate_is_a_pass(cfg):
    rm = RewardModule(cfg)
    kick, end = tap(rm, rm.new_episode())
    assert end.base == 1.0  # the old reward: any catch


def test_with_a_minimum_a_tap_to_a_teammate_next_to_the_kicker_is_no_pass_and_pays_nothing(cfg):
    cfg = with_overrides(cfg, LONG)
    rm = RewardModule(cfg)
    memory = rm.new_episode()  # blue 1 stands 0.6 m from blue 0
    assert memory.min_distance == 1.0
    kick, end = tap(rm, memory)
    assert kick.base == 0 and memory.passes == 0  # no pass
    # and no near miss on blue 1 either: the only credit is for blue 2, 3 m away, like a kick that went nowhere
    assert memory.closest == pytest.approx(3.0)
    assert end.base == pytest.approx(PASS_MISS_MAX * math.exp(-(3.0 - cfg.shared.player_radius) / PASS_MISS_SCALE))


def test_the_closest_approach_ignores_teammates_next_to_the_kicker():
    memory = PassMemory(min_distance=1.0)  # blue 1 is 1.4 m from the kick point (it could only catch a pass under 1 m)
    blues = [{"id": 0, "pos": [0.0, 0.0]}, {"id": 1, "pos": [1.4, 0.0]}, {"id": 2, "pos": [0.0, 3.0]}]

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
    rm.parts(practice_state(0, blues=blues), {}, practice_state(ball=(0.5, 0.0), blues=blues), False, False,
             practice_info(KICK_0, RELEASE_0), memory)
    end = rm.parts(practice_state(ball=(length - 0.2, 0.0), blues=blues), {}, practice_state(1, blues=blues),
                   True, False, practice_info(caught_by(1)), memory)
    assert memory.passes == 1 and memory.length == pytest.approx(length)
    assert end.base == pytest.approx(min(1.0, PASS_CATCH_MIN + (1 - PASS_CATCH_MIN) * length / PASS_FULL_LENGTH))
    assert PASS_MISS_MAX < end.base <= 1.0  # every catch beats every miss


def test_min_pass_distance_cant_be_negative():
    with pytest.raises(ValueError, match="min_pass_distance"):
        config_from_dict({**PRACTICE, "rl": {**PRACTICE["rl"], "reward": {"base": "pass_practice", "min_pass_distance": -1.0}}})
### END AI GENERATED


# --- Gate and training ------------------------------------------------------------------
def test_practice_stages_gate_on_the_catch_rate():
    stage = Stage(name="catch", total_steps=1000, threshold=0.8, window=10)
    ctl = StageCurriculum(stage, {}, max_episode_steps=80, log=lambda _m: None, catch_gate=True)

    def episodes(caught, n):  # long episodes would pass a survival gate; caught ones are short
        return [EpisodeRecord(length=20 if caught else 80, terminated=caught, truncated=not caught,
                              start_type=START_POSSESSION, mean_reward=0.0, passes=int(caught))] * n

    row = ctl.update(episodes(False, 10))
    assert row["gate_mean"] == 0.0 and not ctl.finished
    ctl.update(episodes(True, 10))
    assert ctl.gate.mean() == 1.0 and ctl.finished


def test_training_pass_practice_runs_end_to_end(tmp_path, cfg):
    small = with_overrides(cfg, {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(small, tmp_path / "practice", log=lambda *_: None)
    assert result.iterations == 2 and (tmp_path / "practice" / "checkpoints" / "final.pt").is_file()
    assert {"passes_mean", "frac_passed"} <= set(result.final_eval)


# --- Pass practice with the blues free to move (2026-11-09) -------------------------------
MOVING = {**PRACTICE, "shared": {**PRACTICE["shared"], "scenario": "pass_practice_moving", "max_episode_steps": 100}}


def soft_pass_to_blue_1(env, cfg, blue_1_runs):
    """Blue 0 kicks softly at blue 1, 3 m away (0.9 m/s: it rolls ~1.6 m, ~0.5 m short); blue 1 runs at it or stands."""
    env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0), step=0)
    rm = RewardModule(cfg)
    memory, s, total = rm.new_episode(), env._state(), 0.0
    move = [(0.0, 0.0), (-1.0, 0.0) if blue_1_runs else (0.0, 0.0), (0.0, 0.0)]
    first = True
    while True:
        act = {"blue": [{"move": list(m), "kick_trigger": int(first and i == 0), "kick_vec": [0.15, 0.0]}
                        for i, m in enumerate(move)]}
        s_next, term, trunc, info = env.step(act)
        total += rm.parts(s, act, s_next, term, trunc, info, memory).base
        s, first = s_next, False
        if term or trunc:
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
        s, term, trunc, total = soft_pass_to_blue_1(env, cfg, blue_1_runs=True)
        assert term and not trunc and s["blue"][1]["has_possession"]
        assert total == 1.0
        assert s["blue"][1]["pos"][0] < 2.9  # it came to meet the ball
    finally:
        env.close()


def test_the_same_short_pass_is_a_miss_for_a_frozen_teammate(env, cfg):
    s, term, trunc, total = soft_pass_to_blue_1(env, cfg, blue_1_runs=True)  # frozen: its run is ignored
    assert term and not trunc and not any(b["has_possession"] for b in s["blue"])
    assert 0 < total < PASS_MISS_MAX


def test_a_moving_miss_ends_once_no_teammate_can_get_there_in_time():
    cfg = config_from_dict(MOVING)
    env = PyBulletEnv(cfg.shared)
    try:
        env.reset(seed=0)
        env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0), step=0)
        s, term, trunc, total, _ = play(env, cfg, actions(kicker=0, kick_vec=(-0.7071, -0.7071)))  # full kick away
        assert term and not trunc and s["step"] < 50  # ~7 m from both teammates, who'd need 3.5 s to run there
        time_left = (cfg.shared.max_episode_steps - s["step"]) * cfg.shared.dt_control
        run = cfg.shared.max_speed * cfg.shared.free_speed_factor * time_left
        roll_left = math.log1p(math.hypot(*s["ball"]["vel"])) / cfg.shared.ball_damping
        gaps = [math.dist(b["pos"], s["ball"]["pos"]) - cfg.shared.player_radius - 0.15 for b in s["blue"][1:]]
        assert roll_left + run < min(gaps)
    finally:
        env.close()


def test_moving_practice_trains_end_to_end(tmp_path):
    small = with_overrides(config_from_dict(MOVING), {
        "rl": {"ppo": {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}},
        "run": {"total_steps": 128, "eval_every": 2, "eval_episodes": 3},
    })
    result = train(small, tmp_path / "moving", log=lambda *_: None)
    assert result.iterations == 2 and result.actor.move_enabled


### AI GENERATED (2026-11-11: kicker_clearance, the kicker keeps clear of its own kick)
CLEAR = {"rl": {"reward": {"shaping": [{"name": "kicker_clearance", "enabled": True, "coef": 0.025}]}}}


def clearance(kicker_pos, ball_pos, kicker=0, held=False):
    """kicker_clearance for blue ``kicker`` at ``kicker_pos`` with the ball at ``ball_pos``."""
    spots = [kicker_pos, (5.0, 5.0), (-5.0, 5.0)]
    s_next = {"blue": [{"id": i, "pos": [*p, 0.0], "has_possession": held and i == kicker} for i, p in enumerate(spots)],
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
        s = env.set_state_for_testing(blue_pos=[[0, 0], [3, 0], [0, 3]], red_pos=[], holder=("blue", 0), step=0)
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
                act["blue"][0]["move"] = [0.0, 1.0]  # the kicker runs after the ball; its teammates stand still
    finally:
        env.close()


def test_chasing_your_own_kick_costs_more_than_standing_back():
    cfg = with_overrides(config_from_dict(MOVING), CLEAR)
    still, chase = kick_and_follow(cfg, chase=False), kick_and_follow(cfg, chase=True)
    assert -3.0 < still < 0  # a firm kick clears 1.5 m in ~0.3 s: a small, unavoidable cost
    assert chase < still - 1.0
### END AI GENERATED
