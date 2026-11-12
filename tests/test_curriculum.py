# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-07
# Author:  Austin Beinder
# Usage:
#   - 2026-11-07: Generated this entire file: tests for the curriculum (the
#     max_ball_dist spawn option, train()'s init_from and curriculum hooks,
#     the confidence gate, stage levels, curriculum loading, and a whole
#     two-stage curriculum on the mock env).
#   - 2026-11-07: Tests for the gate in the printed line and --render.
#   - 2026-11-07: The shipped curriculum kicks, so it refuses the mock.
#   - 2026-11-07: Warm starts across team sizes (1v1 -> 2v1) and the 2v1 stage.
#   - 2026-11-07: The goto_target warm-up: no-gate stages, and the critic
#     starting fresh when the reward changes.
#   - 2026-11-08: The shipped curriculum without possession, plus the
#     slow_two_v_one stage.
#   - 2026-11-08: The passing_3v1 curriculum loads. (Austin Beinder)
#   - 2026-11-09: Teams may shrink in a warm start; the passing curriculum
#     test follows its current stages, catch stages included. (Austin Beinder)
#   - 2026-11-10: The passing curriculum's reworked stages (catch_moving, then
#     1, 2 and 3 m minimum passes).
#   - 2026-11-11: Restarting from a later stage (--start-at, --init-from);
#     the passing curriculum's kicker_clearance and go_to_target_again.
#   - 2026-11-11: ... and keepaway_from_catch (catch_start_mix levels).
# Prompt summary: Lay out a curriculum for keep-away: a possession stage, then
#   a free-ball stage whose ball starts close and moves further away (CACTUS).
# Human review:   Pending review
# ==============================================================================
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from curriculum.runner import (
    ConfidenceGate,
    Stage,
    StageCurriculum,
    load_curriculum,
    run_curriculum,
    stage_number,  # AI GENERATED (2026-11-11)
    stage_configs,
)
from marl.common.config import config_from_dict, config_to_dict
from marl.common.spawn import START_FREE_BALL, sample_spawn
from marl.envs.mock_env import MockEnv
from marl.rl.checkpoint import warm_start
from marl.rl.networks import MOVE, build_networks
from marl.rl.state import parse_state
from marl.rl.reward import RewardModule
from marl.rl.rollout import Featurizer, RolloutCollector, make_buffer
from marl.rl.train import train

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "keepaway_1v1.toml"
SMALL_PPO = {"rollout_steps": 64, "num_minibatches": 4, "update_epochs": 1}


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Ep:  # just the fields the gate reads from an EpisodeRecord
    length: int


# --- The max_ball_dist spawn option --------------------------------------------------------
@pytest.mark.parametrize("n_blue, radius", [(1, 1.0), (1, 3.0), (2, 1.5)])
def test_free_ball_spawns_within_max_ball_dist_of_a_blue(n_blue, radius):
    sc = config_from_dict({"shared": {"n_blue": n_blue, "start": {"start_mix": 0.0, "max_ball_dist": radius}}}).shared
    half = np.array(sc.field_size) / 2 - sc.player_radius
    for seed in range(200):
        sp = sample_spawn(np.random.default_rng(seed), scenario="keepaway", n_blue=n_blue, n_red=1,
                          field_size=sc.field_size, player_radius=sc.player_radius,
                          possession_radius=sc.possession_radius, options=sc.start.to_reset_options())
        assert sp.start_type == START_FREE_BALL
        d_blue = np.linalg.norm(np.array(sp.blue_pos) - sp.ball_pos, axis=1)
        d_red = np.linalg.norm(np.array(sp.red_pos) - sp.ball_pos, axis=1)
        assert sc.possession_radius < d_blue.min() <= radius  # loose, but close to a blue
        assert d_red.min() >= sc.start.min_red_ball_dist
        assert np.all(np.abs(sp.ball_pos) <= half)  # inside the walls


def test_max_ball_dist_zero_means_anywhere():
    sc = config_from_dict({"shared": {"start": {"start_mix": 0.0}}}).shared
    far = 0
    for seed in range(200):
        sp = sample_spawn(np.random.default_rng(seed), scenario="keepaway", n_blue=1, n_red=1,
                          field_size=sc.field_size, player_radius=sc.player_radius,
                          possession_radius=sc.possession_radius, options=sc.start.to_reset_options())
        far += np.linalg.norm(np.array(sp.blue_pos[0]) - sp.ball_pos) > 4.0
    assert far > 50


@pytest.mark.parametrize("value", [-1.0, 0.3, 0.5])
def test_max_ball_dist_must_be_zero_or_beyond_the_possession_radius(value):
    with pytest.raises(ValueError, match="max_ball_dist"):
        config_from_dict({"shared": {"start": {"max_ball_dist": value}}})


def test_rollout_uses_the_options_it_is_given():
    cfg = config_from_dict({"rl": {"ppo": SMALL_PPO}, "shared": {"max_episode_steps": 20}})
    feat = Featurizer(cfg)
    actor, critic = build_networks(cfg, feat.obs_dim, feat.state_dim)
    starts = []

    class Recorder(MockEnv):
        def reset(self, seed, options=None):
            s, info = super().reset(seed, options)
            starts.append(s)
            return s, info

    col = RolloutCollector(Recorder(cfg.shared), cfg, actor, critic, RewardModule(cfg), np.random.default_rng(0))
    col.set_reset_options({**cfg.shared.start.to_reset_options(), "start_mix": 0.0, "max_ball_dist": 1.0})
    col.collect(make_buffer(cfg, feat, actor))
    assert len(starts) >= 3
    for s in starts:
        d = np.linalg.norm(np.array(s["blue"][0]["pos"]) - s["ball"]["pos"])
        assert d <= 1.0 + 1e-9


# --- train() hooks ---------------------------------------------------------------------------
def tiny_cfg(**extra):
    return config_from_dict({"rl": {"ppo": SMALL_PPO, **extra.get("rl", {})},
                             "shared": {"max_episode_steps": 20},
                             "run": {"total_steps": 256, "eval_every": 2, "eval_episodes": 3}})


class FakeCurriculum:
    """Ends the run after `stop_after` rollouts and records what train() asked for."""

    def __init__(self, stop_after):
        self.stop_after, self.calls, self.finished = stop_after, 0, False

    def reset_options(self):
        return {**tiny_cfg().shared.start.to_reset_options(), "start_mix": 1.0}

    def update(self, episodes):
        self.calls += 1
        self.finished = self.calls >= self.stop_after
        return {"cur_level": self.calls, "gate_mean": 0.5}


def test_curriculum_hook_adds_columns_and_can_end_the_run_early(tmp_path):
    fake = FakeCurriculum(stop_after=2)
    res = train(tiny_cfg(), tmp_path / "run", log=lambda _m: None, curriculum=fake)
    assert res.iterations == 2 and fake.calls == 2  # 256 steps / 64 would be 4 iterations
    tr = rows(res.out_dir / "train.csv")
    assert [r["cur_level"] for r in tr] == ["1", "2"]
    assert (res.out_dir / "checkpoints" / "final.pt").exists()
    assert [r["iteration"] for r in rows(res.out_dir / "eval.csv")] == ["0", "2", "2"]  # last stoch + deterministic


def test_init_from_starts_from_the_checkpoint(tmp_path):
    a = train(tiny_cfg(), tmp_path / "a", log=lambda _m: None)
    messages = []
    frozen = tiny_cfg(rl={"ppo": {**SMALL_PPO, "lr": 1e-12}})  # a learning rate this small barely moves the weights
    b = train(frozen, tmp_path / "b", log=messages.append, init_from=a.out_dir / "checkpoints" / "final.pt")
    assert any("warm start" in m for m in messages)
    for p, q in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)
    for p, q in zip(a.critic.parameters(), b.critic.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=0)


# --- Warm start across team sizes -----------------------------------------------------------------
def nets(cfg, seed):
    torch.manual_seed(seed)
    feat = Featurizer(cfg)
    return feat, *build_networks(cfg, feat.obs_dim, feat.state_dim)


@pytest.mark.parametrize("critic_input", ["global", "local"])
def test_growing_1v1_to_2v1_keeps_the_old_outputs(critic_input):
    small = config_from_dict({"rl": {"critic_input": critic_input}})
    big = config_from_dict({"rl": {"critic_input": critic_input}, "shared": {"n_blue": 2}})
    f1, actor1, critic1 = nets(small, 0)
    f2, actor2, critic2 = nets(big, 1)  # a different init, so a match can only come from the warm start
    ckpt = {"actor": actor1.state_dict(), "critic": critic1.state_dict(), "config": config_to_dict(small)}
    assert warm_start(actor2, critic2, ckpt, big) == ([], True)

    s, _ = MockEnv(big.shared).reset(seed=0)
    obs2, cin2 = f2(s)  # [2, 23]: self 0-4 | teammate 5-10 | red 11-16 | ball 17-22
    obs1 = np.delete(obs2, np.s_[5:11], axis=1)  # the same agents as a 1v1 network sees them
    with torch.no_grad():
        torch.testing.assert_close(actor2.distributions(torch.as_tensor(obs2))[MOVE].mean,
                                   actor1.distributions(torch.as_tensor(obs1))[MOVE].mean)
        if critic_input == "global":  # blue 0 0-4 | blue 1 5-9 | red 10-14 | ball 15-18
            state1 = f1.flattener(parse_state(_drop_blue_1(s), 1, 1))
            torch.testing.assert_close(critic2(torch.as_tensor(cin2[:1])), critic1(torch.as_tensor(state1[None])))
        else:
            torch.testing.assert_close(critic2(torch.as_tensor(cin2)), critic1(torch.as_tensor(obs1)))


def _drop_blue_1(s):
    return {**s, "blue": s["blue"][:1]}


def test_shrinking_drops_the_removed_players_and_a_new_critic_mode_raises():
    big = config_from_dict({"shared": {"n_blue": 2}})
    _, actor2, critic2 = nets(big, 0)
    ckpt = {"actor": actor2.state_dict(), "critic": critic2.state_dict(), "config": config_to_dict(big)}
    small = config_from_dict({})
    _, actor1, critic1 = nets(small, 1)
    assert warm_start(actor1, critic1, ckpt, small) == ([], True)  # AI GENERATED (2026-11-09): teams may shrink
    w2, w1 = actor2.state_dict()["trunk.0.weight"], actor1.state_dict()["trunk.0.weight"]
    torch.testing.assert_close(w1, torch.cat([w2[:, :5], w2[:, 11:]], dim=1))  # the teammate's columns 5-10 dropped
    local = config_from_dict({"shared": {"n_blue": 2}, "rl": {"critic_input": "local"}})
    _, actor3, critic3 = nets(local, 0)
    with pytest.raises(ValueError, match="critic"):
        warm_start(actor3, critic3, ckpt, local)


def test_goto_target_to_keepaway_keeps_the_actor_but_not_the_critic():
    goto = config_from_dict({"shared": {"scenario": "goto_target", "n_red": 0}, "rl": {"reward": {"base": "goto_target"}}})
    keep = config_from_dict({})
    f0, actor0, critic0 = nets(goto, 0)
    f1, actor1, critic1 = nets(keep, 1)
    fresh_critic = {k: v.clone() for k, v in critic1.state_dict().items()}
    ckpt = {"actor": actor0.state_dict(), "critic": critic0.state_dict(), "config": config_to_dict(goto)}
    assert warm_start(actor1, critic1, ckpt, keep) == ([], False)

    for k, v in critic1.state_dict().items():  # the critic kept its fresh init
        torch.testing.assert_close(v, fresh_critic[k])
    s, _ = MockEnv(keep.shared).reset(seed=0)
    obs1, _ = f1(s)  # [1, 17]: self 0-4 | red 5-10 | ball 11-16
    obs0 = np.delete(obs1, np.s_[5:11], axis=1)  # goto_target's view: self | ball
    with torch.no_grad():  # red's new inputs start at zero, so the actor acts as it did in goto_target
        torch.testing.assert_close(actor1.distributions(torch.as_tensor(obs1))[MOVE].mean,
                                   actor0.distributions(torch.as_tensor(obs0))[MOVE].mean)


def test_curriculum_from_goto_target_to_keepaway_on_the_mock(tmp_path):
    text = TWO_STAGES.replace(
        '[[stage]]\nname = "possession"\ntotal_steps = 256\nthreshold = 0.0       # always passes once the window is full\nwindow = 2\n'
        '[stage.overrides.shared.start]\nstart_mix = 1.0',
        '[[stage]]\nname = "go_to_target"\ntotal_steps = 128\n'
        '[stage.overrides.shared]\nscenario = "goto_target"\nn_red = 0\n'
        '[stage.overrides.rl.reward]\nbase = "goto_target"')
    assert "go_to_target" in text
    messages = []
    summary = run_curriculum(load_curriculum(write(tmp_path, text)), out_dir=tmp_path / "cur", log=messages.append)
    assert [r["name"] for r in summary] == ["go_to_target", "free_ball"]
    assert summary[0]["iterations"] == 2 and not summary[0]["gate_passed"]  # no gate: ran to its cap
    assert any("1v0 -> 1v1" in m and "critic starts fresh" in m for m in messages)  # red added; new reward


def test_curriculum_from_1v1_to_2v1_on_the_mock(tmp_path):
    text = TWO_STAGES.replace('[[stage]]\nname = "free_ball"', '[[stage]]\nname = "two_v_one"').replace(
        "[stage.overrides.shared.start]\nstart_mix = 0.0", "[stage.overrides.shared]\nn_blue = 2\n"
        "[stage.overrides.shared.start]\nstart_mix = 0.0")
    path = write(tmp_path, text)
    messages = []
    summary = run_curriculum(load_curriculum(path), out_dir=tmp_path / "cur", log=messages.append)
    assert [r["name"] for r in summary] == ["possession", "two_v_one"]
    assert any("1v1 -> 2v1" in m for m in messages)


# --- Gate and levels ---------------------------------------------------------------------------
def test_gate_needs_a_full_window_and_a_confident_mean():
    gate = ConfidenceGate(threshold=0.5, confidence=2.0, window=4)
    gate.add([0.9, 0.9, 0.9])
    assert math.isnan(gate.lower_bound()) and not gate.passed()  # not full yet
    gate.add([0.9])
    assert gate.passed()  # no spread at all: the lower bound is the mean
    gate.add([0.1, 0.9, 0.1, 0.9])  # the window slides: mean 0.5 with a big spread
    assert gate.mean() == pytest.approx(0.5) and gate.lower_bound() < 0.5 and not gate.passed()


def test_stage_steps_through_levels_then_finishes():
    stage = Stage(name="free_ball", total_steps=1000, threshold=0.5, window=2,
                  option="max_ball_dist", levels=[1.0, 2.0, 0.0], stop_when_passed=True)
    ctl = StageCurriculum(stage, {"start_mix": 0.0, "max_ball_dist": 0.0}, max_episode_steps=100, log=lambda _m: None)
    assert ctl.reset_options()["max_ball_dist"] == 1.0
    row = ctl.update([Ep(100)])
    assert ctl.level == 0 and math.isnan(row["gate_lower"])  # one episode doesn't fill the window
    ctl.update([Ep(100)])
    assert ctl.level == 1 and ctl.reset_options()["max_ball_dist"] == 2.0
    ctl.update([Ep(10), Ep(10)])  # failing episodes: stay
    assert ctl.level == 1
    ctl.update([Ep(100), Ep(100)])
    assert ctl.level == 2 and ctl.reset_options()["max_ball_dist"] == 0.0 and not ctl.finished
    row = ctl.update([Ep(100), Ep(100)])
    assert ctl.finished and ctl.passed_last_level and row["cur_level"] == 2


def test_stage_without_a_threshold_never_passes():
    stage = Stage(name="warm_up", total_steps=1000, window=2)
    ctl = StageCurriculum(stage, {"start_mix": 0.5}, max_episode_steps=100, log=lambda _m: None)
    for _ in range(5):
        row = ctl.update([Ep(100), Ep(100)])
    assert not ctl.finished and not ctl.passed_last_level and row["gate_mean"] == 1.0


def test_stage_without_levels_and_keep_training_after_passing():
    stage = Stage(name="possession", total_steps=1000, threshold=0.5, window=2, stop_when_passed=False)
    ctl = StageCurriculum(stage, {"start_mix": 1.0}, max_episode_steps=100, log=lambda _m: None)
    assert ctl.reset_options() == {"start_mix": 1.0} and math.isnan(ctl.value())
    ctl.update([Ep(100), Ep(100)])
    assert ctl.passed_last_level and not ctl.finished


# --- Loading -------------------------------------------------------------------------------------
def write(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_curriculum_loads():
    c = load_curriculum(ROOT / "curriculum" / "keepaway_1v1.toml")
    configs = stage_configs(c)
    # possession is commented out for now (2026-11-08)
    assert [s.name for s in c.stages] == ["go_to_target", "free_ball", "two_v_one", "slow_two_v_one"]
    assert [cfg.shared.scenario for cfg in configs] == ["goto_target", "keepaway", "keepaway", "keepaway"]
    assert [(cfg.shared.n_blue, cfg.shared.n_red) for cfg in configs] == [(1, 0), (1, 1), (2, 1), (2, 1)]
    assert [cfg.shared.start.start_mix for cfg in configs[1:]] == [0.0, 0.5, 0.5]
    assert c.stages[0].threshold is None  # the warm-up has no gate
    assert c.stages[1].levels == [1.0, 2.0, 3.0] and configs[1].shared.start.max_ball_dist == 3.0  # eval at 3 m
    # the last stage: blue is slower than red with or without the ball
    assert [cfg.shared.free_speed_factor for cfg in configs] == [1.0, 1.0, 1.0, 0.875]
    slow = configs[3].shared
    assert slow.max_speed * slow.carry_speed_factor < slow.max_speed * slow.free_speed_factor < slow.red_max_speed
    assert all(cfg.run.env == "pybullet" and cfg.rl.kick_enabled for cfg in configs)
    assert stage_configs(c, seed=3)[0].run.seed == 3
    with pytest.raises(NotImplementedError, match="pybullet"):  # kicking needs PyBullet
        stage_configs(c, env="mock")


### AI GENERATED (2026-11-08; 2026-11-09: the catch stages, and the stage list as it is now)
def test_passing_curriculum_loads():
    c = load_curriculum(ROOT / "curriculum" / "passing_3v1.toml")
    configs = stage_configs(c)
    names = [s.name for s in c.stages]
    assert names == ["go_to_target", "free_ball", "catch_moving", "catch_moving_long", "catch_moving_longer",
                     "catch_moving_longest", "keepaway_from_catch", "go_to_target_again", "keepaway_3v1"]
    catch = configs[2:6]
    assert all(cfg.shared.scenario == "pass_practice_moving" and cfg.rl.reward.base == "pass_practice"
               for cfg in catch)
    assert all((cfg.shared.n_blue, cfg.shared.n_red) == (3, 0) for cfg in catch)
    minimums = [cfg.rl.reward.min_pass_distance for cfg in catch]
    assert minimums == [0.0, 1.0, 2.0, 3.0]  # passes have to go further and further
    assert [cfg.rl.reward.pass_length_bonus for cfg in catch] == [False, True, True, True]
    assert all([(t.name, t.enabled) for t in cfg.rl.reward.shaping] == [("kicker_clearance", True)]
               for cfg in catch)  # 2026-11-11: the kicker keeps clear of its kick
    for cfg in catch:  # the teammates start far enough out for the minimum, and a kick can still reach them
        sc = cfg.shared
        assert cfg.rl.reward.min_pass_distance < sc.start.pass_distance * 0.85
        assert sc.start.pass_distance * 1.15 < math.log(1 + sc.max_kick_speed) / sc.ball_damping
    red_joins = configs[6]  # 2026-11-11: 3v1 from the catch layout, then random positions
    assert red_joins.shared.scenario == "keepaway" and (red_joins.shared.n_blue, red_joins.shared.n_red) == (3, 1)
    assert c.stages[6].option == "catch_start_mix" and c.stages[6].levels == [1.0, 0.5, 0.0]
    assert red_joins.shared.start.pass_distance == catch[-1].shared.start.pass_distance
    last = configs[-1]
    assert last.shared.scenario == "keepaway" and last.rl.reward.base == "survival" and last.rl.reward.shaping == ()
    assert (last.shared.n_blue, last.shared.n_red) == (3, 1)
    assert last.shared.max_speed * last.shared.free_speed_factor < last.shared.red_max_speed
    assert all(cfg.run.env == "pybullet" and cfg.rl.kick_enabled for cfg in configs)
### END AI GENERATED


def test_levels_need_a_threshold(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        load_curriculum(write(tmp_path, 'name="x"\nconfig="configs/keepaway_1v1.toml"\n[[stage]]\nname="a"\n'
                                        'total_steps=2048\noption="max_ball_dist"\nlevels=[1.0]'))


@pytest.mark.parametrize("text, error", [
    ('name="x"\nconfig="configs/keepaway_1v1.toml"\nbogus=1\n[[stage]]\nname="a"\ntotal_steps=2048\nthreshold=0.5', "bogus"),
    ('name="x"\nconfig="configs/keepaway_1v1.toml"', "at least one"),
    ('name="x"\nconfig="configs/keepaway_1v1.toml"\n[[stage]]\nname="a"\ntotal_steps=2048\nthreshold=0.5\noption="red_speed"\nlevels=[1.0]', "reset option"),
    ('name="x"\nconfig="configs/keepaway_1v1.toml"\n[[stage]]\nname="a"\ntotal_steps=2048\nthreshold=0.5\nlevels=[1.0]', "both"),
])
def test_bad_curriculum_files_raise(tmp_path, text, error):
    with pytest.raises(ValueError, match=error):
        load_curriculum(write(tmp_path, text))


def test_an_invalid_level_fails_before_anything_trains(tmp_path):
    c = load_curriculum(write(tmp_path, 'name="x"\nconfig="configs/keepaway_1v1.toml"\n[[stage]]\nname="a"\n'
                                        'total_steps=2048\nthreshold=0.5\noption="max_ball_dist"\nlevels=[1.0, 0.3]'))
    with pytest.raises(ValueError, match="max_ball_dist"):
        stage_configs(c)


# --- A whole curriculum on the mock --------------------------------------------------------------
TWO_STAGES = f"""
name = "tiny"
config = "{BASE_CONFIG.as_posix()}"
env = "mock"

[overrides.shared]
max_episode_steps = 20
[overrides.rl.ppo]
rollout_steps = 64
num_minibatches = 4
update_epochs = 1
[overrides.run]
eval_every = 2
eval_episodes = 3

[[stage]]
name = "possession"
total_steps = 256
threshold = 0.0       # always passes once the window is full
window = 2
[stage.overrides.shared.start]
start_mix = 1.0

[[stage]]
name = "free_ball"
total_steps = 256
option = "max_ball_dist"
levels = [1.0, 0.0]
threshold = 0.0
window = 2
stop_when_passed = false
[stage.overrides.shared.start]
start_mix = 0.0
"""


def test_two_stage_curriculum_on_the_mock(tmp_path):
    path = write(tmp_path, TWO_STAGES)
    messages = []
    out = tmp_path / "cur"
    summary = run_curriculum(load_curriculum(path), out_dir=out, curriculum_path=path, log=messages.append)

    assert [r["name"] for r in summary] == ["possession", "free_ball"]
    assert summary[0]["iterations"] == 1 and summary[0]["gate_passed"]  # stopped as soon as it passed
    assert summary[1]["iterations"] == 4 and summary[1]["value_reached"] == 0.0  # trained on to the cap
    assert (out / "curriculum.toml").exists() and len(rows(out / "curriculum.csv")) == 2
    stage2 = out / "02_free_ball"
    levels = [r["cur_value"] for r in rows(stage2 / "train.csv")]
    assert levels[0] == "1.0" and levels[-1] == "0.0"
    assert any("warm start" in m and "01_possession" in m for m in messages)

    with pytest.raises(FileExistsError):  # no silent overwrite
        run_curriculum(load_curriculum(path), out_dir=out, log=lambda _m: None)



### AI GENERATED (2026-11-11: restarting a curriculum from a later stage)
def test_restart_from_stage_2_keeps_stage_1(tmp_path):
    path = write(tmp_path, TWO_STAGES)
    out = tmp_path / "cur"
    run_curriculum(load_curriculum(path), out_dir=out, curriculum_path=path, log=lambda _m: None)
    stage1 = out / "01_possession" / "checkpoints" / "final.pt"
    before = stage1.read_bytes(), rows(out / "curriculum.csv")[0]

    with pytest.raises(FileExistsError, match="02_free_ball"):  # stage 2 has a run; stage 1's is fine
        run_curriculum(load_curriculum(path), out_dir=out, start_at=2, log=lambda _m: None)
    messages = []
    summary = run_curriculum(load_curriculum(path), out_dir=out, curriculum_path=path, start_at="free_ball",
                             overwrite=True, log=messages.append)
    assert [r["name"] for r in summary] == ["free_ball"]
    assert stage1.read_bytes() == before[0]  # stage 1 untouched
    table = rows(out / "curriculum.csv")
    assert [r["name"] for r in table] == ["possession", "free_ball"] and table[0] == before[1]
    assert any("starting at stage 2/2 free_ball" in m for m in messages)
    assert any("warm start" in m and "01_possession" in m for m in messages)


def test_restart_needs_the_previous_stage_or_a_checkpoint(tmp_path):
    path = write(tmp_path, TWO_STAGES)
    with pytest.raises(FileNotFoundError, match="01_possession"):
        run_curriculum(load_curriculum(path), out_dir=tmp_path / "empty", start_at=2, log=lambda _m: None)
    a = train(tiny_cfg(), tmp_path / "a", log=lambda _m: None)
    messages = []
    summary = run_curriculum(load_curriculum(path), out_dir=tmp_path / "cur", start_at=2,
                             init_from=a.out_dir / "checkpoints" / "final.pt", log=messages.append)
    assert [r["stage"] for r in summary] == [2]
    assert any("warm start" in m and str(tmp_path / "a") in m for m in messages)


@pytest.mark.parametrize("start_at, error", [(0, "stages 1 to 2"), (3, "stages 1 to 2"), ("nope", "no such stage")])
def test_bad_start_at_raises(tmp_path, start_at, error):
    with pytest.raises(ValueError, match=error):
        stage_number(load_curriculum(write(tmp_path, TWO_STAGES)), start_at)
### END AI GENERATED

def test_printed_line_shows_the_gate(tmp_path):
    path = write(tmp_path, TWO_STAGES)
    messages = []
    run_curriculum(load_curriculum(path), out_dir=tmp_path / "cur", log=messages.append)
    lines = [m for m in messages if m.startswith("it ")]
    assert all(" | gate " in m and "(low " in m for m in lines)
    assert any("lvl 0 = 1" in m for m in lines)  # the free-ball stage shows its level


def test_plain_training_line_has_no_gate(tmp_path):
    messages = []
    train(tiny_cfg(), tmp_path / "plain", log=messages.append)
    assert not any("gate" in m for m in messages if m.startswith("it "))


def test_render_needs_pybullet(tmp_path):
    path = write(tmp_path, TWO_STAGES)  # env = "mock"
    with pytest.raises(ValueError, match="PyBullet"):
        run_curriculum(load_curriculum(path), out_dir=tmp_path / "r", render=True, log=lambda _m: None)


def test_cli_runs_as_a_script(tmp_path):
    path = write(tmp_path, TWO_STAGES)
    out = tmp_path / "cli"
    result = subprocess.run([sys.executable, str(ROOT / "curriculum" / "runner.py"), str(path), "--out-dir", str(out)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "free_ball" in result.stdout and (out / "02_free_ball" / "checkpoints" / "final.pt").exists()
