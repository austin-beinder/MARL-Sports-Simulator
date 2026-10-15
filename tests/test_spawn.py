# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: spawn sampler tests (constraints
#     over many seeds, determinism, start mix, infeasible constraints, no torch
#     dependency).
# Prompt summary: Implement Phase 3 step 2 of the issue #7 plan (spawn sampler).
# Human review:   Pending review
# ==============================================================================
import subprocess
import sys

import numpy as np
import pytest

from marl.common.config import config_from_dict
from marl.common.spawn import START_FREE_BALL, START_POSSESSION, START_TARGET, sample_spawn


def spawn(seed, shared=None, **opt_overrides):
    data = {"shared": shared or {}}
    if data["shared"].get("scenario") == "goto_target":
        data["rl"] = {"reward": {"base": "goto_target"}}
    cfg = config_from_dict(data)
    s = cfg.shared
    options = s.start.to_reset_options() | opt_overrides
    rng = np.random.default_rng(seed)
    return s, options, sample_spawn(
        rng,
        scenario=s.scenario,
        n_blue=s.n_blue,
        n_red=s.n_red,
        field_size=s.field_size,
        player_radius=s.player_radius,
        possession_radius=s.possession_radius,
        options=options,
    )


@pytest.mark.parametrize("shared", [{}, {"n_blue": 2, "n_red": 2}, {"n_blue": 3, "n_red": 1}])
def test_constraints_hold_over_many_seeds(shared):
    types = set()
    for seed in range(400):
        s, opt, sp = spawn(seed, shared)
        types.add(sp.start_type)
        blue, red, ball = np.array(sp.blue_pos), np.array(sp.red_pos), np.array(sp.ball_pos)
        players = np.concatenate([blue, red])
        lim = np.array(s.field_size) / 2 - s.player_radius
        assert np.all(np.abs(players) <= lim + 1e-12)
        assert np.all(np.abs(ball) <= lim + 1e-12)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                assert np.linalg.norm(players[i] - players[j]) >= opt["min_clearance"]
        d_red = np.linalg.norm(red - ball, axis=1)
        d_blue = np.linalg.norm(blue - ball, axis=1)
        assert d_red.min() >= opt["min_red_ball_dist"] > s.possession_radius  # never red possession
        if sp.start_type == START_POSSESSION:
            assert sp.holder is not None and np.allclose(ball, blue[sp.holder])
        else:
            assert sp.start_type == START_FREE_BALL and sp.holder is None
            assert d_blue.min() > s.possession_radius
            assert d_blue.min() < d_red.min()  # require_blue_closer
    assert types == {START_POSSESSION, START_FREE_BALL}


def test_same_seed_same_spawn_and_seeds_differ():
    assert spawn(123)[2] == spawn(123)[2]
    assert spawn(123)[2] != spawn(124)[2]


def test_start_mix_fraction():
    n = 2000
    frac = np.mean([spawn(seed)[2].start_type == START_POSSESSION for seed in range(n)])
    assert abs(frac - 0.5) < 4 * np.sqrt(0.25 / n)  # ~4 sigma for a binomial


@pytest.mark.parametrize("mix, expected", [(0.0, START_FREE_BALL), (1.0, START_POSSESSION)])
def test_start_mix_extremes(mix, expected):
    assert all(spawn(seed, start_mix=mix)[2].start_type == expected for seed in range(100))


def test_goto_target():
    for seed in range(100):
        s, opt, sp = spawn(seed, {"scenario": "goto_target", "n_red": 0})
        assert sp.start_type == START_TARGET and sp.red_pos == () and sp.holder is None
        assert np.linalg.norm(np.subtract(sp.blue_pos[0], sp.ball_pos)) >= opt["min_clearance"]


def test_require_blue_closer_off_allows_red_closer():
    # Corner case: with the constraint off, some free-ball starts have red closer.
    red_closer = 0
    for seed in range(300):
        _, _, sp = spawn(seed, start_mix=0.0, require_blue_closer=False)
        d_blue = np.linalg.norm(np.subtract(sp.blue_pos, sp.ball_pos), axis=1).min()
        d_red = np.linalg.norm(np.subtract(sp.red_pos, sp.ball_pos), axis=1).min()
        red_closer += d_red < d_blue
    assert red_closer > 0


def test_infeasible_constraints_raise():
    with pytest.raises(RuntimeError, match="too tight"):
        rng = np.random.default_rng(0)
        sample_spawn(
            rng,
            scenario="keepaway",
            n_blue=4,
            n_red=4,
            field_size=(2.0, 2.0),
            player_radius=0.25,
            possession_radius=0.5,
            options={"start_mix": 0.5, "require_blue_closer": True, "min_clearance": 1.0, "min_red_ball_dist": 3.0},
            max_tries=200,
        )


@pytest.mark.parametrize("bad", [{"start_mixx": 0.5}, {}])
def test_bad_options_raise(bad):
    with pytest.raises(ValueError):
        sample_spawn(
            np.random.default_rng(0),
            scenario="keepaway",
            n_blue=1,
            n_red=1,
            field_size=(16.0, 10.0),
            player_radius=0.25,
            possession_radius=0.5,
            options=bad,
        )


def test_shared_modules_do_not_import_torch():
    # The env owner must be able to use these without torch (design #17).
    code = "import sys, marl.common.spawn, marl.common.config; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
