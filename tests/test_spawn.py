# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-10-10
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-10-10: Generated this entire file: spawn sampler tests (constraints
#     over many seeds, determinism, start mix, infeasible constraints, no torch
#     dependency).
#   - 2026-11-07: The reset options now include max_ball_dist. (Austin Beinder)
#   - 2026-11-08: ... and drill_radius. (Austin Beinder)
#   - 2026-11-09: ... and pass_distance. (Austin Beinder)
#   - 2026-11-11: ... and catch_start_mix: keepaway starts in the pass
#     practice layout, plus red. (Austin Beinder)
#   - 2026-11-15: Pylint cleanup: wrapped lines over 100 characters; no
#     behaviour change. (Austin Beinder)
#   - 2026-11-25: Unused unpacked values replaced with _ (pylint cleanup,
#     backlog task 13); no behaviour change.
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
        _, opt, sp = spawn(seed, {"scenario": "goto_target", "n_red": 0})
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
            options={"start_mix": 0.5, "require_blue_closer": True, "min_clearance": 1.0,
                     "min_red_ball_dist": 3.0,
                     # AI GENERATED (max_ball_dist, drills)
                     "max_ball_dist": 0.0, "drill_radius": 2.0, "pass_distance": 2.0,
                     "catch_start_mix": 0.0},  # AI GENERATED (2026-11-11)
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


### AI GENERATED (2026-11-11: catch_start_mix)
THREE_V_ONE = {"n_blue": 3, "n_red": 1, "start": {"pass_distance": 3.9}}


def test_catch_starts_use_the_pass_practice_layout_with_red():
    for seed in range(200):
        s, opt, sp = spawn(seed, THREE_V_ONE, catch_start_mix=1.0)
        blue, red, ball = np.array(sp.blue_pos), np.array(sp.red_pos), np.array(sp.ball_pos)
        assert (sp.start_type == START_POSSESSION and tuple(blue[sp.holder]) == tuple(ball)
                and len(red) == 1)
        d = np.linalg.norm(np.delete(blue, sp.holder, axis=0) - ball, axis=-1)
        # teammates pass_distance away
        assert np.all((d >= 0.85 * 3.9 - 1e-9) & (d <= 1.15 * 3.9 + 1e-9))
        assert np.linalg.norm(red - ball, axis=-1).min() >= opt["min_red_ball_dist"]
        players = np.concatenate([blue, red])
        gaps = np.linalg.norm(players[:, None] - players[None], axis=-1)[np.triu_indices(4, k=1)]
        assert gaps.min() >= opt["min_clearance"]
        assert np.all(np.abs(players) <= np.array(s.field_size) / 2 - s.player_radius + 1e-12)


def test_catch_start_mix_is_the_share_of_catch_starts():
    def layout(sp):  # a catch start: everyone but the holder is ~pass_distance from the ball
        if sp.holder is None:
            return False
        d = np.linalg.norm(np.delete(np.array(sp.blue_pos), sp.holder, axis=0) - sp.ball_pos,
                           axis=-1)
        return bool(np.all(np.abs(d - 3.9) <= 0.15 * 3.9 + 1e-9))
    shares = [np.mean([layout(spawn(seed, THREE_V_ONE, catch_start_mix=mix, start_mix=0.0)[2])
                       for seed in range(300)]) for mix in (0.0, 0.5, 1.0)]
    assert shares[0] == 0.0 and 0.4 < shares[1] < 0.6 and shares[2] == 1.0


def test_catch_start_mix_zero_draws_exactly_as_before():
    for seed in range(20):  # the option draws nothing at 0, so the usual starts are unchanged
        assert spawn(seed, THREE_V_ONE)[2] == spawn(seed, THREE_V_ONE, catch_start_mix=0.0)[2]
    with pytest.raises(ValueError, match="catch_start_mix"):
        spawn(0, THREE_V_ONE, catch_start_mix=2.0)
### END AI GENERATED


def test_shared_modules_do_not_import_torch():
    # The env owner must be able to use these without torch (design #17).
    code = "import sys, marl.common.spawn, marl.common.config; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
