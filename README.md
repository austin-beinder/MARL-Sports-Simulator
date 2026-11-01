# MARL-Sports-Simulator
A github repo for our Wash U. CSE 5106 Multi Agent Systems project. 

# Project setup
1. Clone the repo
2. One of the dependencies for pybullet is C++ 14. This can be obtained with the C++ installer from the following link. Click "Download Build Tools". Click Modify "Visual Studio Build Tools 2026". Select the checkbox for "Desktop development with C++". In the bottom right click Install.

```
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

3. Install the required python dependencies by navigating through Update_Requirements.md

4. An example GUI pybullet script can be run with the following command.

```shell
python example.py
```

<!--
AI CITATION: Claude Code (Claude Opus 5.5) | 2026-11-01 | JC Stinson III (LifeHackerIII)
Usage: Wrote the "Results & plots" section (backlog task 2: results records and ad-hoc plots).
Modified: 2026-11-01 - Run names now carry the env (`_pybullet_`/`_mock_`); PyBullet is the default (backlog task 3).
-->
# Results & plots
Training writes each run to `runs/<config>_<env>_seed<N>/` (gitignored): `config.json`, `train.csv`,
`eval.csv`, `checkpoints/`. The env is PyBullet unless you pass `--env mock`.

**Look at new runs (ad hoc).** Figures go to `runs/plots/<name>/`:

```shell
python scripts/evaluate.py "runs/new_reward_pybullet_seed*"     # evaluation table (needed for final bars + baseline lines)
python scripts/plot.py --name new_reward --arm "new reward=runs/new_reward_pybullet_seed*" --baselines "new reward"
python scripts/plot.py --name vs_golden --arm "golden=results/golden/data/Old_mock_env/keepaway_1v1_seed*" --arm "new reward=runs/new_reward_pybullet_seed*"
```

Each `--arm` is `LABEL=GLOB` (one arm = the seeds of one config; up to 9 arms per figure). Writing into a
folder that already has figures needs `--overwrite`.

**Results records** live in `results/<record>/` (git-tracked): a `manifest.toml` that lists the figures,
their arms and the table sections; `data/` (frozen CSV + config copies, no checkpoints); and `figures/`.
`results/golden/` holds the for-score results, and plain `python scripts/plot.py` renders it to
`runs/plots/golden/`.

To add an experiment to a record, add an arm to its manifest pointing at the runs
(`runs = "../../runs/new_reward_pybullet_seed*"`), then freeze. That copies those files into the record, rewrites
the path, notes the date and commit, and renders the figures:

```shell
python scripts/freeze.py golden     # then commit results/golden/
```

`tests/test_golden.py` checks that the golden record still re-renders exactly from its own data.
