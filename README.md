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

<!--
AI CITATION: Claude Code (Claude Opus 5.5) | 2026-11-02 | JC Stinson III (LifeHackerIII)
Usage: Wrote the "Regression tests (silver record)" section (backlog task 4: the regression harness).
Modified: 2026-11-02 - The tracked report (results/silver/report.md) and its review workflow.
-->
# Regression tests (silver record)
`results/silver/` is the **regression record**: short training runs on reserved seeds whose outputs are
frozen in git. Fixed seeds on unchanged code reproduce them **bit for bit**, so after any change, one
command tells you whether it touched simulation or training results:

```shell
python scripts/regress.py        # ~1 min; exit 0 = all identical, 1 = deltas, 2 = a run failed
```

It trains every case of `results/silver/suite.toml` (a config x an env; seeds 90-91, 6144 steps each, the
config's own PPO settings) into `runs/regression/`, evaluates them, and compares every output with the
record. The report gives one line per case, under a header naming the date and the commit that was checked
(`+ uncommitted changes` means that commit plus your working changes, the normal case before you commit):

| Status | Meaning |
|---|---|
| `IDENTICAL` | Bit-identical to the record. |
| `CHANGED` | A value differs. Under it: the files that differ, **config differences first** (they usually explain the rest), then the first differing cell and the largest delta. |
| `SCHEMA` | Only columns/keys were added or removed; every shared value is identical. |
| `NEW` / `STALE` | A suite case with no record yet / a record entry no suite case produces. |
| `FAILED` | A run crashed; the end of its log is shown (full logs in `runs/regression/logs/`). |

**The report is tracked: commit it with your work.** Every full run (no `--only`, `--no-run` or `--rtol`)
writes `results/silver/report.md`, so a PR shows its reviewer that the regression check ran, on which commit,
and what it found. (A scratch copy of every report, partial runs included, goes to `runs/regression/report.md`.)

**A change that shouldn't affect results** (a refactor, cleanup, docs) must come back all `IDENTICAL`.
Commit the report with the change. Anything else is a bug to find before merging.

**A change that should affect results** (physics, rewards, RL settings): run it, check that the deltas are
the ones you expect, then re-freeze. The report then shows the deltas against the old record plus your
reason, and you commit it and the record together with the change, so the reviewer and the git history
both see why the baseline moved:

```shell
python scripts/regress.py --update --reason "red keeps full speed near blue (fix-speed)"
git add results/silver   # the record + report, in the same commit (or PR) as the change
```

If two branches both changed `report.md`, a merge conflicts there. It's generated, so after merging, rerun
`python scripts/regress.py` and commit the fresh report.

Other options: `--only "*_mock"` (a subset of cases), `--no-run` (re-compare the last run without
retraining), `--rtol 1e-9` (forgive tiny numeric drift, e.g. on another machine), `--parallel-runs N`
(default: half your logical cores). **Results are exact only on the same software and hardware**: the
record's `stamp.json` notes the machine it was frozen on, and the report warns when yours differs.

**To cover a new config**, add a `[[case]]` to `suite.toml` (the harness lists any `configs/*.toml` no case
uses), then `--update`. **Seeds:** 0-9 are for-score runs; 90-99 are for testing (regression 90-91,
smoke tests 99). The suite's `eval_seed` (20261101) differs from the for-score one (20261019), so a
regression run can't be mixed into a for-score table by accident.
