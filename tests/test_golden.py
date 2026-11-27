# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: guards for the tracked golden
#     record (fully frozen; its table view re-renders exactly from its own
#     data) for backlog task 2, step 8.
#   - 2026-11-25: Citation cleanup (backlog task 9): dropped the Prompt summary
#     and Human review lines; no behaviour change.
#   - 2026-11-26: Guards both records (backlog task 15): the golden v2 PyBullet
#     record in results/golden/, and golden v1, archived to
#     results/archive/golden_v1/ so the legacy mock results stay guarded.
#   - 2026-11-27: Guards every golden record it finds (results/golden*/ and
#     results/archive/golden*/), and fails if the search finds none (backlog
#     task 16: results/golden/ split into experiment sets, the first renamed
#     results/golden_1v1_initial/).
# ==============================================================================
"""The golden records are git-tracked, so every checkout can check them.

Every folder named ``golden*`` in ``results/`` or ``results/archive/`` is a record and is
guarded here, so a new experiment set is covered as soon as it's frozen. Today:

- ``results/golden_1v1_initial/``: the PyBullet keep-away 1v1 reward-design results (golden v2).
- ``results/archive/golden_v1/``: the archived midterm mock-env record.

If a change to the plotting/table code alters what a record's data renders to, these
tests fail: either fix the code, or re-freeze that record on purpose
(``python scripts/freeze.py <record folder>``) and commit the new table view with a reason.
"""

from pathlib import Path

import pytest

from marl.manifest import load_manifest
from marl.plots import render_tables

RESULTS = Path(__file__).resolve().parents[1] / "results"
RECORDS = sorted(p.parent for p in [*RESULTS.glob("golden*/manifest.toml"),
                                    *(RESULTS / "archive").glob("golden*/manifest.toml")])


def test_the_golden_records_are_found():
    # An empty list would quietly skip every guard below (pytest skips an empty parametrize).
    names = {p.name for p in RECORDS}
    assert "golden_v1" in names and len(RECORDS) >= 2, names


@pytest.fixture(params=RECORDS, ids=lambda p: p.name)
def record(request) -> Path:
    return request.param


def test_the_golden_record_is_fully_frozen(record):
    manifest = load_manifest(record / "manifest.toml")
    assert manifest.unfrozen() == []  # every input lives inside the record
    assert all(Path(p.path).exists() or "*" in p.raw for p in manifest.paths())


def test_the_golden_table_view_re_renders_exactly_from_its_own_data(record):
    sections, notes = render_tables(load_manifest(record / "manifest.toml"))
    assert notes == []
    # Compare text, not bytes: git may convert line endings on checkout (core.autocrlf).
    assert "\n".join(sections) == (record / "figures" / "results_tables.md").read_text(encoding="utf-8")
