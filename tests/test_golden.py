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
# ==============================================================================
"""The golden records are git-tracked, so every checkout can check them.

- ``results/golden/``: golden v2, the PyBullet keep-away 1v1 results (the for-score record).
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
RECORDS = [RESULTS / "golden", RESULTS / "archive" / "golden_v1"]


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
