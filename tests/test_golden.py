# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-01
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-11-01: Generated this entire file: guards for the tracked golden
#     record (fully frozen; its table view re-renders exactly from its own
#     data) for backlog task 2, step 8.
# Prompt summary: Freeze the midterm mock-env results as golden record v1 and
#   add a permanent test that it still reproduces.
# Human review:   Pending review
# ==============================================================================
"""The golden record (results/golden/) is git-tracked, so every checkout can check it.

If a change to the plotting/table code alters what the golden data renders to,
these tests fail: either fix the code, or re-freeze the record on purpose
(python scripts/freeze.py golden) and commit the new table view with a reason.
"""

from pathlib import Path

from marl.manifest import load_manifest
from marl.plots import render_tables

GOLDEN = Path(__file__).resolve().parents[1] / "results" / "golden"


def test_the_golden_record_is_fully_frozen():
    manifest = load_manifest(GOLDEN / "manifest.toml")
    assert manifest.unfrozen() == []  # every input lives inside the record
    assert all(Path(p.path).exists() or "*" in p.raw for p in manifest.paths())


def test_the_golden_table_view_re_renders_exactly_from_its_own_data():
    sections, notes = render_tables(load_manifest(GOLDEN / "manifest.toml"))
    assert notes == []
    # Compare text, not bytes: git may convert line endings on checkout (core.autocrlf).
    assert "\n".join(sections) == (GOLDEN / "figures" / "results_tables.md").read_text(encoding="utf-8")
