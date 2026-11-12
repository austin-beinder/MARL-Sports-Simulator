# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-11-07
# Author:  Austin Beinder
# Usage:
#   - 2026-11-07: Created this file: marks curriculum/ as a package.
# Prompt summary: Lay out a curriculum for keep-away: a possession stage, then
#   a free-ball stage whose ball starts close and moves further away (CACTUS).
# Human review:   Pending review
# ==============================================================================
"""Curriculum training: a list of training stages, each warm-started from the one before.

Everything lives in ``runner.py``; the ``*.toml`` files here describe curricula. Run one with
``python curriculum/runner.py curriculum/keepaway_1v1.toml``.
"""
