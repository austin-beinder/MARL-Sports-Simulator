#!/usr/bin/env bash
# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: deactivates the project venv.
#   - 2026-09-26: Moved to Other_OS_Setup_Scripts/Mac_Linux_Zsh.
# Prompt summary: Create _CleanupCloseDevEnv that (for now) deactivates the
#   project's python venv.
# Human review:   Pending review
# ==============================================================================
# Usage:  source Other_OS_Setup_Scripts/Mac_Linux_Zsh/_CleanupCloseDevEnv.sh
# Must be sourced: an executed script runs in a child shell and cannot change
# your shell's environment.
# TODO: add further cleanup steps here as the project grows.

if [ -n "${ZSH_VERSION:-}" ]; then
    case "$ZSH_EVAL_CONTEXT" in *:file*) _marl_sourced=1 ;; *) _marl_sourced=0 ;; esac
else
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then _marl_sourced=1; else _marl_sourced=0; fi
fi

if [ "$_marl_sourced" -eq 0 ]; then
    echo "[cleanup] Please source this script:  source Other_OS_Setup_Scripts/Mac_Linux_Zsh/_CleanupCloseDevEnv.sh"
    exit 1
fi
unset _marl_sourced

if [ -z "${VIRTUAL_ENV:-}" ] || ! command -v deactivate >/dev/null 2>&1; then
    echo "[cleanup] No venv is active. Nothing to deactivate."
elif [ "$(basename "$VIRTUAL_ENV")" != ".venv-CSE-5106-Project-Beinder-Stinson-Wang" ]; then
    echo "[cleanup] Active venv is '$(basename "$VIRTUAL_ENV")', not this project's. Leaving it alone."
else
    deactivate
    unset MARL_VENV_NAME
    echo "[cleanup] Deactivated venv .venv-CSE-5106-Project-Beinder-Stinson-Wang."
fi
