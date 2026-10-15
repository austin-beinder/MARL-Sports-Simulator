#!/usr/bin/env bash
# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: macOS/Linux entry point that sets
#     up and activates the venv, then runs the out-of-the-box demo.
#   - 2026-09-26: Moved to Other_OS_Setup_Scripts/Mac_Linux_Zsh; now resolves
#     the project root two folders up.
# Prompt summary: Create _SetupAndRunDemo that creates/activates the project
#   venv, installs requirements.in if missing, then runs a demo placeholder.
# Human review:   Pending review
# ==============================================================================
# Usage:  Other_OS_Setup_Scripts/Mac_Linux_Zsh/_SetupAndRunDemo.sh [extra demo args...]
#     or: source Other_OS_Setup_Scripts/Mac_Linux_Zsh/_SetupAndRunDemo.sh [extra demo args...]   # venv stays active
# The demo script and its default flags are set in _env_setup.py
# (DEMO_SCRIPT / DEMO_ARGS -- currently a placeholder running example.py).

if [ -n "${ZSH_VERSION:-}" ]; then
    case "$ZSH_EVAL_CONTEXT" in *:file*) _marl_sourced=1 ;; *) _marl_sourced=0 ;; esac
    _marl_self="${(%):-%x}"
else
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then _marl_sourced=1; else _marl_sourced=0; fi
    _marl_self="${BASH_SOURCE[0]}"
fi
_marl_dir="$(cd "$(dirname "$_marl_self")" && pwd)"
# This file lives in Other_OS_Setup_Scripts/Mac_Linux_Zsh; the project root is two levels up.
_marl_root="$(cd "$_marl_dir/../.." && pwd)"

. "$_marl_dir/_env_common.sh"

if _marl_env_setup "$_marl_root"; then
    python "$_marl_root/_env_setup.py" run-demo "$@"
    _marl_rc=$?
else
    echo "[setup] Environment setup FAILED. See messages above."
    _marl_rc=1
fi

_marl_was_sourced=$_marl_sourced
unset _marl_sourced _marl_self _marl_dir _marl_root
unset -f _marl_find_python _marl_env_setup
if [ "$_marl_was_sourced" -eq 1 ]; then
    unset _marl_was_sourced
    return $_marl_rc
fi
exit $_marl_rc
