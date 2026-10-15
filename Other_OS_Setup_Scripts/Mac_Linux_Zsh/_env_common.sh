# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: shared macOS/Linux steps (find
#     system Python, create venv, activate it, install requirements.in).
# Prompt summary: Create all-in-one setup scripts that create/activate the
#   project venv and install requirements.in if missing.
# Human review:   Pending review
# ==============================================================================
# Sourced (not executed) by _DevEnvSetup.sh and _SetupAndRunDemo.sh.
# Works in bash and zsh. Defines _marl_env_setup <project_root>.

# Keep in sync with _env_setup.py / _env_common.ps1 / _env_common.bat / _CleanupCloseDevEnv.*
MARL_VENV_NAME=".venv-CSE-5106-Project-Beinder-Stinson-Wang"

_marl_find_python() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

_marl_env_setup() {
    local root="$1"
    local py

    if ! py="$(_marl_find_python)"; then
        echo "[setup] ERROR: Python 3.8+ not found (tried python3, python)."
        echo "[setup] macOS: brew install python  |  Debian/Ubuntu: sudo apt install python3 python3-venv"
        return 1
    fi

    "$py" "$root/_env_setup.py" ensure-venv || return 1

    # shellcheck disable=SC1090
    . "$root/$MARL_VENV_NAME/bin/activate" || {
        echo "[setup] ERROR: failed to activate $MARL_VENV_NAME"
        return 1
    }
    echo "[setup] Activated venv: $VIRTUAL_ENV"

    python "$root/_env_setup.py" ensure-deps || return 1
}
