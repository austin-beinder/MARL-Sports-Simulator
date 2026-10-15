# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: deactivates the project venv in
#     the current PowerShell session.
# Prompt summary: Add a PowerShell version of _CleanupCloseDevEnv that (for
#   now) deactivates the project's python venv.
# Human review:   Pending review
# ==============================================================================
# Usage:  .\_CleanupCloseDevEnv.ps1
# TODO: add further cleanup steps here as the project grows.

# Keep in sync with _env_common.ps1 / _env_setup.py
$venvName = '.venv-CSE-5106-Project-Beinder-Stinson-Wang'

if (-not $env:VIRTUAL_ENV) {
    Write-Host '[cleanup] No venv is active. Nothing to deactivate.'
    return
}

$activeName = Split-Path $env:VIRTUAL_ENV -Leaf
if ($activeName -ne $venvName) {
    Write-Host "[cleanup] Active venv is '$activeName', not this project's. Leaving it alone."
    return
}

if (-not (Get-Command deactivate -CommandType Function -ErrorAction SilentlyContinue)) {
    # VS Code's Python extension can activate a venv by injecting environment
    # variables instead of running Activate.ps1, in which case there is no
    # deactivate function to call.
    Write-Host "[cleanup] $venvName was activated by VS Code, not by a script, so there's"
    Write-Host '[cleanup] nothing to deactivate here. Close this terminal or open a new one instead.'
    return
}

deactivate
Write-Host "[cleanup] Deactivated venv $venvName."
