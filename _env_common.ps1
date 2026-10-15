# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: shared PowerShell steps (find
#     system Python, create venv, activate it in the current session, install
#     requirements.in).
# Prompt summary: Add PowerShell versions of the setup scripts so the venv is
#   activated directly in VS Code's (PowerShell) terminal.
# Human review:   Pending review
# ==============================================================================
# Dot-sourced by __DevEnvSetup.ps1 and _SetupAndRunDemo.ps1.
# Works in Windows PowerShell 5.1 and PowerShell 7+.

# Keep in sync with _env_setup.py / _env_common.sh / Win_Bat\_env_common.bat / _CleanupCloseDevEnv.*
$MarlVenvName = '.venv-CSE-5106-Project-Beinder-Stinson-Wang'

# Returns @(exe, args...) for a working system Python 3, or $null.
# Tries the py launcher first, then python (skipping the Microsoft Store stub,
# which exists on PATH but fails when run).
function Find-MarlSystemPython {
    foreach ($candidate in @(@('py', '-3'), @('python'))) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $extra = @($candidate | Select-Object -Skip 1)
        try {
            & $candidate[0] @extra --version *> $null
            if ($LASTEXITCODE -eq 0) { return , $candidate }
        } catch { }
    }
    return $null
}

# Creates (if needed) and activates the venv in the *current* PowerShell session,
# then installs requirements.in if anything is missing. Throws on failure.
function Invoke-MarlEnvSetup {
    param([Parameter(Mandatory)] [string] $Root)

    $setupPy = Join-Path $Root '_env_setup.py'

    $py = Find-MarlSystemPython
    if (-not $py) {
        throw 'Python 3 not found. Install it from https://www.python.org/downloads/'
    }
    $pyArgs = @($py | Select-Object -Skip 1)

    & $py[0] @pyArgs $setupPy ensure-venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the venv.' }

    $activate = Join-Path $Root "$MarlVenvName\Scripts\Activate.ps1"
    . $activate
    if (-not $env:VIRTUAL_ENV) { throw "Failed to activate $MarlVenvName." }
    Write-Host "[setup] Activated venv: $env:VIRTUAL_ENV"

    & python $setupPy ensure-deps
    if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }
}
