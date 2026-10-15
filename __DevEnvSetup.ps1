# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: PowerShell entry point that sets
#     up the venv, activates it in the current terminal, then notifies the user.
# Prompt summary: Add PowerShell versions of the setup scripts so the venv is
#   activated directly in VS Code's (PowerShell) terminal.
# Human review:   Pending review
# ==============================================================================
# Usage (VS Code terminal / PowerShell; the venv stays active afterwards):
#     .\__DevEnvSetup.ps1 [-NoPopup]
#
# If PowerShell says "running scripts is disabled on this system", run this
# once (per user), then try again:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param(
    # Console message only, no popup window.
    [switch] $NoPopup
)

. (Join-Path $PSScriptRoot '_env_common.ps1')

try {
    Invoke-MarlEnvSetup -Root $PSScriptRoot
} catch {
    Write-Host "[setup] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '[setup] Environment setup FAILED. See messages above.' -ForegroundColor Red
    $global:LASTEXITCODE = 1
    return  # not "exit": that would close the terminal if this file was dot-sourced
}

$notifyArgs = @('notify')
if ($NoPopup) { $notifyArgs += '--no-popup' }
& python (Join-Path $PSScriptRoot '_env_setup.py') @notifyArgs
Write-Host "[setup] Run .\_CleanupCloseDevEnv.ps1 (or 'deactivate') when you're done."
