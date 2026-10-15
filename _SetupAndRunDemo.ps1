# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: PowerShell entry point that sets
#     up and activates the venv, then runs the out-of-the-box demo.
# Prompt summary: Add PowerShell versions of the setup scripts so the venv is
#   activated directly in VS Code's (PowerShell) terminal.
# Human review:   Pending review
# ==============================================================================
# Usage:  .\_SetupAndRunDemo.ps1 [extra demo args...]
# The demo script and its default flags are set in _env_setup.py
# (DEMO_SCRIPT / DEMO_ARGS -- currently a placeholder running example.py).
# The venv stays active in this terminal afterwards.
#
# If PowerShell says "running scripts is disabled on this system", run this
# once (per user), then try again:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# No param() block on purpose: every argument ($args) is passed to the demo.

. (Join-Path $PSScriptRoot '_env_common.ps1')

try {
    Invoke-MarlEnvSetup -Root $PSScriptRoot
} catch {
    Write-Host "[setup] ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '[setup] Environment setup FAILED. See messages above.' -ForegroundColor Red
    $global:LASTEXITCODE = 1
    return  # not "exit": that would close the terminal if this file was dot-sourced
}

& python (Join-Path $PSScriptRoot '_env_setup.py') run-demo @args
