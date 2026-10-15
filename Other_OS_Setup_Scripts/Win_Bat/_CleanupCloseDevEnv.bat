@echo off
rem ==============================================================================
rem AI CITATION
rem Tool:    Claude Code (Claude Opus 5.5)
rem Date:    2026-09-26
rem Author:  JC Stinson III (LifeHackerIII)
rem Usage:
rem   - 2026-09-26: Generated this entire file: deactivates the project venv.
rem   - 2026-09-26: Moved to Other_OS_Setup_Scripts\Win_Bat.
rem Prompt summary: Create _CleanupCloseDevEnv that (for now) deactivates the
rem   project's python venv.
rem Human review:   Pending review
rem ==============================================================================
rem Usage (from the same cmd.exe session that ran _DevEnvSetup.bat):
rem     Other_OS_Setup_Scripts\Win_Bat\_CleanupCloseDevEnv.bat
rem In PowerShell / VS Code use _CleanupCloseDevEnv.ps1 in the project root instead.
rem TODO: add further cleanup steps here as the project grows.

if not defined VIRTUAL_ENV (
    echo [cleanup] No venv is active. Nothing to deactivate.
    exit /b 0
)

for %%I in ("%VIRTUAL_ENV%") do set "_ENV_ACTIVE_NAME=%%~nxI"
if /i not "%_ENV_ACTIVE_NAME%"==".venv-CSE-5106-Project-Beinder-Stinson-Wang" (
    echo [cleanup] Active venv is "%_ENV_ACTIVE_NAME%", not this project's. Leaving it alone.
    set "_ENV_ACTIVE_NAME="
    exit /b 0
)
set "_ENV_ACTIVE_NAME="

call "%VIRTUAL_ENV%\Scripts\deactivate.bat"
set "_ENV_ROOT="
set "_ENV_VENV_NAME="
echo [cleanup] Deactivated venv .venv-CSE-5106-Project-Beinder-Stinson-Wang.
exit /b 0
