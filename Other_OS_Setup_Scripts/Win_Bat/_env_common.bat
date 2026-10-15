@echo off
rem ==============================================================================
rem AI CITATION
rem Tool:    Claude Code (Claude Opus 5.5)
rem Date:    2026-09-26
rem Author:  JC Stinson III (LifeHackerIII)
rem Usage:
rem   - 2026-09-26: Generated this entire file: shared Windows steps (find system
rem     Python, create venv, activate it, install requirements.in).
rem   - 2026-09-26: Moved to Other_OS_Setup_Scripts\Win_Bat; now resolves the
rem     project root two folders up.
rem Prompt summary: Create all-in-one setup scripts that create/activate the
rem   project venv and install requirements.in if missing.
rem Human review:   Pending review
rem ==============================================================================
rem Shared by _DevEnvSetup.bat and _SetupAndRunDemo.bat via "call".
rem No setlocal on purpose: the venv activation must survive into the caller's
rem cmd session. Exits with errorlevel 1 on any failure.

rem This file lives in Other_OS_Setup_Scripts\Win_Bat; the project root is two levels up.
for %%I in ("%~dp0..\..") do set "_ENV_ROOT=%%~fI\"
rem Keep in sync with _env_setup.py / _env_common.ps1 / _env_common.sh / _CleanupCloseDevEnv.*
set "_ENV_VENV_NAME=.venv-CSE-5106-Project-Beinder-Stinson-Wang"

rem --- Find a working system Python (py launcher first; skip the MS Store stub) ---
set "_ENV_PY="
py -3 --version >nul 2>nul && set "_ENV_PY=py -3"
if not defined _ENV_PY (python --version >nul 2>nul && set "_ENV_PY=python")
if not defined _ENV_PY (
    echo [setup] ERROR: Python 3 not found. Install it from https://www.python.org/downloads/
    exit /b 1
)

rem --- Create the venv if needed ---
%_ENV_PY% "%_ENV_ROOT%_env_setup.py" ensure-venv
if errorlevel 1 exit /b 1

rem --- Activate it ---
call "%_ENV_ROOT%%_ENV_VENV_NAME%\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    echo [setup] ERROR: failed to activate %_ENV_VENV_NAME%
    exit /b 1
)
echo [setup] Activated venv: %VIRTUAL_ENV%

rem --- Install dependencies if needed (now running the venv's python) ---
python "%_ENV_ROOT%_env_setup.py" ensure-deps
if errorlevel 1 exit /b 1

set "_ENV_PY="
exit /b 0
