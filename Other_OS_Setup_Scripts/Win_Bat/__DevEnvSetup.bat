@echo off
rem ==============================================================================
rem AI CITATION
rem Tool:    Claude Code (Claude Opus 5.5)
rem Date:    2026-09-26
rem Author:  JC Stinson III (LifeHackerIII)
rem Usage:
rem   - 2026-09-26: Generated this entire file: Windows entry point that sets up
rem     and activates the venv, then notifies the user it is ready.
rem   - 2026-09-26: Moved to Other_OS_Setup_Scripts\Win_Bat; runs
rem     _env_setup.py from the project root.
rem Prompt summary: Create _DevEnvSetup that creates/activates the project venv,
rem   installs requirements.in if missing, and notifies the user when ready.
rem Human review:   Pending review
rem ==============================================================================
rem Usage (from cmd.exe, so the venv stays active afterwards):
rem     Other_OS_Setup_Scripts\Win_Bat\__DevEnvSetup.bat [--no-popup]
rem In PowerShell / VS Code use __DevEnvSetup.ps1 in the project root instead.

call "%~dp0_env_common.bat"
if errorlevel 1 goto :failed

python "%_ENV_ROOT%_env_setup.py" notify %*
exit /b 0

:failed
echo [setup] Environment setup FAILED. See messages above.
rem Keep the window open if launched by double-click so the error can be read.
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b 1
