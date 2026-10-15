@echo off
rem ==============================================================================
rem AI CITATION
rem Tool:    Claude Code (Claude Opus 5.5)
rem Date:    2026-09-26
rem Author:  JC Stinson III (LifeHackerIII)
rem Usage:
rem   - 2026-09-26: Generated this entire file: Windows entry point that sets up
rem     and activates the venv, then runs the out-of-the-box demo.
rem   - 2026-09-26: Moved to Other_OS_Setup_Scripts\Win_Bat; runs
rem     _env_setup.py from the project root.
rem Prompt summary: Create _SetupAndRunDemo that creates/activates the project
rem   venv, installs requirements.in if missing, then runs a demo placeholder.
rem Human review:   Pending review
rem ==============================================================================
rem Usage:  Other_OS_Setup_Scripts\Win_Bat\_SetupAndRunDemo.bat [extra demo args...]
rem The demo script and its default flags are set in _env_setup.py
rem (DEMO_SCRIPT / DEMO_ARGS -- currently a placeholder running example.py).

call "%~dp0_env_common.bat"
if errorlevel 1 goto :failed

python "%_ENV_ROOT%_env_setup.py" run-demo %*
exit /b %errorlevel%

:failed
echo [setup] Environment setup FAILED. See messages above.
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b 1
