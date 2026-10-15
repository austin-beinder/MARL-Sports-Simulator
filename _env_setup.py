# ==============================================================================
# AI CITATION
# Tool:    Claude Code (Claude Opus 5.5)
# Date:    2026-09-26
# Author:  JC Stinson III (LifeHackerIII)
# Usage:
#   - 2026-09-26: Generated this entire file: venv creation, requirements.in
#     dependency check/install, "environment ready" notification (console +
#     Tk popup), and the out-of-the-box demo launcher placeholder.
#   - 2026-09-26: Updated wrapper names/locations in the docstring and comments
#     for the new PowerShell scripts and Other_OS_Setup_Scripts folder.
# Prompt summary: Create all-in-one setup scripts that create/activate the
#   project venv, install requirements.in if missing, then notify the user or
#   run a demo; do as much as possible in native Python.
# Human review:   Pending review
# ==============================================================================
"""Shared Python logic for the project's environment scripts.

The OS wrappers only do what Python cannot do for them -- locate a system
Python and activate the venv in the calling shell. Everything else lives here.

Wrappers:
    Windows (PowerShell / VS Code):  __DevEnvSetup.ps1, _SetupAndRunDemo.ps1
    Windows (cmd.exe):               Other_OS_Setup_Scripts/Win_Bat/*.bat
    macOS / Linux:                   Other_OS_Setup_Scripts/Mac_Linux_Zsh/*.sh

Subcommands:
    ensure-venv   Create the project venv if it does not exist yet.
    ensure-deps   Install requirements.in into the venv if anything is missing.
    notify        Tell the user the environment is ready (console + popup).
    run-demo      Run the out-of-the-box demo inside the venv.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Keep in sync with _env_common.ps1 / _CleanupCloseDevEnv.ps1 and the copies in
# Other_OS_Setup_Scripts/ (_env_common.bat/.sh, _CleanupCloseDevEnv.bat/.sh)
VENV_NAME = ".venv-CSE-5106-Project-Beinder-Stinson-Wang"
VENV_DIR = PROJECT_ROOT / VENV_NAME
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.in"

# ------------------------------------------------------------------------------
# DEMO PLACEHOLDER
# TODO: Point DEMO_SCRIPT at our real main file once it exists, and fill in
#       DEMO_ARGS with the flags for the out-of-the-box demo, e.g.
#       ["--sport", "soccer", "--agents", "4", "--render", "gui"].
#       Extra args given to _SetupAndRunDemo are appended after DEMO_ARGS.
# ------------------------------------------------------------------------------
DEMO_SCRIPT = PROJECT_ROOT / "example.py"
DEMO_ARGS: list[str] = []


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_in_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return False


def reexec_in_venv(argv: list[str]) -> int:
    """Re-run this script with the venv's interpreter (when called from outside it)."""
    py = venv_python()
    if not py.exists():
        log(f"ERROR: venv not found at {VENV_DIR}. Run ensure-venv first.")
        return 1
    return subprocess.call([str(py), str(Path(__file__).resolve()), *argv])


def ensure_venv() -> int:
    if venv_python().exists():
        log(f"Found venv: {VENV_DIR}")
        return 0
    if VENV_DIR.exists():
        log(f"ERROR: {VENV_DIR} exists but has no Python interpreter in it.")
        log("It may be broken or made on another OS. Delete that folder and re-run.")
        return 1

    log(f"Creating venv: {VENV_DIR}")
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as exc:  # e.g. ensurepip missing on Debian/Ubuntu
        log(f"ERROR: failed to create venv: {exc}")
        log("On Debian/Ubuntu you may need: sudo apt install python3-venv")
        return 1
    log("venv created.")
    return 0


def _load_requirement_class():
    try:
        from packaging.requirements import InvalidRequirement, Requirement  # pyright: ignore[reportMissingImports]
    except ImportError:  # pip always vendors packaging, and pip is in every venv
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    return Requirement, InvalidRequirement


def missing_requirements() -> list[str] | None:
    """Return unmet requirement lines, or None if requirements.in can't be checked line by line."""
    from importlib import metadata

    Requirement, InvalidRequirement = _load_requirement_class()
    missing = []
    for raw in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):  # -r, -e, --index-url, ... -> let pip sort it out
            return None
        try:
            req = Requirement(line)
        except InvalidRequirement:
            return None
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            missing.append(line)
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            missing.append(f"{line} (installed: {installed})")
    return missing


def ensure_deps() -> int:
    if not running_in_venv():
        return reexec_in_venv(["ensure-deps"])
    if not REQUIREMENTS_FILE.exists():
        log(f"ERROR: {REQUIREMENTS_FILE.name} not found in {PROJECT_ROOT}")
        return 1

    missing = missing_requirements()
    if missing == []:
        log(f"All dependencies from {REQUIREMENTS_FILE.name} are installed.")
        return 0
    if missing is None:
        log(
            f"{REQUIREMENTS_FILE.name} has options pip must resolve; running pip install."
        )
    else:
        log("Missing or mismatched dependencies:")
        for item in missing:
            log(f"  - {item}")

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    log("Running: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        log(f"ERROR: pip install failed (exit code {rc}).")
        log("If a package had to be built from source (pybullet ships no Windows/macOS")
        log("wheels), install a C++ compiler and re-run:")
        log("  Windows: Microsoft C++ Build Tools (Desktop development with C++)")
        log("           https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        log("  macOS:   xcode-select --install")
        log("  Linux:   sudo apt install build-essential python3-dev")
    return rc


def notify(popup: bool) -> int:
    msg = (
        "Development environment is ready.\n\n"
        f"venv:  {VENV_DIR}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Dependencies from {REQUIREMENTS_FILE.name} are installed."
    )
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70 + "\n", flush=True)

    if popup:
        try:
            import tkinter
            from tkinter import messagebox

            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showinfo("CSE 5106 Dev Environment", msg, parent=root)
            root.destroy()
        except Exception:  # no tkinter / no display (SSH, headless) -> console only
            pass
    return 0


def run_demo(extra_args: list[str]) -> int:
    if not running_in_venv():
        return reexec_in_venv(["run-demo", *extra_args])
    if not DEMO_SCRIPT.exists():
        log(f"ERROR: demo script not found: {DEMO_SCRIPT}")
        return 1
    cmd = [sys.executable, str(DEMO_SCRIPT), *DEMO_ARGS, *extra_args]
    log("Running demo: " + " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ensure-venv")
    sub.add_parser("ensure-deps")
    p_notify = sub.add_parser("notify")
    p_notify.add_argument(
        "--no-popup", action="store_true", help="console message only"
    )
    p_demo = sub.add_parser("run-demo")
    p_demo.add_argument(
        "extra", nargs=argparse.REMAINDER, help="extra args passed to the demo"
    )
    args = parser.parse_args(argv)

    if args.command == "ensure-venv":
        return ensure_venv()
    if args.command == "ensure-deps":
        return ensure_deps()
    if args.command == "notify":
        return notify(popup=not args.no_popup)
    return run_demo(args.extra)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
