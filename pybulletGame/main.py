"""
Main application entry point for the infrastructure issue.

TEAM NOTES
----------
Keep this file boring.

main.py should coordinate startup/shutdown and should NOT contain:
- Player/Ball/Field physics
- reward logic
- observation/action design
- PPO/IPPO/MAPPO
- PyBullet stepping

Those responsibilities already belong to other layers/issues.

The intended startup flow is:

    CLI args
      -> YAML config
      -> CLI overrides
      -> random seed
      -> run/output directory
      -> runner.run(...)
      -> demo or RL integration
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from runner import run
from runtime import create_run_context, set_global_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line options.

    Command-line values override YAML values when both are supplied. This is
    convenient for quick debugging without creating many nearly-identical YAML
    files.
    """
    parser = argparse.ArgumentParser(
        description="MARL Sports Simulator infrastructure entry point."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file.",
    )

    # A mutually-exclusive group prevents "--gui --headless" from being used
    # together.
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument(
        "--gui",
        action="store_true",
        help="Force PyBullet GUI mode regardless of YAML.",
    )
    gui_group.add_argument(
        "--headless",
        action="store_true",
        help="Force PyBullet DIRECT/headless mode regardless of YAML.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override project.seed.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Override model.resume_from with a checkpoint path.",
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "rl"],
        default=None,
        help="Override run.mode.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Perform process-level setup, then delegate execution to runner.py.

    Returning an integer exit code makes this function easier to test and lets
    the bottom-level SystemExit communicate success/interruption to the shell.
    """
    args = parse_args(argv)

    # 1) Load the reproducible experiment configuration.
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # 2) Apply command-line overrides.
    #
    # Priority is:
    #     dataclass defaults < YAML < CLI
    if args.gui:
        cfg.simulation.gui = True
    if args.headless:
        cfg.simulation.gui = False
    if args.seed is not None:
        cfg.project.seed = args.seed
    if args.resume is not None:
        cfg.model.resume_from = args.resume
    if args.mode is not None:
        cfg.run.mode = args.mode

    # 3) Seed the libraries we know about before creating the environment or
    #    trainer. This improves reproducibility between runs.
    set_global_seed(cfg.project.seed)

    # 4) Create one standardized output directory for this run and copy the
    #    exact YAML used into it.
    context = create_run_context(cfg, config_path)

    # 5) Delegate the actual run. runner.py owns environment creation and
    #    chooses demo vs. RL mode.
    try:
        run(cfg, context)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    # Keep all meaningful startup logic inside main() instead of placing a
    # large block directly under this condition.
    raise SystemExit(main())
