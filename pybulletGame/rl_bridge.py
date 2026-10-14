"""
Small integration boundary between infrastructure and the future RL code.

TEAM NOTES
----------
This file deliberately does not import PPO/IPPO/MAPPO implementations.

Instead, YAML contains a string such as:

    training:
      entrypoint: "train:run"

and this module dynamically loads that function.

This keeps main.py stable even if the RL implementation changes modules or
algorithms later.

IMPORTANT: the proposed function signature below is provisional because the
repository has a separate "Define Gym/RL layer API/ICD" issue. The owners of
that issue should review the interface before it becomes permanent.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from config import AppConfig
from runtime import RunContext


def _load_entrypoint(spec: str) -> Callable[..., Any]:
    """
    Resolve a string of the form "module:function" into a Python callable.

    Example:
        "train:run"

    roughly corresponds to:
        import train
        func = train.run
    """
    if ":" not in spec:
        raise ValueError(
            "training.entrypoint must have the form 'module:function', "
            "for example 'train:run'."
        )

    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)

    try:
        func = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(
            f"Function {function_name!r} was not found in module {module_name!r}."
        ) from exc

    if not callable(func):
        raise TypeError(f"Configured RL entrypoint {spec!r} is not callable.")

    return func


def invoke_rl_loop(env, cfg: AppConfig, context: RunContext) -> Any:
    """
    Invoke the future RL training/evaluation loop.

    Proposed contract:

        def run(env, config, context):
            ...

    Parameters
    ----------
    env:
        The already-created KeepAwayEnv. The RL layer should use reset()/step()
        rather than creating a second environment connection.

    config:
        Parsed application configuration, including training hyperparameters.

    context:
        Standardized run paths such as:
            context.run_dir
            context.checkpoints_dir
            context.metrics_path
    """
    if not cfg.training.entrypoint:
        raise RuntimeError(
            "run.mode is 'rl', but training.entrypoint is null. "
            "Set it after the RL implementation exposes a training function."
        )

    func = _load_entrypoint(cfg.training.entrypoint)

    # The bridge contains the dependency on the provisional RL interface.
    # If the API/ICD issue chooses a different signature, change this file
    # rather than spreading that change through main.py.
    return func(env, cfg, context)
