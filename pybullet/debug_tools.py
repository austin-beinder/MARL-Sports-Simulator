"""
Small optional visual-debug helpers.

KeepAwayEnv already owns the PyBullet connection and sets a default camera in
human mode. These functions only customize that existing GUI after reset().
"""

from __future__ import annotations

from config import AppConfig


def apply_debug_view(cfg: AppConfig, env) -> None:
    """
    Apply optional camera settings and labels to the existing environment.

    Do not call p.connect() here. KeepAwayEnv already created the connection.
    """
    if not cfg.simulation.gui:
        return

    import pybullet as p

    if not p.isConnected():
        return

    camera = cfg.debug.camera

    # Match the environment's existing logic when YAML does not explicitly set
    # a camera distance.
    if camera.distance is None:
        env_cfg = env.config
        distance = max(
            env_cfg.field_length,
            env_cfg.field_width * 1.6,
        )
    else:
        distance = camera.distance

    # Hide PyBullet's default side panel to keep the demo view cleaner.
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=camera.yaw,
        cameraPitch=camera.pitch,
        cameraTargetPosition=camera.target,
    )

    if cfg.debug.draw_labels:
        # Parent each label to link 2, which is the visible/moving player link
        # in the current Entity implementation.
        p.addUserDebugText(
            "BLUE",
            [0, 0, 0.5],
            textColorRGB=[0.2, 0.4, 1.0],
            parentObjectUniqueId=env.blue.body_id,
            parentLinkIndex=2,
        )

        p.addUserDebugText(
            "RED",
            [0, 0, 0.5],
            textColorRGB=[1.0, 0.2, 0.2],
            parentObjectUniqueId=env.red.body_id,
            parentLinkIndex=2,
        )
