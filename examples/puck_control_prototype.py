"""
BEGIN AI GENERATED w/ CLAUDE EDU

Prototype: a puck (cylinder) slides on a friction-covered pybullet ground
plane and is pushed in 2D by a small tkinter directional pad (forward /
backward / left / right). Holding a button applies a steady force in that
direction each physics step; releasing it lets ground friction slow the
puck back down. Holding two adjacent buttons (e.g. forward + right) pushes
diagonally, so the pad covers any direction, not just the four cardinals.
"""

import tkinter as tk

import pybullet as p
import pybullet_data

FORCE_MAGNITUDE = 15.0  # N, world-frame push applied while a direction is held
PUCK_RADIUS = 0.3
PUCK_HEIGHT = 0.1
PUCK_MASS = 1.0
GROUND_FRICTION = 0.4
PUCK_FRICTION = 0.4
PHYSICS_HZ = 240
GUI_HZ = 60
SUBSTEPS_PER_FRAME = PHYSICS_HZ // GUI_HZ

held = {"forward": False, "backward": False, "left": False, "right": False}


def setup_world():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0, 0, 0])

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=GROUND_FRICTION)

    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=PUCK_RADIUS, height=PUCK_HEIGHT)
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=PUCK_RADIUS, length=PUCK_HEIGHT, rgbaColor=[0.9, 0.2, 0.2, 1]
    )
    puck_id = p.createMultiBody(
        baseMass=PUCK_MASS,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, PUCK_HEIGHT / 2 + 0.01],
    )
    p.changeDynamics(puck_id, -1, lateralFriction=PUCK_FRICTION, angularDamping=0.4)
    return puck_id


def apply_held_forces(puck_id):
    fx = fy = 0.0
    if held["forward"]:
        fy += FORCE_MAGNITUDE
    if held["backward"]:
        fy -= FORCE_MAGNITUDE
    if held["left"]:
        fx -= FORCE_MAGNITUDE
    if held["right"]:
        fx += FORCE_MAGNITUDE
    if fx == 0.0 and fy == 0.0:
        return
    pos, _ = p.getBasePositionAndOrientation(puck_id)
    p.applyExternalForce(puck_id, -1, forceObj=[fx, fy, 0], posObj=pos, flags=p.WORLD_FRAME)


def build_control_pad(root):
    def bind_hold(button, key):
        button.bind("<ButtonPress-1>", lambda _event: held.__setitem__(key, True))
        button.bind("<ButtonRelease-1>", lambda _event: held.__setitem__(key, False))

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    btn_forward = tk.Button(frame, text="Forward", width=10, height=2)
    btn_left = tk.Button(frame, text="Left", width=10, height=2)
    btn_right = tk.Button(frame, text="Right", width=10, height=2)
    btn_backward = tk.Button(frame, text="Backward", width=10, height=2)

    btn_forward.grid(row=0, column=1)
    btn_left.grid(row=1, column=0)
    btn_right.grid(row=1, column=2)
    btn_backward.grid(row=2, column=1)

    bind_hold(btn_forward, "forward")
    bind_hold(btn_backward, "backward")
    bind_hold(btn_left, "left")
    bind_hold(btn_right, "right")


def main():
    puck_id = setup_world()

    root = tk.Tk()
    root.title("Puck Controller")
    build_control_pad(root)

    def loop():
        try:
            for _ in range(SUBSTEPS_PER_FRAME):
                apply_held_forces(puck_id)
                p.stepSimulation()
                pos, _ = p.getBasePositionAndOrientation(puck_id)
                print(f"puck position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        except p.error:
            # pybullet window was closed by the user; close the pad too.
            root.destroy()
            return
        root.after(int(1000 / GUI_HZ), loop)

    root.after(int(1000 / GUI_HZ), loop)
    root.mainloop()

    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()

# END AI GENERATED