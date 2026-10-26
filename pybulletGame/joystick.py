"""
This file was AI generated, see the ### AI GENERATED markers.

An on-screen joystick (a tkinter canvas) that outputs a velocity command, the same action the
RL layer sends to a player. Drag the knob with the mouse; let go and it springs back to the centre.
Up on the screen is +y on the field and right is +x.
"""

### AI GENERATED
import math
import tkinter as tk

import numpy as np


def knob_to_velocity(dx: float, dy: float, radius: float, max_speed: float) -> np.ndarray:
    """
    Convert the knob's offset from the centre, in screen pixels, into a field velocity.

    Screen y grows downwards, so it is flipped to make up on the screen +y on the field.
    Pushing the knob to the edge of the base (or past it) gives max_speed.
    """
    distance = math.hypot(dx, dy)
    if distance > radius:
        dx, dy = dx * radius / distance, dy * radius / distance
    return np.array([dx / radius * max_speed, -dy / radius * max_speed])


class Joystick(tk.Canvas):

    def __init__(self, master, max_speed: float, size: int = 220, knob_radius: int = 18):
        super().__init__(master, width=size, height=size, highlightthickness=0)
        self.max_speed = max_speed
        self.centre = size / 2
        self.knob_radius = knob_radius
        self.radius = size / 2 - knob_radius - 4  # how far the knob can travel from the centre

        c, r = self.centre, self.radius
        self.create_oval(c - r - knob_radius, c - r - knob_radius, c + r + knob_radius, c + r + knob_radius,
                         fill="#e8e8e8", outline="#999999", width=2)
        self.create_line(c - r, c, c + r, c, fill="#bbbbbb")
        self.create_line(c, c - r, c, c + r, fill="#bbbbbb")
        self.knob = self.create_oval(0, 0, 0, 0, fill="#2060d0", outline="#103070", width=2)
        self.velocity = np.zeros(2)
        self._move_knob(0, 0)

        self.bind("<ButtonPress-1>", self._on_drag)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_drag(self, event):
        self._move_knob(event.x - self.centre, event.y - self.centre)

    def _on_release(self, event=None):
        self._move_knob(0, 0)  # spring back to the centre: stop

    def _move_knob(self, dx, dy):
        self.velocity = knob_to_velocity(dx, dy, self.radius, self.max_speed)
        # Draw the knob where the velocity says it is (clamped to the base)
        x = self.centre + self.velocity[0] / self.max_speed * self.radius
        y = self.centre - self.velocity[1] / self.max_speed * self.radius
        k = self.knob_radius
        self.coords(self.knob, x - k, y - k, x + k, y + k)
### END AI GENERATED
