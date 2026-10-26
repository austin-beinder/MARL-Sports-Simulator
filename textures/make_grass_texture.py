### AI GENERATED
"""
Generates textures/grass.png, the tiling grass texture used by pybulletGame/field.py.

The image is one tile of mowed grass: a light stripe on the left half and a dark stripe on
the right half, with noise on top. The noise wraps around the edges, so the tile repeats
without visible seams. Only numpy is needed (the PNG is written by hand with zlib).

Run from anywhere:
    python textures/make_grass_texture.py
"""

import os
import struct
import zlib

import numpy as np

SIZE = 256
LIGHT_GREEN = np.array([92, 158, 62], dtype=float)
DARK_GREEN = np.array([72, 132, 48], dtype=float)


def periodic_blur(image, radius):
    """Box blur that wraps around the edges, so the result still tiles seamlessly."""
    out = np.zeros_like(image)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            out += np.roll(np.roll(image, dx, axis=0), dy, axis=1)
    return out / (2 * radius + 1) ** 2


def make_grass(size=SIZE, seed=0):
    rng = np.random.default_rng(seed)

    # Mowing stripes: light left half, dark right half
    image = np.empty((size, size, 3))
    image[:, : size // 2] = LIGHT_GREEN
    image[:, size // 2 :] = DARK_GREEN

    # Large soft patches, then fine per-pixel grain
    patches = periodic_blur(rng.normal(0, 1, (size, size)), radius=6)
    patches /= np.abs(patches).max()
    grain = rng.normal(0, 1, (size, size))

    brightness = 1.0 + 0.08 * patches + 0.05 * grain
    image *= brightness[:, :, None]
    return np.clip(image, 0, 255).astype(np.uint8)


def write_png(path, rgb):
    """Write an RGB uint8 array as a PNG using only the standard library."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)  # filter type 0 on every row

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n")
        out.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        out.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        out.write(chunk(b"IEND", b""))


if __name__ == "__main__":
    write_png(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grass.png"), make_grass())
### END AI GENERATED
