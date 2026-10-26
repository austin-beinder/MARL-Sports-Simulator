"""
This file was AI generated, see the ### AI GENERATED markers.

The pybullet keep away game as a package, so other code (the marl RL layer) can import it with
`from pybulletGame import KeepAwayEnv`. The folder isn't called "pybullet" because a package
with that name would hide the real pybullet library.

The modules in this folder import each other by bare name (`from ball import Ball`) so they can
still be run as scripts. This file puts the folder on sys.path so those imports keep working
when the folder is imported as a package.

Import the game's classes from here (`from pybulletGame import Entity`), not from submodules
like `pybulletGame.entity`: that would load a second copy of the module, whose classes (and
settings such as Entity.print_collisions) the game itself never sees.
"""

### AI GENERATED
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from entity import Entity  # noqa: E402
from keep_away_env import KeepAwayConfig, KeepAwayEnv  # noqa: E402
from teams import Team  # noqa: E402

__all__ = ["Entity", "KeepAwayConfig", "KeepAwayEnv", "Team"]
### END AI GENERATED
