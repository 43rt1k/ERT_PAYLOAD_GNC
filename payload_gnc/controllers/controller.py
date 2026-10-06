import numpy as np
from abc import ABC, abstractmethod

import pfh.glidersim as gsim


def deg2rad(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def q2euler(q):
    return gsim.orientation.quaternion_to_euler(q)


class Controller(ABC):

    @abstractmethod
    def reset(self) -> None:
        """Clear controller memory before a new simulation."""
        pass

    @abstractmethod
    def update(self, state) -> tuple[float, float]:
        """Return normalized (left_brake, right_brake), each in [0, 1]."""
        pass
