import numpy as np

from dataclasses import dataclass
from enum import Enum, auto

import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.pid import PID_Controller

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def deg2rad(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def q2euler(q):
    return gsim.orientation.quaternion_to_euler(q)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class FP(Enum):
    COLLECT = auto()
    HOLD = auto()
    APPROACH = auto()
    LANDED = auto()


class GN_Statistics:

    def __init__(self):
        self.times = []
        self.states = []
        self.brakes = []
        self.errors = []

    def reset(self):
        self.times = []
        self.states = []
        self.brakes = []
        self.errors = []

    def get(self):
        return (np.asarray(self.times), np.asarray(self.states),
                np.asarray(self.brakes), np.asarray(self.errors))

    def update(self, time, state, brake, error):
        self.times.append(float(time))
        self.states.append(state.copy())
        self.brakes.append(brake.copy())
        self.errors.append(error.copy())


class GN_Controller:

    def __init__(self, config):
        self.controller = PID_Controller(config["pid"], config["dt"],
                                         config["max_brake"])

        self.dt = config["dt"]
        self.arrival_margin = config["arrival_margin"]

        self.brakes = [0, 0]
        self.state = None
        self.goal_position = config["goal_position"]

        self.phase = FP.COLLECT

        self.time = 0

        self.statistics = GN_Statistics()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)

    def reset(self):
        self.phase = FP.COLLECT
        self.statistics.reset()
        self.controller.reset(self.goal_position)

    def __update_phase(self):

        if self.state[ERR_POS][Z] >= 0: self.phase = FP.LANDED

    def __update_state(self, state, time):
        self.state = state.copy()
        self.time = float(time)

        self.position = self.state[ERR_POS]
        self.velocity = self.state[VEL]
        self.orientation = q2euler(self.state[ORI])

    def update(self, state, time):
        self.__update_state(state, time)
        self.__update_phase()

        self.brakes, self.error = self.controller.update(
            self.position, self.orientation)

        self.statistics.update(self.time, self.state, self.brakes, self.error)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
