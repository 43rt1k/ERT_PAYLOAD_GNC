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
    STABILISATION = auto()
    DEFAULT = auto()
    HOLD = auto()
    APPROACH = auto()
    LANDED = auto()


class GN_Statistics:

    def __init__(self):
        self.VRB_ID = "––STA | "
        self.vrb_update = False
        self.times = []
        self.states = []
        self.brakes = []
        self.errors = []
        self.goal_positions = []

    def reset(self):
        self.times = []
        self.states = []
        self.brakes = []
        self.errors = []
        self.goal_positions = []

    def get(self):
        return (np.asarray(self.times), np.asarray(self.states),
                np.asarray(self.brakes), np.asarray(self.errors),
                np.asarray(self.goal_positions))

    def update(self, time, state, brake, error, goal_position):
        self.times.append(float(time))
        self.states.append(state.copy())
        self.brakes.append(brake.copy())
        self.errors.append(error.copy())

        if not self.goal_positions:
            self.goal_positions.append(goal_position.copy())
            if self.vrb_update: print(self.VRB_ID, "First goal position added")
            return

        if not np.array_equal(goal_position, self.goal_positions[-1]):
            if self.vrb_update: print(self.VRB_ID, "New goal position added")
            self.goal_positions.append(goal_position.copy())


class GN_Controller:

    def __init__(self, config, initial_position):
        self.VRB_ID = "––GNC | "
        self.vrb_phase = True
        self.vrb_goal = True

        self.controller = PID_Controller(config["pid"], config["dt"],
                                         config["max_brake"])

        self.dt = config["dt"]
        self.arrival_margin = config["arrival_margin"]

        self.brakes = [0, 0]
        self.state = None
        self.goal_position_ultimate = config["goal_position"]
        self.goal_position = self.goal_position_ultimate
        self.ex_goal_position = self.goal_position_ultimate

        self.initial_position = initial_position
        self.phase = FP.STABILISATION
        self.ex_phase = self.phase

        self.time = 0
        self.stabilisation_time = config["stabilisation_time"]

        self.statistics = GN_Statistics()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)

        self.mean_forward_velocity = 0
        self.mean_Z_velocity = 0

        self.dist_to_goal_ultimate_init = np.inf
        self.dist_to_goal_ultimate = np.inf
        self.dist_to_goal = np.inf
        self.dist_hold = np.inf
        self.hold_coeff = config["hold_coeff"]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def reset(self):
        self.time = 0

        self.phase = FP.STABILISATION
        self.statistics.reset()
        self.controller.reset(self.goal_position_ultimate)

        self.mean_forward_velocity = 0
        self.mean_Z_velocity = 0
        self.dist_to_goal_ultimate_init = np.linalg.norm(
            self.goal_position_ultimate - self.position)
        self.dist_to_goal_ultimate = np.linalg.norm(
            self.goal_position_ultimate - self.position)
        self.dist_hold = self.dist_to_goal_ultimate_init * self.hold_coeff

    def __set_goal_poosition(self, goal_position, grounded=False):

        self.ex_goal_position = self.goal_position
        self.goal_position = goal_position.copy()

        if grounded: self.goal_position[Z] = 0

        self.controller.reset(self.goal_position)

        if self.vrb_goal:
            print(self.VRB_ID, "Goal changed: ", self.ex_goal_position, " -> ",
                  self.goal_position, self.__get_time_print())

    def __set_phase(self, phase):
        self.ex_phase = self.phase
        self.phase = phase

        if self.vrb_phase:
            print(self.VRB_ID, "Phase changed: ", self.ex_phase, " -> ",
                  self.phase, self.__get_time_print())

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_time_print(self):
        return f" | at {int(self.time)}s"

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    def __is_landed(self):
        return (self.state[POS][Z] >= 0)

    def __is_at_hold_distance(self):
        return self.dist_to_goal <= self.dist_hold

    def __is_hold_done(self):
        return False

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_phase(self):

        # LAST PHASE: LANDING
        if self.__is_landed():
            self.__set_phase(FP.LANDED)

            # PHASE STABILISATION: Wait for the payload to stabilise
        elif self.phase is FP.STABILISATION and self.__is_stabilisation_done():
            self.__set_phase(FP.DEFAULT)

        # PHASE HOLD: Turn around at a nearer point to lose height
        elif self.phase is FP.DEFAULT and self.ex_phase is FP.STABILISATION and self.__is_at_hold_distance(
        ):
            self.__set_phase(FP.HOLD)
            self.__set_goal_poosition(self.position, grounded=True)

        # PHASE APPROACH: Approach the ultimate goal position in a arc-shaped trajectory
        elif self.phase is FP.HOLD and self.ex_phase is FP.DEFAULT and self.__is_hold_done(
        ):
            self.__set_phase(FP.APPROACH)

    def __update_state(self, state, time):
        self.state = state.copy()
        self.time = float(time)

        self.position = self.state[POS]
        self.velocity = self.state[VEL]
        self.orientation = q2euler(self.state[ORI])

        self.dist_to_goal_ultimate = np.linalg.norm(
            self.goal_position_ultimate - self.position)
        self.dist_to_goal = np.linalg.norm(self.goal_position - self.position)

        if self.phase == FP.STABILISATION: return

        recent_states = self.statistics.states[-5:]
        velocities = np.array([state[VEL] for state in recent_states])
        self.mean_forward_velocity = float(
            np.mean(np.linalg.norm(velocities[:, :2], axis=1)))

        self.mean_z_velocity = np.mean(velocities[:, 2])

    def update(self, state, time):
        if self.phase is FP.LANDED: return

        self.__update_state(state, time)
        self.__update_phase()

        # if self.phase == FP.COLLECT and self.time >= self.collect_time: self.__set_goal_poosition(self.goal_position_ultimate)
        # if self.phase ==

        self.brakes, self.error = self.controller.update(
            self.position, self.orientation)

        self.statistics.update(self.time, self.state, self.brakes, self.error,
                               self.goal_position)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
