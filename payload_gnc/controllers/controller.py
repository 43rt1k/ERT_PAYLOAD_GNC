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


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class GN_Controller:

    def __init__(self, config, initial_position):

        # VERBOSE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.VRB_ID = "––GNC | "
        self.vrb_phase = True
        self.vrb_goal = True

        # CONFIG –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dt = config["dt"]
        self.goal_position_final = np.asarray(config["goal_position"])

        self.arrival_margin = config["arrival_margin"]
        self.stabilisation_time = config["stabilisation_time"]
        self.hold_coeff = config["hold_coeff"]
        self.goal_margin = config["goal_margin"]

        # PHASE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.phase = FP.STABILISATION
        self.ex_phase = self.phase

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.controller = PID_Controller(
            config["pid"],
            config["dt"],
            config["max_brake"],
            goal_position=self.goal_position_final)

        self.statistics = GN_Statistics()

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.brakes = [0, 0]
        self.state = None

        self.initial_position = np.asarray(initial_position)
        self.goal_position = self.goal_position_final
        self.ex_goal_position = self.goal_position_final

        self.time = 0

        self.mean_speed = 0
        self.mean_Z_speed = 0

        self.orientation = None

        self.dist_to_goal_final_init = np.linalg.norm(
            self.goal_position_final - self.initial_position)
        self.dist_to_goal_final = np.inf
        self.dist_to_goal = np.inf
        self.dist_hold = self.dist_to_goal_final_init * self.hold_coeff

        self.arrival_Z = 0

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_goal_poosition(self, goal_position, grounded=False):

        self.ex_goal_position = self.goal_position
        self.goal_position = goal_position.copy()

        if grounded: self.goal_position[Z] = 0

        self.controller.set_goal_position(self.goal_position)

        if self.vrb_goal:
            print(self.VRB_ID, "Goal changed: ", self.ex_goal_position, " -> ",
                  self.goal_position, self.__get_time_print())

    def __set_phase(self, phase):
        self.ex_phase = self.phase
        self.phase = phase

        if self.vrb_phase:
            print(self.VRB_ID, "Phase changed: ", self.ex_phase, " -> ",
                  self.phase, self.__get_time_print())

    def __set_mean(self, n=5):

        recent_states = self.statistics.states[-n:]
        velocities = np.array([state[VEL] for state in recent_states])

        self.mean_speed = float(
            np.mean(np.linalg.norm(velocities[:, :2], axis=1)))

        self.mean_Z_speed = np.mean(velocities[:, 2])

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_time_print(self):
        return f" | at {int(self.time)}s"

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_goal_reached(self, only_XY=False):

        if only_XY:
            goal_position = self.goal_position[:2]
            position = self.state[POS][:2]
            velocity = self.state[VEL][:2]
        else:
            goal_position = self.goal_position
            position = self.state[POS]
            velocity = self.state[VEL]

        offset = goal_position - position

        offset_norm = np.linalg.norm(offset)
        speed_norm = np.linalg.norm(velocity)

        cosine = np.dot(velocity, offset) / (speed_norm * offset_norm)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))

        print(self.VRB_ID, "offset_norm: ", offset_norm, " , angle ", angle,
              self.__get_time_print())
        return offset_norm <= self.goal_margin and angle <= 1

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    def __is_landed(self):
        return (self.state[POS][Z] >= 0)

    def __is_at_hold_distance(self):
        return self.dist_to_goal <= self.dist_hold

    def __is_hold_done(self):

        return self.__is_goal_reached(only_XY=True)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def __update_arrival_Z(self):

        horizontal_distance = np.linalg.norm(self.goal_position_final[:2] -
                                             self.state[POS][:2])

        time_to_goal = horizontal_distance / self.mean_speed
        self.arrival_Z = self.state[POS][Z] - self.mean_Z_speed * time_to_goal

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
            self.__set_goal_poosition(self.state[POS])

        # PHASE APPROACH: Approach the final goal position in a arc-shaped trajectory
        elif self.phase is FP.HOLD and self.ex_phase is FP.DEFAULT and self.__is_hold_done(
        ):
            self.__set_phase(FP.APPROACH)
            self.__set_goal_poosition(self.goal_position_final)

    def __update_state(self, state, time):
        self.state = state.copy()
        self.orientation = q2euler(state[ORI]).copy()
        self.time = float(time)

        self.dist_to_goal_final = np.linalg.norm(self.goal_position_final -
                                                 self.state[POS])
        self.dist_to_goal = np.linalg.norm(self.goal_position -
                                           self.state[POS])

        if self.phase == FP.STABILISATION: return

        self.__set_mean()

    def update(self, state, time):
        if self.phase is FP.LANDED: return

        self.__update_state(state, time)
        self.__update_phase()

        self.__update_arrival_Z()
        # print(self.VRB_ID, "Arrival Z: ", self.arrival_Z)
        self.__is_goal_reached()
        self.brakes, self.error = self.controller.update(
            self.state[POS], self.orientation)

        self.statistics.update(self.time, self.state, self.brakes, self.error,
                               self.goal_position)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
