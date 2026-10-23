import numpy as np

from dataclasses import dataclass
from enum import Enum, auto

import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.pid import PID_Controller, PT
from payload_gnc.logger import Log
from payload_gnc.controllers.body import Body
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class FP(Enum):
    STABILISATION = auto()
    DEFAULT = auto()
    HOLD = auto()
    APPROACH = auto()
    LANDED = auto()


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class GN_Controller:

    def __init__(self, log, config, initial_position, brakes=[0, 0]):

        # VERBOSE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.VRB_ID = "––GNC | "
        self.vrb_phase = True

        # CONFIG –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dt = config["dt"]

        self.arrival_margin = config["arrival_margin"]
        self.stabilisation_time = config["stabilisation_time"]
        self.goal_margin = config["goal_margin"]

        # PHASE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.phase = FP.HOLD
        self.ex_phase = FP.DEFAULT
        # self.phase = FP.STABILISATION
        # self.ex_phase = self.phase

        # Status ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.is_landed = False
        self.is_goal_reached = False
        self.is_goal_reached_XY = False

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = 0
        self.brakes = brakes

        self.B = Body(goal_position=np.asarray(config["goal_position"]),
                      initial_position=np.asarray(initial_position),
                      hold_coeff=config["hold_coeff"])

        self.C_XY = PID_Controller(config["pid_XY"],
                                   config["dt"],
                                   config["max_brake"],
                                   type=PT.XY,
                                   goal_pos=self.B.goal_pos_final)

        self.C_YZ = PID_Controller(config["pid_YZ"],
                                   config["dt"],
                                   config["max_brake"],
                                   type=PT.YZ,
                                   goal_pos=self.B.goal_pos_final)
        self.L = log

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_phase(self, phase):
        self.ex_phase = self.phase
        self.phase = phase

        if self.vrb_phase:
            print(self.VRB_ID, "Phase changed: ", self.ex_phase, " -> ",
                  self.phase, get_time_str(self.time))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_phase(self):

        # LAST PHASE: LANDING
        if self.B.is_landed():
            self.__set_phase(FP.LANDED)

            # PHASE STABILISATION: Wait for the payload to stabilise
        elif self.phase is FP.STABILISATION and self.__is_stabilisation_done():
            self.__set_phase(FP.DEFAULT)

        # PHASE HOLD: Turn around at a nearer point to lose height
        elif self.phase is FP.DEFAULT and self.ex_phase is FP.STABILISATION and self.B.is_at_hold(
        ):
            self.__set_phase(FP.HOLD)
            self.B.set_goal(self.B.state[POS])
            self.C_XY.set_goal(self.B.state[POS])

        # PHASE APPROACH: Approach the final goal position in a arc-shaped trajectory
        elif self.phase is FP.HOLD and self.ex_phase is FP.DEFAULT and self.B.is_hold_done(
        ):
            self.__set_phase(FP.APPROACH)
            self.B.set_goal()
            self.C_XY.set_goal(self.B.goal_pos_final)

    def __update_controller(self):

        brakes_XY, error_XY, brake_command_XY, integral_XY = self.C_XY.update(
            self.B)

        # if self.phase != FP.APPROACH:
        return brakes_XY, error_XY, brake_command_XY, integral_XY

        brakes_YZ, error_YZ, brake_command_YZ, integral_YZ = self.C_YZ.update(
            self.B)

        command = brakes_XY + brakes_YZ
        brakes = np.clip(
            np.asarray(brakes_XY, dtype=float) +
            np.asarray(brakes_YZ, dtype=float),
            0.0,
            self.C_XY.max_brake,
        )
        brake_command = brake_command_YZ
        integral = integral_YZ
        error = error_YZ

        return brakes, error, brake_command, integral

    def update(self, state, time):
        if self.phase is FP.LANDED: return

        self.B.update(state, time, self.L[Log.ID.STATES])
        self.time = float(time)

        self.__update_phase()
        self.brakes, self.error, brake_command, integral = self.__update_controller(
        )

        # Update Logger ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.L.update(self.time, self.B.state, self.brakes, brake_command,
                      integral, self.error, self.B.goal_pos)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
