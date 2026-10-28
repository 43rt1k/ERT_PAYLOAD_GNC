import numpy as np

from dataclasses import dataclass
from enum import Enum, auto

import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.pid import PID_Header, PID_Rotation
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





class GN_Controller:

    @dataclass
    class Phase:
        now: FP
        ex: FP
        time_now: float
        time_ex: float
        pos_now: np.ndarray
        pos_ex: np.ndarray
        goal_pos: np.ndarray

        def __eq__(self, other):
            if isinstance(other, GN_Controller.Phase):
                return self.now == other.now

            if isinstance(other, FP):
                return self.now == other

        def __str__(self):
            return str(self.now)

        def set(self, phase, time, pos, goal_pos):

            self.ex = self.now
            self.time_ex = self.time_now
            self.pos_ex = self.pos_now

            self.now = phase
            self.time_now = float(time)

            self.pos_now = np.array(pos, dtype=float, copy=True)
            self.goal_pos = np.array(goal_pos, dtype=float, copy=True)

    def __init__(self, log, config, initial_position, brakes=[0, 0]):

        # VERBOSE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.VRB_ID = "GNC"
        self.vrb_phase = True
        self.vrb_landed = True

        # CONFIG –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dt = config["dt"]

        self.arrival_margin = config["arrival_margin"]
        self.stabilisation_time = config["stabilisation_time"]
        self.goal_margin = config["goal_margin"]

        # Status ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.is_landed = False
        self.is_goal_reached = False
        self.is_goal_reached_XY = False

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = 0

        self.brakes = brakes

        self.B = Body(goal_position=np.asarray(config["goal_position"]),
                      initial_position=np.asarray(initial_position),
                      hold_coeff=config["hold_coeff"],
                      goal_margin=config["goal_margin"])

        self.PID_H = PID_Header(config["pid_H"],
                                config["dt"],
                                config["max_brake"],
                                goal_pos=self.B.goal_pos_final)

        self.PID_C = PID_Rotation(config["pid_C"],
                                  config["dt"],
                                  config["max_brake"],
                                  goal_pos=self.B.goal_pos_final)

        self.C = self.PID_H

        self.L = log

        # PHASE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        now = FP.STABILISATION
        time_now = self.time
        self.P = self.Phase(now=now,
                            ex=now,
                            time_now=time_now,
                            time_ex=time_now,
                            pos_ex=np.array(initial_position),
                            pos_now=np.array(initial_position),
                            goal_pos=self.B.goal_pos)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_phase(self, phase):

        self.P.set(phase, self.time, self.B.state[POS], self.B.goal_pos)

        if self.vrb_phase:
            print_change(self.VRB_ID, self.P.ex, self.P.now, self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_phase(self):

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # LAST PHASE: LANDING
        if self.B.is_landed():
            self.__set_phase(FP.LANDED)
            if self.vrb_landed:
                print_message(self.VRB_ID, "! –––––––––– LANDED –––––––––– !", self.time)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE STABILISATION: Wait for the payload to stabilise
        elif self.P == FP.STABILISATION and self.__is_stabilisation_done():
            self.__set_phase(FP.DEFAULT)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE HOLD: Turn around at a nearer point to get the min turning radius
        elif self.P == FP.DEFAULT and self.P.ex == FP.STABILISATION and self.B.is_at_hold():
            self.__set_phase(FP.HOLD)
            self.B.set_goal(self.B.state[POS])
            self.C.reset(self.B.state[POS])

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE DEFAULT: Approach the final goal position
        elif self.P == FP.HOLD and self.P.ex == FP.DEFAULT and self.B.is_at_goal():
            self.__set_phase(FP.DEFAULT)
            self.B.set_goal()
            self.C.reset(self.B.goal_pos_final)
            self.B.get_min_rad(self.P, self.L)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE APPROACH: Approach the final goal position in circles to lose altitude
        elif self.P == FP.DEFAULT and self.P.ex == FP.HOLD and self.B.is_at_goal():
            self.__set_phase(FP.APPROACH)
            self.C = self.PID_C
            self.C.init(self.B)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE DEFAULT: Return to standard controlling for the best goal matcing (optional)

    def update(self, state, time):
        if self.P == FP.LANDED: return

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = float(time)
        self.B.update(state, time, self.L[Log.ID.STATES])

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.__update_phase()

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.brakes, self.error, brake_command, integral = self.C.update(self.B)

        # Update Logger ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.L.update(self.time,
                      self.B.state,
                      self.brakes,
                      brake_command,
                      integral,
                      self.error,
                      self.P)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
