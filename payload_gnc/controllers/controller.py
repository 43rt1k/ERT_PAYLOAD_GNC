import numpy as np

from dataclasses import dataclass
from enum import Enum, auto

import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.pid import PID_Rotation
from payload_gnc.logger import Log
from payload_gnc.controllers.body import Body

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

R_BASIC = 200
R_MIN = 50
INCREASE_COEFF = 1.25





class FP(Enum):
    STABILISATION = auto()
    APPROACH = auto()
    LOOP = auto()
    LANDED = auto()





class GN_Controller:

    class Phase:

        def __init__(self, now, time, pos, goal_pos):
            self.now: FP = now
            self.ex: FP = now
            self.time_now: float = time
            self.time_ex: float = time
            self.pos_now: np.ndarray = pos
            self.pos_ex: np.ndarray = pos
            self.goal_pos: np.ndarray = goal_pos

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

    def __init__(self, log: Log, config, initial_position):

        # VERBOSE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.VRB_ID = "GNC"
        self.vrb_phase = True
        self.vrb_landed = True
        self.vrb_init = True

        # CONFIG –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dt = config["dt"]

        self.arrival_margin = config["arrival_margin"]
        self.stabilisation_time = config["stabilisation_time"]
        self.goal_margin = config["goal_margin"]

        # Status ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = 0
        self.is_landed = False
        self.is_goal_reached = False
        self.is_goal_reached_XY = False

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.loops_nb = None
        self.center_pos = None
        self.r = None
        self.r_basic = R_BASIC
        self.init_circle_pos = None
        self.init_circle_side = None

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.B = Body(goal_position=np.asarray(config["goal_position"]),
                      initial_position=np.asarray(initial_position),
                      hold_coeff=config["hold_coeff"],
                      goal_margin=config["goal_margin"])

        self.PID_C = PID_Rotation(config["pid_C"],
                                  config["dt"],
                                  config["max_brake"],
                                  goal_pos=self.B.goal_pos_final)

        self.C = self.PID_C

        self.L = log

        self.P = self.Phase(FP.STABILISATION, self.time, self.B.init_pos, goal_pos=self.B.goal_pos)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __print_landed(self):
        if self.vrb_landed:
            print_message(self.VRB_ID, "! –––––––––– LANDED –––––––––– !", self.time)

    def __print_center(self):
        if self.vrb_init:
            text = f"Init center=[{', '.join(f'{pos:.2f}' for pos in self.center_pos)}], R={self.r:.1f}"
            print_message(self.VRB_ID, text, self.B.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_phase(self, phase):

        self.P.set(phase, self.time, self.B.pos, self.B.goal_pos)

        if self.vrb_phase:
            print_change(self.VRB_ID, self.P.ex, self.P.now, self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_h_loop(self, R, angle=(2 * np.pi)):

        speed_xy = self.B.get_speed(Body.ST.XY)

        duration = R * abs(angle) / speed_xy

        return self.B.vel[Z] * duration

    def __get_r(self, h, angle=(2 * np.pi)):

        speed_xy = self.B.get_speed(Body.ST.XY)
        duration = h / self.B.vel[Z]
        return duration * speed_xy / abs(angle)

    def __get_center_dir(self):
        vel_xy = self.B.vel[: XY]
        speed_xy = np.linalg.norm(vel_xy)

        # Unit vector pointing right of the motion in NED coordinates.
        self.center_dir = np.array([-vel_xy[Y], vel_xy[X]], dtype=float) / speed_xy
        return self.center_dir

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __init_circle(self):
        pos = self.B.pos[: XY]
        goal = self.B.goal_pos_final[: XY]

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.center_pos = (pos + goal) / 2.0
        self.r = float(np.linalg.norm(goal - pos) / 2.0)

        self.__print_center()

    def __init_loops_nb(self):

        B = self.B
        h = max(float(B.goal_pos_final[Z] - B.pos[Z]), 0.0)

        r = float(self.r_basic)
        print(f"r={r}, h={h}")

        counter = 100
        while True:
            counter -= 1
            if counter <= 0: return

            h_loop = self.__get_h_loop(r)

            loop_nb = int(h / h_loop)
            print(f"r={r}, h_loop={h_loop}, loop_nb={loop_nb}")

            h_top = h - loop_nb * h_loop

            if np.isclose(h_top, 0.0, atol=10.0): break

            r -= 1.0

            if r < R_MIN: r = float(self.r_basic * INCREASE_COEFF)

        print("r: ", r)
        self.r = float(r)
        self.center_pos = (np.asarray(self.B.goal_pos_final[: XY], dtype=float) +
                           self.__get_center_dir() * self.r)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_loops(self):
        pass

    def __update_phase(self):

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # LAST PHASE: LANDING
        if self.B.is_landed():
            self.__set_phase(FP.LANDED)
            self.__print_landed()

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE STABILISATION: Wait for the payload to stabilise
        elif self.P == FP.STABILISATION and self.__is_stabilisation_done():
            self.__set_phase(FP.APPROACH)
            self.__init_circle()
            self.C.reset()
            self.L.update_centers(self.center_pos, self.r)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # PHASE DEFAULT: Approach the final goal position
        # elif self.P == FP.APPROACH and self.B.is_goal_passed():
        #     self.__set_phase(FP.LOOP)
        #     self.__init_loops_nb()
        #     self.C.reset()
        #     self.L.update_centers(self.C.center_pos)

    def update(self, state, time):

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        if self.P == FP.LANDED: return
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = float(time)
        self.B.update(state, time)
        self.__update_phase()

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        if self.P == FP.STABILISATION: return
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.B.brakes, self.error, brake_command, integral = self.C.update(self.B, self.center_pos, self.r)

        # Update Logger ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.L.update(self.B, brake_command, integral, self.error, self.P)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––





# self.PID_H = PID_Header(config["pid_H"],
#                         config["dt"],
#                         config["max_brake"],
#                         goal_pos=self.B.goal_pos_final)
