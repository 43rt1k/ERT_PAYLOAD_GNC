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

R_MIN = 150
INCREASE_COEFF = 1.25
INIT_CENTER_COEFF = 2.5
OPTIMAL_LOOP_NB = 2





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
        self.vrb_loop = True

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
        self.h_loop = None
        self.center_pos = None
        self.r = None
        self.r_min = R_MIN
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
                                  goal_pos=self.B.goal_pos)

        self.C = self.PID_C

        self.L = log

        self.P = self.Phase(FP.STABILISATION, self.time, self.B.init_pos, goal_pos=self.B.goal_pos)

        self.loop_direction = None
        self.radius_gain = 0.1    # Example correction gain per update.

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

    def __print_loop_no_found(self, r_pot):
        if self.vrb_loop and len(r_pot) == 0:
            print_message(self.VRB_ID, "NO LOOP FOUND, BLYAT'!", self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_phase(self, phase):

        self.P.set(phase, self.time, self.B.pos, self.B.goal_pos)

        if self.vrb_phase:
            print_change(self.VRB_ID, self.P.ex, self.P.now, self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_h_loop_r(self, R, angle=(2 * np.pi)):

        speed_xy = self.B.get_speed(Body.ST.XY)

        duration = R * abs(angle) / speed_xy

        return self.B.vel[Z] * duration

    def __get_r_loop_h(self, h, angle=2 * np.pi):
        speed_xy = self.B.get_speed(Body.ST.XY)

        duration = h / self.B.vel[Z]

        return float(duration * speed_xy / abs(angle))

    def __get_r(self, h, angle=(2 * np.pi)):

        speed_xy = self.B.get_speed(Body.ST.XY)
        duration = h / self.B.vel[Z]
        return duration * speed_xy / abs(angle)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __init_circle(self):
        pos = self.B.pos[: XY]
        goal = self.B.goal_pos[: XY]
        self.center_pos = [0, 0, self.B.pos[Z]]

        to_goal = pos - goal
        to_goal_norm = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / to_goal_norm

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.r = to_goal_norm / INIT_CENTER_COEFF
        self.center_pos[: XY] = goal + to_goal_dir * self.r

        self.__print_center()

    def __init_loops_nb(self):

        direction = self.center_pos[: XY] - self.B.goal_pos[: XY]
        self.center_dir = direction / np.linalg.norm(direction)

        B = self.B
        self.center_pos = self.B.pos.copy()

        h = max(float(B.goal_pos[Z] - B.pos[Z]), 0.0)

        loop_nb = 1
        r_pot = []
        while True:

            h_loop = h / loop_nb
            r = self.__get_r_loop_h(h_loop)

            if r < R_MIN: break
            # print(f"r={r}, h_loop={h_loop}, loop_nb={loop_nb}")

            r_pot.append(r)
            loop_nb += 1

        self.__print_loop_no_found(r_pot)

        r_pot_nb = len(r_pot)

        if r_pot_nb == 0:
            self.r = R_MIN
            self.loops_nb = 1
        elif r_pot_nb <= OPTIMAL_LOOP_NB:
            self.r = r_pot[-1]
            self.loops_nb = r_pot_nb
        else:
            r_pot_id = int(r_pot_nb / 2)
            self.r = r_pot[r_pot_id]
            self.loops_nb = r_pot_id + 1

        self.h_loop = h / self.loops_nb

        print_message(self.VRB_ID,
                      f"Found {len(r_pot)} possible R. Best: R={self.r:.1f}",
                      self.time)

        self.center_pos[: XY] = (np.asarray(self.B.goal_pos[: XY], dtype=float) +
                                 self.center_dir * self.r)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_r(self):
        two_pi = 2 * np.pi

        if self.h_loop <= 0:
            return

        center = np.asarray(self.center_pos[: 2], dtype=float)
        position = np.asarray(self.B.pos[: 2], dtype=float)
        goal = np.asarray(self.B.goal_pos[: 2], dtype=float)
        velocity = np.asarray(self.B.vel[: 2], dtype=float)

        to_position = position - center
        to_goal = goal - center

        if min(np.linalg.norm(to_position), np.linalg.norm(to_goal),
               np.linalg.norm(velocity)) <= 1e-6:
            return

        if self.B.vel[Z] <= 1e-6:
            return

        # 1. Determine the orbit direction once per loop phase.
        if self.loop_direction is None:
            cross = (to_position[0] * velocity[1] - to_position[1] * velocity[0])

            if abs(cross) <= 1e-12:
                return

            self.loop_direction = 1 if cross > 0 else -1

        # Directed angle from the goal radius to the current radius.
        cross = (to_goal[0] * to_position[1] - to_goal[1] * to_position[0])
        dot = np.dot(to_goal, to_position)

        progress_angle = (self.loop_direction * np.arctan2(cross, dot)) % two_pi

        self.angular_progress = progress_angle / two_pi
        remaining_angle = two_pi - progress_angle

        # 2. Height above the final goal, using NED coordinates.
        height = max(
                    float(self.B.goal_pos[Z] - self.B.pos[Z]), 0.0,
        )

        loop_nb = int(height / self.h_loop)
        remaining_height = height - self.h_loop * loop_nb

        # At an exact multiple, target the next lower level.
        if height > 0 and np.isclose(remaining_height, 0.0, atol=1e-6):
            remaining_height = min(self.h_loop, height)

        # 3. Predicted height loss over the remaining arc.
        estimated_loss = self.__get_h_loop_r(
                    self.r, angle=remaining_angle,
        )

        if not np.isfinite(estimated_loss):
            return

        # 4. Positive error means more height must be lost.
        self.height_loss_error = remaining_height - estimated_loss

        # 5. Increase radius to lengthen the path, or decrease it.
        self.r = max(
                    R_MIN, self.r + self.radius_gain * self.height_loss_error,
        )

        # Keep the goal on the circle.
        self.center_pos[: 2] = goal + self.center_dir * self.r

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
        elif self.P == FP.APPROACH and self.B.is_goal_passed(self.center_pos):
            self.__set_phase(FP.LOOP)
            self.__init_loops_nb()

            self.L.update_centers(self.center_pos, self.r)

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
        if self.P == FP.LOOP: self.__update_r()
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.B.brakes, self.error, brake_command, integral = self.C.update(self.B, self.center_pos, self.r)

        # Update Logger ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.L.update(self.B, brake_command, integral, self.error, self.P)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––





# self.PID_H = PID_Header(config["pid_H"],
#                         config["dt"],
#                         config["max_brake"],
#                         goal_pos=self.B.goal_pos_final)
