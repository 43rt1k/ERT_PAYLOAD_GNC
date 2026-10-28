import numpy as np

from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod

from payload_gnc.gsim_helper import *

from payload_gnc.controllers.body import Body





R_MAX = 300





class Controller(ABC):

    def __init__(self, config, VRB_ID, dt, unity, max_brake, goal_pos=None):

        self.VRB_ID = VRB_ID
        self.vrb_init = True
        self.vrb_err = False
        self.unity = unity

        self.dt = dt

        self.kp = config["kp"]
        self.ki = config["ki"]
        self.kd = config["kd"]

        self.ki_max = config["ki_max"]
        self.max_brake = max_brake

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.error: float
        self.brake: list[float]
        self.integral: float
        self.prev_error: float
        self.goal_pos = None

        self.reset(goal_pos)

        self._print_init()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    @abstractmethod
    def update(self, body):
        pass

    @abstractmethod
    def _update_error(self, body):
        pass

    def reset(self, goal_pos=None):

        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        if goal_pos is None: return

        self.goal_pos = np.array(goal_pos, dtype=float, copy=True)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _update_ki(self):
        self.integral = float(
                    np.clip(self.integral + self.error * self.dt, -self.ki_max, self.ki_max))

    def update(self, B: Body):

        self.error = self._update_error(B)

        self._print_error()

        self._update_ki()

        derivative = (self.error - self.prev_error) / self.dt
        self.prev_error = self.error

        brake_command = (self.kp * self.error + self.ki * self.integral + self.kd * derivative)

        command = float(np.clip(brake_command, -self.max_brake, self.max_brake))

        self.brake = [max(-command, 0.0), max(command, 0.0)]

        return self.brake, self.error, brake_command, self.integral

    def _print_init(self):
        if self.vrb_init:
            text = f"kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            print_message(self.VRB_ID, text)

    def _print_error(self):
        if self.vrb_err:
            print(f"Error = {self.error} [{self.unity}]")





class PID_Header(Controller):

    def __init__(self, config, dt, max_brake, goal_pos=None, verbose=True):
        super().__init__(config, "C_H", dt, "deg", max_brake, goal_pos)

        self.error_xy_max = config["error_xy_max"]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __clip_error_yaw(self, error_yaw):
        yaw_max = self.error_xy_max * np.pi

        return float(np.clip(error_yaw, -yaw_max, yaw_max))

    def _update_error(self, B):

        offset = np.asarray(self.goal_pos[: XY]) - np.asarray(B.state[POS][: XY])

        # Horizontal bearing toward the goal [rad].
        desired_yaw = np.arctan2(offset[Y], offset[X])

        # Shortest signed heading error, in [-pi, pi).
        error = (desired_yaw - B.orientation[2] + np.pi) % (2 * np.pi) - np.pi

        error = self.__clip_error_yaw(error)

        return error





# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––





class PID_Rotation(Controller):

    def __init__(self, config, dt, max_brake, goal_pos=None, verbose=True):
        super().__init__(config, "C_C", dt, "m", max_brake, goal_pos)
        self.vrb_init
        self.r = None
        self.r_min = None
        self.r_max = R_MAX

        self.center = None
        self.center_dir = None

        self.error_prev = None
        self.loops = []

    def __get_h_loop(self, speed, R, angle=(2 * np.pi)):

        speed_xy = np.linalg.norm(speed[: XY])

        duration = R * abs(angle) / speed_xy

        return speed[Z] * duration

    def __get_r(self, speed, h, angle=(2 * np.pi)):

        speed_xy = np.linalg.norm(speed[: XY])
        duration = h / speed[Z]
        return duration * speed_xy / abs(angle)

    def __init_loops(self, B: Body):

        h = max(float(B.goal_pos_final[Z] - B.state[POS][Z]), 0.0)
        # print("h=", h)
        speed = B.state[VEL]
        r_basic = (self.r_min + self.r_max) / 2

        while r_basic > self.r_min:

            # print("r_basic=", r_basic)

            h_loop = self.__get_h_loop(speed, r_basic)

            loop_nb = int(h / h_loop)

            h_top = h - loop_nb * h_loop

            if np.isclose(h_top, 0.0, atol=10.0):
                self.loops = [r_basic] * loop_nb
                return

            r_top = self.__get_r(speed, h_top)

            # print("r_top=", r_top, ", loop_nb=", loop_nb)

            # if self.r_min <= r_top <= self.r_max:
            #     self.loops = [r_basic] * loop_nb + [r_top]
            #     return

            r_basic -= 1.0

        raise ValueError("No valid loop plan found with this radius search.")

    def __init_center(self, body: Body):
        velocity_xy = np.asarray(body.state[VEL], dtype=float)[: XY]
        speed_xy = np.linalg.norm(velocity_xy)

        self.center_dir = np.array([-velocity_xy[Y], velocity_xy[X]]) / speed_xy

        self.center = np.array(self.goal_pos, dtype=float, copy=True)
        self.center[: XY] += self.center_dir * self.loops[-1]

    def __print_init_loops(self):
        if self.vrb_init:
            text = (f"Init: loops=[{', '.join(f'{v:.1f}' for v in self.loops)}]")
            print_message(self.VRB_ID, text)

    def init(self, body):
        self.r_min = body.minR
        self.__init_loops(body)
        self.__init_center(body)

        self.__print_init_loops()

    def _update_error(self, body):

        position = np.asarray(body.state[POS], dtype=float)

        distance = np.linalg.norm(position[: 2] - self.center[: 2])

        radius = self.loops[0]

        return float(distance - radius)
