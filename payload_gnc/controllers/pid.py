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
        self.brake_command: list[float]
        self.integral: float
        self.prev_error: float
        self.goal_pos = None

        self.reset(goal_pos)

        self._print_init()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    @abstractmethod
    def _update_error(self, body):
        pass

    @abstractmethod
    def update(self):
        pass

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def reset(self, goal_pos=None):

        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        if goal_pos is None: return

        self.goal_pos = np.array(goal_pos, dtype=float, copy=True)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _set_brake(self, command=None):
        if command is None: command = self.brake_command
        brake_command = float(np.clip(command, -self.max_brake, self.max_brake))
        self.brake = [max(-brake_command, 0.0), max(brake_command, 0.0)]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _update_ki(self):
        self.integral = float(
                    np.clip(self.integral + self.error * self.dt, -self.ki_max, self.ki_max))

    def _update_basic(self, B: Body):

        self._update_ki()

        derivative = (self.error - self.prev_error) / self.dt
        self.prev_error = self.error

        self.brake_command = (self.kp * self.error + self.ki * self.integral + self.kd * derivative)
        self._set_brake()
        return self.brake, self.error, self.brake_command, self.integral

    def _print_init(self):
        if self.vrb_init:
            text = f"kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            print_message(self.VRB_ID, text)

    def _print_error(self):
        if self.vrb_err:
            print(f"Error = {self.error} [{self.unity}]")





# class PID_Header(Controller):

#     def __init__(self, config, dt, max_brake, goal_pos=None, verbose=True):
#         super().__init__(config, "C_H", dt, "deg", max_brake, goal_pos)

#         self.error_xy_max = config["error_xy_max"]

#     # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

#     def __clip_error_yaw(self, error_yaw):
#         yaw_max = self.error_xy_max * np.pi

#         return float(np.clip(error_yaw, -yaw_max, yaw_max))

#     def _update_error(self, B):

#         offset = np.asarray(self.goal_pos[: XY]) - np.asarray(B.state[POS][: XY])

#         # Horizontal bearing toward the goal [rad].
#         desired_yaw = np.arctan2(offset[Y], offset[X])

#         # Shortest signed heading error, in [-pi, pi).
#         error = (desired_yaw - B.orientation[2] + np.pi) % (2 * np.pi) - np.pi

#         error = self.__clip_error_yaw(error)

#         return error

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––





class PID_Rotation(Controller):

    def __init__(self, config, dt, max_brake, goal_pos=None):
        super().__init__(config, "C_C", dt, "m", max_brake, goal_pos)
        self.min_ang = np.pi / 5

        self.center_pos = None
        self.to_center = None
        self.r = None

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _update_error(self, B: Body, center_pos, r):

        self.to_center = np.asarray(center_pos, dtype=float)[: XY] - B.pos[: XY]
        distance = np.linalg.norm(self.to_center)

        self.error = float(distance - r)


    def __update_brake_side(self, B: Body):

        brake = self.brake.copy()

        # Signed 2D cross product, using North–East coordinates.
        side = (B.vel[X] * self.to_center[Y] - B.vel[Y] * self.to_center[X])

        # Centre on the left: reverse steering.
        if side <= 0:
            self.brake = [brake[Y], brake[X]]
            self.brake_command = -self.brake_command

    def __update_extreme(self, B: Body):
        vel = np.asarray(B.vel[: XY], dtype=float)
        speed = np.linalg.norm(vel)
        distance = np.linalg.norm(self.to_center)

        cosine = np.clip(np.dot(vel, self.to_center) / (speed * distance), -1.0, 1.0)
        angle = abs(np.arccos(np.clip(cosine, -1.0, 1.0)))

        if angle > np.pi - self.min_ang:
            self.brake_command = self.max_brake
            self._set_brake()

    def update(self, B, center_pos, r):
        self._update_error(B, center_pos, r)
        self._print_error()

        self._update_basic(B)

        self.__update_extreme(B)

        self.__update_brake_side(B)

        return self.brake, self.error, self.brake_command, self.integral

    # def __print_init_loops(self):
    #     if self.vrb_init:
    #         text = (f"Init: loops=[{', '.join(f'{v:.1f}' for v in self.loops)}]")
    #         print_message(self.VRB_ID, text)

    # def init(self, body):
    #     self.r_min = body.minR
    #     self.__init_loops(body)
    #     self.__init_center(body)

    #     self.__print_init_loops()

    # def _update_error(self, body):

    #     position = np.asarray(body.state[POS], dtype=float)

    #     distance = np.linalg.norm(position[: 2] - self.center[: 2])

    #     radius = self.loops[0]

    #     return float(distance - radius)
