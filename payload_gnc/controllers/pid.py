import numpy as np

from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod

from payload_gnc.gsim_helper import *

from payload_gnc.controllers.body import Body





ERR_POS = 0
ERR_YAW = 1





class Controller(ABC):

    def __init__(self, config, VRB_ID, dt, max_brake, goal_pos=None):

        self.VRB_ID = VRB_ID
        self.vrb_init = True
        self.vrb_err = False

        self.dt = dt

        self.kp = config["kp"]
        self.ki = config["ki"]
        self.kd = config["kd"]

        self.ki_max = config["ki_max"]
        self.max_brake = max_brake

        self.goal_pos = (np.array(goal_pos, dtype=float, copy=True)
                         if goal_pos is not None else None)

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.error = [np.nan, np.nan]
        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = 0

        self._print_init()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    @abstractmethod
    def update(self, body):
        pass

    @abstractmethod
    def _update_error(self, body):
        pass

    def reset(self, goal_pos):
        self.goal_pos = goal_pos

        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = 0
        self.error = [np.nan, np.nan]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def _update_ki(self):
        self.integral = float(
                    np.clip(self.integral + self.error[ERR_YAW] * self.dt,
                            -self.ki_max,
                            self.ki_max))

    def _print_init(self):
        if self.vrb_init:
            text = f"kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            print_message(self.VRB_ID, text)





class PID_Header(Controller):

    def __init__(self, config, dt, max_brake, goal_pos=None, verbose=True):
        super().__init__(config, "C_H", dt, max_brake, goal_pos)

        self.error_xy_max = config["error_xy_max"]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __clip_error_yaw(self, error_yaw):
        yaw_max = self.error_xy_max * np.pi

        return float(np.clip(error_yaw, -yaw_max, yaw_max))

    def _update_error(self, position, orientation):

        offset = np.asarray(self.goal_pos) - np.asarray(position)

        # Straight-line distance to the goal [m].
        error_position = float(np.linalg.norm(offset))

        # Horizontal bearing toward the goal [rad].
        desired_yaw = np.arctan2(offset[Y], offset[X])

        # Shortest signed heading error, in [-pi, pi).
        error_yaw = (desired_yaw - orientation[2] + np.pi) % (2 * np.pi) - np.pi

        error_yaw = self.__clip_error_yaw(error_yaw)

        return [error_position, error_yaw]

    def update(self, B: Body):

        self.error = self._update_error(B.state[POS], B.orientation)
        if self.vrb_err:
            print(f"Error XY: {self.error[ERR_POS]} [m], Error Euler: {self.error[ERR_YAW]} [rad]")

        self._update_ki()

        derivative = (self.error[ERR_YAW] - self.prev_error) / self.dt
        self.prev_error = self.error[ERR_YAW]

        brake_command = (self.kp * self.error[ERR_YAW] + self.ki * self.integral +
                         self.kd * derivative)

        command = float(np.clip(brake_command, -self.max_brake, self.max_brake))

        self.brake = [max(-command, 0.0), max(command, 0.0)]

        return self.brake, self.error, brake_command, self.integral





# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––





class PID_Circle(Controller):

    def __init__(self, config, dt, max_brake, goal_pos=None, verbose=True):
        super().__init__(config, "C_C", dt, max_brake, goal_pos)

        self.center = None
        self.radius = None
        self.radial_gain = 1.0
        self.error_prev = None

    def start_circle(self, body, height_loss=100.0):
        """Call once when switching into circle mode."""

        orbit_time = height_loss / body.speed.Z
        self.radius = body.speed.XY * orbit_time / (2 * np.pi)

        yaw = float(body.orientation[2])

        # Right of the entry heading in the North/East plane.
        right = np.array([-np.sin(yaw), np.cos(yaw)])

        self.center = body.goal_pos_final[: XY] + self.radius * right
        self.error_prev = None

    def _update_error(self, body):

        yaw = float(body.orientation[2])

        radial = body.state[POS][: XY] - self.center
        distance = np.linalg.norm(radial)

        if distance <= 1e-6:
            # At the centre, choose an outward direction.
            outward = np.array([np.cos(yaw), np.sin(yaw)])
        else:
            outward = radial / distance

        # Positive means outside the desired circle.
        radius_error = distance - self.radius

        # Clockwise tangent in North/East coordinates.
        tangent = np.array([-outward[1], outward[0]])

        # Outside: steer inward. Inside: steer outward.
        direction = (tangent - self.radial_gain * (radius_error / self.radius) * outward)

        desired_yaw = np.arctan2(direction[1], direction[0])
        heading_error = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        return [float(radius_error), float(heading_error)]

    def update(self, body):

        self.error = self._update_error(body)
        self._update_ki()

        heading_error = self.error[ERR_YAW]

        if self.error_prev is None:
            derivative = 0.0
        else:
            difference = (heading_error - self.error_prev + np.pi) % (2 * np.pi) - np.pi
            derivative = difference / self.dt

        self.error_prev = heading_error

        brake_command = (self.kp * heading_error + self.ki * self.integral + self.kd * derivative)

        command = float(np.clip(brake_command, -self.max_brake, self.max_brake))

        self.brake = [max(-command, 0.0), max(command, 0.0)]

        return self.brake, self.error, brake_command, self.integral
