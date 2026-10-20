import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.body import Body

ERR_POS = 0
ERR_YAW = 1


#Pid type
class PT(Enum):
    XY = auto()
    YZ = auto()


class PID_Controller():

    def __init__(self,
                 config,
                 dt,
                 max_brake,
                 type,
                 goal_pos=None,
                 verbose=True):

        self.VRB_ID = "––PID | "
        self.vrb_err_XY = False
        self.vrb_err_YZ = False

        self.kp = config["kp"]
        self.ki = config["ki"]
        self.kd = config["kd"]
        self.ki_max = config["ki_max"]
        self.max_brake = max_brake

        self.type = type

        self.dt = dt

        self.goal_pos = goal_pos

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.error = [np.nan, np.nan]
        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = [0, 0]

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        if verbose:
            print(
                self.VRB_ID,
                f"Initialized PID controller with kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def set_goal(self, goal_pos):
        self.brake = [0, 0]
        self.integral = 0
        self.prev_error = [0, 0]
        self.error = [np.nan, np.nan]
        self.goal_pos = goal_pos

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_error_XY(self, position, orientation):

        offset = np.asarray(self.goal_pos) - np.asarray(position)

        # Straight-line distance to the goal [m].
        error_position = float(np.linalg.norm(offset))

        # Horizontal bearing toward the goal [rad].
        desired_yaw = np.arctan2(offset[1], offset[0])

        # Shortest signed heading error, in [-pi, pi).
        error_yaw = (desired_yaw - orientation[2] + np.pi) % (2 *
                                                              np.pi) - np.pi

        return [error_position, error_yaw]

    def __update_error_YZ(self, position, arrival_Z):

        arrival = self.goal_pos.copy()
        arrival[Z] = arrival_Z

        to_arrival = arrival - position
        to_goal = self.goal_pos - position

        denominator = np.linalg.norm(to_arrival) * np.linalg.norm(to_goal)

        if denominator < 1e-12:
            error = [0, 0]
        else:
            cosine = np.dot(to_arrival, to_goal) / denominator
            error = np.arccos(np.clip(cosine, -1.0, 1.0))

        return [arrival_Z, error]

    def __update_ki(self):
        self.integral = float(
            np.clip(
                self.integral + self.error[ERR_YAW] * self.dt,
                -self.ki_max,
                self.ki_max,
            ))

    def update(self, B: Body):

        if self.type == PT.XY:
            self.error = self.__update_error_XY(B.state[POS], B.orientation)
            if self.vrb_err_XY:
                print(
                    f"Error XY: {self.error[ERR_POS]} [m], Error Euler: {self.error[ERR_YAW]} [deg]"
                )

        elif self.type == PT.YZ:
            if not np.isfinite(B.arrival_Z):
                return self.brake, self.error, 0.0, self.integral

            self.error = self.__update_error_YZ(B.state[POS], B.arrival_Z)
            if self.vrb_err_YZ:
                print(
                    f"Error YZ: {self.error[ERR_POS]} [m], Error Euler: {self.error[ERR_YAW]} [deg]"
                )

        self.__update_ki()

        derivative = (self.error[ERR_YAW] - self.prev_error[ERR_YAW]) / self.dt
        self.prev_error[ERR_YAW] = self.error[ERR_YAW]

        brake_command = (self.kp * self.error[ERR_YAW] +
                         self.ki * self.integral + self.kd * derivative)

        command = float(np.clip(brake_command, -self.max_brake,
                                self.max_brake))

        self.brake = [max(-command, 0.0), max(command, 0.0)]

        return self.brake, self.error, brake_command, self.integral
