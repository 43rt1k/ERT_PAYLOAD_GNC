import numpy as np

from payload_gnc.gsim_helper import *

ERR_POS = 0
ERR_YAW = 1


class PID_Controller():

    def __init__(self, config, dt, max_brake, verbose=True):
        self.kp = config["kp"]
        self.ki = config["ki"]
        self.kd = config["kd"]
        self.max_brake = max_brake

        self.dt = dt
        self.integral = 0.0

        self.goal_position = None

        self.prev_error = [0, 0]

        self.position = np.zeros(3)
        self.orientation = np.zeros(3)

        if verbose:
            print(
                f"Initialized PID controller with kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def reset(self, goal_position):
        self.integral = 0.0
        self.goal_position = goal_position
        self.prev_error = [0, 0]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_error(self, position, orientation):

        offset = np.asarray(self.goal_position) - np.asarray(position)

        # Straight-line distance to the goal [m].
        error_position = float(np.linalg.norm(offset))

        # Horizontal bearing toward the goal [rad].
        desired_yaw = np.arctan2(offset[1], offset[0])

        # Shortest signed heading error, in [-pi, pi).
        error_yaw = (desired_yaw - orientation[2] + np.pi) % (2 *
                                                              np.pi) - np.pi

        return [error_position, error_yaw]

    def update(self, position, orientation):

        error = self.__update_error(position, orientation)
        # print(f"Error Position: {error[POS]}, Error Euler: {error[YAW]}")

        self.integral += error[ERR_YAW] * self.dt

        derivative = (error[ERR_YAW] - self.prev_error[ERR_YAW]) / self.dt
        self.prev_error[ERR_YAW] = error[ERR_YAW]

        command = (self.kp * error[ERR_YAW] + self.ki * self.integral +
                   self.kd * derivative)
        command = float(np.clip(command, -self.max_brake, self.max_brake))

        left_brake = max(-command, 0.0)
        right_brake = max(command, 0.0)

        # self.pritn_state(time_step=2)
        return [left_brake, right_brake], error
