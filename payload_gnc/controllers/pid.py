import numpy as np

from payload_gnc.controllers.controller import Controller, q2euler

POS = 0
YAW = 1


class PID_Controller(Controller):

    def __init__(self, config, dt, verbose=True):
        self.kp = config["pid"]["kp"]
        self.ki = config["pid"]["ki"]
        self.kd = config["pid"]["kd"]
        self.max_brake = config["max_brake"]

        self.dt = dt
        self.time = 0.0
        self.integral = 0.0

        self.goal_position = config["goal_position"]

        self.prev_error = [0, 0]

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.euler = np.zeros(3)

        if verbose:
            print(
                f"Initialized PID controller with kp={self.kp}, ki={self.ki}, kd={self.kd}, max_brake={self.max_brake}."
            )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def reset(self):
        self.integral = 0.0

    def pritn_state(self, time_step=100):
        if int(self.time / time_step) == self.time / time_step:
            print(
                f"Position: {self.position}, Velocity: {self.velocity}, Euler: {self.euler}"
            )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def __update_state(self, state, time):
        self.position = state["r_RM2O"]
        self.velocity = state["v_RM2e"]

        self.euler = q2euler(state["q_b2e"])
        self.time = time

    def __update_error(self):

        offset = np.asarray(self.goal_position) - np.asarray(self.position)

        # Straight-line distance to the goal [m].
        error_position = float(np.linalg.norm(offset))

        # Horizontal bearing toward the goal [rad].
        desired_yaw = np.arctan2(offset[1], offset[0])

        # Shortest signed heading error, in [-pi, pi).
        error_yaw = (desired_yaw - self.euler[2] + np.pi) % (2 * np.pi) - np.pi

        return [error_position, error_yaw]

    def update(self, state, time):

        self.__update_state(state, time)
        error = self.__update_error()
        # print(f"Error Position: {error[POS]}, Error Euler: {error[YAW]}")

        self.integral += error[YAW] * self.dt

        derivative = (error[YAW] - self.prev_error[YAW]) / self.dt
        self.prev_error[YAW] = error[YAW]

        command = (self.kp * error[YAW] + self.ki * self.integral +
                   self.kd * derivative)
        command = float(np.clip(command, -self.max_brake, self.max_brake))

        left_brake = max(-command, 0.0)
        right_brake = max(command, 0.0)

        # self.pritn_state(time_step=2)
        return [left_brake, right_brake], error
