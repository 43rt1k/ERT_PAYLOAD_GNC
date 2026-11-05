import numpy as np
from dataclasses import dataclass
from payload_gnc.gsim_helper import *
from payload_gnc.logger import *





ARRIVAL_Z = 100
Z_COEFF = 3
DEG_TOL = 15





class Body:

    class ST(Enum):
        Z = auto()
        XY = auto()
        XYZ = auto()

    def __init__(self,
                 goal_position,
                 initial_position,
                 goal_margin,
                 brakes=[0, 0],
                 n=5,
                 time=0,
                 hold_coeff=0.4):

        # Verboooooooose ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.VRB_ID = "BOD"
        self.vrb_goal = True
        self.vrb_arr = False
        self.vrb_rad = True

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.time = time    # s
        self.brakes = brakes

        self.pos: np.ndarray    # m
        self.vel: np.ndarray    # m/s
        self.ori: np.ndarray    # rad
        self.rot: np.ndarray    # rad/s

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.init_pos = initial_position
        self.goal_pos_final = goal_position
        self.goal_pos: np.ndarray
        self.set_goal()

        self.goal_margin = goal_margin
        # self.n = n

        # Intrinsic data ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        # self.arrival_Z = 0
        # self.minR = 0
        # self.minZloss = 0

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def set_state(self, state):
        state_new = state.copy()

        self.pos = state_new[POS]
        self.vel = state_new[VEL]
        self.ori = q2euler(state_new[ORI])
        # self.rot = None    # rad/s

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def get_dist(self, goal, pos=None, XY=True):
        if pos is None: pos = self.pos

        dist = np.asarray(goal, dtype=float) - np.asarray(pos, dtype=float)

        if XY: dist = dist[: XY]

        return float(np.linalg.norm(dist))

    def get_speed(self, st):

        if st == self.ST.Z: return float(self.vel[Z])    # Signed: positive down in NED.
        if st == self.ST.XY: return float(np.linalg.norm(self.vel[: XY]))
        if st == self.ST.XYZ: return float(np.linalg.norm(self.vel))
        else: return self.vel.copy()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def set_goal(self, goal_pos=None, grounded=False):

        if goal_pos is None: goal_pos = self.goal_pos_final

        self.goal_pos = goal_pos.copy()

        if grounded: self.goal_pos[Z] = 0

        if self.vrb_goal: print_change(self.VRB_ID, f"New Goal is: {self.goal_pos}", self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def is_landed(self):
        return self.pos[Z] >= 0

    def is_goal_passed(self):

        to_goal = self.goal_pos_final[: XY] - self.pos[: XY]

        yaw = float(self.ori[YAW])
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        dist = self.get_dist(self.goal_pos_final)

        if np.dot(forward, to_goal) < 0 and dist < self.goal_margin:
            return True

        return False

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def update(self, state, time):

        self.time = float(time)
        self.set_state(state)

        # self.state = state.copy()
        # self.orientation = q2euler(state[ORI]).copy()

        # self.dist.to_end_goal = np.linalg.norm(self.goal_pos_final - state[POS])
        # self.dist.to_end_goal_XY = np.linalg.norm(self.goal_pos_final[: XY] - state[POS][: XY])

        # self.dist.to_goal = np.linalg.norm(self.goal_pos - state[POS])
        # self.dist.to_goal_XY = np.linalg.norm(self.goal_pos[: XY] - state[POS][: XY])

        # self.__update_speed(stat_states)
        # self.__update_arrival_Z()





# def __update_speed(self, stat_states):

#         if not stat_states or len(stat_states) < 5: return

#         recent_states = stat_states[-self.n :]

#         recent_speeds = np.array([state[VEL] for state in recent_states])

#         # Mean total speed (3D magnitude).
#         self.speed.XYZ = np.mean(np.linalg.norm(recent_speeds, axis=1))

#         # Mean horizontal speed.
#         self.speed.XY = np.mean(np.linalg.norm(recent_speeds[:, : XY], axis=1))

#         # Mean signed vertical velocity: positive down in NED.
#         self.speed.Z = np.mean(recent_speeds[:, Z])

# def __update_arrival_Z(self):

#         if self.speed.XY <= 1e-6:
#             self.arrival_Z = np.inf
#             return

#         time_to_goal = self.dist.to_goal_XY / self.speed.XY

#         self.arrival_Z = self.state[POS][Z] - self.speed.Z * time_to_goal

#         # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

#         if self.vrb_arr:
#             print(self.VRB_ID, f"Arrival Z: {self.arrival_Z:.2f}", get_time_str(self.time))

# def is_at_goal(self, only_XY=True):

#     if only_XY:
#         if (abs(self.dist.to_goal_XY) <= self.dist.goal_margin and
#                         abs(self.dist.to_goal) >= self.dist.goal_margin):
#             return True

#     return False
