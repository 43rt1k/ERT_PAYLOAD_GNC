import numpy as np
from dataclasses import dataclass
from payload_gnc.gsim_helper import *
from payload_gnc.logger import *





ARRIVAL_Z = 100
Z_COEFF = 3
DEG_TOL = 15





class Body:

    @dataclass
    class Dist:
        init_to_end_goal = np.inf    #rename to init_to_goal_final and pthers
        init_to_end_goal_XY = np.inf    #rename to init_to_goal_final and pthers

        to_end_goal = np.inf
        to_end_goal_XY = np.inf

        to_goal = np.inf
        to_goal_XY = np.inf

        hold_to_end_goal = np.inf
        hold_to_end_goal_XY = np.inf

        init_to_hold_Z = np.inf
        goal_margin = np.inf

    @dataclass
    class Speed:
        XYZ = 0
        XY = 0
        Z = 0

    def __init__(self, goal_position, initial_position, goal_margin, n=5, time=0, hold_coeff=0.4):

        # Verboooooooose ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.VRB_ID = "BOD"
        self.vrb_goal = True
        self.vrb_arr = False
        self.vrb_rad = True

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = time
        self.state = None
        self.orientation = None

        self.init_pos = initial_position

        self.goal_pos_final = goal_position
        self.goal_pos = self.goal_pos_final
        self.ex_goal_pos = self.goal_pos_final

        self.n = n

        # Intrinsic data ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dist = self._init_dist(hold_coeff, goal_margin)
        self.speed = self.Speed()

        self.arrival_Z = 0
        self.minR = 0

    def _init_dist(self, hold_coeff, goal_margin):
        dist = self.Dist()

        dist.init_to_end_goal = np.linalg.norm(self.goal_pos_final - self.init_pos)
        dist.init_to_end_goal_XY = np.linalg.norm(self.goal_pos_final[: XY] - self.init_pos[: XY])

        dist.to_end_goal = np.inf
        dist.to_end_goal_XY = np.inf

        dist.to_goal = np.inf
        dist.to_goal_XY = np.inf

        dist.hold_to_end_goal = dist.init_to_end_goal * hold_coeff
        dist.hold_to_end_goal_XY = dist.init_to_end_goal_XY * hold_coeff

        dist.goal_margin = goal_margin

        return dist

    def set_goal(self, goal_pos=None, grounded=False):

        if goal_pos is None: goal_pos = self.goal_pos_final

        self.ex_goal_pos = self.goal_pos
        self.goal_pos = goal_pos.copy()

        if grounded: self.goal_pos[Z] = 0

        if self.vrb_goal: print_change(self.VRB_ID, self.ex_goal_pos, self.goal_pos, self.time)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def is_landed(self):
        return self.state[POS][Z] >= 0

    def is_at_hold(self):
        if self.dist.to_goal_XY <= self.dist.hold_to_end_goal_XY: return True

    def is_at_goal(self, only_XY=True):
        if only_XY:
            if (abs(self.dist.to_goal_XY) <= self.dist.goal_margin and
                            abs(self.dist.to_goal) >= self.dist.goal_margin):
                return True

        return False

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_speed(self, stat_states):

        if not stat_states or len(stat_states) < 5: return

        recent_states = stat_states[-self.n :]

        recent_speeds = np.array([state[VEL] for state in recent_states])

        # Mean total speed (3D magnitude).
        self.speed.XYZ = np.mean(np.linalg.norm(recent_speeds, axis=1))

        # Mean horizontal speed.
        self.speed.XY = np.mean(np.linalg.norm(recent_speeds[:, : XY], axis=1))

        # Mean signed vertical velocity: positive down in NED.
        self.speed.Z = np.mean(recent_speeds[:, Z])

    def __update_arrival_Z(self):

        if self.speed.XY <= 1e-6:
            self.arrival_Z = np.inf
            return

        time_to_goal = self.dist.to_goal_XY / self.speed.XY

        self.arrival_Z = self.state[POS][Z] - self.speed.Z * time_to_goal

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        if self.vrb_arr:
            print(self.VRB_ID, f"Arrival Z: {self.arrival_Z:.2f}", get_time_str(self.time))

    def get_min_rad(self, P, L):

        times = np.asarray(L[Log.ID.TIMES], dtype=float)
        states = np.asarray(L[Log.ID.STATES])

        mask = (times >= P.time_ex) & (times <= P.time_now)
        states = states[mask]

        yaw = q2euler(states[ORI])[:, XY]
        opp_yaw = yaw[0] + np.pi

        error = np.abs(deg2rad(yaw - opp_yaw))

        index = np.argmin(error)

        min_error = error[index]
        opposite_position = states[POS][index]
        # Aircraft position when HOLD began.

        diameter = np.linalg.norm(opposite_position[: XY] - P.pos_ex[: XY])
        radius = float(diameter / 2)

        if self.vrb_rad:
            print_message(self.VRB_ID, f"Minimal turning radius = {int(radius)}", self.time)

        return radius

    def update(self, state, time, stat_states):

        self.time = float(time)
        self.state = state.copy()
        self.orientation = q2euler(state[ORI]).copy()

        self.dist.to_end_goal = np.linalg.norm(self.goal_pos_final - state[POS])
        self.dist.to_end_goal_XY = np.linalg.norm(self.goal_pos_final[: XY] - state[POS][: XY])

        self.dist.to_goal = np.linalg.norm(self.goal_pos - state[POS])
        self.dist.to_goal_XY = np.linalg.norm(self.goal_pos[: XY] - state[POS][: XY])

        self.__update_speed(stat_states)
        self.__update_arrival_Z()
