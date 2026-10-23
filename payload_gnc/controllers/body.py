import numpy as np
from dataclasses import dataclass
from payload_gnc.gsim_helper import *


class Body:

    @dataclass
    class Dist:
        init_to_end_goal = np.inf  #rename to init_to_goal_final and pthers
        init_to_end_goal_XY = np.inf  #rename to init_to_goal_final and pthers

        to_end_goal = np.inf
        to_end_goal_XY = np.inf

        to_goal = np.inf
        to_goal_XY = np.inf

        hold_to_end_goal = np.inf
        hold_to_end_goal_XY = np.inf

        init_to_hold_Z = np.inf

    @dataclass
    class Speed:
        XYZ = 0
        XY = 0
        Z = 0

    def __init__(self,
                 goal_position,
                 initial_position,
                 n=5,
                 time=0,
                 hold_coeff=0.4):

        # Verboooooooose ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.VRB_ID = "––BOD | "
        self.vrb_goal = True
        self.vrb_arr = False

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

        self.dist = self._init_dist(hold_coeff)
        self.speed = self.Speed()

        self.arrival_Z = 0

    def _init_dist(self, hold_coeff):
        dist = self.Dist()

        dist.init_to_end_goal = np.linalg.norm(self.goal_pos_final -
                                               self.init_pos)
        dist.init_to_end_goal_XY = np.linalg.norm(self.goal_pos_final[:XY] -
                                                  self.init_pos[:XY])

        dist.to_end_goal = np.inf
        dist.to_end_goal_XY = np.inf

        dist.to_goal = np.inf
        dist.to_goal_XY = np.inf

        dist.hold_to_end_goal = dist.init_to_end_goal * hold_coeff
        dist.hold_to_end_goal_XY = dist.init_to_end_goal_XY * hold_coeff

        return dist

    def set_goal(self, goal_pos=None, grounded=False):

        if goal_pos is None: goal_pos = self.goal_pos_final

        self.ex_goal_pos = self.goal_pos
        self.goal_pos = goal_pos.copy()

        if grounded: self.goal_pos[Z] = 0

        self.is_goal_reached = False
        self.is_goal_reached_XY = False

        if self.vrb_goal:
            print(self.VRB_ID, "Goal changed: ",
                  np.array(self.ex_goal_pos, dtype=int, copy=False), " -> ",
                  np.array(self.goal_pos, dtype=int, copy=False),
                  get_time_str(self.time))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def is_at_hold(self):
        if self.dist.to_goal_XY <= self.dist.hold_to_end_goal_XY:
            self.dist.init_to_hold_Z = abs(self.init_pos[Z] -
                                           self.state[POS][Z])

            return True

    def is_landed(self):
        return self.state[POS][Z] >= 0

    def is_hold_done(self):
        if abs(self.state[POS][Z]) <= abs(self.init_pos[Z] * 0.3):
            return True

    def is_goal_reached(self, only_XY=False):

        if only_XY:
            goal_position = self.goal_position[:XY]
            position = self.state[POS][:XY]
            velocity = self.state[VEL][:XY]
        else:
            goal_position = self.goal_position
            position = self.state[POS]
            velocity = self.state[VEL]

        offset = goal_position - position

        offset_norm = np.linalg.norm(offset)
        speed_norm = np.linalg.norm(velocity)

        cosine = np.dot(velocity, offset) / (speed_norm * offset_norm)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))

        print(self.VRB_ID, "offset_norm: ", offset_norm, " , angle ", angle,
              get_time_str(self.time))

        if offset_norm <= self.goal_margin and angle <= 1: return True

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_speed(self, stat_states):

        if not stat_states or len(stat_states) < 5: return

        recent_states = stat_states[-self.n:]

        recent_speeds = np.array([state[VEL] for state in recent_states])

        # Mean total speed (3D magnitude).
        self.speed.XYZ = np.mean(np.linalg.norm(recent_speeds, axis=1))

        # Mean horizontal speed.
        self.speed.XY = np.mean(np.linalg.norm(recent_speeds[:, :XY], axis=1))

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
            print(self.VRB_ID, f"Arrival Z: {self.arrival_Z:.2f}",
                  get_time_str(self.time))

    def update(self, state, time, stat_states):

        self.time = float(time)
        self.state = state.copy()
        self.orientation = q2euler(state[ORI]).copy()

        self.dist.to_end_goal = np.linalg.norm(self.goal_pos_final -
                                               state[POS])
        self.dist.to_end_goal_XY = np.linalg.norm(self.goal_pos_final[:XY] -
                                                  state[POS][:XY])

        self.dist.to_goal = np.linalg.norm(self.goal_pos - state[POS])
        self.dist.to_goal_XY = np.linalg.norm(self.goal_pos[:XY] -
                                              state[POS][:XY])

        self.__update_speed(stat_states)
        self.__update_arrival_Z()
