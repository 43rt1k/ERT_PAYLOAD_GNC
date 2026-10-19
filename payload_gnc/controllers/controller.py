import numpy as np

from dataclasses import dataclass
from enum import Enum, auto

import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.pid import PID_Controller

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def deg2rad(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def q2euler(q):
    return gsim.orientation.quaternion_to_euler(q)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class FP(Enum):
    STABILISATION = auto()
    DEFAULT = auto()
    HOLD = auto()
    APPROACH = auto()
    LANDED = auto()


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class GN_Statistics:

    def __init__(self):
        self.VRB_ID = "––STA | "
        self.vrb_update = False
        self.times = []
        self.states = []
        self.brakes = []
        self.errors = []
        self.goal_positions = []

    def get(self):
        return (np.asarray(self.times), np.asarray(self.states),
                np.asarray(self.brakes), np.asarray(self.errors),
                np.asarray(self.goal_positions))

    def update(self, time, state, brake, error, goal_position):
        self.times.append(float(time))
        self.states.append(state.copy())
        self.brakes.append(brake.copy())
        self.errors.append(error.copy())

        if not self.goal_positions:
            self.goal_positions.append(goal_position.copy())
            if self.vrb_update: print(self.VRB_ID, "First goal position added")
            return

        if not np.array_equal(goal_position, self.goal_positions[-1]):
            if self.vrb_update: print(self.VRB_ID, "New goal position added")
            self.goal_positions.append(goal_position.copy())


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class GN_Body:

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

        self.VRB_ID = "––GNB | "
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
            print(self.VRB_ID, "Goal changed: ", self.ex_goal_pos, " -> ",
                  self.goal_pos, get_time_str(self.time))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def is_at_hold(self):
        if self.dist.to_goal_XY <= self.dist.hold_to_end_goal_XY:
            self.dist.init_to_hold_Z = abs(self.init_pos[Z] -
                                           self.state[POS][Z])

            return True

    def is_landed(self):
        return self.state[POS][Z] >= 0

    def is_hold_done(self):
        if abs(self.state[POS][Z]) <= self.dist.init_to_hold_Z * 1.25:
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

        if self.vrb_arr:
            print(self.VRB_ID, "Arrival Z: ", self.arrival_Z,
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


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


class GN_Controller:

    def __init__(self, config, initial_position, brakes=[0, 0]):

        # VERBOSE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        self.VRB_ID = "––GNC | "
        self.vrb_phase = True

        # CONFIG –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.dt = config["dt"]

        self.arrival_margin = config["arrival_margin"]
        self.stabilisation_time = config["stabilisation_time"]
        self.goal_margin = config["goal_margin"]

        # PHASE ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.phase = FP.STABILISATION
        self.ex_phase = self.phase

        # Status ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.is_landed = False
        self.is_goal_reached = False
        self.is_goal_reached_XY = False

        # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

        self.time = 0
        self.brakes = brakes

        self.B = GN_Body(goal_position=np.asarray(config["goal_position"]),
                         initial_position=np.asarray(initial_position),
                         hold_coeff=config["hold_coeff"])

        self.C = PID_Controller(config["pid"],
                                config["dt"],
                                config["max_brake"],
                                goal_position=self.B.goal_pos_final)

        self.S = GN_Statistics()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_phase(self, phase):
        self.ex_phase = self.phase
        self.phase = phase

        if self.vrb_phase:
            print(self.VRB_ID, "Phase changed: ", self.ex_phase, " -> ",
                  self.phase, get_time_str(self.time))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __is_stabilisation_done(self):
        return self.time >= self.stabilisation_time

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __update_phase(self):

        # LAST PHASE: LANDING
        if self.B.is_landed():
            self.__set_phase(FP.LANDED)

            # PHASE STABILISATION: Wait for the payload to stabilise
        elif self.phase is FP.STABILISATION and self.__is_stabilisation_done():
            self.__set_phase(FP.DEFAULT)

        # PHASE HOLD: Turn around at a nearer point to lose height
        elif self.phase is FP.DEFAULT and self.ex_phase is FP.STABILISATION and self.B.is_at_hold(
        ):
            self.__set_phase(FP.HOLD)
            self.B.set_goal(self.B.state[POS])

        # PHASE APPROACH: Approach the final goal position in a arc-shaped trajectory
        elif self.phase is FP.HOLD and self.ex_phase is FP.DEFAULT and self.B.is_hold_done(
        ):
            self.__set_phase(FP.APPROACH)
            self.B.set_goal()

    def update(self, state, time):
        if self.phase is FP.LANDED: return

        self.B.update(state, time, self.S.states)
        self.time = float(time)

        self.__update_phase()

        self.brakes, self.error = self.C.update(self.B.goal_pos,
                                                self.B.state[POS],
                                                self.B.orientation)

        self.S.update(self.time, self.B.state, self.brakes, self.error,
                      self.B.goal_pos)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
