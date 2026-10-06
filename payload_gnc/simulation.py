import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *

L = 0
R = 1


class Simulator:

    @dataclass
    class Stats:
        time = []
        state = []
        brake = []

    def __init__(
            self,
            aircraft,
            *,
            configs,
            controller,
            wind=(0, 0, 0),
    ):
        # Commands currently applied to the aircraft.
        self.model = self.__init_model(aircraft, configs, wind)
        self.controller = controller

        self.brake = np.zeros(2)
        self.init_pos = configs["init_pos"]
        self.init_euler = configs["init_euler"]

        self.time = 0
        self.state = None

        self.stats = self.Stats()

        self.duration = configs["t_sim"]
        self.dt = configs["dt"]

        self.end_time = self.time + self.duration

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_bl(self, t):
        return self.brake[L]

    def __get_br(self, t):
        return self.brake[R]

    def get_totals(self):
        return (np.asarray(self.stats.time),
                np.asarray(self.stats.state, dtype=self.model.state_dtype),
                np.asarray(self.stats.brake), np.asarray(self.stats.error))

    def update_totals(self, time, state, brake, error):
        self.stats.time.append(time)

        self.stats.state.append(state.copy())

        self.stats.brake.append(brake.copy())

        self.stats.error.append(error.copy())

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __init_model(self, aircraft, configs, wind):

        return gsim.simulator.ParagliderStateDynamics6a(
            aircraft,
            delta_a=0,
            delta_bl=self.__get_bl,
            delta_br=self.__get_br,
            delta_w=0,
            rho_air=configs["rho_air"],
            v_W2e=wind,
        )

    def reset(self, init_pos=None):
        """Start a new flight in steady glide at the given NED position."""

        if init_pos is not None: self.init_pos = init_pos

        # Equilibrium initialization uses neutral controls.
        self.brake = np.zeros(2)

        state = self.model.starting_equilibrium()
        state[POS] = self.init_pos

        self.state = state
        self.time = 0

        self.stats.time = [self.time]
        self.stats.state = [self.state.copy()]
        self.stats.brake = []
        self.stats.error = []

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def step(self, brake):

        self.brake = brake

        with redirect_stdout(io.StringIO()):
            times, states = gsim.simulator.simulate(
                self.model,
                self.state.copy(),
                t0=self.time,
                T=self.dt,
                dt=self.dt,
            )

        self.state = states[-1].copy()
        self.time = float(times[-1])

        return self.state.copy()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def run(self):
        self.reset()
        self.controller.reset()

        while self.time < self.end_time:

            brake, error = self.controller.update(self.state, self.time)
            state = self.step(brake=brake)

            if state[POS][Z] >= 0: break

            self.update_totals(self.time, state, brake, error)

        return self.get_totals()
