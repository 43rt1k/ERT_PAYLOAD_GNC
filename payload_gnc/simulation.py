import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import pfh.glidersim as gsim

from payload_gnc.gsim_helper import *
from payload_gnc.controllers.controller import GN_Controller, FP


class Simulator:

    def __init__(self, aircraft, *, configs, dt, GNC, wind=(0, 0, 0)):
        self.VRB_ID = "––SIM | "
        self.vrb_progress = False

        self.init_pos = configs["init_pos"]
        self.init_euler = configs["init_euler"]

        self.duration = configs["t_sim"]
        self.dt = dt
        self.time = 0

        self.GNC = GNC

        self.end_time = self.time + self.duration
        self.brake = np.zeros(2)

        self.model = self.__init_model(aircraft, configs, wind)
        self.state = self.__init_state()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __print_progress(self, step=100):
        if not self.vrb_progress: return

        if self.time % step == 0:
            print(self.VRB_ID, int(self.time), "/", int(self.end_time))

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __get_bl(self, t):
        return self.brake[L]

    def __get_br(self, t):
        return self.brake[R]

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

    def __init_state(self):

        state = self.model.starting_equilibrium()
        state[POS] = self.init_pos.copy()

        return state

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def step(self, brakes):

        self.brake = brakes

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

        while self.time < self.end_time:

            state = self.step(self.GNC.brakes)

            self.GNC.update(state, self.time)

            if self.GNC.phase == FP.LANDED:
                print(self.VRB_ID, "Reached ground at: ", int(self.time), " s")
                break

            self.__print_progress()

        return self.GNC.S.get()
