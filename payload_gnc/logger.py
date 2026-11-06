import numpy as np
from enum import Enum, auto
from copy import deepcopy
from payload_gnc.gsim_helper import *





class Log:

    class ID(Enum):

        BODY = auto()
        BRAKE_COMMANDS = auto()
        ERRORS = auto()
        INTEGRALS = auto()
        PHASES = auto()
        CENTERS = auto()

    def __init__(self):
        self.VRB_ID = "––LOG | "
        self.vrb_update = False

        self.data = {
                    key: [] for key in self.ID }

    def __getitem__(self, key):
        return self.data[key]

    def pad_all(self):
        target_len = len(self.data[self.ID.BODY])

        for key, values in self.data.items():
            if key == self.ID.PHASES or key == self.ID.CENTERS:
                # Keep phase snapshots as a list; do not pad them.
                continue

            if key == self.ID.BODY:
                self.data[key] = np.asarray(values)
                continue

            data = np.asarray(values, dtype=float)
            missing_len = target_len - len(data)

            if missing_len < 0:
                raise ValueError(f"{key.name} has more samples than STATES.")

            padding = np.full((missing_len, ) + data.shape[1 :], np.nan,
                             )

            self.data[key] = np.concatenate((padding, data), axis=0)

    def update_centers(self, center, r):

        data = np.array([center[X], center[Y], center[Z], r], dtype=float)
        self.data[self.ID.CENTERS].append(data)

    def update(self, B, brake_command, integral, error, phase):

        self.data[self.ID.BODY].append(deepcopy(B))
        self.data[self.ID.BRAKE_COMMANDS].append(float(brake_command))
        self.data[self.ID.ERRORS].append(np.array(error, dtype=float, copy=True))
        self.data[self.ID.INTEGRALS].append(float(integral))

        phases = self.data[self.ID.PHASES]

        if (not phases or phase.now != phases[-1].now or phase.time_now != phases[-1].time_now):
            phases.append(deepcopy(phase))

            if self.vrb_update:
                print(self.VRB_ID, "Phase recorded:", phase.now)
