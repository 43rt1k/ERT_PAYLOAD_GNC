import numpy as np
from enum import Enum, auto


class Log:

    class ID(Enum):
        TIMES = auto()
        STATES = auto()
        BRAKES = auto()
        BRAKE_COMMANDS = auto()
        ERRORS = auto()
        INTEGRALS = auto()
        GOAL_POSITIONS = auto()

    def __init__(self):
        self.VRB_ID = "––LOG | "
        self.vrb_update = False

        self.data = {key: [] for key in self.ID}

    def __getitem__(self, key):
        return self.data[key]

    def pad_all(self):
        target_len = len(self.data[self.ID.STATES])

        for key, values in self.data.items():
            # Convert states without forcing a numeric dtype.
            if key == self.ID.STATES:
                self.data[key] = np.asarray(values)
                continue

            data = np.asarray(values, dtype=float)

            # Goal history is not one entry per timestep.
            if key == self.ID.GOAL_POSITIONS:
                self.data[key] = data
                continue

            missing_len = target_len - len(data)

            if missing_len < 0:
                raise ValueError(f"{key.name} has more samples than STATES.")

            padding = np.full(
                (missing_len, ) + data.shape[1:],
                np.nan,
            )

            self.data[key] = np.concatenate((padding, data), axis=0)

    def update(self, time, state, brakes, brake_command, integral, error,
               goal_position):
        self.data[self.ID.TIMES].append(float(time))
        self.data[self.ID.STATES].append(state.copy())
        self.data[self.ID.BRAKES].append(brakes.copy())
        self.data[self.ID.BRAKE_COMMANDS].append(float(brake_command))
        self.data[self.ID.ERRORS].append(
            np.array(error, dtype=float, copy=True))
        self.data[self.ID.INTEGRALS].append(float(integral))

        goals = self.data[self.ID.GOAL_POSITIONS]

        if not goals:
            goals.append(goal_position.copy())
            if self.vrb_update: print(self.VRB_ID, "First goal position added")
            return

        if not np.array_equal(goal_position, goals[-1]):
            if self.vrb_update: print(self.VRB_ID, "New goal position added")
            goals.append(goal_position.copy())
