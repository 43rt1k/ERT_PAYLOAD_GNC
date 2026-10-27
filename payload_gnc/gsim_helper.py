import numpy as np
import pfh.glidersim as gsim





POS = "r_RM2O"
ORI = "q_b2e"
VEL = "v_RM2e"

X = 0
Y = 1
Z = 2

XY = 2
XYZ = 3

L = 0
R = 1

LOG_LEN = 55





def get_time_str(time=0):
    return f" | {int(time)} [s]"





def print_change(VRB_ID, old_val, new_val, time=0, to_int=True):
    id = f"–{VRB_ID} | "

    def format_value(value):
        if to_int:
            if isinstance(value, (float, np.floating)):
                return int(value) if np.isfinite(value) else value

            if isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)

                if (np.issubdtype(array.dtype, np.floating) and np.all(np.isfinite(array))):
                    return array.astype(int)

        return value

    text = (f"{id}{format_value(old_val)}"
            f" ––> {format_value(new_val)}")

    print(text.ljust(LOG_LEN), get_time_str(time))





def print_message(VRB_ID, str, time=0):
    id = f"–{VRB_ID} | "
    text = (f"{id}{str}")
    print(text.ljust(LOG_LEN), get_time_str(time))





def deg2rad(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi





def q2euler(q):
    return gsim.orientation.quaternion_to_euler(q)
