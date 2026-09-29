from scipy.interpolate import interp1d


def right_brake_sequence():
    """Ramp up the right brake, hold it, then release it."""
    return interp1d(
        [0, 3, 5, 15, 17, 25],
        [0, 0, 0.25, 0.25, 0, 0],
        bounds_error=False,
        fill_value=0.0,
    )