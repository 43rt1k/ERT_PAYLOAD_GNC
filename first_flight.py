from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pfh.glidersim as gsim
from scipy.interpolate import interp1d

# Demonstration aircraft: NOT the RocketTeam payload.
wing = gsim.extras.wings.niviuk_hook3(size=23, verbose=False)

harness = gsim.paraglider_harness.Spherical(
    mass=75,
    z_riser=0.5,
    S=0.55,
    CD=0.8,
    kappa_w=0.1,
)

aircraft = gsim.paraglider.ParagliderSystemDynamics6a(wing, harness)

# Normalized right-brake input: gentle application, hold, release.
right_brake = interp1d(
    [0, 3, 5, 15, 17, 25],
    [0, 0, 0.25, 0.25, 0, 0],
    bounds_error=False,
    fill_value=0.0,
)

model = gsim.simulator.ParagliderStateDynamics6a(
    aircraft,
    delta_a=0.0,  # Accelerator
    delta_bl=0.0,  # Left brake
    delta_br=right_brake,
    delta_w=0.0,  # Weight shift
    rho_air=1.225,
    v_W2e=(0.0, 0.0, 0.0),
)

initial_state = model.starting_equilibrium()

print("Running 25 seconds of simulated flight...")
times, states = gsim.simulator.simulate(
    model,
    initial_state,
    dt=0.5,  # Output sampling interval
    T=25,
)

output = Path("results")
output.mkdir(exist_ok=True)

np.savez(output / "first_flight.npz", times=times, states=states)

points = gsim.extras.simulation.sample_paraglider_positions(
    model, states, times)
gsim.extras.plots.plot_3d_simulation_path(**points)
plt.savefig(output / "first_flight.png", dpi=180)
plt.show()
