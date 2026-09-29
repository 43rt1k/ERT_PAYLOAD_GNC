from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from payload_gnc.aircraft import build_demo_aircraft
from payload_gnc.controllers.scripted import right_brake_sequence
from payload_gnc.plotting import plot_trajectory
from payload_gnc.simulation import run_simulation


def main():
    aircraft = build_demo_aircraft()
    right_brake = right_brake_sequence()

    print("Running 25 seconds of simulated flight...")

    model, times, states = run_simulation(
        aircraft,
        right_brake=right_brake,
        duration=25.0,
        sample_interval=0.5,
    )

    # Resolve results relative to the project, regardless of where
    # the terminal command is executed.
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "results"
    output.mkdir(parents=True, exist_ok=True)

    np.savez(
        output / "first_flight.npz",
        times=times,
        states=states,
    )

    figure = plot_trajectory(model, times, states)
    figure.savefig(output / "first_flight.png", dpi=180)

    print(f"Results saved to {output}")
    plt.show()


if __name__ == "__main__":
    main()
