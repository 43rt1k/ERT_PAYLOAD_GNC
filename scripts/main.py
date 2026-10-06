from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import pfh.glidersim as gsim

from payload_gnc.aircraft import build_basic_aircraft
from payload_gnc.controllers.scripted import right_brake_sequence
from payload_gnc.plotting import plot_trajectory
from payload_gnc.simulation import Simulator
from payload_gnc.controllers.pid import PID_Controller

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def load_config(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def init_config():
    """Initialize the simulation configuration."""
    project_root_path = Path(__file__).resolve().parents[1]

    output_path = project_root_path / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(project_root_path / "payload_gnc" / "configs" /
                         "simulation.toml")

    return project_root_path, output_path, config


def plot_results(output_path, model, times, states):

    np.savez(
        output_path / "main.npz",
        times=times,
        states=states,
    )

    figure = plot_trajectory(
        model,
        times,
        states,
        goal_position=config["controller"]["goal_position"])
    figure.savefig(output_path / "main.png", dpi=180)

    print(f"Results saved to {output_path}")
    plt.show()


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

project_root_path, output_path, config = init_config()


def main():

    aircraft = build_basic_aircraft(config["aircraft"])
    controller = PID_Controller(
        config["controller"],
        dt=config["simulation"]["dt"],
    )

    sim = Simulator(
        aircraft,
        configs=config["simulation"],
        controller=controller,
    )

    times, states = sim.run()

    plot_results(output_path, sim.model, times, states)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == "__main__":
    main()
