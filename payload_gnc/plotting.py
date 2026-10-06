import matplotlib.pyplot as plt
import numpy as np
import pfh.glidersim as gsim


def plot_trajectory(model, times, states, goal_position):
    # Copy so plotting does not modify the simulation data.
    positions = np.array(states["r_RM2O"], dtype=float, copy=True)
    print(positions)
    goal = np.array(goal_position, dtype=float, copy=True)

    # North, East, Down → North, East, Altitude
    positions[:, 2] *= -1
    goal[2] *= -1

    initial = positions[0]

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="black",
        label="Trajectory",
    )

    ax.scatter(
        *initial,
        color="blue",
        marker="x",
        s=100,
        linewidths=3,
        label="Initial position",
    )

    ax.scatter(
        *goal,
        color="red",
        marker="x",
        s=100,
        linewidths=3,
        label="Goal",
    )

    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_zlabel("Altitude above origin [m]")
    ax.set_title("Flight trajectory")
    ax.legend()

    figure.tight_layout()
    return figure


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def basic_plot(model, times, states):
    points = gsim.extras.simulation.sample_paraglider_positions(
        model, states, times)

    gsim.extras.plots.plot_3d_simulation_path(**points)

    return plt.gcf()
