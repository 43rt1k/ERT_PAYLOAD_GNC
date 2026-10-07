import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pfh.glidersim as gsim
from matplotlib.widgets import Slider

from payload_gnc.gsim_helper import *

YAW = 2

POS_ERROR = 0
YAW_ERROR = 1


class Plotter:

    def __init__(self, model, times, states, brakes, errors, output_path,
                 file_name, goal_position):
        self.model = model
        self.times = times
        self.states = states

        self.brakes = brakes
        self.errors = errors

        self.positions = np.array(states[POS], dtype=float, copy=True)
        self.positions[:, Z] *= -1

        self.velocities = np.array(states[VEL], dtype=float, copy=True)

        self.init_position = self.positions[0]

        self.goal_position = goal_position
        self.goal_position[Z] *= -1

        self.file_name = file_name
        self.output_path = output_path

        self.window_title = "Payload GNC — Simulation"

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def save_npz(self):
        np.savez(
            self.output_path / f"{self.file_name}.npz",
            times=self.times,
            states=self.states,
        )

    def save_png(self, figure):
        figure.savefig(self.output_path / f"{self.file_name}.png", dpi=180)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __set_titles(self, figure, ax):
        figure.canvas.manager.set_window_title(self.window_title)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Flight trajectory")

    def __plot_cross(self, ax, position, label, color="black"):
        label_n_pos = (
            f"{label} "
            f"[{position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}]")
        x, y, z = position
        point, = ax.plot(
            [x],
            [y],
            [z],
            marker="X",
            color=color,
            markersize=10,
            linestyle="None",
        )

        text = ax.text(
            x,
            y,
            z + 20,  # 20 m above the cross in plotted altitude
            label_n_pos,
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
        )
        return point, text

    def __add_slider(self, figure, ax):

        def label(position):
            return f"Payload position\n{position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}"

        point, text = self.__plot_cross(ax,
                                        self.init_position,
                                        label(self.init_position),
                                        color="purple")

        slider_ax = figure.add_axes([0.10, 0.06, 0.32, 0.025])
        times = np.asarray(self.times)

        self.slider = Slider(
            ax=slider_ax,
            label="Time [s]",
            valmin=times[0],
            valmax=times[-1],
            valinit=times[0],
            valstep=times,
            valfmt="%.2f",
        )

        def update(time):
            index = np.argmin(np.abs(times - time))
            position = self.positions[index]
            x, y, z = position

            self.time_line.set_xdata([time, time])
            point.set_data_3d([x], [y], [z])

            text.set_position((x, y))
            text.set_3d_properties(z + 20, zdir=None)
            text.set_text(label(position))

            figure.canvas.draw_idle()

        self.slider.on_changed(update)

    def plot_trajectory(self, ax):
        figure = ax.figure

        self.__set_titles(figure, ax)

        ax.plot(
            self.positions[:, X],
            self.positions[:, Y],
            self.positions[:, Z],
            color="black",
            label="Trajectory",
        )

        self.__plot_cross(ax, self.init_position, "Initial position", "blue")
        self.__plot_cross(ax, self.goal_position, "Goal position", "red")

        ax.view_init(elev=40, azim=-60)
        ax.tick_params(labelsize=8)

        figure.tight_layout()
        ax.legend()
        ax.legend(loc="upper left")
        # Leave room below the plot for the slider.
        figure.tight_layout(rect=[0, 0.12, 1, 1])
        self.__add_slider(figure, ax)

        self.save_png(figure)

    def plot_flight_data(self, ax):

        euler = gsim.orientation.quaternion_to_euler(self.states["q_b2e"])
        yaw = np.rad2deg(euler[:, YAW])

        yaw_error = np.rad2deg(self.errors[:, YAW_ERROR])

        self.time_line = ax.axvline(
            x=self.times[0],
            color="black",
            linestyle="--",
            linewidth=1,
        )

        # First graph: brakes on the left axis.
        ax.stairs(
            self.brakes[:, 0],
            self.times,
            baseline=None,
            color="blue",
            linestyle="--",
            label="Left brake",
        )
        ax.stairs(
            self.brakes[:, 1],
            self.times,
            baseline=None,
            color="green",
            linestyle="--",
            label="Right brake",
        )
        ax.set_ylabel("Brake [0–1]")
        ax.set_ylim(-0.05, 1.05)

        # Angles on the right axis, sharing the same graph.
        ax_angles = ax.twinx()
        ax_angles.plot(
            self.times[:-1],
            self.errors[:, POS_ERROR],
            color="purple",
            label="XY error",
        )
        ax_angles.plot(
            self.times[:-1],
            yaw_error,
            color="red",
            label="Yaw error",
        )
        ax_angles.set_ylabel("Angle [°]")

        handles, labels = ax.get_legend_handles_labels()
        angle_handles, angle_labels = ax_angles.get_legend_handles_labels()
        ax.legend(handles + angle_handles, labels + angle_labels)
        ax.grid(True, alpha=0.3)

    def plot(self):
        figure = plt.figure(figsize=(16, 8))
        figure.subplots_adjust(
            left=0.07,
            right=0.95,
            bottom=0.17,
            top=0.92,
            wspace=0.4,
            hspace=0.15,
        )
        grid = figure.add_gridspec(1, 2, width_ratios=[1, 1.4])

        # Left: flight data.
        ax_controls = figure.add_subplot(grid[0, 0])

        # Right: trajectory.
        ax_trajectory = figure.add_subplot(grid[:, 1], projection="3d")

        self.plot_flight_data(ax_controls)
        self.plot_trajectory(ax_trajectory)

        figure.savefig(
            self.output_path / f"{self.file_name}_overview.png",
            dpi=180,
        )

        plt.show()
        return figure


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def basic_plot(model, times, states):
    points = gsim.extras.simulation.sample_paraglider_positions(
        model, states, times)

    gsim.extras.plots.plot_3d_simulation_path(**points)

    return plt.gcf()
