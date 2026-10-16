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

    def __init__(self, times, states, brakes, errors, output_path, file_name,
                 goal_position):
        self.times = times
        self.states = states

        self.brakes = self.__make_same_size(brakes)

        self.errors = self.__make_same_size(errors)

        self.positions = np.array(states[POS], dtype=float, copy=True)
        self.positions[:, Z] *= -1

        self.orientation = gsim.orientation.quaternion_to_euler(states[ORI])

        self.velocities = np.array(states[VEL], dtype=float, copy=True)

        self.init_position = self.positions[0]

        self.goal_position = goal_position
        self.goal_position[Z] *= -1

        self.file_name = file_name
        self.output_path = output_path

        self.window_title = "Payload GNC — Simulation"

        self.vrb_save_fig = False
        self.time_line = None

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def save_npz(self):
        np.savez(
            self.output_path / f"{self.file_name}.npz",
            times=self.times,
            states=self.states,
        )

    def save_png(self, figure):
        figure.savefig(self.output_path / f"{self.file_name}.png", dpi=180)

    def __save_fig(self, figure):
        if self.vrb_save_fig is False: return

        figure.savefig(
            self.output_path / f"{self.file_name}_overview.png",
            dpi=180,
        )

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __make_same_size(self, control_data):

        missing_len = len(self.states) - len(control_data)

        missing_values = np.full(
            (missing_len, ) + control_data.shape[1:],
            np.nan,
            dtype=float,
        )

        return np.concatenate((missing_values, control_data), axis=0)

    def __set_orientation(self):
        pass

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


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def plot_series(
        self,
        ax,
        times,
        data,
        *,
        color,
        label="",
        ylabel="",
        stepped=False,
    ):
        if stepped:
            # Requires one more timestamp than data values.
            ax.step(times, data, where="post", color=color, label=label)
        else:
            ax.plot(times, data, color=color, label=label)

        cursor = ax.axvline(
            self.times[0],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        self.time_lines.append(cursor)

        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def plot_left_side(self, figure, slot):

        plots = [
            dict(
                times=self.times,
                data=self.brakes[:, 0],
                color="blue",
                label="Left brake",
                stepped=True,
            ),
            dict(
                times=self.times,
                data=self.brakes[:, 1],
                color="green",
                label="Right brake",
                stepped=True,
            ),
            dict(
                times=self.times,
                data=np.rad2deg(self.errors[:, YAW_ERROR]),
                color="red",
                label="Yaw error",
                ylabel="[°]",
            ),
            dict(
                times=self.times,
                data=self.errors[:, POS_ERROR],
                color="purple",
                label="Position error",
                ylabel="[m]",
            ),
        ]

        grid = slot.subgridspec(len(plots), 1, hspace=0.1)
        self.time_lines = []
        axes = []

        for row, settings in enumerate(plots):
            ax = figure.add_subplot(
                grid[row, 0],
                sharex=axes[0] if axes else None,
            )
            self.plot_series(ax, **settings)
            axes.append(ax)

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel("Time [s]")

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __plot_trajectory(self, ax):

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Flight trajectory")

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

    def __plot_info(self, ax):
        ax.set_title("State data")

    def plot_right_side(self, figure, slot):
        grid = slot.subgridspec(2, 1, height_ratios=[8, 1])

        # Top right: 3D trajectory.
        ax_trajectory = figure.add_subplot(grid[0, 0], projection="3d")
        self.__plot_trajectory(ax_trajectory)

        ax_info = figure.add_subplot(grid[1, 0])
        self.__plot_info(ax_info)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def __plot_init(self):
        figure = plt.figure(figsize=(16, 8))
        figure.canvas.manager.set_window_title(self.window_title)
        figure.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.05,
            top=0.95,
            wspace=0.05,
            hspace=0.15,
        )

        return figure

    def plot(self):
        figure = self.__plot_init()

        # Main grid: one row, two columns.
        grid = figure.add_gridspec(1, 2, width_ratios=[1, 1])

        self.plot_right_side(figure, grid[0, 1])
        self.plot_left_side(figure, grid[0, 0])

        self.__save_fig(figure)
        plt.show()

        return figure
