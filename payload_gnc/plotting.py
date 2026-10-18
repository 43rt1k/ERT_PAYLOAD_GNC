import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pfh.glidersim as gsim
from matplotlib.widgets import Slider
from dataclasses import dataclass, field

from payload_gnc.gsim_helper import *

YAW = 2

POS_ERROR = 0
YAW_ERROR = 1


class Plotter:

    @dataclass
    class Snap:
        position: np.ndarray
        position_point: object = None
        position_text: object = None

        time_lines: list = field(default_factory=list)

    def __init__(self, times, states, brakes, errors, goal_positions,
                 output_path, file_name):
        self.times = times
        self.states = states

        self.brakes = self.__make_same_size(brakes)

        self.errors = self.__make_same_size(errors)

        self.positions = np.array(states[POS], dtype=float, copy=True)
        self.positions[:, Z] *= -1

        self.orientation = gsim.orientation.quaternion_to_euler(states[ORI])

        self.velocities = np.array(states[VEL], dtype=float, copy=True)

        self.init_position = self.positions[0]

        self.goal_positions = goal_positions
        self.goal_positions[:, Z] *= -1

        self.file_name = file_name
        self.output_path = output_path

        self.window_title = "Payload GNC — Simulation"

        self.vrb_save_fig = False
        self.plots = self.__define_plots()
        self.table_rows = self.__define_table_rows()

        self.slider = None

        self.snap = self.Snap(position=self.positions[0].copy())

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __define_plots(self):
        return [
            dict(
                times=self.times,
                data=self.positions[:, Z],
                color="blue",
                label="Position: Z",
                stepped=False,
            ),
            dict(
                times=self.times,
                data=self.velocities[:, Z],
                color="blue",
                label="Speed: Z",
                stepped=False,
            ),
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

    def __define_table_rows(self):
        return [
            dict(
                label="Position",
                data=self.positions,
                components="X , Y , Z",
                unit="[m]",
            ),
            dict(
                label="Velocity",
                data=self.velocities,
                components="X , Y , Z",
                unit="[m/s]",
            ),
            dict(
                label="Orientation",
                data=np.rad2deg(self.orientation),
                components="Roll , Pitch , Yaw",
                unit="[°]",
            ),
            dict(
                label="Angular velocity",
                data=np.rad2deg(self.states["omega_b2e"]),
                components="Body p , q , r",
                unit="[°/s]",
            ),
        ]

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


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    def __plot_series(
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

        time_line = ax.axvline(
            self.times[0],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        self.snap.time_lines.append(time_line)

        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def __plot_graphs(self, figure, slot):

        grid = slot.subgridspec(len(self.plots), 1)

        axes = []

        for row, settings in enumerate(self.plots):
            ax = figure.add_subplot(grid[row, 0])
            ax.tick_params(labelbottom=False)

            self.__plot_series(ax, **settings)

            axes.append(ax)

        axes[-1].set_xlabel("Time [s]")

    def __plot_slider(self, figure, slot):
        slider_ax = figure.add_subplot(slot)

        self.slider = Slider(
            ax=slider_ax,
            label="Time [s]",
            valmin=self.times[0],
            valmax=self.times[-1],
            valinit=self.times[0],
            valstep=np.asarray(self.times),
            valfmt="%.2f",
        )

        self.__update(self.times[0])
        self.slider.on_changed(self.__update)

    def plot_left_side(self, figure, slot):

        grid = slot.subgridspec(2, 1, height_ratios=[20, 1])

        self.__plot_graphs(figure, grid[0, 0])
        self.__plot_slider(figure, grid[1, 0])

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
        for index, position in enumerate(self.goal_positions):
            self.__plot_cross(ax, position, f"Goal {index + 1}", "green")

        self.snap.position_point, self.snap.position_text = self.__plot_cross(
            ax, self.snap.position, "Payload", color="red")

        ax.view_init(elev=40, azim=-60)
        ax.tick_params(labelsize=8)

    def __format_table_value(self, data, index):
        values = np.atleast_1d(data[index])
        return " | ".join(f"{value:.2f}" for value in values)

    def __plot_info(self, ax):
        ax.set_axis_off()

        self.info_ax = ax

        cells = [[
            f'{row["label"]} [{row["components"].replace(" / ", ", ")}]',
            self.__format_table_value(row["data"], 0),
            row["unit"],
        ] for row in self.table_rows]

        self.state_table = ax.table(
            cellText=cells,
            colWidths=[0.45, 0.45, 0.10],
            cellLoc="left",
            bbox=[0, 0, 1, 0.9],
        )
        for cell in self.state_table.get_celld().values():
            cell.set_linewidth(0.2)

        self.state_table.auto_set_font_size(False)
        self.state_table.set_fontsize(8)

        self.__update_table(self.times[0])

    def plot_right_side(self, figure, slot):
        grid = slot.subgridspec(2, 1, height_ratios=[8, 1])

        # Top right: 3D trajectory.
        ax_trajectory = figure.add_subplot(grid[0, 0], projection="3d")
        self.__plot_trajectory(ax_trajectory)

        ax_info = figure.add_subplot(grid[1, 0])
        self.__plot_info(ax_info)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def __update_table(self, time):
        index = np.argmin(np.abs(np.asarray(self.times) - time))

        for row_index, settings in enumerate(self.table_rows):
            text = self.__format_table_value(settings["data"], index)
            self.state_table[row_index, 1].get_text().set_text(text)

        self.info_ax.set_title(f"State at t = {self.times[index]:.2f} s")

    def __update_time_line(self, time):
        for line in self.snap.time_lines:
            line.set_xdata([time, time])

    def __update_position(self, index):
        self.snap.position = self.positions[index].copy()
        x, y, z = self.snap.position

        self.snap.position_point.set_data_3d([x], [y], [z])

        self.snap.position_text.set_position((x, y))
        self.snap.position_text.set_3d_properties(z + 20, zdir=None)
        self.snap.position_text.set_text(
            f"Payload [{x:.0f}, {y:.0f}, {z:.0f}]")

    def __update(self, time):
        index = np.argmin(np.abs(self.times - time))
        selected_time = self.times[index]

        self.__update_time_line(selected_time)
        self.__update_position(index)
        self.__update_table(selected_time)

        # Refresh the entire figure.
        self.snap.position_point.figure.canvas.draw_idle()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def __plot_init(self):
        figure = plt.figure(figsize=(16, 8))
        figure.canvas.manager.set_window_title(self.window_title)
        figure.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.05,
            top=0.95,
            wspace=0.1,
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
