import matplotlib.pyplot as plt
import pfh.glidersim as gsim


def plot_trajectory(model, times, states):
    points = gsim.extras.simulation.sample_paraglider_positions(
        model, states, times
    )

    gsim.extras.plots.plot_3d_simulation_path(**points)

    return plt.gcf()