from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import pfh.glidersim as gsim

from payload_gnc.aircraft import build_basic_aircraft
from payload_gnc.controllers.scripted import right_brake_sequence
from payload_gnc.plotting import Plotter
from payload_gnc.simulation import Simulator
from payload_gnc.controllers.controller import GN_Controller
from payload_gnc.logger import Log





FILE_NAME = Path(__file__).name

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

    config = load_config(project_root_path / "payload_gnc" / "configs" / "simulation.toml")

    return project_root_path, output_path, config





# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

project_root_path, output_path, config = init_config()

VRB_ID = "––MAI | "





def main():

    AIR = build_basic_aircraft(config["aircraft"])

    LOG = Log()
    GNC = GN_Controller(LOG, config["controller"], config["simulation"]["init_pos"])

    SIM = Simulator(AIR, configs=config["simulation"], dt=config["controller"]["dt"], GNC=GNC)

    SIM.run()
    LOG.pad_all()

    plotter = Plotter(log=LOG, output_path=output_path, file_name=FILE_NAME)
    plotter.plot()





# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == "__main__":
    main()
