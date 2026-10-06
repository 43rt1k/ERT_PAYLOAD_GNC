import io
from contextlib import redirect_stdout
import pfh.glidersim as gsim


def build_basic_aircraft(config, verbose=True):

    wing_config = config["wing"]

    with redirect_stdout(io.StringIO()):
        wing = gsim.extras.wings.niviuk_hook3(size=wing_config["default_size"],
                                              verbose=False)

    payload_config = config["payload"]
    payload = gsim.paraglider_harness.Spherical(
        mass=payload_config["mass"],
        z_riser=payload_config["z_riser"],
        S=payload_config["S"],
        CD=payload_config["CD"],
        kappa_w=payload_config["kappa_w"],
    )
    paraglider = gsim.paraglider.ParagliderSystemDynamics6a(wing, payload)

    if verbose:
        print(
            f"Built paraglider with wing size {wing_config['default_size']} m² and payload mass {payload_config['mass']} kg."
        )

    return paraglider


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def build_demo_aircraft():
    """Build the example aircraft, not the RocketTeam payload."""
    wing = gsim.extras.wings.niviuk_hook3(size=23, verbose=False)

    harness = gsim.paraglider_harness.Spherical(
        mass=75,
        z_riser=0.5,
        S=0.55,
        CD=0.8,
        kappa_w=0.1,
    )

    return gsim.paraglider.ParagliderSystemDynamics6a(wing, harness)

    #     UPPER_FABRIC_DENSITY = 0.039  # kg/m² = 39 g/m²
    #     LOWER_FABRIC_DENSITY = 0.035  # kg/m²
    #     RIB_FABRIC_DENSITY = 0.041  # kg/m²
    #     CELL_COUNT = 52

    # wing = gsim.paraglider_wing.ParagliderWing(
    #     canopy=canopy,
    #     lines=lines,
    #     rho_upper=wing_config.UPPER_FABRIC_DENSITY,
    #     rho_lower=wing_config.LOWER_FABRIC_DENSITY,
    #     rho_ribs=wing_config.RIB_FABRIC_DENSITY,
    #     N_cells=wing_config.CELL_COUNT,
    # )
