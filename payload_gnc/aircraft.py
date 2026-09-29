import pfh.glidersim as gsim


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