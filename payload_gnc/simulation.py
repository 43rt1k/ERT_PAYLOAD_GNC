import pfh.glidersim as gsim


def run_simulation(
    aircraft,
    *,
    left_brake=0.0,
    right_brake=0.0,
    wind=(0.0, 0.0, 0.0),
    air_density=1.225,
    duration=25.0,
    sample_interval=0.5,
):
    model = gsim.simulator.ParagliderStateDynamics6a(
        aircraft,
        delta_a=0.0,
        delta_bl=left_brake,
        delta_br=right_brake,
        delta_w=0.0,
        rho_air=air_density,
        v_W2e=wind,
    )

    initial_state = model.starting_equilibrium()

    times, states = gsim.simulator.simulate(
        model,
        initial_state,
        dt=sample_interval,
        T=duration,
    )

    return model, times, states