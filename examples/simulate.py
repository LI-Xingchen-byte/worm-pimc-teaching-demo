"""A small Worm run entirely in Python, without input or output files."""

from wormpimc import Simulation, SimulationConfig


def main() -> None:
    config = SimulationConfig.small_system(
        n_slices=8,
        warmup_sweeps=5,
        measurement_sweeps=40,
    )
    simulation = Simulation(config)
    simulation.advance()  # Warm up, then measure at the configured stride.
    result = simulation.results()
    print("In-memory Worm PIMC demonstration")
    print("Z measurements:", result["summary"]["measurement_count"])
    for name, estimate in result["scalars"].items():
        print(
            name,
            estimate["mean"],
            estimate["standard_error"],
            estimate["uncertainty_status"],
        )
    print("Short demonstration only; these estimates are not a convergence claim.")


if __name__ == "__main__":
    main()
