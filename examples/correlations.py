"""Small optional g2/S(k) measurement; deliberately not a convergence claim."""

from dataclasses import replace

from wormpimc import Simulation, SimulationConfig
from wormpimc.estimators import EqualTimeCorrelations


def main():
    config = SimulationConfig.small_system(
        n_slices=8,
        chemical_potential=-0.7,
        seed=2718,
        warmup_sweeps=128,
        measurement_sweeps=1024,
        steps_per_sweep=8,
    )
    config = replace(
        config, moves=replace(config.moves, worm_sector_weight=0.5, max_segment_links=3)
    )
    simulation = Simulation(config)
    correlations = EqualTimeCorrelations(config, spatial_bins=8, modes=(1, 2, 3))

    def measure(sim):
        if (
            sim.phase == "measurement"
            and sim.measurement_completed % config.run.measurement_stride == 0
        ):
            correlations.measure(sim.state)

    simulation.advance(after_sweep=measure)
    result = correlations.results()
    for row in result["g2"]:
        print(
            "g2",
            row["left_edge"],
            row["right_edge"],
            row["mean"],
            row["standard_error"],
            row["uncertainty_status"],
        )
    for row in result["structure_factor"]:
        print(
            "S",
            row["mode"],
            row["mean"],
            row["standard_error"],
            row["uncertainty_status"],
        )
    print("Exploratory only: None means an uncertainty is not available.")


if __name__ == "__main__":
    main()
