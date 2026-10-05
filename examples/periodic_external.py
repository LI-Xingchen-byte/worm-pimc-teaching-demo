"""Periodic Fourier field and optional density measurement; a teaching pilot."""

import argparse
from pathlib import Path

from wormpimc import Simulation, SimulationConfig
from wormpimc.estimators import DensityProfile
from wormpimc.output import write_json_atomic
from wormpimc.potentials import external_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="use a TOML instead of the demo model"
    )
    parser.add_argument(
        "--sweeps", type=int, default=1024, help="demo measurement sweeps"
    )
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, help="save results, paired checkpoints and PNGs"
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="run without Matplotlib")
    args = parser.parse_args()
    config = (
        SimulationConfig.from_toml(args.config)
        if args.config
        else SimulationConfig.small_system(
            n_slices=8,
            chemical_potential=-0.7,
            particle_count=1,
            external="fourier",
            external_offset=0.5,
            external_cosine=(-0.5,),
            seed=2718,
            warmup_sweeps=128,
            measurement_sweeps=args.sweeps,
            steps_per_sweep=8,
        )
    )
    simulation = Simulation(config, verify_local_delta=True)
    density = DensityProfile(config, spatial_bins=args.bins)
    simulation.advance(after_sweep=density.after_sweep)
    result = density.results()
    print("Periodic Fourier external field demonstration")
    print("Z measurements:", result["n_z_measurements"])
    print("Integral rho dx = <N>:", result["integrated_density"])
    for row in result["rows"]:
        print(
            "rho",
            row["left_edge"],
            row["right_edge"],
            row["mean"],
            row["standard_error"],
            row["uncertainty_status"],
        )
    print(
        "Teaching pilot: missing error bars and a short chain do not establish convergence."
    )
    external = external_from_config(config)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "input.toml").write_text(config.to_toml(), encoding="utf-8")
        simulation.checkpoint(args.output / "simulation.npz")
        write_json_atomic(args.output / "density_snapshot.json", density.snapshot())
        write_json_atomic(args.output / "density.json", result)
    if args.no_plot:
        return
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from worldline_plot import plot_density, plot_step

    figure = plot_density(density, external)
    if args.output:
        figure.savefig(args.output / "density.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(figure)
    # A subsequent demonstration step is not part of the saved measured history.
    event = simulation.step(trace=True)
    figure = plot_step(event, tau=config.tau, external=external)
    if args.output:
        figure.savefig(args.output / "update.png", dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
