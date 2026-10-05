"""Show selected Worm moves or a random short chain. Close each figure to continue."""

import argparse
from dataclasses import replace
from pathlib import Path

from wormpimc import Simulation, SimulationConfig
from wormpimc.config import MOVE_NAMES
from wormpimc.potentials import external_from_config
from worldline_plot import plot_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moves", nargs="+", choices=("random", *MOVE_NAMES), default=["open", "close"]
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="attempt count; cycles through --moves"
    )
    parser.add_argument("--seed", type=int, help="override the configured RNG seed")
    parser.add_argument("--config", type=Path, help="optional TOML, including a periodic external field")
    parser.add_argument("--output", type=Path, help="save one PNG per attempted move")
    parser.add_argument("--no-show", action="store_true", help="render headlessly")
    parser.add_argument("--ids", action="store_true", help="label all bead IDs")
    args = parser.parse_args()
    steps = len(args.moves) if args.steps is None else args.steps
    if steps < 1:
        parser.error("--steps must be positive")
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = (
        SimulationConfig.from_toml(args.config) if args.config
        else SimulationConfig.small_system(n_slices=8)
    )
    if args.seed is not None:
        config = replace(config, run=replace(config.run, seed=args.seed))
    external = external_from_config(config) if config.potential.external != "none" else None
    simulation = Simulation(config)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    for index in range(steps):
        move = args.moves[index % len(args.moves)]
        event = simulation.step(move=None if move == "random" else move, trace=True)
        figure = plot_step(event, tau=config.tau, show_ids=args.ids, external=external)
        print(
            index + 1,
            event.move_name,
            event.status.value,
            "accepted:",
            event.accepted,
            event.selection_mode,
        )
        if args.output:
            figure.savefig(
                args.output / f"{index + 1:03d}_{event.move_name}.png", dpi=150
            )
        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
