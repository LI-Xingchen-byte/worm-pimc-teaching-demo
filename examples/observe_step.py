"""Observe real random proposals and decisions; step() does not measure."""

from wormpimc import Simulation, SimulationConfig


def main() -> None:
    simulation = Simulation(SimulationConfig.small_system(n_slices=8))
    for index in range(20):
        event = simulation.step(trace=True)
        print(
            index, event.move_name, event.before["sector"], "->", event.after["sector"]
        )
        if event.breakdown is None:
            print(" ", event.status.value, event.patch.invalid_reason)
            continue
        decision = event.breakdown
        candidate = event.candidate()  # Replays the recorded patch, never resamples.
        print("  beads:", candidate.number_of_beads, "log ratio:", decision.log_ratio)
        print(
            "  kinetic/potential/chemical/sector/proposal:",
            decision.kinetic,
            decision.potential,
            decision.chemical,
            decision.sector_measure,
            decision.proposal_selection,
        )
        print("  log(u):", decision.log_uniform, "accepted:", event.accepted)
    print(
        "Measurement opportunities:", simulation.summary()["measurement_opportunities"]
    )


if __name__ == "__main__":
    main()
