"""Optional Matplotlib view of the completed P1 matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .report_p1 import GROUPS, SEEDS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    if report["missing"] or report["incomplete"]:
        raise ValueError("plot requires the complete predeclared run matrix")
    colors = ("#0072b2", "#009e73", "#d55e00", "#9b59b6")
    labels = ("C0 = 0.5", "C0 = 1", "C0 = 2", "Asymmetric moves")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for axis, observable, title in zip(
        axes.flat,
        ("N", "E", "exchange_particles"),
        ("Mean particle number", "Total energy", "Particles in exchange cycles"),
    ):
        for index, (group, color) in enumerate(zip(GROUPS, colors)):
            for offset, seed in zip((-0.15, 0, 0.15), SEEDS):
                row = report["runs"][f"{group}_s{seed}"]["comparisons"][observable]
                se = row["blocking"]["standard_error"]
                axis.errorbar(
                    index + offset,
                    row["estimate"],
                    yerr=se,
                    fmt="o",
                    ms=5,
                    color=color,
                    capsize=3,
                    alpha=0.85,
                )
        reference = report["groups"]["c1"][observable]["reference"]
        axis.axhline(
            reference,
            color="#333333",
            linestyle="--",
            linewidth=1.3,
            label=f"Independent reference: {reference:.4f}",
        )
        axis.set_xticks(range(4), labels, fontsize=9)
        axis.set_title(title, loc="left", weight="bold")
        axis.legend(fontsize=9, frameon=False)
        axis.grid(axis="y", alpha=0.2)
    axis = axes.flat[3]
    names = ("P0", "P1", "P2", "P3", "P4", "Pge5")
    x = np.arange(len(names))
    for index, (group, color, label) in enumerate(zip(GROUPS, colors, labels)):
        rows = [report["groups"][group][name] for name in names]
        axis.errorbar(
            x + (index - 1.5) * 0.13,
            [row["equal_seed_mean"] for row in rows],
            yerr=[row["within_chain_SE"] for row in rows],
            color=color,
            fmt="o",
            ms=4,
            capsize=2,
            label=label,
        )
    axis.plot(
        x,
        [report["groups"]["c1"][name]["reference"] for name in names],
        "_",
        color="#333333",
        ms=15,
        mew=2,
        label="Independent reference",
    )
    axis.set_xticks(x, ("0", "1", "2", "3", "4", ">=5"))
    axis.set_xlabel("Particle number N")
    axis.set_ylabel("Probability")
    axis.set_title(
        "Particle-number distribution (three-seed mean)", loc="left", weight="bold"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "P1: full Worm chain vs finite-box ideal Bose gas\n"
        "L=4, beta=1, lambda=0.5, mu=-0.7, M=8; error bars: 1 blocking SE",
        fontsize=12,
        weight="bold",
    )
    output = args.root / "comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
