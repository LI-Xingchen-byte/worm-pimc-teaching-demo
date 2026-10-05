"""Plot every predeclared seed and finite-bin reference for the P2 pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    seeds = (2718, 3141, 5772)
    runs = []
    for seed in seeds:
        directory = args.root / f"s{seed}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        samples = np.load(directory / "samples.npy", allow_pickle=False)
        if hashlib.sha256(samples.tobytes()).hexdigest() != result["samples_sha256"]:
            raise ValueError("raw sample hash mismatch")
        if result["status"] != "COMPLETED":
            raise ValueError(f"seed {seed} did not complete; do not silently omit it")
        runs.append(result)
    summary = {
        "seeds": list(seeds),
        "purpose": "exploratory; not a precision certification",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, quantity in zip(axes, ("g2", "structure_factor")):
        rows = runs[0]["correlations"][quantity]
        x = (
            np.array([(row["left_edge"] + row["right_edge"]) / 2 for row in rows])
            if quantity == "g2"
            else np.array([row["mode"] for row in rows])
        )
        reference = runs[0]["ideal_reference"][quantity]
        axis.plot(
            x,
            reference,
            "k--",
            marker="_",
            markersize=10,
            label=(
                "Ideal reference (bin average)"
                if quantity == "g2"
                else "Ideal reference"
            ),
        )
        for seed_index, (run, color) in enumerate(
            zip(runs, ("#0072b2", "#d55e00", "#009e73"))
        ):
            estimates = run["correlations"][quantity]
            for i, row in enumerate(estimates):
                error = row["standard_error"]
                axis.errorbar(
                    x[i] + (seed_index - 1) * 0.025,
                    row["mean"],
                    yerr=error,
                    marker="o" if error is not None else "x",
                    ms=4,
                    color=color,
                    capsize=3,
                    label=f"seed {run['seed']}" if i == 0 else None,
                )
        aggregate = []
        for i, exact in enumerate(reference):
            values = [run["correlations"][quantity][i]["mean"] for run in runs]
            errors = [
                run["correlations"][quantity][i]["standard_error"] for run in runs
            ]
            aggregate.append(
                {
                    "x": float(x[i]),
                    "reference": exact,
                    "equal_seed_mean": (
                        float(np.mean(values))
                        if all(v is not None for v in values)
                        else None
                    ),
                    "within_chain_SE": (
                        float(np.sqrt(np.sum(np.square(errors))) / len(seeds))
                        if all(e is not None for e in errors)
                        else None
                    ),
                    "between_seed_SE": (
                        float(np.std(values, ddof=1) / np.sqrt(len(seeds)))
                        if all(v is not None for v in values)
                        else None
                    ),
                }
            )
        summary[quantity] = aggregate
        axis.set_xlabel(
            "Periodic separation r" if quantity == "g2" else "Mode l  (k = 2 pi l / L)"
        )
        axis.set_ylabel("g2(r), ordered pairs" if quantity == "g2" else "S(k)")
        axis.set_title(
            "Pair correlation" if quantity == "g2" else "Static structure factor",
            loc="left",
            weight="bold",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[1].axhline(1, color="gray", linewidth=0.8, alpha=0.6)
    axes[1].set_xticks([1, 2, 3])
    fig.suptitle(
        "Exploratory ideal Bose gas: L=4, beta=1, mu=-0.7, M=8\n"
        "3 seeds; 147,456 attempts each; error bars: 1 blocking SE; x: SE unavailable",
        fontsize=11,
    )
    fig.savefig(args.root / "correlations.png", dpi=170)
    plt.close(fig)
    summary["total_sampling_seconds"] = sum(run["elapsed_seconds"] for run in runs)
    summary["throughput_range"] = [
        min(run["attempts_per_second"] for run in runs),
        max(run["attempts_per_second"] for run in runs),
    ]
    summary["unavailable_errors"] = [
        (run["seed"], name, i)
        for run in runs
        for name in ("g2", "structure_factor")
        for i, row in enumerate(run["correlations"][name])
        if row["standard_error"] is None
    ]
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
