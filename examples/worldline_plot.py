"""Optional 1D worldline drawing; only consumes recorded data, never RNG.

Keep this helper outside the solver. Install wormpimc[plot] to use plot_step.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np

from wormpimc import StepResult


def split_periodic_link(x0, x1, image, box_length, t0, t1):
    """Split an image-resolved link at every spatial seam.

    Time is already lifted across the final time slice (t1 may equal beta).
    Return straight pieces inside [0,L], preserving even multiple windings.
    The midpoint determines each piece's image, including boundary ties.
    """

    end = x1 + image * box_length
    displacement = end - x0
    cuts = [0.0, 1.0]
    if displacement:
        lower, upper = sorted((x0, end))
        for winding in range(
            math.floor(lower / box_length), math.ceil(upper / box_length) + 1
        ):
            fraction = (winding * box_length - x0) / displacement
            if 0.0 < fraction < 1.0:
                cuts.append(fraction)
    cuts.sort()
    pieces = []
    for start, stop in zip(cuts, cuts[1:]):
        image_offset = box_length * math.floor(
            (x0 + (start + stop) * displacement / 2) / box_length
        )
        pieces.append(
            (
                (x0 + start * displacement - image_offset, t0 + start * (t1 - t0)),
                (x0 + stop * displacement - image_offset, t0 + stop * (t1 - t0)),
            )
        )
    return pieces


def _links(snapshot):
    """Link identity includes endpoint coordinates and the explicit image."""

    result = {}
    for bead, active in enumerate(snapshot["active"]):
        next_bead = int(snapshot["next_of"][bead])
        if active and next_bead >= 0:
            result[bead] = (
                next_bead,
                float(snapshot["positions"][bead, 0]),
                float(snapshot["positions"][next_bead, 0]),
                int(snapshot["image_to_next"][bead, 0]),
            )
    return result


def changed_links(before, candidate):
    """Return source bead IDs whose connectivity, images or geometry change."""

    original, proposed = _links(before), _links(candidate)
    return {
        bead
        for bead in original.keys() | proposed.keys()
        if original.get(bead) != proposed.get(bead)
    }


def _draw_state(ax, snapshot, tau, changed, highlight, show_ids):
    length, slices = snapshot["box_length"], snapshot["n_slices"]
    beta = slices * tau
    ax.set(xlim=(0, length), ylim=(-0.035 * beta, 1.055 * beta), xlabel="Position x")
    ax.set_xticks([0, length / 2, length], ["0", "L/2", "L"])
    ax.set_yticks([0, beta / 2, beta], ["0", "beta/2", "beta"])
    ax.grid(alpha=0.12)
    for time in (0, beta):
        ax.axhline(time, color="#8794a4", linewidth=0.8, linestyle=":")
    for bead, (next_bead, x0, x1, image) in _links(snapshot).items():
        time = int(snapshot["slice_of"][bead]) * tau
        color = highlight if bead in changed else "#8895a7"
        for start, stop in split_periodic_link(x0, x1, image, length, time, time + tau):
            ax.plot(
                [start[0], stop[0]],
                [start[1], stop[1]],
                color=color,
                linewidth=2.4 if bead in changed else 1.15,
                zorder=2,
            )
        if snapshot["slice_of"][bead] == slices - 1:
            ax.scatter(x1, beta, s=24, facecolors="white", edgecolors=color, zorder=3)
    ids = [bead for bead, active in enumerate(snapshot["active"]) if active]
    ax.scatter(
        snapshot["positions"][ids, 0],
        snapshot["slice_of"][ids] * tau,
        s=14,
        color=[
            highlight
            if bead in changed or int(snapshot["prev_of"][bead]) in changed
            else "#536174"
            for bead in ids
        ],
        zorder=3,
    )
    if show_ids:
        for bead in ids:
            ax.annotate(
                str(bead),
                (snapshot["positions"][bead, 0], snapshot["slice_of"][bead] * tau),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="#536174",
            )
    for name, marker, color in (("head", "o", "#00897b"), ("tail", "D", "#9b458e")):
        bead = snapshot[f"worm_{name}"]
        if bead >= 0:
            x, time = (
                float(snapshot["positions"][bead, 0]),
                int(snapshot["slice_of"][bead]) * tau,
            )
            ax.scatter(
                x, time, s=90, marker=marker, color=color, edgecolors="white", zorder=5
            )
            # Slice zero and beta are the same time seam.
            if time == 0:
                ax.scatter(
                    x,
                    beta,
                    s=90,
                    marker=marker,
                    facecolors="white",
                    edgecolors=color,
                    zorder=5,
                )
            ax.annotate(
                f"{name} #{bead}",
                (x, time),
                xytext=(6 if x < length / 2 else -6, -13 if name == "tail" else 8),
                textcoords="offset points",
                ha="left" if x < length / 2 else "right",
                fontsize=8,
                color=color,
            )
    if not ids:
        ax.text(
            0.5,
            0.5,
            "Vacuum (no beads)",
            transform=ax.transAxes,
            ha="center",
            color="#536174",
        )


def plot_configuration(state, *, tau: float, show_ids: bool = False):
    """Draw one Configuration or snapshot, without sampling or displaying it."""

    snapshot = state.snapshot() if hasattr(state, "snapshot") else state
    if snapshot["positions"].shape[1] != 1:
        raise ValueError("plot_configuration currently supports only 1D configurations")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    _draw_state(ax, snapshot, tau, set(), "#287bb5", show_ids)
    ax.set(ylabel="Imaginary time", title=f"Worldline configuration  [{snapshot['sector']}]")
    return fig


def plot_step(event: StepResult, *, tau: float, show_ids: bool = False, external=None):
    """Plot before / candidate / actual after and return a Matplotlib Figure.

    Nothing is displayed or saved automatically. Invalid/inapplicable attempts
    have an explicitly empty candidate panel; rejection keeps the after state.
    """

    if event.before is None or event.after is None:
        raise ValueError("plot_step requires step(trace=True)")
    if event.before["positions"].shape[1] != 1:
        raise ValueError("plot_step currently supports only 1D configurations")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    if external is not None and external.box_length != event.before["box_length"]:
        raise ValueError("external field and trace use different box lengths")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    candidate = event.candidate()
    proposed = candidate.snapshot() if candidate is not None else None
    changed = changed_links(event.before, proposed) if proposed is not None else set()
    outcome = (
        ("ACCEPTED" if event.accepted else "REJECTED")
        if event.breakdown
        else event.status.value.upper()
    )
    mode = (
        "Specified move demo (configured acceptance weights)"
        if event.selection_mode == "specified"
        else "Random move selection"
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 7.5) if external is not None else (12, 6),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        top=0.77, bottom=0.43 if external is not None else 0.24,
        left=0.07, right=0.97, wspace=0.19,
    )
    fig.suptitle(
        f"{event.move_name.upper()}  |  {outcome}",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    fig.text(0.5, 0.91, mode, ha="center", color="#536174", fontsize=10)
    panels = (
        (event.before, "Before", "#d46b36"),
        (proposed, "Candidate", "#287bb5"),
        (event.after, "After decision", "#287bb5" if event.accepted else "#d46b36"),
    )
    for ax, (snapshot, title, highlight) in zip(axes, panels):
        if snapshot is None:
            ax.text(
                0.5,
                0.5,
                "No valid proposal",
                transform=ax.transAxes,
                ha="center",
                color="#536174",
            )
            ax.set_xlabel("Position x")
        else:
            _draw_state(ax, snapshot, tau, changed, highlight, show_ids)
            title += f"  [{snapshot['sector']}]"
        ax.set_title(title, fontsize=11, pad=11)
    if external is not None:
        x = np.linspace(0.0, external.box_length, 401)
        values = external.energy(x)
        for ax in axes:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
            bounds = ax.get_position()
            field_ax = fig.add_axes([bounds.x0, 0.255, bounds.width, 0.105], sharex=ax)
            field_ax.plot(x, values, color="#7152a3", linewidth=1.7)
            field_ax.set(xlabel="Position x", ylabel="V_ext(x)")
            field_ax.tick_params(labelsize=8)
            field_ax.grid(alpha=0.15)
    axes[0].set_ylabel("Imaginary time")
    fig.legend(
        handles=[
            Line2D([], [], color="#8895a7", label="Unchanged"),
            Line2D([], [], color="#d46b36", linewidth=2.4, label="Old / removed links"),
            Line2D([], [], color="#287bb5", linewidth=2.4, label="Proposed links"),
            Line2D([], [], color="#00897b", marker="o", linestyle="none", label="Head"),
            Line2D([], [], color="#9b458e", marker="D", linestyle="none", label="Tail"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    if event.breakdown:
        b = event.breakdown
        detail = (
            f"log R = {b.log_ratio:.4g}    |    log u = {b.log_uniform:.4g}    |    "
            f"log u < min(0, log R): {event.accepted}"
        )
        terms = (
            f"kinetic {b.kinetic:+.4g}    potential {b.potential:+.4g}    chemical {b.chemical:+.4g}    "
            f"sector {b.sector_measure:+.4g}    proposal/selection {b.proposal_selection:+.4g}"
        )
    else:
        detail = "\n".join(
            textwrap.wrap(
                event.patch.invalid_reason if event.patch else "No proposal", 110
            )
        )
        terms = "No Metropolis draw; state unchanged."
    fig.text(0.5, 0.13, detail, ha="center", fontsize=10)
    fig.text(0.5, 0.085, terms, ha="center", fontsize=9, color="#536174")
    fig.text(
        0.5,
        0.025,
        "x = 0 and L are identified; time = 0 and beta are identified. Hollow top beads repeat slice 0.",
        ha="center",
        fontsize=8,
        color="#536174",
    )
    return fig


def plot_density(profile, external):
    """Separate aligned field and density axes; short-chain SEs stay missing."""
    if profile.config.system.box_length != external.box_length:
        raise ValueError("external field and density profile use different box lengths")
    import matplotlib.pyplot as plt

    result = profile.results()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True, layout="constrained")
    x = np.linspace(0.0, external.box_length, 401)
    axes[0].plot(x, external.energy(x), color="#7152a3")
    axes[0].set(ylabel="V_ext(x) [energy]", title="Periodic field and spatial density")
    rows = result["rows"]
    centers = [(row["left_edge"] + row["right_edge"]) / 2 for row in rows]
    density = [np.nan if row["mean"] is None else row["mean"] for row in rows]
    axes[1].stairs(density, profile.bin_edges, color="#287bb5", linewidth=1.8)
    valid = [(center, row) for center, row in zip(centers, rows)
             if row["standard_error"] is not None]
    if valid:
        axes[1].errorbar(
            [center for center, _ in valid], [row["mean"] for _, row in valid],
            yerr=[row["standard_error"] for _, row in valid],
            fmt="none", color="#287bb5", capsize=3,
        )
    integral = result["integrated_density"]
    text = "No Z reference" if integral is None else f"Integral rho dx = <N> = {integral:.4g}"
    axes[1].set(xlim=(0, external.box_length), xlabel="Position x", ylabel="rho(x) [1/length]")
    axes[1].set_title(text + "; error bars only with sufficient blocking history", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
    return fig
