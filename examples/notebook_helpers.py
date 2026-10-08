"""Small display helpers for worm_walkthrough.ipynb; no sampling logic."""

from __future__ import annotations

from html import escape

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from examples.worldline_plot import plot_configuration, plot_step


def display_table(headers, rows):
    """Render escaped values without adding a pandas dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            value = f"{value:.6g}"
        elif value is None:
            value = "—"
        return escape(str(value))

    head = "".join(f"<th>{cell(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    display(HTML(
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        'font-size:14px;text-align:left;margin:10px 0">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    ))


def show_configuration(state, *, tau, show_ids=True):
    figure = plot_configuration(state, tau=tau, show_ids=show_ids)
    display(figure)
    plt.close(figure)
    snapshot = state.snapshot()
    ids = np.flatnonzero(snapshot["active"])
    display_table(
        ("bead", "slice", "x", "prev", "next", "image → next"),
        [(int(i), int(snapshot["slice_of"][i]), snapshot["positions"][i, 0],
          int(snapshot["prev_of"][i]), int(snapshot["next_of"][i]),
          int(snapshot["image_to_next"][i, 0])) for i in ids],
    )


def show_step(event, *, tau, show_ids=True, details=False):
    """Display an already recorded event; never mutate state or consume RNG."""
    figure = plot_step(event, tau=tau, show_ids=show_ids)
    display(figure)
    plt.close(figure)
    patch = event.patch
    if patch is None:
        return
    moved = tuple(change.bead_id for change in patch.position_changes)
    added = tuple(bead.bead_id for bead in patch.added_beads)
    removed = tuple(bead.bead_id for bead in patch.removed_beads)
    reconnected = tuple(
        change.bead_id for change in patch.link_changes
        if change.before_next != change.after_next or change.before_prev != change.after_prev
    )
    display_table(("本次提议", "记录"), [
        ("状态 / 接受", f"{event.status.value} / {event.accepted}"),
        ("移动的 beads", moved), ("新增 / 删除的 beads", f"{added} / {removed}"),
        ("连接关系改变的 beads", reconnected),
        ("head / tail（更新前 → 候选）",
         f"({patch.endpoints_before.worm_head}, {patch.endpoints_before.worm_tail}) → "
         f"({patch.endpoints_after.worm_head}, {patch.endpoints_after.worm_tail})"),
    ])
    if details and event.breakdown is not None:
        b = event.breakdown
        display_table(("log R 的贡献", "数值"), [
            ("kinetic", b.kinetic), ("potential", b.potential),
            ("chemical", b.chemical), ("sector / measure", b.sector_measure),
            ("proposal / selection", b.proposal_selection), ("总和 log R", b.log_ratio),
            ("阈值 min(0, log R)", min(0, b.log_ratio)), ("log u", b.log_uniform),
        ])
        q = event.proposal_ratio
        if q is not None:
            display_table(("提议概率分解", "数值"), [
                ("log q forward", q.log_q_forward), ("log q reverse", q.log_q_reverse),
                ("log p(move) forward", q.log_move_selection_forward),
                ("log p(move) reverse", q.log_move_selection_reverse),
            ])
