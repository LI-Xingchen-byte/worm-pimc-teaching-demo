"""Reconstruct topology proposal densities from graphs, without patch metadata.

Only raw state fields are consumed; no production move or propagator helpers.
The Gaussian image sum retains exponents out to at least 50 plus two images.
This is a numerical 1D audit, not a new implementation of simulation moves.
"""

import math


REVERSE = {
    "open": "close",
    "close": "open",
    "insert": "remove",
    "remove": "insert",
    "advance": "recede",
    "recede": "advance",
    "swap": "swap",
}


def logsum(values):
    top = max(values)
    return top + math.log(math.fsum(math.exp(value - top) for value in values))


def endpoint_log(state, start, end, links, config):
    length = state.box_length
    duration = links * config.tau
    diffusion = config.system.lambda_kin
    displacement = float(state.positions[end, 0] - state.positions[start, 0])
    center = round(-displacement / length)
    width = math.ceil(math.sqrt(4 * diffusion * duration * 50) / length) + 2
    return logsum(
        [
            -((displacement + image * length) ** 2) / (4 * diffusion * duration)
            - 0.5 * math.log(4 * math.pi * diffusion * duration)
            for image in range(center - width, center + width + 1)
        ]
    )


def link_log(state, bead, config):
    next_bead = int(state.next_of[bead])
    if next_bead < 0:
        raise ValueError("attempted to read absent link")
    displacement = float(
        state.positions[next_bead, 0]
        - state.positions[bead, 0]
        + state.image_to_next[bead, 0] * state.box_length
    )
    diffusion_time = config.system.lambda_kin * config.tau
    return -0.5 * math.log(4 * math.pi * diffusion_time) - displacement**2 / (
        4 * diffusion_time
    )


def segment(state, start, links, config):
    values = []
    end = start
    for _ in range(links):
        values.append(link_log(state, end, config))
        end = int(state.next_of[end])
    return end, math.fsum(values)


def swap_table(state, config):
    """Enumerate eligible endpoints by raw predecessor traversal."""
    links = config.moves.max_segment_links
    target_slice = (int(state.slice_of[state.worm_head]) + links) % state.n_slices
    result = {}
    for bead in range(len(state.active)):
        if not state.active[bead] or state.slice_of[bead] != target_slice:
            continue
        previous = bead
        for _ in range(links):
            previous = int(state.prev_of[previous])
            if previous < 0:
                break
        if previous >= 0 and previous != state.worm_tail:
            result[bead] = endpoint_log(state, state.worm_head, bead, links, config)
    return result


def densities(name, before, after, config):
    """Actual forward and reverse log q, excluding move-selection weights."""
    smax = config.moves.max_segment_links
    uniform_length = -math.log(smax)
    if name in ("close", "remove", "recede"):
        backward, forward = densities(REVERSE[name], after, before, config)
        return forward, backward
    if name == "open":
        links = (
            int(after.slice_of[after.worm_tail]) - int(after.slice_of[after.worm_head])
        ) % after.n_slices
        assert 1 <= links <= smax
        endpoint, kinetic = segment(before, after.worm_head, links, config)
        assert endpoint == after.worm_tail
        return (
            uniform_length - math.log(before.number_of_beads),
            kinetic - endpoint_log(before, after.worm_head, endpoint, links, config),
        )
    if name == "insert":
        bead = after.worm_tail
        values = []
        while bead != after.worm_head:
            values.append(link_log(after, bead, config))
            bead = int(after.next_of[bead])
            if len(values) > smax:
                raise ValueError("inserted open component exceeds smax")
        return (
            math.fsum(values)
            + uniform_length
            - math.log(after.n_slices * after.box_length),
            0.0,
        )
    if name == "advance":
        links = after.number_of_beads - before.number_of_beads
        assert 1 <= links <= smax
        endpoint, kinetic = segment(after, before.worm_head, links, config)
        assert endpoint == after.worm_head
        return kinetic + uniform_length, uniform_length
    if name == "swap":
        endpoint, new_kinetic = segment(after, before.worm_head, smax, config)
        old_endpoint, old_kinetic = segment(before, after.worm_head, smax, config)
        assert endpoint == old_endpoint
        forward = swap_table(before, config)
        reverse = swap_table(after, config)
        assert endpoint in forward and endpoint in reverse
        return new_kinetic - logsum(list(forward.values())), old_kinetic - logsum(
            list(reverse.values())
        )
    raise ValueError(f"unsupported audit move: {name}")


def ideal_log_weight(state, config):
    if config.potential.pair != "none" or state.ndim != 1:
        raise ValueError("audit target is restricted to the 1D ideal gas")
    links = [
        int(bead)
        for bead in range(len(state.active))
        if state.active[bead] and state.next_of[bead] >= 0
    ]
    result = math.fsum(link_log(state, bead, config) for bead in links)
    result += config.system.chemical_potential * config.tau * len(links)
    if state.sector.value == "G":
        result += math.log(
            config.moves.worm_sector_weight
            / (state.box_length * state.n_slices * config.moves.max_segment_links)
        )
    return result
