"""Sanity checks on a parsed Board: is it complete enough to trust for measurements?"""
from __future__ import annotations

import collections
import math

from ..board import Board


def _cells(bbox, size):
    x0, y0, x1, y1 = bbox
    for cx in range(int(x0 // size), int(x1 // size) + 1):
        for cy in range(int(y0 // size), int(y1 // size) + 1):
            yield cx, cy


def board_health(b: Board) -> dict:
    """Counts that reveal work-in-progress boards: unplaced parts, unrouted nets, overlapping pads."""
    unplaced = [f.ref for f in b.footprints if f.x == 0 and f.y == 0]
    copper_nets = ({s.net for s in b.segments if s.net} | {v.net for v in b.vias if v.net}
                   | {z.net for z in b.zones if z.net and z.filled})
    signal = [p for p in b.pads if p.net and p.net != "GND"]
    no_copper = [p for p in signal if p.net not in copper_nets]

    seen: set[tuple[int, int]] = set()
    grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    boxes = [p.bbox() for p in b.pads]
    for i, bx in enumerate(boxes):
        for c in _cells(bx, 2.0):
            grid[c].append(i)
    for idxs in grid.values():
        for a in range(len(idxs)):
            for c in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[c]
                pi, pj = b.pads[i], b.pads[j]
                if pi.ref == pj.ref or not set(pi.layers) & set(pj.layers):
                    continue
                A, B = boxes[i], boxes[j]
                if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                    seen.add((min(i, j), max(i, j)))
    return {
        "footprints_at_origin": unplaced,
        "signal_pads": len(signal),
        "signal_pads_on_nets_with_no_copper": len(no_copper),
        "overlapping_pad_pairs": len(seen),
    }


def pad_alignment(b: Board, search_mm: float = 1.5) -> dict[int, tuple[int, int]]:
    """For track ends that start within `search_mm` of a same-net pad, how many land on the pad?

    Returns {footprint rotation: (hits, total)}. On a routed board that is consistent with its
    footprints this is ~100% for every rotation; a low value for rotated footprints only means the
    pad transform is wrong if unrotated footprints score high.
    """
    fprot = {f.ref: int(round(f.rot_deg)) % 360 for f in b.footprints}
    by_net: dict[str, list] = collections.defaultdict(list)
    for p in b.pads:
        if p.net:
            by_net[p.net].append((p, p.bbox()))
    out: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for s in b.segments:
        if not s.net:
            continue
        for x, y in ((s.x1, s.y1), (s.x2, s.y2)):
            near = []
            for p, bx in by_net.get(s.net, []):
                if s.layer not in p.layers:
                    continue
                dx = max(bx[0] - x, 0, x - bx[2])
                dy = max(bx[1] - y, 0, y - bx[3])
                near.append((math.hypot(dx, dy), p))
            if not near:
                continue
            d, p = min(near, key=lambda t: t[0])
            if d > search_mm:
                continue
            row = out[fprot.get(p.ref, 0)]
            row[1] += 1
            row[0] += 1 if d <= s.width / 2 + 0.02 else 0
    return {k: (v[0], v[1]) for k, v in out.items()}
