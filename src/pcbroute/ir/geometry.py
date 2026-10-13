"""Integer geometry primitives for the IR. All coordinates are integer nanometres.

KiCad stores millimetres with at most 6 decimals, which is exactly 1 nm, so nothing is lost on import
and the IR can be hashed, compared, and snapped to a manufacturing grid without float noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

NM_PER_MM = 1_000_000


def mm_to_nm(v: float) -> int:
    return int(round(v * NM_PER_MM))


def nm_to_mm(v: int | float) -> float:
    return v / NM_PER_MM


@dataclass(frozen=True)
class Pt:
    x: int
    y: int

    def mm(self) -> tuple[float, float]:
        return (self.x / NM_PER_MM, self.y / NM_PER_MM)

    def snapped(self, grid_nm: int) -> "Pt":
        """Nearest point on a manufacturing grid with pitch `grid_nm`."""
        return Pt(int(round(self.x / grid_nm)) * grid_nm, int(round(self.y / grid_nm)) * grid_nm)


def pt_mm(x: float, y: float) -> Pt:
    return Pt(mm_to_nm(x), mm_to_nm(y))


Ring = tuple[Pt, ...]          # closed implicitly: the last point connects back to the first


@dataclass(frozen=True)
class Polygon:
    outer: Ring
    holes: tuple[Ring, ...] = ()


def ring_area2(ring: Ring) -> int:
    """Twice the signed area in nm^2 (positive for counter-clockwise in a y-up frame). Exact integer math."""
    return sum(a.x * b.y - b.x * a.y for a, b in zip(ring, ring[1:] + ring[:1]))


def ring_bbox(ring: Ring) -> tuple[int, int, int, int]:
    xs = [p.x for p in ring]
    ys = [p.y for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_ring(x: float, y: float, ring: Ring) -> bool:
    """Even-odd ray casting. Points exactly on an edge may fall either way."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = ring[i].x, ring[i].y, ring[j].x, ring[j].y
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    return point_in_ring(x, y, poly.outer) and not any(point_in_ring(x, y, h) for h in poly.holes)


def circle_ring(cx: int, cy: int, r: int, n: int = 24) -> Ring:
    return tuple(Pt(int(round(cx + r * math.cos(2 * math.pi * i / n))),
                    int(round(cy + r * math.sin(2 * math.pi * i / n)))) for i in range(n))


def rotated_rect_ring(cx: int, cy: int, w: int, h: int, rot_deg: float) -> Ring:
    """Rectangle of size w x h centred on (cx, cy), rotated the KiCad way (CCW on screen, y down)."""
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x, y = sx * w / 2, sy * h / 2
        pts.append(Pt(int(round(cx + x * c + y * s)), int(round(cy - x * s + y * c))))
    return tuple(pts)


def chain_segments(segs: list[tuple[tuple[int, int], tuple[int, int]]], tol_nm: int = 1000
                   ) -> tuple[list[Ring], int]:
    """Join loose line segments into closed rings. Returns (rings, number_of_open_chains).

    Endpoints within `tol_nm` of each other are treated as the same point (checked across neighbouring
    hash bins, so points that straddle a bin boundary still match).
    """
    bins: dict[tuple[int, int], list[tuple[int, int]]] = {}     # bin -> [(segment index, end 0|1)]
    for i, seg in enumerate(segs):
        for e in (0, 1):
            p = seg[e]
            bins.setdefault((p[0] // tol_nm, p[1] // tol_nm), []).append((i, e))

    def near(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= tol_nm * tol_nm

    def find_unused(p, used):
        bx, by = p[0] // tol_nm, p[1] // tol_nm
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i, e in bins.get((bx + dx, by + dy), []):
                    if not used[i] and near(segs[i][e], p):
                        return i, e
        return None

    used = [False] * len(segs)
    rings: list[Ring] = []
    open_chains = 0
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        pts = [segs[start][0], segs[start][1]]
        closed = False
        while True:
            if len(pts) > 2 and near(pts[-1], pts[0]):
                closed = True
                break
            hit = find_unused(pts[-1], used)
            if hit is None:
                break
            i, e = hit
            used[i] = True
            pts.append(segs[i][1 - e])
        if closed:
            rings.append(tuple(Pt(*p) for p in pts[:-1]))
        else:
            open_chains += 1
    return rings, open_chains
