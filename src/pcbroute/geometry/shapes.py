"""Exact integer geometry for the kernel. Every decision is made with integer arithmetic on nanometres.

A *core* is the zero-width skeleton of a copper feature: a segment (a point is a degenerate segment) or a
polygon region (with optional holes). A feature's copper is its core grown by a *radius* (half the trace width,
half the via diameter, 0 for polygons). Distances between cores are kept as exact rationals num/den, so
"is this closer than R" is the integer test num < R^2 * den, with no square roots and no rounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isqrt

Point = tuple[int, int]


@dataclass(frozen=True)
class SegCore:
    a: Point
    b: Point


@dataclass(frozen=True)
class PolyCore:
    ring: tuple[Point, ...]
    holes: tuple[tuple[Point, ...], ...] = ()

    @cached_property
    def _locators(self):
        """Banded edge tables, built once per polygon: big copper pours have tens of thousands of vertices."""
        return _RingLocator(self.ring), tuple(_RingLocator(h) for h in self.holes)


Core = SegCore | PolyCore


def cross(o: Point, a: Point, b: Point) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _sgn(v: int) -> int:
    return (v > 0) - (v < 0)


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Closed segments ab and cd share at least one point (touching counts)."""
    d1, d2 = _sgn(cross(a, b, c)), _sgn(cross(a, b, d))
    d3, d4 = _sgn(cross(c, d, a)), _sgn(cross(c, d, b))
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and _on_segment(a, b, c)) or (d2 == 0 and _on_segment(a, b, d))
            or (d3 == 0 and _on_segment(c, d, a)) or (d4 == 0 and _on_segment(c, d, b)))


# ---- squared distances as exact fractions (num, den), den > 0 ----
Frac = tuple[int, int]


def _less(x: Frac, y: Frac) -> bool:
    return x[0] * y[1] < y[0] * x[1]


def pt_seg_d2(p: Point, a: Point, b: Point) -> Frac:
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    l2 = abx * abx + aby * aby
    if l2 == 0:
        return (apx * apx + apy * apy, 1)
    t = apx * abx + apy * aby
    if t <= 0:
        return (apx * apx + apy * apy, 1)
    if t >= l2:
        bpx, bpy = p[0] - b[0], p[1] - b[1]
        return (bpx * bpx + bpy * bpy, 1)
    cr = apx * aby - apy * abx
    return (cr * cr, l2)


def seg_seg_d2(a: Point, b: Point, c: Point, d: Point) -> Frac:
    if segments_intersect(a, b, c, d):
        return (0, 1)
    best = pt_seg_d2(a, c, d)
    for cand in (pt_seg_d2(b, c, d), pt_seg_d2(c, a, b), pt_seg_d2(d, a, b)):
        if _less(cand, best):
            best = cand
    return best


def ring_locate(p: Point, ring: tuple[Point, ...]) -> int:
    """1 strictly inside, 0 on the boundary, -1 outside. Exact (even-odd)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        a, b = ring[j], ring[i]
        cr = cross(a, b, p)
        if cr == 0 and _on_segment(a, b, p):
            return 0
        if (a[1] > p[1]) != (b[1] > p[1]) and (cr > 0) == (b[1] > a[1]):
            inside = not inside
        j = i
    return 1 if inside else -1


class _RingLocator:
    """Exact point location in a ring, scanning only the edges whose y-range contains the query's y."""

    def __init__(self, ring: tuple[Point, ...]):
        self.ring = ring
        n = len(ring)
        ys = [p[1] for p in ring]
        xs = [p[0] for p in ring]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        nb = max(1, isqrt(n))
        self.h = max(1, (self.y1 - self.y0) // nb + 1)
        self.bands: list[list[tuple[Point, Point]]] = [[] for _ in range((self.y1 - self.y0) // self.h + 1)]
        for i in range(n):
            a, b = ring[i - 1], ring[i]
            lo, hi = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
            for bi in range((lo - self.y0) // self.h, (hi - self.y0) // self.h + 1):
                self.bands[bi].append((a, b))

    def locate(self, p: Point) -> int:
        if p[1] < self.y0 or p[1] > self.y1 or p[0] < self.x0 or p[0] > self.x1:
            return -1
        inside = False
        for a, b in self.bands[(p[1] - self.y0) // self.h]:
            cr = cross(a, b, p)
            if cr == 0 and _on_segment(a, b, p):
                return 0
            if (a[1] > p[1]) != (b[1] > p[1]) and (cr > 0) == (b[1] > a[1]):
                inside = not inside
        return 1 if inside else -1


def point_in_polygon(p: Point, poly: PolyCore) -> bool:
    """Copper region test: inside or on the boundary of the outer ring, and not strictly inside a hole."""
    outer, holes = poly._locators
    if outer.locate(p) < 0:
        return False
    return not any(h.locate(p) > 0 for h in holes)


def _edges(poly: PolyCore):
    for ring in (poly.ring, *poly.holes):
        n = len(ring)
        for i in range(n):
            yield ring[i - 1], ring[i]


def seg_poly_d2(a: Point, b: Point, poly: PolyCore) -> Frac:
    if point_in_polygon(a, poly) or point_in_polygon(b, poly):
        return (0, 1)
    best: Frac | None = None
    for c, d in _edges(poly):
        cand = seg_seg_d2(a, b, c, d)
        if cand[0] == 0:
            return cand
        if best is None or _less(cand, best):
            best = cand
    return best if best is not None else (0, 1)


def poly_poly_d2(p: PolyCore, q: PolyCore) -> Frac:
    if any(point_in_polygon(v, q) for v in p.ring) or any(point_in_polygon(v, p) for v in q.ring):
        return (0, 1)
    best: Frac | None = None
    for a, b in _edges(p):
        for c, d in _edges(q):
            cand = seg_seg_d2(a, b, c, d)
            if cand[0] == 0:
                return cand
            if best is None or _less(cand, best):
                best = cand
    return best if best is not None else (0, 1)


def d2(x: Core, y: Core) -> Frac:
    if isinstance(x, SegCore):
        return seg_seg_d2(x.a, x.b, y.a, y.b) if isinstance(y, SegCore) else seg_poly_d2(x.a, x.b, y)
    return seg_poly_d2(y.a, y.b, x) if isinstance(y, SegCore) else poly_poly_d2(x, y)


def closer_than(x: Core, rx: int, y: Core, ry: int, gap: int = 0) -> bool:
    """True when the copper gap between (x grown by rx) and (y grown by ry) is smaller than `gap`.

    Exactly touching at the required gap is legal (returns False).
    """
    num, den = d2(x, y)
    r = rx + ry + gap
    return num < r * r * den if r > 0 else False


def touches(x: Core, rx: int, y: Core, ry: int, gap: int = 0) -> bool:
    """True when the grown shapes overlap or touch, i.e. gap between them is <= `gap`."""
    num, den = d2(x, y)
    r = rx + ry + gap
    return num <= r * r * den


def dist_floor(x: Core, y: Core) -> int:
    """floor(distance between cores) in nm; for reporting, never for decisions."""
    num, den = d2(x, y)
    return isqrt(num // den)


def core_bbox(c: Core) -> tuple[int, int, int, int]:
    pts = (c.a, c.b) if isinstance(c, SegCore) else c.ring
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def convex_hull(points: list[Point]) -> tuple[Point, ...]:
    """Andrew's monotone chain; exact. Counter-clockwise in a y-up frame."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return tuple(pts)
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])
