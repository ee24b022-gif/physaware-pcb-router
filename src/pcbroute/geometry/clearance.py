"""Clearance expansion.

A trace of width w needs a copper gap of at least c to every other-net feature. Equivalently its centreline, a
zero-width path, must stay at least w/2 + c from the other feature's copper. `InflatedObstacle` carries exactly that
radius, so A* and maze routers test centrelines against inflated obstacles with the exact predicates below.

Half widths round up (w=251 nm counts as 126 nm), so every inflation is conservative by at most 0.5 nm.
Inflation is represented by an exact distance predicate (the Minkowski sum with a disc), and `outline()` gives an outer
polygon approximation with integer vertices for rasterising or drawing.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ir import Design
from .shapes import Point, PolyCore, SegCore, closer_than, convex_hull
from .spatial_index import BBox, Item


def half_width_nm(width_nm: int) -> int:
    return (width_nm + 1) // 2


def inflation_nm(width_nm: int, clearance_nm: int) -> int:
    """w/2 + c, rounded up to a whole nanometre."""
    return half_width_nm(width_nm) + clearance_nm


@dataclass(frozen=True)
class Violation:
    code: str                  # clearance | keepout | board_edge | hole_clearance | hole_to_hole | outside_board | ...
    layer: str                 # layer id, or "*" for board-level findings
    obj_id: str                # the object that was hit (empty for board-level findings)
    message: str
    required_nm: int | None = None
    gap_nm: int | None = None  # actual copper gap, floored; negative means overlap
    severity: str = "error"    # "error" | "warning"


# Unit vectors of a regular 16-gon, scaled by 1e9 and rounded up in magnitude, so the polygon built from
# them circumscribes the disc. Constants only: no floating point at runtime.
_C0, _C1, _C2, _C3 = 1_000_000_000, 923_879_533, 707_106_782, 382_683_433
_UNIT = ((_C0, 0), (_C1, _C3), (_C2, _C2), (_C3, _C1), (0, _C0), (-_C3, _C1), (-_C2, _C2), (-_C1, _C3),
         (-_C0, 0), (-_C1, -_C3), (-_C2, -_C2), (-_C3, -_C1), (0, -_C0), (_C3, -_C1), (_C2, -_C2), (_C1, -_C3))
_SEC16 = 1_019_591_159         # sec(pi/16) * 1e9, rounded up
_E9 = 1_000_000_000


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def disc_polygon(c: Point, r: int) -> list[Point]:
    """16-gon with integer vertices that contains the disc of radius r around c."""
    if r <= 0:
        return [c]
    rc = _ceil_div(r * _SEC16, _E9)
    out = []
    for ux, uy in _UNIT:
        dx = _ceil_div(rc * ux, _E9) if ux >= 0 else -_ceil_div(rc * -ux, _E9)
        dy = _ceil_div(rc * uy, _E9) if uy >= 0 else -_ceil_div(rc * -uy, _E9)
        out.append((c[0] + dx, c[1] + dy))
    return out


@dataclass(frozen=True)
class InflatedObstacle:
    item: Item
    radius: int                # item copper radius + w/2 + required clearance

    @property
    def bbox(self) -> BBox:
        return self.item.bbox.inflate(self.radius - self.item.radius)

    def blocks_point(self, p: Point) -> bool:
        """A trace centre at p would violate clearance (strictly inside the inflated boundary)."""
        return closer_than(SegCore(p, p), 0, self.item.core, self.radius)

    def blocks_segment(self, a: Point, b: Point) -> bool:
        """A trace centreline from a to b would violate clearance."""
        return closer_than(SegCore(a, b), 0, self.item.core, self.radius)

    def outline(self) -> tuple[Point, ...]:
        """Outer polygon approximation of the inflated shape (convex hull; concave obstacles are filled in)."""
        c = self.item.core
        verts = [c.a, c.b] if isinstance(c, SegCore) else list(c.ring)
        pts: list[Point] = []
        for v in verts:
            pts += disc_polygon(v, self.radius)
        return convex_hull(pts)


class ClearanceModel:
    """Resolves the exact clearance rule between a net and any indexed item."""

    def __init__(self, design: Design, edge_clearance_nm: int | None = None, hole_clearance_nm: int | None = None):
        self.min_nm = design.rules.min_clearance_nm
        self._net = {n.name: n.clearance_nm for n in design.nets}
        self.edge_nm = self.min_nm if edge_clearance_nm is None else edge_clearance_nm
        self.hole_nm = self.min_nm if hole_clearance_nm is None else hole_clearance_nm
        self._max = max([self.min_nm, self.edge_nm, self.hole_nm, *self._net.values()])

    def net_clearance(self, net: str | None) -> int:
        return self._net.get(net, 0) if net else 0

    def max_clearance(self) -> int:
        return self._max

    def required(self, net: str | None, other: Item) -> int:
        """Copper gap a feature of `net` must keep from `other`: the larger of the two nets' class clearances."""
        if other.kind == "edge":
            return self.edge_nm
        if other.kind == "hole":
            return self.hole_nm
        if other.kind == "keepout":
            return 0
        return max(self.min_nm, self.net_clearance(net), self.net_clearance(other.net))

    def inflate(self, item: Item, width_nm: int, net: str | None) -> InflatedObstacle:
        return InflatedObstacle(item, item.radius + inflation_nm(width_nm, self.required(net, item)))
