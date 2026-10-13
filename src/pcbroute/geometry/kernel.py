"""GeometryKernel: one object built from the canonical IR that answers every spatial question.

All stored coordinates and all decisions are integer nanometres. The only floating point is used once, at build time,
to flatten track arcs and to rotate pads by angles that are not multiples of 90 degrees; both round to whole
nanometres and err on the conservative side (arc pieces carry an extra radius equal to the flattening tolerance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..ir import Design, PadShape, Pin, PinKind, ViaKind, ZoneKind
from ..ir.copper import TrackArc, TrackSeg, Via
from .clearance import ClearanceModel, InflatedObstacle, Violation, half_width_nm
from .shapes import (Core, Point, PolyCore, SegCore, closer_than, dist_floor, ring_locate)
from .spatial_index import BBox, Item, LayeredIndex
from .via_legality import ViaCheck, ViaLegality


@dataclass(frozen=True)
class KernelConfig:
    cell_nm: int = 2_000_000                  # spatial-grid cell size
    arc_tolerance_nm: int = 1_000             # max sagitta when flattening arcs; also added to arc radii
    edge_clearance_nm: int | None = None      # copper to board edge; default = rules.min_clearance_nm
    hole_clearance_nm: int | None = None      # copper to drilled hole; default = rules.min_clearance_nm
    hole_to_hole_nm: int | None = None        # drill wall to drill wall; default = rules.min_clearance_nm
    allow_blind_buried: bool = False          # beyond the via types listed in the rules
    allow_plane_cuts: bool = True             # different-net vias may pass through a plane with an antipad


def _rot(x: int, y: int, deg: float) -> Point:
    """Rotate the KiCad way (CCW on screen, y down). Exact for multiples of 90 degrees."""
    d = deg % 360.0
    if d == 0.0:
        return x, y
    if d == 90.0:
        return y, -x
    if d == 180.0:
        return -x, -y
    if d == 270.0:
        return -y, x
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return round(x * c + y * s), round(-x * s + y * c)


def pad_geometry(pin: Pin) -> tuple[Core, int]:
    """Core and copper radius of a pad. Round and oval pads are exact; rounded rectangles are treated as rectangles."""
    cx, cy = pin.position.x, pin.position.y
    w, h = pin.width_nm, pin.height_nm
    if pin.shape == PadShape.CIRCLE:
        return SegCore((cx, cy), (cx, cy)), (max(w, h) + 1) // 2
    if pin.shape == PadShape.OVAL:
        long_, short = max(w, h), min(w, h)
        r = (short + 1) // 2
        half = max(0, (long_ + 1) // 2 - r)
        ax, ay = (1, 0) if w >= h else (0, 1)
        dx, dy = _rot(ax * half, ay * half, pin.rotation_deg)
        return SegCore((cx - dx, cy - dy), (cx + dx, cy + dy)), r
    hw, hh = (w + 1) // 2, (h + 1) // 2
    ring = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        dx, dy = _rot(sx * hw, sy * hh, pin.rotation_deg)
        ring.append((cx + dx, cy + dy))
    return PolyCore(tuple(ring)), 0


def flatten_arc(start: Point, mid: Point, end: Point, tol_nm: int) -> list[Point]:
    """Polyline through the arc with sagitta <= tol_nm, vertices rounded to nm (build-time float use)."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if d == 0:
        return [start, end]
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in (start, mid, end))
    tau = 2 * math.pi
    ccw = (a3 - a1) % tau
    sweep = ccw if (a2 - a1) % tau <= ccw else ccw - tau
    step = 2 * math.acos(max(-1.0, min(1.0, 1 - tol_nm / r))) if r > tol_nm else math.pi / 4
    n = max(2, math.ceil(abs(sweep) / max(step, 1e-9)))
    pts = [(round(ux + r * math.cos(a1 + sweep * i / n)), round(uy + r * math.sin(a1 + sweep * i / n))) for i in range(n + 1)]
    pts[0], pts[-1] = start, end
    return pts


class GeometryKernel:
    def __init__(self, design: Design, config: KernelConfig = KernelConfig()):
        self.design = design
        self.config = config
        self.layer_ids: tuple[str, ...] = design.board.stackup.layer_ids()
        self.index = LayeredIndex(self.layer_ids, config.cell_nm)
        self.clearance = ClearanceModel(design, config.edge_clearance_nm, config.hole_clearance_nm)
        self.edge_items: list[Item] = []      # one representative per board/cutout edge (they exist on every layer)
        self.hole_items: list[Item] = []      # one representative per mounting hole
        self._outline: tuple[Point, ...] = tuple((p.x, p.y) for p in design.board.outline)
        self._cutouts: tuple[tuple[Point, ...], ...] = tuple(tuple((p.x, p.y) for p in c.ring) for c in design.board.cutouts)
        self._build()
        self.via = ViaLegality(self)

    # ------------------------------------------------------------------ build
    @staticmethod
    def _pt(p) -> Point:
        return (p.x, p.y) if hasattr(p, "x") else (int(p[0]), int(p[1]))

    def _add_all_layers(self, make) -> Item:
        first = None
        for lid in self.layer_ids:
            it = make(lid)
            self.index.add(it)
            first = first or it
        return first

    def _build(self) -> None:
        d = self.design
        for ri, ring in enumerate((self._outline, *self._cutouts)):
            for i in range(len(ring)):
                a, b = ring[i - 1], ring[i]
                oid = f"edge:{ri}:{i}"
                self.edge_items.append(self._add_all_layers(
                    lambda lid, a=a, b=b, oid=oid: Item(f"{oid}@{lid}", oid, "edge", lid, None, SegCore(a, b), 0, locked=True)))
        locked_ref = {c.ref: c.locked for c in d.components}
        for pin in d.pins():
            pos = self._pt(pin.position)
            if pin.kind == PinKind.NPTH:
                r = half_width_nm(pin.drill_nm or max(pin.width_nm, pin.height_nm))
                oid = f"hole:{pin.id}"
                self.hole_items.append(self._add_all_layers(
                    lambda lid, oid=oid, pos=pos, r=r: Item(f"{oid}@{lid}", oid, "hole", lid, None, SegCore(pos, pos), r,
                                                            locked=True)))
                continue
            core, r = pad_geometry(pin)
            for lid in pin.layers:
                self.index.add(Item(f"{pin.id}@{lid}", pin.id, "pad", lid, pin.net, core, r, pin.drill_nm,
                                    pos if pin.drill_nm else None, locked_ref.get(pin.ref, False)))
        for s in d.copper.segments:
            self.add_track(s)
        for a in d.copper.arcs:
            self.add_arc(a)
        for v in d.copper.vias:
            self.add_via(v)
        for z in d.copper.zones:
            ring = tuple(self._pt(p) for p in z.outline)
            if z.kind == ZoneKind.KEEPOUT:
                self.index.add(Item(f"{z.id}@{z.layer}", z.id, "keepout", z.layer, None, PolyCore(ring), 0,
                                    locked=z.locked, forbids=frozenset(z.forbids)))
            else:
                polys = [PolyCore(tuple(self._pt(p) for p in f.outer), tuple(tuple(self._pt(p) for p in h) for h in f.holes))
                         for f in z.fills] or [PolyCore(ring)]
                for i, pc in enumerate(polys):
                    self.index.add(Item(f"{z.id}#{i}@{z.layer}", z.id, "pour", z.layer, z.net, pc, 0, locked=z.locked))

    # ------------------------------------------------------------------ mutation (the router adds and rips up copper)
    def add_track(self, s: TrackSeg) -> None:
        self.index.add(Item(f"{s.id}@{s.layer}", s.id, "track", s.layer, s.net, SegCore(self._pt(s.a), self._pt(s.b)),
                            half_width_nm(s.width_nm), locked=s.locked))

    def add_arc(self, a: TrackArc) -> None:
        tol = self.config.arc_tolerance_nm
        pts = flatten_arc(self._pt(a.start), self._pt(a.mid), self._pt(a.end), tol)
        for i, (p, q) in enumerate(zip(pts, pts[1:])):
            self.index.add(Item(f"{a.id}#{i}@{a.layer}", a.id, "track", a.layer, a.net, SegCore(p, q),
                                half_width_nm(a.width_nm) + tol, locked=a.locked))

    def add_via(self, v: Via) -> None:
        ids = self.layer_ids
        i, j = sorted((ids.index(v.layer_from), ids.index(v.layer_to)))
        pos = self._pt(v.position)
        for lid in ids[i:j + 1]:
            self.index.add(Item(f"{v.id}@{lid}", v.id, "via", lid, v.net, SegCore(pos, pos), half_width_nm(v.diameter_nm),
                                v.drill_nm, pos, v.locked))

    def remove_object(self, obj_id: str) -> int:
        return self.index.remove_object(obj_id)

    # ------------------------------------------------------------------ board
    def board_location(self, p) -> str:
        """"inside" (boundary counts), "outside", or "cutout"."""
        q = self._pt(p)
        if ring_locate(q, self._outline) < 0:
            return "outside"
        if any(ring_locate(q, c) > 0 for c in self._cutouts):
            return "cutout"
        return "inside"

    def inside_board(self, p) -> bool:
        return self.board_location(p) == "inside"

    # ------------------------------------------------------------------ spatial queries
    def items_in_bbox(self, layer: str, bbox: BBox, kinds: Iterable[str] | None = None) -> list[Item]:
        return self.index.query_bbox(layer, bbox, kinds)

    def items_at_point(self, layer: str, p, kinds: Iterable[str] | None = None) -> list[Item]:
        return self.index.query_point(layer, self._pt(p), kinds)

    def items_hit_by_segment(self, layer: str, a, b, margin: int = 0, kinds: Iterable[str] | None = None) -> list[Item]:
        return self.index.query_segment(layer, self._pt(a), self._pt(b), margin, kinds)

    def plane_net_at(self, layer: str, p) -> str | None:
        hits = self.index.query_polygon_point(layer, self._pt(p), "pour")
        return hits[0].net if hits else None

    # ------------------------------------------------------------------ clearance for routers
    def obstacles_for(self, net: str | None, layer: str, width_nm: int, window: BBox | None = None,
                      object_kind: str = "track") -> list[InflatedObstacle]:
        """Inflated obstacles a centreline of `net` and `width_nm` must avoid on `layer`, optionally within a window."""
        hw = half_width_nm(width_nm)
        if window is None:
            xs = [p[0] for p in self._outline]
            ys = [p[1] for p in self._outline]
            window = BBox(min(xs), min(ys), max(xs), max(ys))
        q = window.inflate(hw + self.clearance.max_clearance())
        out = []
        for it in self.index.query_bbox(layer, q):
            if not it.applies_to(object_kind):
                continue
            if net is not None and it.net == net and it.kind in ("pad", "track", "via"):
                continue
            out.append(self.clearance.inflate(it, width_nm, net))
        return out

    def point_blocked(self, layer: str, p, net: str | None, width_nm: int) -> bool:
        q = self._pt(p)
        w = BBox(q[0], q[1], q[0], q[1])
        return (not self.inside_board(q)) or any(o.blocks_point(q) for o in self.obstacles_for(net, layer, width_nm, w))

    def trace_violations(self, layer: str, a, b, width_nm: int, net: str | None,
                         ignore: Iterable[str] = ()) -> list[Violation]:
        """Every rule a trace of `width_nm` from a to b on `layer` would break. Empty means legal."""
        pa, pb = self._pt(a), self._pt(b)
        hw = half_width_nm(width_nm)
        q = SegCore(pa, pb)
        ignore = set(ignore)
        out: list[Violation] = []
        for pt in (pa, pb):
            loc = self.board_location(pt)
            if loc != "inside":
                out.append(Violation("in_cutout" if loc == "cutout" else "outside_board", layer, "",
                                     f"trace end {pt} is {loc}"))
        window = BBox.of_points(pa, pb).inflate(hw + self.clearance.max_clearance())
        for it in self.index.query_bbox(layer, window):
            if it.obj_id in ignore or not it.applies_to("track"):
                continue
            if net is not None and it.net == net and it.kind in ("pad", "track", "via"):
                continue
            req = self.clearance.required(net, it)
            if closer_than(q, hw, it.core, it.radius, req):
                code = {"keepout": "keepout", "edge": "board_edge", "hole": "hole_clearance"}.get(it.kind, "clearance")
                out.append(Violation(code, layer, it.obj_id, f"trace too close to {it.kind} {it.obj_id}", req,
                                     dist_floor(q, it.core) - hw - it.radius))
        return out

    def is_trace_legal(self, layer: str, a, b, width_nm: int, net: str | None, ignore: Iterable[str] = ()) -> bool:
        return not [v for v in self.trace_violations(layer, a, b, width_nm, net, ignore) if v.severity == "error"]

    # ------------------------------------------------------------------ vias
    def check_via(self, pos, net: str | None, diameter_nm: int, drill_nm: int, layer_from: str, layer_to: str,
                  kind: ViaKind | None = None, ignore: Iterable[str] = ()) -> ViaCheck:
        return self.via.check(pos, net, diameter_nm, drill_nm, layer_from, layer_to, kind, ignore)

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.index.items.values():
            out[it.kind] = out.get(it.kind, 0) + 1
        return out
