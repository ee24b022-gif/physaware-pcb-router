"""Via legality: layer spans, drill and annular ring, board edge, and clearance on every layer the via crosses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..ir import LayerKind, ViaKind
from .clearance import Violation, half_width_nm
from .shapes import PolyCore, SegCore, closer_than, dist_floor
from .spatial_index import BBox

if TYPE_CHECKING:
    from .kernel import GeometryKernel


@dataclass(frozen=True)
class PlaneInteraction:
    layer: str
    plane_net: str | None
    kind: str                      # "antipad" (different net, copper is cut away) | "thermal_relief" (same net)
    radius_nm: int                 # antipad radius, or the outer radius of the relief ring
    spokes: int | None = None      # thermal relief: how many of 4 probe directions still have plane copper


@dataclass(frozen=True)
class ViaCheck:
    legal: bool
    kind: ViaKind | None
    layers: tuple[str, ...]
    violations: tuple[Violation, ...] = ()
    plane: tuple[PlaneInteraction, ...] = ()

    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "error")

    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "warning")

    def codes(self) -> set[str]:
        return {v.code for v in self.errors()}


class ViaLegality:
    def __init__(self, kernel: "GeometryKernel"):
        self.k = kernel

    def _span_allowed(self, a: int, b: int, kind: ViaKind) -> bool:
        ids = self.k.layer_ids
        if kind == ViaKind.THROUGH:
            return (a, b) == (0, len(ids) - 1)
        index = {lid: i for i, lid in enumerate(ids)}
        for vt in self.k.design.rules.via_types:
            if vt.kind == kind and vt.layer_from in index and vt.layer_to in index:
                if tuple(sorted((index[vt.layer_from], index[vt.layer_to]))) == (a, b):
                    return True
        return kind in (ViaKind.BLIND, ViaKind.BURIED) and self.k.config.allow_blind_buried

    def check(self, pos, net: str | None, diameter_nm: int, drill_nm: int, layer_from: str, layer_to: str,
              kind: ViaKind | None = None, ignore: Iterable[str] = ()) -> ViaCheck:
        k = self.k
        p = k._pt(pos)
        ids = k.layer_ids
        out: list[Violation] = []
        seen: set[tuple[str, str, str]] = set()

        def add(v: Violation) -> None:
            key = (v.code, v.obj_id, v.layer)
            if key not in seen:
                seen.add(key)
                out.append(v)

        # ---- 1. layer span against the stackup ----
        if layer_from not in ids or layer_to not in ids:
            bad = layer_from if layer_from not in ids else layer_to
            add(Violation("unknown_layer", "*", "", f"layer {bad} is not in the stackup"))
            return ViaCheck(False, kind, (), tuple(out))
        a, b = sorted((ids.index(layer_from), ids.index(layer_to)))
        span = ids[a:b + 1]
        if a == b:
            add(Violation("span_too_short", "*", "", "a via must span at least two layers"))
            return ViaCheck(False, kind, span, tuple(out))
        derived = (ViaKind.THROUGH if (a, b) == (0, len(ids) - 1)
                   else ViaKind.BLIND if a == 0 or b == len(ids) - 1 else ViaKind.BURIED)
        kind = kind or derived
        if kind == ViaKind.MICRO:
            if b - a != 1:
                add(Violation("span_not_allowed", "*", "", "a microvia may only join two adjacent layers"))
        elif kind != derived:
            add(Violation("span_kind_mismatch", "*", "", f"declared {kind.value} but the span L{a}..L{b} is {derived.value}"))
        if not self._span_allowed(a, b, kind):
            add(Violation("span_not_allowed", "*", "", f"{kind.value} via {ids[a]}..{ids[b]} is not allowed by the via types / policy"))

        # ---- 2. drill and annular ring ----
        rules = k.design.rules
        r = half_width_nm(diameter_nm)
        dr = half_width_nm(drill_nm)
        if drill_nm >= diameter_nm:
            add(Violation("drill_ge_diameter", "*", "", f"drill {drill_nm} nm is not smaller than the pad diameter {diameter_nm} nm"))
        if diameter_nm < rules.min_via_diameter_nm:
            add(Violation("diameter_too_small", "*", "", "via diameter below the minimum",
                          required_nm=rules.min_via_diameter_nm, gap_nm=diameter_nm))
        if drill_nm < rules.min_via_drill_nm:
            add(Violation("drill_too_small", "*", "", "via drill below the minimum",
                          required_nm=rules.min_via_drill_nm, gap_nm=drill_nm))
        if diameter_nm - drill_nm < 2 * rules.min_annulus_nm:
            add(Violation("annulus_too_small", "*", "", "annular ring below the minimum",
                          required_nm=rules.min_annulus_nm, gap_nm=(diameter_nm - drill_nm) // 2))

        # ---- 3. board outline, cutouts, copper-to-edge ----
        loc = k.board_location(p)
        if loc != "inside":
            add(Violation("in_cutout" if loc == "cutout" else "outside_board", "*", "", f"via centre is {loc}"))
        else:
            pc = SegCore(p, p)
            for e in k.edge_items:
                if closer_than(pc, r, e.core, 0, k.clearance.edge_nm):
                    add(Violation("board_edge", "*", e.obj_id, "via too close to the board edge",
                                  k.clearance.edge_nm, dist_floor(pc, e.core) - r))
        pc = SegCore(p, p)
        h2h = k.config.hole_to_hole_nm if k.config.hole_to_hole_nm is not None else k.clearance.min_nm
        hole_clr = k.clearance.hole_nm

        # ---- 4. every layer the via crosses ----
        ignore = set(ignore)
        window = BBox(p[0], p[1], p[0], p[1]).inflate(r + k.clearance.max_clearance())
        for layer in span:
            for it in k.index.query_bbox(layer, window):
                if it.obj_id in ignore or it.kind in ("pour", "edge", "hole") or not it.applies_to("via"):
                    continue
                same_net = net is not None and it.net == net and it.kind in ("pad", "track", "via")
                if not same_net:
                    req = k.clearance.required(net, it)
                    if closer_than(pc, r, it.core, it.radius, req):
                        add(Violation("keepout" if it.kind == "keepout" else "clearance", layer, it.obj_id,
                                      f"via copper too close to {it.kind} {it.obj_id} on {layer}", req,
                                      dist_floor(pc, it.core) - r - it.radius))
                    elif it.kind != "keepout" and closer_than(pc, dr, it.core, it.radius, hole_clr):
                        add(Violation("hole_clearance", layer, it.obj_id, f"via drill too close to {it.kind} {it.obj_id}",
                                      hole_clr, dist_floor(pc, it.core) - dr - it.radius))
                if it.hole_xy is not None and it.drill_nm:
                    other = SegCore(it.hole_xy, it.hole_xy)
                    if closer_than(pc, dr, other, half_width_nm(it.drill_nm), h2h):
                        add(Violation("hole_to_hole", layer, it.obj_id, f"drill too close to the drill of {it.obj_id}", h2h,
                                      dist_floor(pc, other) - dr - half_width_nm(it.drill_nm)))
        for h in k.hole_items:
            if closer_than(pc, r, h.core, h.radius, hole_clr):
                add(Violation("hole_clearance", "*", h.obj_id, f"via too close to mounting hole {h.obj_id}", hole_clr,
                              dist_floor(pc, h.core) - r - h.radius))
            elif closer_than(pc, dr, h.core, h.radius, h2h):
                add(Violation("hole_to_hole", "*", h.obj_id, f"via drill too close to mounting hole {h.obj_id}", h2h,
                              dist_floor(pc, h.core) - dr - h.radius))

        # ---- 5. plane layers crossed: antipads for other nets, thermal reliefs for the via's own net ----
        plane: list[PlaneInteraction] = []
        st = k.design.board.stackup
        for layer in span:
            if st.layer(layer).kind not in (LayerKind.POWER, LayerKind.GROUND):
                continue
            for pour in k.index.query_polygon_point(layer, p, "pour"):
                if net is not None and pour.net == net:
                    gap = max(rules.min_clearance_nm, k.clearance.net_clearance(net))
                    probe = r + gap + rules.min_track_nm
                    spokes = sum(1 for dx, dy in ((probe, 0), (-probe, 0), (0, probe), (0, -probe))
                                 if any(q.net == net for q in k.index.query_polygon_point(layer, (p[0] + dx, p[1] + dy), "pour")))
                    plane.append(PlaneInteraction(layer, pour.net, "thermal_relief", r + gap, spokes))
                    if spokes < 2:
                        add(Violation("thermal_relief_spokes", layer, pour.obj_id,
                                      f"only {spokes} of 4 directions keep plane copper around the via for thermal spokes",
                                      severity="warning"))
                else:
                    gap = max(rules.min_clearance_nm, k.clearance.net_clearance(net), k.clearance.net_clearance(pour.net))
                    plane.append(PlaneInteraction(layer, pour.net, "antipad", r + gap))
                    if not k.config.allow_plane_cuts:
                        add(Violation("plane_cut", layer, pour.obj_id, f"via cuts the {pour.net} plane on {layer}"))

        return ViaCheck(not any(v.severity == "error" for v in out), kind, span, tuple(out), tuple(plane))
