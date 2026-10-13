"""The versioned canonical IR container and its structural validator."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .board import Board
from .components import Component, Pin
from .copper import Copper
from .geometry import point_in_ring, ring_area2
from .nets import Net
from .relations import Relations
from .rules import Rules
from .serde import from_jsonable, to_jsonable

IR_VERSION = "0.1.0"


class IRVersionError(ValueError):
    pass


@dataclass(frozen=True)
class Provenance:
    source_format: str = ""            # e.g. "kicad_pcb"
    source_version: str = ""           # file format version as written by the CAD tool
    generator: str = ""
    units: str = "nm"
    warnings: tuple[str, ...] = ()     # import-time caveats, e.g. "no stackup in source"


@dataclass(frozen=True)
class Issue:
    level: str                         # "error" | "warning"
    code: str
    message: str
    object_id: str = ""


@dataclass(frozen=True)
class Design:
    board: Board
    components: tuple[Component, ...]
    nets: tuple[Net, ...]
    copper: Copper
    relations: Relations = field(default_factory=Relations)
    rules: Rules = field(default_factory=Rules)
    provenance: Provenance = field(default_factory=Provenance)
    name: str = "design"
    ir_version: str = IR_VERSION

    # ---- lookups (built on demand; the IR itself stays plain immutable data) ----
    def pins(self) -> list[Pin]:
        return [p for c in self.components for p in c.pins]

    def pin(self, pin_id: str) -> Pin | None:
        return next((p for p in self.pins() if p.id == pin_id), None)

    def net(self, name: str) -> Net | None:
        return next((n for n in self.nets if n.name == name), None)

    def component(self, ref: str) -> Component | None:
        return next((c for c in self.components if c.ref == ref), None)

    # ---- validation: the compiler rejects, it does not repair ----
    def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        layer_ids = set(self.board.stackup.layer_ids())
        net_names = {n.name for n in self.nets}

        def err(code, msg, oid=""):
            issues.append(Issue("error", code, msg, oid))

        def warn(code, msg, oid=""):
            issues.append(Issue("warning", code, msg, oid))

        if self.ir_version != IR_VERSION:
            err("ir_version", f"IR version {self.ir_version} != supported {IR_VERSION}")
        if len(self.board.outline) < 3 or ring_area2(self.board.outline) == 0:
            err("outline", "board outline is not a closed polygon with area")
        if self.board.outline_inferred:
            warn("outline_inferred", "no Edge.Cuts outline in the source; a bounding box was used")
        if not self.board.stackup.defined_in_source:
            warn("stackup_missing", "dielectric thicknesses are unknown; impedance and plane-gap physics are not available")

        outside = [c.ref for c in self.components
                   if not point_in_ring(c.position.x, c.position.y, self.board.outline)]
        if outside:
            warn("component_outside_outline",
                 f"{len(outside)} component(s) sit outside the board outline (unplaced?): {', '.join(outside[:8])}")

        seen: set[str] = set()
        pin_ids: set[str] = set()
        for c in self.components:
            for p in c.pins:
                if p.id in pin_ids:
                    err("duplicate_pin", f"duplicate pin id {p.id}", p.id)
                pin_ids.add(p.id)
                if p.net is not None and p.net not in net_names:
                    err("pin_unknown_net", f"pin {p.id} references unknown net {p.net!r}", p.id)
                for ly in p.layers:
                    if ly not in layer_ids:
                        err("pin_unknown_layer", f"pin {p.id} on unknown layer {ly}", p.id)
        for n in self.nets:
            for pid in n.pins:
                if pid not in pin_ids:
                    err("net_unknown_pin", f"net {n.name} lists unknown pin {pid}", n.name)
            for ly in n.layer_mask:
                if ly not in layer_ids:
                    err("net_unknown_layer", f"net {n.name} layer mask has unknown layer {ly}", n.name)
            if not (0.0 <= n.criticality <= 1.0):
                err("criticality_range", f"net {n.name} criticality {n.criticality} outside 0..1", n.name)
            if n.width_nm <= 0 or n.clearance_nm < 0:
                err("net_rule", f"net {n.name} has non-positive width or negative clearance", n.name)
        for group in (self.copper.segments, self.copper.arcs, self.copper.vias, self.copper.zones):
            for o in group:
                if o.id in seen:
                    err("duplicate_id", f"duplicate copper id {o.id}", o.id)
                seen.add(o.id)
                if o.net is not None and o.net not in net_names:
                    err("copper_unknown_net", f"{o.id} references unknown net {o.net!r}", o.id)
        for o in (*self.copper.segments, *self.copper.arcs, *self.copper.zones):
            if o.layer not in layer_ids:
                err("copper_unknown_layer", f"{o.id} on unknown layer {o.layer}", o.id)
        for v in self.copper.vias:
            for ly in (v.layer_from, v.layer_to):
                if ly not in layer_ids:
                    err("via_unknown_layer", f"{v.id} spans unknown layer {ly}", v.id)
        for pid in self.copper.protected:
            if pid not in seen:
                err("protected_unknown", f"protected id {pid} does not exist", pid)
        for dp in self.relations.diff_pairs:
            for n in (dp.positive, dp.negative):
                if n not in net_names:
                    err("diffpair_unknown_net", f"diff pair {dp.name} references unknown net {n!r}", dp.name)
        return issues

    # ---- serialization ----
    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Design":
        v = data.get("ir_version")
        if v != IR_VERSION:
            raise IRVersionError(f"cannot read IR version {v!r}; this code reads {IR_VERSION}. "
                                 "Write a migration before bumping the version.")
        return from_jsonable(cls, data)

    @classmethod
    def from_json(cls, text: str) -> "Design":
        return cls.from_dict(json.loads(text))
