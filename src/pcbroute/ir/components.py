"""IR Components: reference, footprint, placement, rotation, keepouts, pins."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometry import Pt, Ring, circle_ring, rotated_rect_ring


class PadShape(str, Enum):
    RECT = "rect"
    ROUNDRECT = "roundrect"
    OVAL = "oval"
    CIRCLE = "circle"
    CUSTOM = "custom"


class PinKind(str, Enum):
    SMD = "smd"
    THROUGH = "through"
    NPTH = "npth"           # mounting hole, no net
    OTHER = "other"


class PinRole(str, Enum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    UNKNOWN = "unknown"


class Side(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


class KeepoutKind(str, Enum):
    COURTYARD = "courtyard"
    NO_COPPER = "no_copper"
    NO_TRACKS = "no_tracks"
    NO_VIAS = "no_vias"


@dataclass(frozen=True)
class Pin:
    id: str                       # unique in the design, normally "REF.NUMBER"
    ref: str
    number: str
    function: str                 # pin function name when the source provides it, else ""
    net: str | None
    layers: tuple[str, ...]       # IR copper layer ids the pad exists on
    position: Pt                  # pad centre, board coordinates
    shape: PadShape
    width_nm: int                 # pad size before rotation
    height_nm: int
    rotation_deg: float           # absolute orientation on the board
    kind: PinKind = PinKind.SMD
    drill_nm: int | None = None
    role: PinRole = PinRole.UNKNOWN
    source_id: str = ""

    def polygon(self) -> Ring:
        """Pad outline for geometry work (circles and ovals are approximated by polygons)."""
        if self.shape == PadShape.CIRCLE:
            return circle_ring(self.position.x, self.position.y, max(self.width_nm, self.height_nm) // 2)
        return rotated_rect_ring(self.position.x, self.position.y, self.width_nm, self.height_nm, self.rotation_deg)


@dataclass(frozen=True)
class Keepout:
    id: str
    kind: KeepoutKind
    ring: Ring
    layers: tuple[str, ...] = ()  # empty means all layers


@dataclass(frozen=True)
class Component:
    ref: str
    footprint: str                # footprint library id, e.g. "Capacitor_SMD:C_1206"
    value: str
    side: Side
    position: Pt
    rotation_deg: float
    pins: tuple[Pin, ...]
    keepouts: tuple[Keepout, ...] = ()
    locked: bool = False
    properties: dict = field(default_factory=dict)
    source_id: str = ""

    def pin(self, number: str) -> Pin | None:
        return next((p for p in self.pins if p.number == number), None)
