"""IR Copper: segments, arcs, vias, zones, locked/protected objects."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .geometry import Polygon, Pt, Ring


class ViaKind(str, Enum):
    THROUGH = "through"
    BLIND = "blind"
    BURIED = "buried"
    MICRO = "micro"


class ZoneKind(str, Enum):
    POUR = "pour"
    KEEPOUT = "keepout"


@dataclass(frozen=True)
class TrackSeg:
    id: str                    # "track:<n>", stable within one import
    net: str | None
    layer: str
    a: Pt
    b: Pt
    width_nm: int
    locked: bool = False
    source_id: str = ""


@dataclass(frozen=True)
class TrackArc:
    id: str                    # "arc:<n>"
    net: str | None
    layer: str
    start: Pt
    mid: Pt
    end: Pt
    width_nm: int
    locked: bool = False
    source_id: str = ""


@dataclass(frozen=True)
class Via:
    id: str                    # "via:<n>"
    net: str | None
    position: Pt
    diameter_nm: int
    drill_nm: int
    layer_from: str
    layer_to: str
    kind: ViaKind = ViaKind.THROUGH
    locked: bool = False
    source_id: str = ""


@dataclass(frozen=True)
class Zone:
    id: str                    # "zone:<n>:<layer id>"
    net: str | None
    layer: str
    kind: ZoneKind
    outline: Ring
    fills: tuple[Polygon, ...] = ()           # computed copper; empty for keepouts or unfilled zones
    forbids: tuple[str, ...] = ()             # keepouts: subset of {"tracks", "vias", "copper", "pads", "footprints"}
    locked: bool = False
    source_id: str = ""


@dataclass(frozen=True)
class Copper:
    segments: tuple[TrackSeg, ...] = ()
    arcs: tuple[TrackArc, ...] = ()
    vias: tuple[Via, ...] = ()
    zones: tuple[Zone, ...] = ()
    protected: tuple[str, ...] = ()           # ids the router must never modify; adds to anything locked

    def protected_ids(self) -> frozenset[str]:
        locked = {o.id for group in (self.segments, self.arcs, self.vias, self.zones) for o in group if o.locked}
        return frozenset(locked) | frozenset(self.protected)
