"""Board data model shared by the KiCad reader, FastHenry emitter, grid, and router.

All lengths are in millimetres. Coordinates are KiCad board coordinates
(x right, y down); convert at the I/O boundary only.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    kind: str                  # "copper" or "dielectric"
    thickness_mm: float
    copper_index: int | None = None   # 0 = top copper layer


@dataclass
class Pad:
    ref: str                   # footprint reference, e.g. "Q1"
    number: str                # pad number, e.g. "5"
    net: str | None
    layer: str
    x: float
    y: float
    w: float
    h: float
    rot_deg: float = 0.0
    shape: str = "rect"


@dataclass
class Segment:
    net: str | None
    layer: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: float


@dataclass
class Via:
    net: str | None
    x: float
    y: float
    diameter: float
    drill: float
    layers: tuple[str, str]


@dataclass
class Zone:
    net: str | None
    layer: str
    polygon: list[tuple[float, float]]
    keepout: bool = False


@dataclass
class Board:
    stackup: list[Layer] = field(default_factory=list)
    pads: list[Pad] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)

    def nets(self) -> set[str]:
        names = {p.net for p in self.pads} | {s.net for s in self.segments} | {v.net for v in self.vias}
        return {n for n in names if n}

    def pads_on_net(self, net: str) -> list[Pad]:
        return [p for p in self.pads if p.net == net]
