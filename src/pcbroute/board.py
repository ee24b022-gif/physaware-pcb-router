"""Board data model shared by the KiCad reader, FastHenry emitter, grid, and router.

All lengths are in millimetres. Coordinates are KiCad board coordinates
(x right, y down); convert at the I/O boundary only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def rotate_kicad(x: float, y: float, deg: float) -> tuple[float, float]:
    """Rotate (x, y) about the origin the way KiCad does.

    A positive angle is counter-clockwise as seen on screen. Because KiCad's y axis
    points down, this is x' = x cos + y sin, y' = -x sin + y cos.
    """
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
    return x * c + y * s, -x * s + y * c


@dataclass
class Layer:
    name: str
    kind: str                  # "copper" or "dielectric"
    thickness_mm: float
    copper_index: int | None = None   # 0 = top copper layer
    epsilon_r: float | None = None
    material: str | None = None


@dataclass
class Pad:
    ref: str                   # footprint reference, e.g. "Q1"
    number: str                # pad number, e.g. "5"
    net: str | None
    layer: str                 # primary copper layer ("" if the pad has no copper)
    x: float                   # board coordinates of the pad centre
    y: float
    w: float
    h: float
    rot_deg: float = 0.0       # absolute orientation on the board
    shape: str = "rect"
    layers: tuple[str, ...] = ()   # all copper layers the pad exists on
    kind: str = "smd"          # smd | thru_hole | np_thru_hole | connect
    drill: float | None = None

    def corners(self) -> list[tuple[float, float]]:
        """Corners of the pad's rectangle (bounding box for round and oval shapes)."""
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            dx, dy = rotate_kicad(sx * self.w / 2, sy * self.h / 2, self.rot_deg)
            pts.append((self.x + dx, self.y + dy))
        return pts

    def bbox(self) -> tuple[float, float, float, float]:
        c = self.corners()
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        return min(xs), min(ys), max(xs), max(ys)


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
    filled: bool = False       # True for a computed fill polygon, False for the outline
    keepout_rules: dict = field(default_factory=dict)


@dataclass
class Footprint:
    ref: str
    value: str
    lib_id: str
    x: float
    y: float
    rot_deg: float
    layer: str
    properties: dict = field(default_factory=dict)


@dataclass
class Board:
    stackup: list[Layer] = field(default_factory=list)
    pads: list[Pad] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    footprints: list[Footprint] = field(default_factory=list)
    outline: list[tuple[float, float, float, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def nets(self) -> set[str]:
        names = {p.net for p in self.pads} | {s.net for s in self.segments} | {v.net for v in self.vias}
        return {n for n in names if n}

    def pads_on_net(self, net: str) -> list[Pad]:
        return [p for p in self.pads if p.net == net]

    def copper_layers(self) -> list[str]:
        names = [ly.name for ly in self.stackup if ly.kind == "copper"]
        return names or list(self.metadata.get("copper_layers", []))

    def footprint(self, ref: str) -> Footprint | None:
        return next((f for f in self.footprints if f.ref == ref), None)
