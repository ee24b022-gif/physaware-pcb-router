"""IR Board: outline, cutouts, stackup, dielectric data, layer types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometry import Ring


class LayerKind(str, Enum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    MIXED = "mixed"


@dataclass(frozen=True)
class CopperLayer:
    id: str                 # IR id, "L0" is the top layer; independent of any CAD naming
    index: int              # 0 = top
    kind: LayerKind
    thickness_nm: int
    source_name: str        # name in the source CAD file, kept for export ("F.Cu", "In1.Cu", ...)
    side: str               # "top" | "inner" | "bottom"


@dataclass(frozen=True)
class Dielectric:
    name: str
    thickness_nm: int
    above: str              # copper layer id above this dielectric
    below: str              # copper layer id below it
    epsilon_r: float | None = None      # None means the source did not say; never invented
    loss_tangent: float | None = None
    material: str | None = None


@dataclass(frozen=True)
class Stackup:
    copper: tuple[CopperLayer, ...]
    dielectrics: tuple[Dielectric, ...] = ()
    defined_in_source: bool = True      # False when dielectric data were absent in the CAD file

    def layer_ids(self) -> tuple[str, ...]:
        return tuple(ly.id for ly in self.copper)

    def layer(self, layer_id: str) -> CopperLayer:
        for ly in self.copper:
            if ly.id == layer_id:
                return ly
        raise KeyError(layer_id)

    def index(self, layer_id: str) -> int:
        return self.layer(layer_id).index

    def gap_nm(self, a: str, b: str) -> int | None:
        """Total dielectric thickness between two adjacent copper layers, or None if unknown."""
        total = [d.thickness_nm for d in self.dielectrics if {d.above, d.below} == {a, b}]
        return sum(total) if total else None


@dataclass(frozen=True)
class Cutout:
    id: str
    ring: Ring
    plated: bool = False


@dataclass(frozen=True)
class Board:
    outline: Ring
    stackup: Stackup
    cutouts: tuple[Cutout, ...] = ()
    grid_nm: int = 10_000               # manufacturing grid (10 um); coordinates stay exact, this is for snapping
    thickness_nm: int | None = None
    outline_inferred: bool = False      # True when the source had no Edge.Cuts and a bounding box was used
