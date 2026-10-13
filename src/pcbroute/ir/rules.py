"""IR Rules: manufacturer minima, via types, net classes."""
from __future__ import annotations

from dataclasses import dataclass

from .copper import ViaKind
from .nets import NetClass


@dataclass(frozen=True)
class ViaType:
    name: str
    diameter_nm: int
    drill_nm: int
    layer_from: str
    layer_to: str
    kind: ViaKind = ViaKind.THROUGH


@dataclass(frozen=True)
class Rules:
    min_track_nm: int = 100_000
    min_clearance_nm: int = 100_000
    min_via_diameter_nm: int = 400_000
    min_via_drill_nm: int = 200_000
    min_annulus_nm: int = 100_000
    net_classes: tuple[NetClass, ...] = ()
    via_types: tuple[ViaType, ...] = ()
    default_via: str | None = None
    defaults_assumed: bool = True      # True when the source gave no design rules and typical values were used

    def net_class(self, name: str) -> NetClass | None:
        return next((c for c in self.net_classes if c.name == name), None)
