"""IR Nets: pins, class, width, clearance, layer mask, criticality."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NetKind(str, Enum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    CLOCK = "clock"


@dataclass(frozen=True)
class NetClass:
    name: str
    width_nm: int
    clearance_nm: int
    via_diameter_nm: int | None = None
    via_drill_nm: int | None = None
    diff_pair_width_nm: int | None = None
    diff_pair_gap_nm: int | None = None


@dataclass(frozen=True)
class Net:
    name: str
    index: int                       # stable 0-based index, used for graph/embedding lookups
    pins: tuple[str, ...]            # Pin ids
    net_class: str                   # NetClass name
    width_nm: int                    # resolved routing width
    clearance_nm: int                # resolved clearance
    layer_mask: tuple[str, ...]      # layer ids this net may be routed on
    criticality: float = 0.3         # 0..1 routing priority hint
    criticality_source: str = "default"   # "default" | "heuristic" | "user"
    kind: NetKind = NetKind.SIGNAL
    source_id: str = ""
