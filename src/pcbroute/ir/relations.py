"""IR Relations: differential pairs, buses, length/skew constraints."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffPair:
    name: str
    positive: str                    # net names
    negative: str
    gap_nm: int | None = None
    width_nm: int | None = None
    max_skew_nm: int | None = None
    source: str = "inferred"         # "user" | "inferred"; inferred pairs need approval before they change routing


@dataclass(frozen=True)
class Bus:
    name: str
    nets: tuple[str, ...]
    source: str = "inferred"


@dataclass(frozen=True)
class LengthMatch:
    name: str
    nets: tuple[str, ...]
    tolerance_nm: int
    source: str = "user"


@dataclass(frozen=True)
class Relations:
    diff_pairs: tuple[DiffPair, ...] = ()
    buses: tuple[Bus, ...] = ()
    length_matches: tuple[LengthMatch, ...] = ()
