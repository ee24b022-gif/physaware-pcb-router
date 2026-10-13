"""Stage A: the canonical, versioned board IR. Independent of KiCad, Altium, or DSN."""
from .board import Board, Cutout, CopperLayer, Dielectric, LayerKind, Stackup
from .components import Component, Keepout, KeepoutKind, PadShape, Pin, PinKind, PinRole, Side
from .copper import Copper, TrackArc, TrackSeg, Via, ViaKind, Zone, ZoneKind
from .design import IR_VERSION, Design, Issue, IRVersionError, Provenance
from .geometry import NM_PER_MM, Polygon, Pt, Ring, mm_to_nm, nm_to_mm, pt_mm
from .nets import Net, NetClass, NetKind
from .relations import Bus, DiffPair, LengthMatch, Relations
from .rules import Rules, ViaType

__all__ = [n for n in dir() if not n.startswith("_")]
from .from_kicad import ir_from_kicad  # noqa: E402  (imports the KiCad parser model; keep last)
