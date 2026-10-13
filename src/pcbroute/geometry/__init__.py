"""Geometry kernel: exact integer-nanometre spatial queries, clearance expansion, and via legality.

    from pcbroute.geometry import GeometryKernel
    k = GeometryKernel(design)                       # design: pcbroute.ir.Design
    k.trace_violations("L0", (0, 0), (5_000_000, 0), 250_000, "SW")
    k.check_via((20_000_000, 20_000_000), "SW", 600_000, 300_000, "L0", "L3")
"""
from .clearance import ClearanceModel, InflatedObstacle, Violation, disc_polygon, half_width_nm, inflation_nm
from .kernel import GeometryKernel, KernelConfig, flatten_arc, pad_geometry
from .shapes import (PolyCore, SegCore, closer_than, convex_hull, dist_floor, point_in_polygon, ring_locate,
                     segments_intersect, touches)
from .spatial_index import BBox, GridIndex, Item, LayeredIndex
from .via_legality import PlaneInteraction, ViaCheck, ViaLegality

__all__ = [n for n in dir() if not n.startswith("_")]
