"""Spatial index separated by layer, tuned for integer nanometres.

`GridIndex` is a hashed uniform grid: O(cells touched) insert/remove, which suits a router that adds and rips up
copper constantly. Very large boxes go to a small "big" list that every query scans. Candidates are filtered by exact
bounding-box overlap, then by the exact predicates in `shapes`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable

from .shapes import Core, Point, PolyCore, SegCore, core_bbox, point_in_polygon, touches


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def intersects(self, o: "BBox") -> bool:
        return self.x0 <= o.x1 and o.x0 <= self.x1 and self.y0 <= o.y1 and o.y0 <= self.y1

    def inflate(self, d: int) -> "BBox":
        return BBox(self.x0 - d, self.y0 - d, self.x1 + d, self.y1 + d)

    def contains_point(self, p: Point) -> bool:
        return self.x0 <= p[0] <= self.x1 and self.y0 <= p[1] <= self.y1

    def union(self, o: "BBox") -> "BBox":
        return BBox(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    @staticmethod
    def of_points(*pts: Point) -> "BBox":
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return BBox(min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Item:
    """One copper (or restricting) feature on one layer."""
    key: str                       # unique, "<obj_id>@<layer>" plus "#n" for pieces
    obj_id: str                    # IR id: "Q1.5", "track:3", "via:0", "zone:1:L0", "edge:0", "hole:H1."
    kind: str                      # pad | track | via | keepout | hole | edge | pour
    layer: str
    net: str | None
    core: Core
    radius: int                    # copper half-width in nm, grown around the core
    drill_nm: int | None = None
    hole_xy: Point | None = None   # drill centre for pads and vias
    locked: bool = False
    forbids: frozenset[str] = frozenset()   # keepouts: {"tracks", "vias", "copper", "pads", "footprints"}

    @cached_property
    def bbox(self) -> BBox:
        x0, y0, x1, y1 = core_bbox(self.core)
        return BBox(x0 - self.radius, y0 - self.radius, x1 + self.radius, y1 + self.radius)

    def applies_to(self, object_kind: str) -> bool:
        """Does this item restrict a new `track`, `via` or `pad`? Pours yield (they are refilled), keepouts are selective."""
        if self.kind == "pour":
            return False
        if self.kind == "keepout":
            return {"track": "tracks", "via": "vias", "pad": "pads"}.get(object_kind, "") in self.forbids
        return True


class GridIndex:
    def __init__(self, cell_nm: int = 2_000_000, big_cells: int = 64):
        self.cell = cell_nm
        self.big_cells = big_cells
        self._cells: dict[tuple[int, int], set[str]] = {}
        self._boxes: dict[str, BBox] = {}
        self._big: set[str] = set()

    def _range(self, b: BBox):
        c = self.cell
        return b.x0 // c, b.x1 // c, b.y0 // c, b.y1 // c

    def insert(self, key: str, b: BBox) -> None:
        if key in self._boxes:
            self.remove(key)
        self._boxes[key] = b
        i0, i1, j0, j1 = self._range(b)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > self.big_cells:
            self._big.add(key)
            return
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self._cells.setdefault((i, j), set()).add(key)

    def remove(self, key: str) -> None:
        b = self._boxes.pop(key, None)
        if b is None:
            return
        if key in self._big:
            self._big.discard(key)
            return
        i0, i1, j0, j1 = self._range(b)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                s = self._cells.get((i, j))
                if s:
                    s.discard(key)
                    if not s:
                        del self._cells[(i, j)]

    def query(self, b: BBox) -> list[str]:
        i0, i1, j0, j1 = self._range(b)
        found: set[str] = set(self._big)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > 4 * self.big_cells:
            found.update(self._boxes)                     # huge query window: scanning everything is cheaper
        else:
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    s = self._cells.get((i, j))
                    if s:
                        found.update(s)
        return sorted(k for k in found if self._boxes[k].intersects(b))

    def __len__(self) -> int:
        return len(self._boxes)


class LayeredIndex:
    """One GridIndex per layer id (L0..Ln) plus the item table."""

    def __init__(self, layer_ids: Iterable[str], cell_nm: int = 2_000_000):
        self.layer_ids = tuple(layer_ids)
        self.cell_nm = cell_nm
        self._layers = {lid: GridIndex(cell_nm) for lid in self.layer_ids}
        self.items: dict[str, Item] = {}
        self._by_obj: dict[str, list[str]] = {}

    def add(self, item: Item) -> None:
        if item.layer not in self._layers:
            raise KeyError(f"unknown layer {item.layer}")
        self.items[item.key] = item
        self._layers[item.layer].insert(item.key, item.bbox)
        keys = self._by_obj.setdefault(item.obj_id, [])
        if item.key not in keys:
            keys.append(item.key)

    def remove_object(self, obj_id: str) -> int:
        keys = self._by_obj.pop(obj_id, [])
        for k in keys:
            it = self.items.pop(k, None)
            if it is not None:
                self._layers[it.layer].remove(k)
        return len(keys)

    def has_object(self, obj_id: str) -> bool:
        return obj_id in self._by_obj

    def items_of(self, obj_id: str) -> list[Item]:
        return [self.items[k] for k in self._by_obj.get(obj_id, [])]

    def __len__(self) -> int:
        return len(self.items)

    # ---- queries ----
    def query_bbox(self, layer: str, b: BBox, kinds: Iterable[str] | None = None,
                   nets: Iterable[str | None] | None = None) -> list[Item]:
        ks = set(kinds) if kinds is not None else None
        ns = set(nets) if nets is not None else None
        out = []
        for key in self._layers[layer].query(b):
            it = self.items[key]
            if (ks is None or it.kind in ks) and (ns is None or it.net in ns):
                out.append(it)
        return out

    def query_point(self, layer: str, p: Point, kinds: Iterable[str] | None = None) -> list[Item]:
        """Items whose copper contains the point (boundary included). Exact."""
        pc = SegCore(p, p)
        return [it for it in self.query_bbox(layer, BBox(p[0], p[1], p[0], p[1]), kinds)
                if touches(pc, 0, it.core, it.radius)]

    def query_segment(self, layer: str, a: Point, b: Point, margin: int = 0,
                      kinds: Iterable[str] | None = None) -> list[Item]:
        """Items whose copper touches the segment ab grown by `margin`. Exact."""
        sc = SegCore(a, b)
        bb = BBox.of_points(a, b).inflate(margin)
        return [it for it in self.query_bbox(layer, bb, kinds) if touches(sc, margin, it.core, it.radius)]

    def query_polygon_point(self, layer: str, p: Point, kind: str = "pour") -> list[Item]:
        """Polygon items (pours, keepouts) whose region contains p, using the exact point-in-polygon test."""
        return [it for it in self.query_bbox(layer, BBox(p[0], p[1], p[0], p[1]), [kind])
                if isinstance(it.core, PolyCore) and point_in_polygon(p, it.core)]
