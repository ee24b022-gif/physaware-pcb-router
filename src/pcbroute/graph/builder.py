"""Stage C: canonical IR -> heterogeneous graph for the global routing model.

Node types   pad, component, net, region, layer, via, obstacle
Edge groups  electrical membership   (pad,member_of,net) (pad,part_of,component) (via,on_net,net) (obstacle,on_net,net) + rev_*
             spatial proximity       (pad,near,pad) (component,near,component)
             corridor adjacency      (region,corridor,region)             same layer, 4-neighbour cells
             layer transition        (region,via_site,region) (layer,stack,layer) (via,spans,layer) (region,in_layer,layer)
             pair relationships      (net,paired_with,net)
             geometry coupling       (pad,in_region,region) (via,in_region,region) (obstacle,overlaps,region) + rev_*

A *region* is one routing-grid cell on one layer. That is the routing resource the model reasons about:
it carries density, capacity, plane ownership, current overflow, and estimated demand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..ir import Design, KeepoutKind, LayerKind, NetKind, PadShape, PinKind, PinRole, ViaKind, ZoneKind
from ..ir.geometry import Pt, Ring, point_in_ring, ring_bbox
from .features import Cols, edge_arrays
from .hetero import EdgeType, HeteroGraph

MM = 1e6
SHAPES = list(PadShape)
PINKINDS = list(PinKind)
ROLES = list(PinRole)
NETKINDS = list(NetKind)
LAYERKINDS = list(LayerKind)
VIAKINDS = list(ViaKind)
OBSTACLE_KINDS = ["locked_track", "locked_via", "plane_zone", "keepout_zone", "mounting_hole", "cutout"]


@dataclass(frozen=True)
class GraphConfig:
    max_cells_per_side: int = 24            # region grid resolution cap
    region_size_mm: float | None = None     # overrides the cap when set
    knn: int = 8                            # pad-pad proximity neighbours
    near_radius_mm: float = 5.0
    component_knn: int = 6
    component_near_radius_mm: float = 15.0
    fourier_bands: int = 0                  # 0 disables Fourier position features
    plane_nets_excluded_from_demand: bool = True   # assume GND/power nets on a plane layer are plane-routed


@dataclass
class _Obstacle:
    id: str
    kind: str
    bbox: tuple[int, int, int, int]
    ring: Ring | None
    layers: tuple[str, ...]        # empty = all layers
    net: str | None
    locked: bool
    blocks_routing: bool


def _bbox_of_points(pts) -> tuple[int, int, int, int]:
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _knn(points: np.ndarray, k: int, radius: float):
    """Directed neighbour lists via a spatial hash. points: [N,2] float. Returns (src, dst, dist)."""
    n = len(points)
    cell = max(radius, 1.0)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(points):
        grid.setdefault((int(x // cell), int(y // cell)), []).append(i)
    src, dst, dist = [], [], []
    for i in range(n):
        cx, cy = int(points[i, 0] // cell), int(points[i, 1] // cell)
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    if j != i:
                        d = float(np.hypot(*(points[i] - points[j])))
                        if d <= radius:
                            cand.append((d, j))
        cand.sort()
        for d, j in cand[:k]:
            src.append(i)
            dst.append(j)
            dist.append(d)
    return src, dst, dist


def build_graph(d: Design, cfg: GraphConfig = GraphConfig()) -> HeteroGraph:
    st = d.board.stackup
    lay_ids = list(st.layer_ids())
    L = len(lay_ids)
    lay_idx = {lid: i for i, lid in enumerate(lay_ids)}
    x0, y0, x1, y1 = ring_bbox(d.board.outline)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    scale = max(x1 - x0, y1 - y0) / 2 or 1.0

    def nx_(x, y):
        return (x - cx) / scale, (y - cy) / scale

    pins = d.pins()
    pin_idx = {p.id: i for i, p in enumerate(pins)}
    comp_idx = {c.ref: i for i, c in enumerate(d.components)}
    net_idx = {n.name: n.index for n in d.nets}
    nets = sorted(d.nets, key=lambda n: n.index)
    class_names = [c.name for c in d.rules.net_classes] or ["Default"]
    pins_per_net = {n.name: len(n.pins) for n in nets}
    default_cls = d.rules.net_class("Default") or (d.rules.net_classes[0] if d.rules.net_classes else None)
    pitch_nm = (default_cls.width_nm + default_cls.clearance_nm) if default_cls else 450_000

    # ---------------- obstacles ----------------
    obs: list[_Obstacle] = []
    for c in d.copper.segments:
        if c.locked:
            r = c.width_nm // 2
            obs.append(_Obstacle(c.id, "locked_track", (min(c.a.x, c.b.x) - r, min(c.a.y, c.b.y) - r,
                                                        max(c.a.x, c.b.x) + r, max(c.a.y, c.b.y) + r),
                                 None, (c.layer,), c.net, True, False))
    for c in d.copper.arcs:
        if c.locked:
            b = _bbox_of_points([c.start, c.mid, c.end])
            r = c.width_nm // 2
            obs.append(_Obstacle(c.id, "locked_track", (b[0] - r, b[1] - r, b[2] + r, b[3] + r), None, (c.layer,),
                                 c.net, True, False))
    for v in d.copper.vias:
        if v.locked:
            r = v.diameter_nm // 2
            span = tuple(lay_ids[lay_idx[v.layer_from]: lay_idx[v.layer_to] + 1])
            obs.append(_Obstacle(v.id, "locked_via", (v.position.x - r, v.position.y - r, v.position.x + r,
                                                      v.position.y + r), None, span, v.net, True, True))
    for z in d.copper.zones:
        if z.kind == ZoneKind.KEEPOUT:
            blocks = "tracks" in z.forbids
            obs.append(_Obstacle(z.id, "keepout_zone", ring_bbox(z.outline), z.outline, (z.layer,), None, z.locked, blocks))
        elif st.layer(z.layer).kind in (LayerKind.POWER, LayerKind.GROUND):
            obs.append(_Obstacle(z.id, "plane_zone", ring_bbox(z.outline), z.outline, (z.layer,), z.net, z.locked, False))
    for p in pins:
        if p.kind == PinKind.NPTH:
            poly = p.polygon()
            obs.append(_Obstacle(f"hole:{p.id}", "mounting_hole", ring_bbox(poly), poly, (), None, True, True))
    for c in d.board.cutouts:
        obs.append(_Obstacle(c.id, "cutout", ring_bbox(c.ring), c.ring, (), None, True, True))

    # ---------------- region grid ----------------
    w_nm, h_nm = x1 - x0, y1 - y0
    cell = int(cfg.region_size_mm * MM) if cfg.region_size_mm else int(math.ceil(max(w_nm, h_nm) / cfg.max_cells_per_side))
    cell = max(cell, 1)
    gx, gy = int(math.ceil(w_nm / cell)), int(math.ceil(h_nm / cell))
    cell_area = float(cell) * cell

    def cell_range(bb):
        i0 = max(0, int((bb[0] - x0) // cell))
        i1 = min(gx - 1, int((bb[2] - x0) // cell))
        j0 = max(0, int((bb[1] - y0) // cell))
        j1 = min(gy - 1, int((bb[3] - y0) // cell))
        return range(i0, i1 + 1), range(j0, j1 + 1)

    def sample_cover(i, j, ring: Ring | None, bb) -> float:
        """Fraction of a cell covered by a ring (or bbox when ring is None), by 3x3 point sampling."""
        hits = 0
        for a in range(3):
            for b in range(3):
                px, py = x0 + (i + (a + 0.5) / 3) * cell, y0 + (j + (b + 0.5) / 3) * cell
                if ring is not None:
                    hits += point_in_ring(px, py, ring)
                else:
                    hits += bb[0] <= px <= bb[2] and bb[1] <= py <= bb[3]
        return hits / 9.0

    inside = np.zeros((gx, gy), np.float32)
    for i in range(gx):
        for j in range(gy):
            f = sample_cover(i, j, d.board.outline, None)
            for co in d.board.cutouts:
                f -= sample_cover(i, j, co.ring, None) * (f > 0)
            inside[i, j] = max(f, 0.0)
    valid = [(i, j) for i in range(gx) for j in range(gy) if inside[i, j] >= 0.05]

    pad_area = np.zeros((L, gx, gy)); blocked = np.zeros((L, gx, gy)); used = np.zeros((L, gx, gy))
    for p in pins:
        bb = ring_bbox(p.polygon())
        ii, jj = cell_range(bb)
        for i in ii:
            for j in jj:
                ox = min(bb[2], x0 + (i + 1) * cell) - max(bb[0], x0 + i * cell)
                oy = min(bb[3], y0 + (j + 1) * cell) - max(bb[1], y0 + j * cell)
                if ox > 0 and oy > 0:
                    for ly in p.layers:
                        pad_area[lay_idx[ly], i, j] += ox * oy
    for o in obs:
        if not o.blocks_routing:
            continue
        ii, jj = cell_range(o.bbox)
        for i in ii:
            for j in jj:
                cov = sample_cover(i, j, o.ring, o.bbox)
                if cov > 0:
                    for ly in (o.layers or lay_ids):
                        blocked[lay_idx[ly], i, j] = min(1.0, blocked[lay_idx[ly], i, j] + cov)

    def add_used(layer, pts):
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            length = math.hypot(bx - ax, by - ay)
            n = max(1, int(math.ceil(length / (cell / 4))))
            for s in range(n):
                t = (s + 0.5) / n
                i = int((ax + (bx - ax) * t - x0) // cell); j = int((ay + (by - ay) * t - y0) // cell)
                if 0 <= i < gx and 0 <= j < gy:
                    used[lay_idx[layer], i, j] += length / n

    for s in d.copper.segments:
        add_used(s.layer, [(s.a.x, s.a.y), (s.b.x, s.b.y)])
    for a in d.copper.arcs:
        add_used(a.layer, [(a.start.x, a.start.y), (a.mid.x, a.mid.y), (a.end.x, a.end.y)])

    # plane ownership
    plane_net = np.full((L, gx, gy), -1, np.int64)
    for z in d.copper.zones:
        if z.kind == ZoneKind.POUR and st.layer(z.layer).kind in (LayerKind.POWER, LayerKind.GROUND) and z.net:
            for (i, j) in valid:
                if point_in_ring(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell, z.outline):
                    plane_net[lay_idx[z.layer], i, j] = net_idx.get(z.net, -1)

    # estimated demand: RUDY over each net's pin bounding box
    plane_kinds = {st.layer(l).kind for l in lay_ids}
    skip_kinds = set()
    if cfg.plane_nets_excluded_from_demand:
        if LayerKind.GROUND in plane_kinds:
            skip_kinds.add(NetKind.GROUND)
        if LayerKind.POWER in plane_kinds:
            skip_kinds.add(NetKind.POWER)
    demand = np.zeros((gx, gy))
    pitch_mm = pitch_nm / MM
    for n in nets:
        if len(n.pins) < 2 or n.kind in skip_kinds:
            continue
        pts = [pins[pin_idx[pid]].position for pid in n.pins]
        bb = _bbox_of_points(pts)
        bw, bh = max((bb[2] - bb[0]) / MM, pitch_mm), max((bb[3] - bb[1]) / MM, pitch_mm)
        density = (bw + bh) * pitch_mm / (bw * bh)
        ii, jj = cell_range(bb)
        for i in ii:
            for j in jj:
                ox = min(bb[2], x0 + (i + 1) * cell) - max(bb[0], x0 + i * cell)
                oy = min(bb[3], y0 + (j + 1) * cell) - max(bb[1], y0 + j * cell)
                if ox > 0 and oy > 0:
                    demand[i, j] += density * (ox * oy) / cell_area
    n_routable = max(1, sum(1 for l in lay_ids if st.layer(l).kind in (LayerKind.SIGNAL, LayerKind.MIXED)))

    # ---------------- node tables ----------------
    node_ids: dict[str, list[str]] = {}
    X: dict[str, np.ndarray] = {}
    XN: dict[str, list[str]] = {}
    CAT: dict[str, dict[str, np.ndarray]] = {}
    bands = cfg.fourier_bands

    # pads
    N = len(pins)
    c = Cols(N)
    pos = np.array([nx_(p.position.x, p.position.y) for p in pins], np.float64).reshape(-1, 2)
    c.add("x", pos[:, 0]).add("y", pos[:, 1]).fourier("pos", pos, bands)
    c.add("w_mm", [p.width_nm / MM for p in pins]).add("h_mm", [p.height_nm / MM for p in pins])
    c.add("rot_sin", [math.sin(math.radians(p.rotation_deg)) for p in pins])
    c.add("rot_cos", [math.cos(math.radians(p.rotation_deg)) for p in pins])
    c.onehot("shape", [SHAPES.index(p.shape) for p in pins], [s.value for s in SHAPES])
    c.onehot("kind", [PINKINDS.index(p.kind) for p in pins], [s.value for s in PINKINDS])
    c.onehot("role", [ROLES.index(p.role) for p in pins], [s.value for s in ROLES])
    mask = np.array([[ly in p.layers for ly in lay_ids] for p in pins], np.float32).reshape(N, L)
    c.multihot("layer", mask, lay_ids)
    c.add("drill_mm", [(p.drill_nm or 0) / MM for p in pins])
    on_board = [point_in_ring(p.position.x, p.position.y, d.board.outline) for p in pins]
    c.add("on_board", on_board)                                   # 0 for parts parked outside the outline
    c.add("has_net", [p.net is not None for p in pins])
    c.add("net_degree_log", [math.log1p(pins_per_net.get(p.net, 0)) if p.net else 0.0 for p in pins])
    X["pad"], XN["pad"] = c.build()
    node_ids["pad"] = [p.id for p in pins]
    CAT["pad"] = {"net": np.array([net_idx.get(p.net, -1) if p.net else -1 for p in pins], np.int64),
                  "component": np.array([comp_idx[p.ref] for p in pins], np.int64)}

    # components
    comps = d.components
    N = len(comps)
    c = Cols(N)
    pos = np.array([nx_(k.position.x, k.position.y) for k in comps], np.float64).reshape(-1, 2)
    c.add("x", pos[:, 0]).add("y", pos[:, 1]).fourier("pos", pos, bands)
    c.add("rot_sin", [math.sin(math.radians(k.rotation_deg)) for k in comps])
    c.add("rot_cos", [math.cos(math.radians(k.rotation_deg)) for k in comps])
    c.add("on_board", [point_in_ring(k.position.x, k.position.y, d.board.outline) for k in comps])
    c.add("bottom", [k.side.value == "bottom" for k in comps])
    c.add("n_pins_log", [math.log1p(len(k.pins)) for k in comps])
    ext = []
    for k in comps:
        rings = [ko.ring for ko in k.keepouts if ko.kind == KeepoutKind.COURTYARD]
        pts = [pt for r in rings for pt in r] or [p.position for p in k.pins] or [k.position]
        bb = _bbox_of_points(pts)
        ext.append(((bb[2] - bb[0]) / scale, (bb[3] - bb[1]) / scale))
    c.add("ext_w", [e[0] for e in ext]).add("ext_h", [e[1] for e in ext]).add("locked", [k.locked for k in comps])
    X["component"], XN["component"] = c.build()
    node_ids["component"] = [k.ref for k in comps]
    CAT["component"] = {}

    # nets
    N = len(nets)
    c = Cols(N)
    in_pair = {n for dp in d.relations.diff_pairs for n in (dp.positive, dp.negative)}
    routed_len: dict[str, float] = {}
    for s in d.copper.segments:
        if s.net:
            routed_len[s.net] = routed_len.get(s.net, 0.0) + math.hypot(s.b.x - s.a.x, s.b.y - s.a.y) / MM
    hp, ext = [], []
    for n in nets:
        pts = [pins[pin_idx[pid]].position for pid in n.pins]
        if len(pts) >= 2:
            bb = _bbox_of_points(pts)
            hp.append((bb[2] - bb[0] + bb[3] - bb[1]) / MM); ext.append(((bb[2] - bb[0]) / scale, (bb[3] - bb[1]) / scale))
        else:
            hp.append(0.0); ext.append((0.0, 0.0))
    c.add("width_mm", [n.width_nm / MM for n in nets]).add("clearance_mm", [n.clearance_nm / MM for n in nets])
    c.add("criticality", [n.criticality for n in nets]).add("n_pins_log", [math.log1p(len(n.pins)) for n in nets])
    c.add("hpwl_mm", hp).add("bbox_w", [e[0] for e in ext]).add("bbox_h", [e[1] for e in ext])
    c.onehot("kind", [NETKINDS.index(n.kind) for n in nets], [k.value for k in NETKINDS])
    c.add("in_diff_pair", [n.name in in_pair for n in nets])
    c.multihot("layer_ok", np.array([[ly in n.layer_mask for ly in lay_ids] for n in nets], np.float32).reshape(N, L), lay_ids)
    c.add("offboard_frac", [(sum(1 for pid in n.pins if not on_board[pin_idx[pid]]) / len(n.pins)) if n.pins else 0.0 for n in nets])
    c.add("routed_ratio", [min(4.0, routed_len.get(n.name, 0.0) / h) if h > 0 else 0.0 for n, h in zip(nets, hp)])
    X["net"], XN["net"] = c.build()
    node_ids["net"] = [n.name for n in nets]
    CAT["net"] = {"net_class": np.array([class_names.index(n.net_class) if n.net_class in class_names else 0 for n in nets], np.int64),
                  "kind": np.array([NETKINDS.index(n.kind) for n in nets], np.int64)}

    # layers
    N = L
    c = Cols(N)
    layers = list(st.copper)
    gaps = [st.gap_nm(layers[i].id, layers[i + 1].id) if i + 1 < L else None for i in range(L)]
    eps = []
    for i in range(L):
        e = [dd.epsilon_r for dd in st.dielectrics if dd.above == layers[i].id and dd.epsilon_r]
        eps.append(e[0] if e else 0.0)
    c.add("index_norm", [ly.index / max(1, L - 1) for ly in layers]).onehot("kind", [LAYERKINDS.index(ly.kind) for ly in layers], [k.value for k in LAYERKINDS])
    c.add("thickness_mm", [ly.thickness_nm / MM for ly in layers]).add("gap_below_mm", [(g or 0) / MM for g in gaps])
    c.add("gap_known", [g is not None for g in gaps]).add("epsilon_r", eps).add("is_outer", [ly.side != "inner" for ly in layers])
    X["layer"], XN["layer"] = c.build()
    node_ids["layer"] = lay_ids
    CAT["layer"] = {}

    # vias
    vias = d.copper.vias
    N = len(vias)
    c = Cols(N)
    pos = np.array([nx_(v.position.x, v.position.y) for v in vias], np.float64).reshape(-1, 2)
    c.add("x", pos[:, 0]).add("y", pos[:, 1]).fourier("pos", pos, bands)
    c.add("diameter_mm", [v.diameter_nm / MM for v in vias]).add("drill_mm", [v.drill_nm / MM for v in vias])
    c.add("span_from", [lay_idx[v.layer_from] / max(1, L - 1) for v in vias]).add("span_to", [lay_idx[v.layer_to] / max(1, L - 1) for v in vias])
    c.onehot("kind", [VIAKINDS.index(v.kind) for v in vias], [k.value for k in VIAKINDS])
    c.add("locked", [v.locked for v in vias]).add("has_net", [v.net is not None for v in vias])
    X["via"], XN["via"] = c.build()
    node_ids["via"] = [v.id for v in vias]
    CAT["via"] = {"net": np.array([net_idx.get(v.net, -1) if v.net else -1 for v in vias], np.int64)}

    # obstacles
    N = len(obs)
    c = Cols(N)
    pos = np.array([nx_((o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2) for o in obs], np.float64).reshape(-1, 2)
    c.add("x", pos[:, 0]).add("y", pos[:, 1]).fourier("pos", pos, bands)
    c.add("w", [(o.bbox[2] - o.bbox[0]) / scale for o in obs]).add("h", [(o.bbox[3] - o.bbox[1]) / scale for o in obs])
    c.onehot("kind", [OBSTACLE_KINDS.index(o.kind) for o in obs], OBSTACLE_KINDS)
    c.multihot("layer", np.array([[(not o.layers) or ly in o.layers for ly in lay_ids] for o in obs], np.float32).reshape(N, L), lay_ids)
    c.add("has_net", [o.net is not None for o in obs]).add("locked", [o.locked for o in obs]).add("blocks_routing", [o.blocks_routing for o in obs])
    X["obstacle"], XN["obstacle"] = c.build()
    node_ids["obstacle"] = [o.id for o in obs]
    CAT["obstacle"] = {"net": np.array([net_idx.get(o.net, -1) if o.net else -1 for o in obs], np.int64)}

    # regions: one node per (valid cell, layer)
    reg_idx: dict[tuple[int, int, int], int] = {}
    reg_ids: list[str] = []
    for li, lid_ in enumerate(lay_ids):
        for (i, j) in valid:
            reg_idx[(i, j, li)] = len(reg_ids)
            reg_ids.append(f"R{i}.{j}.{lid_}")
    N = len(reg_ids)
    keys = list(reg_idx)
    c = Cols(N)
    ri = np.array([k[0] for k in keys]); rj = np.array([k[1] for k in keys]); rl = np.array([k[2] for k in keys])
    pos = np.array([nx_(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell) for i, j, _ in keys], np.float64).reshape(-1, 2)
    c.add("x", pos[:, 0]).add("y", pos[:, 1]).fourier("pos", pos, bands)
    c.add("cell_size", np.full(N, cell / scale)).add("inside_frac", inside[ri, rj])
    c.add("layer_index_norm", rl / max(1, L - 1))
    dens = pad_area[rl, ri, rj] / cell_area
    blk = np.minimum(blocked[rl, ri, rj], 1.0)
    free = np.clip(inside[ri, rj] * (1 - blk) - dens, 0.0, 1.0)
    cap_mm = free * cell_area / pitch_nm / MM
    used_mm = used[rl, ri, rj] / MM
    overflow = np.where(cap_mm > 1e-9, np.maximum(0.0, used_mm / np.maximum(cap_mm, 1e-9) - 1.0), (used_mm > 0).astype(float))
    pflag = (plane_net[rl, ri, rj] >= 0).astype(np.float32)
    c.add("pad_density", dens).add("blocked_frac", blk).add("free_frac", free)
    c.add("capacity_mm", cap_mm).add("used_mm", used_mm).add("overflow_ratio", np.minimum(overflow, 10.0))
    c.add("plane_flag", pflag).add("demand", demand[ri, rj]).add("demand_per_layer", demand[ri, rj] / n_routable)
    X["region"], XN["region"] = c.build()
    node_ids["region"] = reg_ids
    CAT["region"] = {"layer": rl.astype(np.int64), "plane_net": plane_net[rl, ri, rj].astype(np.int64)}
    free_of = lambda k: float(free[reg_idx[k]])  # noqa: E731
    cap_of = lambda k: float(cap_mm[reg_idx[k]])  # noqa: E731
    blk_of = lambda k: float(blk[reg_idx[k]])  # noqa: E731
    plane_of = lambda k: float(pflag[reg_idx[k]])  # noqa: E731

    # ---------------- edges ----------------
    E: dict[EdgeType, tuple] = {}

    def put(et: EdgeType, src, dst, attrs, names):
        E[et] = edge_arrays(src, dst, attrs, names)

    def with_reverse(name, a, b, src, dst, attrs, names):
        put((a, name, b), src, dst, attrs, names)
        put((b, f"rev_{name}", a), dst, src, attrs, names)

    # membership
    s_, t_ = zip(*[(i, net_idx[p.net]) for i, p in enumerate(pins) if p.net]) if any(p.net for p in pins) else ((), ())
    with_reverse("member_of", "pad", "net", list(s_), list(t_), [[]] * len(s_), [])
    with_reverse("part_of", "pad", "component", list(range(len(pins))), [comp_idx[p.ref] for p in pins], [[]] * len(pins), [])
    vi = [(i, net_idx[v.net]) for i, v in enumerate(vias) if v.net]
    with_reverse("on_net", "via", "net", [a for a, _ in vi], [b for _, b in vi], [[]] * len(vi), [])
    oi = [(i, net_idx[o.net]) for i, o in enumerate(obs) if o.net]
    with_reverse("on_net", "obstacle", "net", [a for a, _ in oi], [b for _, b in oi], [[]] * len(oi), [])

    # spatial proximity: pads and components
    ppos = np.array([(p.position.x, p.position.y) for p in pins], np.float64).reshape(-1, 2)
    src, dst, dist = _knn(ppos, cfg.knn, cfg.near_radius_mm * MM)
    attrs = []
    for a, b, dd in zip(src, dst, dist):
        pa, pb = pins[a], pins[b]
        attrs.append([dd / scale, (pb.position.x - pa.position.x) / scale, (pb.position.y - pa.position.y) / scale,
                      float(pa.net is not None and pa.net == pb.net), float(pa.ref == pb.ref), float(bool(set(pa.layers) & set(pb.layers)))])
    put(("pad", "near", "pad"), src, dst, attrs, ["dist", "dx", "dy", "same_net", "same_component", "shares_layer"])
    cpos = np.array([(k.position.x, k.position.y) for k in comps], np.float64).reshape(-1, 2)
    src, dst, dist = _knn(cpos, cfg.component_knn, cfg.component_near_radius_mm * MM)
    attrs = [[dd / scale, (comps[b].position.x - comps[a].position.x) / scale, (comps[b].position.y - comps[a].position.y) / scale]
             for a, b, dd in zip(src, dst, dist)]
    put(("component", "near", "component"), src, dst, attrs, ["dist", "dx", "dy"])

    # corridor adjacency (same layer) and layer transition (same cell, adjacent layers)
    src, dst, attrs = [], [], []
    for (i, j, li), a in reg_idx.items():
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k2 = (i + di, j + dj, li)
            if k2 in reg_idx:
                src.append(a); dst.append(reg_idx[k2])
                attrs.append([di, dj, min(free_of((i, j, li)), free_of(k2)), min(cap_of((i, j, li)), cap_of(k2))])
    put(("region", "corridor", "region"), src, dst, attrs, ["dir_x", "dir_y", "min_free", "min_capacity_mm"])
    src, dst, attrs = [], [], []
    for (i, j, li), a in reg_idx.items():
        for lj in (li - 1, li + 1):
            k2 = (i, j, lj)
            if k2 in reg_idx:
                lo = min(li, lj)
                g = gaps[lo]
                src.append(a); dst.append(reg_idx[k2])
                attrs.append([abs(lj - li), (g or 0) / MM, float(g is not None),
                              1.0 - max(blk_of((i, j, li)), blk_of(k2)), max(plane_of((i, j, li)), plane_of(k2))])
    put(("region", "via_site", "region"), src, dst, attrs, ["layer_dist", "gap_mm", "gap_known", "via_allowed", "plane_crossing"])
    src, dst = [], []
    for (i, j, li), a in reg_idx.items():
        src.append(a); dst.append(li)
    with_reverse("in_layer", "region", "layer", src, dst, [[]] * len(src), [])
    src, dst, attrs = [], [], []
    for li in range(L - 1):
        for a, b in ((li, li + 1), (li + 1, li)):
            g = gaps[li]
            src.append(a); dst.append(b)
            attrs.append([(g or 0) / MM, float(g is not None), eps[li]])
    put(("layer", "stack", "layer"), src, dst, attrs, ["gap_mm", "gap_known", "epsilon_r"])
    src, dst, attrs = [], [], []
    for vi_, v in enumerate(vias):
        for li in range(lay_idx[v.layer_from], lay_idx[v.layer_to] + 1):
            src.append(vi_); dst.append(li)
            attrs.append([float(li in (lay_idx[v.layer_from], lay_idx[v.layer_to]))])
    put(("via", "spans", "layer"), src, dst, attrs, ["is_end"])

    # geometry coupling: pad/via/obstacle <-> region
    src, dst, attrs = [], [], []
    for pi, p in enumerate(pins):
        bb = ring_bbox(p.polygon())
        area = max(1.0, float(bb[2] - bb[0]) * (bb[3] - bb[1]))
        ii, jj = cell_range(bb)
        for i in ii:
            for j in jj:
                ox = min(bb[2], x0 + (i + 1) * cell) - max(bb[0], x0 + i * cell)
                oy = min(bb[3], y0 + (j + 1) * cell) - max(bb[1], y0 + j * cell)
                if ox <= 0 or oy <= 0:
                    continue
                for ly in p.layers:
                    k2 = (i, j, lay_idx[ly])
                    if k2 in reg_idx:
                        src.append(pi); dst.append(reg_idx[k2]); attrs.append([ox * oy / area])
    with_reverse("in_region", "pad", "region", src, dst, attrs, ["overlap_frac"])
    src, dst, attrs = [], [], []
    for vi_, v in enumerate(vias):
        i, j = int((v.position.x - x0) // cell), int((v.position.y - y0) // cell)
        for li in range(lay_idx[v.layer_from], lay_idx[v.layer_to] + 1):
            k2 = (i, j, li)
            if k2 in reg_idx:
                src.append(vi_); dst.append(reg_idx[k2]); attrs.append([1.0])
    with_reverse("in_region", "via", "region", src, dst, attrs, ["overlap_frac"])
    src, dst, attrs = [], [], []
    for oi_, o in enumerate(obs):
        ii, jj = cell_range(o.bbox)
        for i in ii:
            for j in jj:
                cov = sample_cover(i, j, o.ring, o.bbox)
                if cov <= 0:
                    continue
                for ly in (o.layers or lay_ids):
                    k2 = (i, j, lay_idx[ly])
                    if k2 in reg_idx:
                        src.append(oi_); dst.append(reg_idx[k2]); attrs.append([cov])
    with_reverse("overlaps", "obstacle", "region", src, dst, attrs, ["coverage"])

    # pair relationships
    src, dst, attrs = [], [], []
    for dp in d.relations.diff_pairs:
        a, b = net_idx[dp.positive], net_idx[dp.negative]
        row = [(dp.gap_nm or 0) / MM, (dp.width_nm or 0) / MM, float(dp.source == "inferred")]
        src += [a, b]; dst += [b, a]; attrs += [row, row]
    put(("net", "paired_with", "net"), src, dst, attrs, ["gap_mm", "width_mm", "inferred"])

    g = HeteroGraph(
        node_ids=node_ids, x=X, x_names=XN, cat=CAT,
        edge_index={et: v[0] for et, v in E.items()}, edge_attr={et: v[1] for et, v in E.items()},
        edge_attr_names={et: v[2] for et, v in E.items()},
        norm={"center_x_nm": cx, "center_y_nm": cy, "scale_nm": scale},
        meta={"design": d.name, "ir_version": d.ir_version, "layer_ids": lay_ids, "grid": (gx, gy), "cell_nm": cell,
              "net_classes": class_names, "obstacle_kinds": OBSTACLE_KINDS, "plane_nets_excluded_from_demand": sorted(k.value for k in skip_kinds)})
    return g
