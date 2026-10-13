"""KiCad parser model (pcbroute.board.Board) -> canonical IR (pcbroute.ir.Design).

This is the only place that knows both worlds. Everything downstream of it sees only the IR.
"""
from __future__ import annotations

from .. import board as k
from ..kicad_io.project import ProjectSettings
from .board import Board, CopperLayer, Cutout, Dielectric, LayerKind, Stackup
from .components import Component, Keepout, KeepoutKind, PadShape, Pin, PinKind, PinRole, Side
from .copper import Copper, TrackArc, TrackSeg, Via, ViaKind, Zone, ZoneKind
from .design import Design, Provenance
from .geometry import Polygon, Pt, Ring, chain_segments, mm_to_nm, ring_area2, ring_bbox
from .inference import infer_criticality, infer_diff_pairs, infer_net_kind
from .nets import Net, NetClass, NetKind
from .relations import DiffPair, Relations
from .rules import Rules, ViaType

_SHAPES = {"rect": PadShape.RECT, "roundrect": PadShape.ROUNDRECT, "oval": PadShape.OVAL,
           "circle": PadShape.CIRCLE}
_PINKIND = {"smd": PinKind.SMD, "thru_hole": PinKind.THROUGH, "np_thru_hole": PinKind.NPTH}
_FORBID = {"tracks": "tracks", "vias": "vias", "copperpour": "copper", "pads": "pads", "footprints": "footprints"}

DEFAULT_TRACK_MM, DEFAULT_CLEARANCE_MM, DEFAULT_VIA_MM, DEFAULT_DRILL_MM = 0.25, 0.2, 0.8, 0.4


def _pt(x: float, y: float) -> Pt:
    return Pt(mm_to_nm(x), mm_to_nm(y))


def _ring(points) -> Ring:
    return tuple(_pt(x, y) for x, y in points)


def _edges_to_rings(edges) -> tuple[list[Ring], int]:
    segs = [((mm_to_nm(a), mm_to_nm(b)), (mm_to_nm(c), mm_to_nm(d))) for a, b, c, d in edges]
    return chain_segments(segs)


def _stackup(kb: k.Board, kinds: dict[str, LayerKind]) -> tuple[Stackup, dict[str, str]]:
    copper_src = [ly for ly in kb.stackup if ly.kind == "copper"]
    if not copper_src:
        copper_src = [k.Layer(n, "copper", 0.035, i) for i, n in enumerate(kb.copper_layers())]
    n = len(copper_src)
    name_to_id = {ly.name: f"L{i}" for i, ly in enumerate(copper_src)}
    copper = tuple(CopperLayer(
        id=f"L{i}", index=i, kind=kinds.get(ly.name, LayerKind.SIGNAL), thickness_nm=mm_to_nm(ly.thickness_mm),
        source_name=ly.name, side="top" if i == 0 else "bottom" if i == n - 1 else "inner")
        for i, ly in enumerate(copper_src))
    diels: list[Dielectric] = []
    last: str | None = None
    pending: list[k.Layer] = []
    for ly in kb.stackup:
        if ly.kind == "dielectric":
            pending.append(ly)
        elif ly.kind == "copper":
            cid = name_to_id[ly.name]
            if last is not None:
                diels += [Dielectric(d.name, mm_to_nm(d.thickness_mm), last, cid, d.epsilon_r, None, d.material)
                          for d in pending]
            last, pending = cid, []
    return Stackup(copper, tuple(diels), defined_in_source=bool(kb.metadata.get("stackup_defined", False))), name_to_id


def _infer_layer_kinds(kb: k.Board, board_area_mm2: float) -> dict[str, LayerKind]:
    """Plane = a zone covering at least half the board while the layer holds almost no tracks (inner layers only)."""
    layers = [ly.name for ly in kb.stackup if ly.kind == "copper"] or kb.copper_layers()
    tracks = {n: sum(1 for s in kb.segments if s.layer == n) for n in layers}
    total = max(1, sum(tracks.values()))
    out: dict[str, LayerKind] = {}
    for i, name in enumerate(layers):
        zs = [z for z in kb.zones if z.layer == name and not z.filled and not z.keepout and z.net and len(z.polygon) >= 3]
        cover = 0.0
        net = None
        for z in zs:
            a = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(z.polygon, z.polygon[1:] + z.polygon[:1]))) / 2
            if a > cover:
                cover, net = a, z.net
        if board_area_mm2 > 0 and cover / board_area_mm2 >= 0.5:
            outer = i in (0, len(layers) - 1)
            if outer or tracks[name] > 0.05 * total:
                out[name] = LayerKind.MIXED
            else:
                out[name] = LayerKind.GROUND if infer_net_kind(net) == NetKind.GROUND else LayerKind.POWER
        else:
            out[name] = LayerKind.SIGNAL
    return out


def _outline(kb: k.Board) -> tuple[Ring, tuple[Cutout, ...], bool, list[str]]:
    warns: list[str] = []
    rings, open_chains = _edges_to_rings(kb.outline)
    if open_chains:
        warns.append(f"{open_chains} open chain(s) on Edge.Cuts were ignored")
    if not rings:
        xs = [p.x for p in kb.pads] + [s.x1 for s in kb.segments]
        ys = [p.y for p in kb.pads] + [s.y1 for s in kb.segments]
        if not xs:
            xs, ys = [0.0, 1.0], [0.0, 1.0]
        x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
        return _ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), (), True, warns
    rings.sort(key=lambda r: -abs(ring_area2(r)))
    cut = tuple(Cutout(f"cutout:{i}", r) for i, r in enumerate(rings[1:]))
    return rings[0], cut, False, warns


def ir_from_kicad(kb: k.Board, settings: ProjectSettings | None = None, *, name: str = "design",
                  layer_kinds: dict[str, LayerKind] | None = None, grid_nm: int = 10_000) -> Design:
    warnings: list[str] = list(kb.warnings)

    # ---- outline first: layer-kind inference needs the board area
    outline, cutouts, inferred, w = _outline(kb)
    warnings += w
    x0, y0, x1, y1 = ring_bbox(outline)
    board_area_mm2 = abs(ring_area2(outline)) / 2 / 1e12

    # ---- stackup and layer kinds
    kinds = _infer_layer_kinds(kb, board_area_mm2)
    if layer_kinds:
        kinds.update(layer_kinds)
    stackup, lid = _stackup(kb, kinds)
    inferred_planes = [n for n, v in kinds.items() if v in (LayerKind.POWER, LayerKind.GROUND, LayerKind.MIXED)]
    if inferred_planes and not layer_kinds:
        warnings.append(f"layer kinds inferred from zone coverage (verify): "
                        + ", ".join(f"{n}={kinds[n].value}" for n in inferred_planes))
    all_layers = stackup.layer_ids()

    def lids(names) -> tuple[str, ...]:
        return tuple(lid[n] for n in names if n in lid)

    # ---- nets: collect names, then rules
    names = sorted({n for n in kb.nets()})
    pairs = infer_diff_pairs(names)
    paired = {n for _, p, m in pairs for n in (p, m)}

    cls_src = (settings.net_classes if settings else {}) or {}
    classes = tuple(NetClass(
        c.name, mm_to_nm(c.track_width if c.track_width is not None else DEFAULT_TRACK_MM),
        mm_to_nm(c.clearance if c.clearance is not None else DEFAULT_CLEARANCE_MM),
        mm_to_nm(c.via_diameter) if c.via_diameter is not None else None,
        mm_to_nm(c.via_drill) if c.via_drill is not None else None,
        mm_to_nm(c.diff_pair_width) if c.diff_pair_width is not None else None,
        mm_to_nm(c.diff_pair_gap) if c.diff_pair_gap is not None else None) for c in cls_src.values())
    defaults_assumed = not classes
    if not classes:
        classes = (NetClass("Default", mm_to_nm(DEFAULT_TRACK_MM), mm_to_nm(DEFAULT_CLEARANCE_MM),
                            mm_to_nm(DEFAULT_VIA_MM), mm_to_nm(DEFAULT_DRILL_MM)),)
        warnings.append("no project settings: typical net-class values assumed (0.25 mm track, 0.2 mm clearance)")
    cls_by_name = {c.name: c for c in classes}
    default_cls = cls_by_name.get("Default", classes[0])

    dr = (settings.design_rules if settings else {}) or {}

    def rule_nm(key, fallback_mm):
        v = dr.get(key)
        return mm_to_nm(v if isinstance(v, (int, float)) else fallback_mm)

    via_types = tuple(ViaType(c.name, c.via_diameter_nm, c.via_drill_nm, all_layers[0], all_layers[-1])
                      for c in classes if c.via_diameter_nm and c.via_drill_nm)
    rules = Rules(
        min_track_nm=rule_nm("min_track_width", 0.1), min_clearance_nm=rule_nm("min_clearance", 0.1),
        min_via_diameter_nm=rule_nm("min_via_diameter", 0.4), min_via_drill_nm=rule_nm("min_through_hole_diameter", 0.2),
        min_annulus_nm=rule_nm("min_via_annular_width", 0.1), net_classes=classes, via_types=via_types,
        default_via=via_types[0].name if via_types else None,
        defaults_assumed=defaults_assumed and not dr)

    # ---- components and pins
    refs_seen: set[str] = set()
    pin_ids_seen: set[str] = set()
    pads_by_fp: dict[int, list[k.Pad]] = {}
    for p in kb.pads:
        pads_by_fp.setdefault(p.fp_index, []).append(p)
    net_pins: dict[str, list[str]] = {n: [] for n in names}
    components: list[Component] = []
    for fi, f in enumerate(kb.footprints):
        ref = f.ref or f"FP{fi}"
        if ref in refs_seen:
            ref = f"{ref}@{fi}"
            warnings.append(f"duplicate reference replaced by {ref}")
        refs_seen.add(ref)
        pins = []
        for p in pads_by_fp.get(fi, []):
            pid = f"{ref}.{p.number}"
            while pid in pin_ids_seen:
                pid += "'"
            pin_ids_seen.add(pid)
            nk = infer_net_kind(p.net) if p.net else None
            role = (PinRole.UNKNOWN if nk is None else PinRole.GROUND if nk == NetKind.GROUND
                    else PinRole.POWER if nk == NetKind.POWER else PinRole.SIGNAL)
            pins.append(Pin(
                id=pid, ref=ref, number=p.number, function=p.function, net=p.net, layers=lids(p.layers),
                position=_pt(p.x, p.y), shape=_SHAPES.get(p.shape, PadShape.CUSTOM), width_nm=mm_to_nm(p.w),
                height_nm=mm_to_nm(p.h), rotation_deg=p.rot_deg, kind=_PINKIND.get(p.kind, PinKind.OTHER),
                drill_nm=mm_to_nm(p.drill) if p.drill else None, role=role, source_id=p.uuid))
            if p.net:
                net_pins[p.net].append(pid)
        rings, _ = _edges_to_rings(f.courtyard)
        keepouts = tuple(Keepout(f"{ref}:courtyard:{i}", KeepoutKind.COURTYARD, r) for i, r in enumerate(rings))
        components.append(Component(
            ref=ref, footprint=f.lib_id, value=f.value, side=Side.BOTTOM if f.layer.startswith("B") else Side.TOP,
            position=_pt(f.x, f.y), rotation_deg=f.rot_deg, pins=tuple(pins), keepouts=keepouts, locked=f.locked,
            properties=dict(f.properties), source_id=f.uuid))

    # ---- nets
    nets: list[Net] = []
    for i, n in enumerate(names):
        c = cls_by_name.get(settings.netclass_for(n).name, default_cls) if settings and settings.netclass_for(n) else default_cls
        kind = infer_net_kind(n)
        crit = infer_criticality(kind, n in paired, c.width_nm / max(1, default_cls.width_nm))
        nets.append(Net(name=n, index=i, pins=tuple(net_pins[n]), net_class=c.name, width_nm=c.width_nm,
                        clearance_nm=c.clearance_nm, layer_mask=all_layers, criticality=crit,
                        criticality_source="heuristic", kind=kind))
    if nets:
        warnings.append("net criticality and kind are name-based heuristics; layer masks default to all layers")

    # ---- copper
    segs = [TrackSeg(f"track:{i}", s.net, lid[s.layer], _pt(s.x1, s.y1), _pt(s.x2, s.y2), mm_to_nm(s.width), s.locked, s.uuid)
            for i, s in enumerate(s for s in kb.segments if s.arc_index is None) if s.layer in lid]
    arcs = [TrackArc(f"arc:{i}", a.net, lid[a.layer], _pt(*a.start), _pt(*a.mid), _pt(*a.end), mm_to_nm(a.width),
                     a.locked, a.uuid) for i, a in enumerate(kb.arcs) if a.layer in lid]
    n_layers = len(all_layers)
    vias = []
    for i, v in enumerate(kb.vias):
        if v.layers[0] not in lid or v.layers[1] not in lid:
            continue
        a, b = sorted((stackup.index(lid[v.layers[0]]), stackup.index(lid[v.layers[1]])))
        kind = (ViaKind.THROUGH if (a, b) == (0, n_layers - 1)
                else ViaKind.BLIND if a == 0 or b == n_layers - 1 else ViaKind.BURIED)
        vias.append(Via(f"via:{i}", v.net, _pt(v.x, v.y), mm_to_nm(v.diameter), mm_to_nm(v.drill),
                        all_layers[a], all_layers[b], kind, v.locked, v.uuid))
    zones: list[Zone] = []
    by_key: dict[tuple[int, str], dict] = {}
    for z in kb.zones:
        if z.layer not in lid:
            continue
        slot = by_key.setdefault((z.zone_index, z.layer), {"outline": None, "fills": [], "z": z})
        if z.filled:
            slot["fills"].append(Polygon(_ring(z.polygon))) if len(z.polygon) >= 3 else None
        else:
            slot["outline"], slot["z"] = z, z
    for (zi, layer), slot in sorted(by_key.items()):
        z = slot["z"]
        if slot["outline"] is None or len(slot["outline"].polygon) < 3:
            continue
        forbids = tuple(sorted({_FORBID[kk] for kk, vv in z.keepout_rules.items() if vv == "not_allowed" and kk in _FORBID}))
        zones.append(Zone(f"zone:{zi}:{lid[layer]}", z.net, lid[layer], ZoneKind.KEEPOUT if z.keepout else ZoneKind.POUR,
                          _ring(slot["outline"].polygon), tuple(slot["fills"]), forbids, z.locked, z.uuid))
    copper = Copper(tuple(segs), tuple(arcs), tuple(vias), tuple(zones))

    # ---- relations
    dps = []
    for base, pos, neg in pairs:
        c = cls_by_name.get(next(n.net_class for n in nets if n.name == pos), default_cls)
        dps.append(DiffPair(base, pos, neg, gap_nm=c.diff_pair_gap_nm, width_nm=c.diff_pair_width_nm, source="inferred"))
    if dps:
        warnings.append(f"{len(dps)} differential pair(s) inferred from net names; confirm before they change routing")

    board = Board(outline=outline, stackup=stackup, cutouts=cutouts, grid_nm=grid_nm,
                  thickness_nm=None, outline_inferred=inferred)
    prov = Provenance(source_format="kicad_pcb", source_version=str(kb.metadata.get("version") or ""),
                      generator=" ".join(str(x) for x in (kb.metadata.get("generator"), kb.metadata.get("generator_version")) if x),
                      warnings=tuple(warnings))
    return Design(board=board, components=tuple(components), nets=tuple(nets), copper=copper,
                  relations=Relations(diff_pairs=tuple(dps)), rules=rules, provenance=prov, name=name)
