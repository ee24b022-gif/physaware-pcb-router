"""KiCad .kicad_pcb / .kicad_mod  ->  Board model.

Supports the S-expression PCB format from KiCad 5 through 9 (net ids as in 5-8,
net names as in 9). Units are millimetres.
"""
from __future__ import annotations

import math
import re

from ..board import Arc, Board, Footprint, Layer, Pad, Segment, Via, Zone, rotate_kicad
from .sexpr import Node, Sym

ARC_TOL_MM = 0.02


def _num(v, default: float = 0.0) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _uuid(item: Node) -> str:
    v = item.value("uuid", item.value("tstamp", ""))
    return str(v) if v else ""


def _locked(item: Node) -> bool:
    if str(item.value("locked", "no")) == "yes":
        return True
    return any(isinstance(a, Sym) and a == "locked" for a in item.atoms())


def _courtyard(fp: Node, fx: float, fy: float, frot: float) -> list[tuple[float, float, float, float]]:
    """Courtyard edges in board coordinates (from fp_line/fp_rect/fp_poly/fp_circle on *.CrtYd)."""
    def tr(p):
        dx, dy = rotate_kicad(p[0], p[1], frot)
        return (fx + dx, fy + dy)

    def ring(pts):
        t = [tr(p) for p in pts]
        return [(a[0], a[1], b[0], b[1]) for a, b in zip(t, t[1:] + t[:1])]

    out: list[tuple[float, float, float, float]] = []
    for n in fp.children():
        if not str(n.value("layer", "")).endswith(".CrtYd"):
            continue
        if n.name == "fp_line":
            a, b = tr(n.xy("start")), tr(n.xy("end"))
            out.append((a[0], a[1], b[0], b[1]))
        elif n.name == "fp_rect":
            (x1, y1), (x2, y2) = n.xy("start"), n.xy("end")
            out += ring([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        elif n.name == "fp_poly":
            out += ring(_pts(n))
        elif n.name == "fp_circle":
            (cx, cy), (ex, ey) = n.xy("center"), n.xy("end")
            r = math.hypot(ex - cx, ey - cy)
            out += ring([(cx + r * math.cos(2 * math.pi * i / 32), cy + r * math.sin(2 * math.pi * i / 32))
                         for i in range(32)])
    return out


def _copper_sort_key(name: str) -> int:
    if name == "F.Cu":
        return 0
    if name == "B.Cu":
        return 10_000
    m = re.match(r"In(\d+)\.Cu$", name)
    return int(m.group(1)) if m else 5_000


def _copper_layer_names(root: Node) -> list[str]:
    names = []
    layers = root.child("layers")
    if layers is not None:
        for item in layers.children(""):
            a = item.atoms()
            if len(a) >= 2 and str(a[1]).endswith(".Cu"):
                names.append(str(a[1]))
    return sorted(names, key=_copper_sort_key)


def _net_of(item: Node, net_map: dict[int, str]) -> str | None:
    n = item.child("net")
    name: str | None = None
    if n is not None:
        a = n.atoms()
        if len(a) >= 2:
            name = str(a[1])
        elif len(a) == 1:
            if isinstance(a[0], int):
                name = net_map.get(a[0], f"net{a[0]}") if a[0] != 0 else ""
            else:
                name = str(a[0])
    if not name:
        nn = item.value("net_name")
        name = str(nn) if nn else ""
    return name or None


def _expand_layers(atoms: list, copper: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for a in atoms:
        s = str(a)
        if s == "*.Cu":
            out.extend(copper or ["F.Cu", "B.Cu"])
        elif s == "F&B.Cu":
            out.extend(["F.Cu", "B.Cu"])
        elif s.endswith(".Cu"):
            out.append(s)
    seen: dict[str, None] = {}
    for s in out:
        seen.setdefault(s, None)
    return tuple(seen)


def _read_stackup(root: Node, board: Board, copper: list[str]) -> None:
    setup = root.child("setup")
    st = setup.child("stackup") if setup is not None else None
    idx = 0
    if st is not None:
        for ly in st.children("layer"):
            a = ly.atoms()
            lname = str(a[0]) if a else ""
            typ = str(ly.value("type", "")).lower()
            if typ == "copper":
                kind = "copper"
            elif typ in ("core", "prepreg", "dielectric") or lname.lower().startswith("dielectric"):
                kind = "dielectric"
            else:
                continue
            th = ly.value("thickness")
            if th is None:
                th = 0.035 if kind == "copper" else 0.0
                board.warnings.append(f"stackup layer {lname!r} has no thickness; assumed {th} mm")
            eps = ly.value("epsilon_r")
            mat = ly.value("material")
            board.stackup.append(Layer(
                name=lname, kind=kind, thickness_mm=float(th),
                copper_index=idx if kind == "copper" else None,
                epsilon_r=float(eps) if isinstance(eps, (int, float)) else None,
                material=str(mat) if mat is not None else None,
            ))
            if kind == "copper":
                idx += 1
        board.metadata["stackup_defined"] = True
    else:
        board.metadata["stackup_defined"] = False
        board.warnings.append(
            "no stackup in file; copper assumed 0.035 mm and dielectric thicknesses are unknown "
            "(set them in KiCad: Board Setup > Physical Stackup)"
        )
        for i, name in enumerate(copper):
            board.stackup.append(Layer(name=name, kind="copper", thickness_mm=0.035, copper_index=i))


def _fp_props(fp: Node) -> dict[str, str]:
    props: dict[str, str] = {}
    for p in fp.children("property"):
        a = p.atoms()
        if len(a) >= 2:
            props[str(a[0])] = str(a[1])
    for t in fp.children("fp_text"):
        a = t.atoms()
        if len(a) >= 2 and str(a[0]) in ("reference", "value"):
            props.setdefault(str(a[0]).capitalize(), str(a[1]))
    return props


def _pts(node: Node | None) -> list[tuple[float, float]]:
    if node is None:
        return []
    pts = node.child("pts")
    src = pts if pts is not None else node
    out = []
    for xy in src.children("xy"):
        n = [v for v in xy.atoms() if isinstance(v, (int, float))]
        if len(n) >= 2:
            out.append((float(n[0]), float(n[1])))
    return out


def _drill(pad: Node) -> float | None:
    d = pad.child("drill")
    if d is None:
        return None
    nums = [v for v in d.atoms() if isinstance(v, (int, float))]
    return float(nums[0]) if nums else None


def _parse_footprint(fp: Node, net_map: dict[int, str], copper: list[str]) -> tuple[Footprint, list[Pad]]:
    a = fp.atoms()
    lib_id = str(a[0]) if a else ""
    fx, fy = fp.xy("at")
    at = fp.child("at")
    nums = [v for v in at.atoms() if isinstance(v, (int, float))] if at is not None else []
    frot = float(nums[2]) if len(nums) >= 3 else 0.0
    layer = str(fp.value("layer", "F.Cu"))
    props = _fp_props(fp)
    ref = props.get("Reference", "")
    footprint = Footprint(ref=ref, value=props.get("Value", ""), lib_id=lib_id,
                          x=fx, y=fy, rot_deg=frot, layer=layer, properties=props,
                          uuid=_uuid(fp), locked=_locked(fp), courtyard=_courtyard(fp, fx, fy, frot))
    pads: list[Pad] = []
    for pad in fp.children("pad"):
        pa = pad.atoms()
        number = str(pa[0]) if pa else ""
        kind = str(pa[1]) if len(pa) > 1 else "smd"
        shape = str(pa[2]) if len(pa) > 2 else "rect"
        pat = pad.child("at")
        pn = [v for v in pat.atoms() if isinstance(v, (int, float))] if pat is not None else []
        lx = float(pn[0]) if len(pn) > 0 else 0.0
        ly = float(pn[1]) if len(pn) > 1 else 0.0
        # Assumption to verify on a real file: pad position is in the footprint's unrotated frame,
        # and the pad angle in a .kicad_pcb is already absolute (it includes the footprint angle).
        pang = float(pn[2]) if len(pn) > 2 else 0.0
        dx, dy = rotate_kicad(lx, ly, frot)
        w, h = pad.xy("size")
        lyrs = _expand_layers(pad.values("layers"), copper)
        pads.append(Pad(
            ref=ref, number=number, net=_net_of(pad, net_map),
            layer=lyrs[0] if lyrs else "", x=fx + dx, y=fy + dy, w=w, h=h,
            rot_deg=pang, shape=shape, layers=lyrs, kind=kind, drill=_drill(pad), uuid=_uuid(pad),
            function=str(pad.value("pinfunction", "")),
        ))
    return footprint, pads


def _arc_points(start, mid, end, tol: float = ARC_TOL_MM, max_seg: int = 256) -> list[tuple[float, float]]:
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        return [start, end]
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)
    tau = 2 * math.pi
    ccw = (a3 - a1) % tau
    sweep = ccw if (a2 - a1) % tau <= ccw else ccw - tau
    step = 2 * math.acos(max(-1.0, min(1.0, 1 - tol / r))) if r > tol else math.pi / 4
    n = max(2, min(max_seg, math.ceil(abs(sweep) / max(step, 1e-6))))
    pts = [(ux + r * math.cos(a1 + sweep * i / n), uy + r * math.sin(a1 + sweep * i / n)) for i in range(n + 1)]
    pts[0], pts[-1] = start, end
    return pts


def _zone_layers(z: Node, copper: list[str]) -> list[str]:
    one = z.value("layer")
    if one is not None:
        return list(_expand_layers([one], copper)) or [str(one)]
    ls = z.values("layers")
    out = list(_expand_layers(ls, copper))
    return out or [str(x) for x in ls]


def _read_zones(root: Node, board: Board, net_map: dict[int, str], copper: list[str]) -> None:
    for zi, z in enumerate(root.children("zone")):
        zu, zl = _uuid(z), _locked(z)
        net = _net_of(z, net_map)
        ko = z.child("keepout")
        rules = {c.name: str(c.atoms()[0]) for c in ko.children() if c.atoms()} if ko is not None else {}
        poly = _pts(z.child("polygon"))
        for layer in _zone_layers(z, copper):
            board.zones.append(Zone(net=net, layer=layer, polygon=poly, keepout=ko is not None,
                                    filled=False, keepout_rules=rules,
                                    uuid=zu, zone_index=zi, locked=zl))
        for fpoly in z.children("filled_polygon"):
            board.zones.append(Zone(net=net, layer=str(fpoly.value("layer", "")), polygon=_pts(fpoly),
                                    keepout=ko is not None, filled=True, keepout_rules=rules,
                                    uuid=zu, zone_index=zi, locked=zl))


def _read_outline(root: Node, board: Board) -> None:
    def add_poly(pts, closed):
        for a, b in zip(pts, pts[1:] + ([pts[0]] if closed and pts else [])):
            board.outline.append((a[0], a[1], b[0], b[1]))

    for n in root.children():
        if n.name not in ("gr_line", "gr_arc", "gr_rect", "gr_circle", "gr_poly", "gr_curve"):
            continue
        if n.value("layer") != "Edge.Cuts":
            continue
        if n.name == "gr_line":
            a, b = n.xy("start"), n.xy("end")
            board.outline.append((a[0], a[1], b[0], b[1]))
        elif n.name == "gr_arc":
            if n.child("mid") is None:
                board.warnings.append("legacy gr_arc (center/angle form) on Edge.Cuts skipped")
                continue
            add_poly(_arc_points(n.xy("start"), n.xy("mid"), n.xy("end")), False)
        elif n.name == "gr_rect":
            (x1, y1), (x2, y2) = n.xy("start"), n.xy("end")
            add_poly([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], True)
        elif n.name == "gr_circle":
            (cx, cy), (ex, ey) = n.xy("center"), n.xy("end")
            r = math.hypot(ex - cx, ey - cy)
            add_poly([(cx + r * math.cos(2 * math.pi * i / 64), cy + r * math.sin(2 * math.pi * i / 64))
                      for i in range(64)], True)
        elif n.name == "gr_poly":
            add_poly(_pts(n), True)
        else:
            board.warnings.append("Bezier gr_curve on Edge.Cuts skipped")


def parse_board(root: Node) -> Board:
    if root.name != "kicad_pcb":
        raise ValueError(f"not a kicad_pcb file (root is {root.name!r})")
    b = Board()
    b.metadata.update(version=root.value("version"), generator=root.value("generator"),
                      generator_version=root.value("generator_version"))
    net_map: dict[int, str] = {}
    for n in root.children("net"):
        a = n.atoms()
        if len(a) >= 2 and isinstance(a[0], int):
            net_map[a[0]] = str(a[1])
    copper = _copper_layer_names(root)
    b.metadata["copper_layers"] = copper
    _read_stackup(root, b, copper)

    for fp in list(root.children("footprint")) + list(root.children("module")):
        f, pads = _parse_footprint(fp, net_map, copper)
        for p in pads:
            p.fp_index = len(b.footprints)
        b.footprints.append(f)
        b.pads.extend(pads)

    for s in root.children("segment"):
        (x1, y1), (x2, y2) = s.xy("start"), s.xy("end")
        b.segments.append(Segment(_net_of(s, net_map), str(s.value("layer", "")),
                                  x1, y1, x2, y2, _num(s.value("width")), uuid=_uuid(s), locked=_locked(s)))
    for a in root.children("arc"):
        pts = _arc_points(a.xy("start"), a.xy("mid"), a.xy("end"))
        net, layer, width = _net_of(a, net_map), str(a.value("layer", "")), _num(a.value("width"))
        idx = len(b.arcs)
        b.arcs.append(Arc(net, layer, a.xy("start"), a.xy("mid"), a.xy("end"), width,
                          uuid=_uuid(a), locked=_locked(a)))
        for p, q in zip(pts, pts[1:]):
            b.segments.append(Segment(net, layer, p[0], p[1], q[0], q[1], width, uuid=_uuid(a),
                                      locked=_locked(a), arc_index=idx))
    for v in root.children("via"):
        x, y = v.xy("at")
        ls = [str(x) for x in v.values("layers")]
        b.vias.append(Via(_net_of(v, net_map), x, y, _num(v.value("size")), _num(v.value("drill")),
                          (ls[0], ls[1]) if len(ls) >= 2 else ("F.Cu", "B.Cu"),
                          uuid=_uuid(v), locked=_locked(v)))
    _read_zones(root, b, net_map, copper)
    _read_outline(root, b)
    return b


def parse_footprint_file(root: Node) -> Board:
    """A .kicad_mod library footprint as a one-footprint Board (pads in local coordinates)."""
    if root.name not in ("footprint", "module"):
        raise ValueError(f"not a footprint file (root is {root.name!r})")
    b = Board()
    b.metadata.update(version=root.value("version"), generator=root.value("generator"))
    f, pads = _parse_footprint(root, {}, [])
    if not f.ref:
        f.ref = "REF**"
    for p in pads:
        p.ref = f.ref
    b.footprints.append(f)
    b.pads.extend(pads)
    return b
