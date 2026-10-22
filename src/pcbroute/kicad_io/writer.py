"""Write KiCad .kicad_pcb files from the Board model."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from .sexpr import Node, Sym, dumps
from ..board import Arc, Board, Footprint, Layer, Pad, Segment, Via, Zone


@dataclass
class WriteConfig:
    version: int = 20240108
    generator: str = "pcbroute"
    generator_version: str = "0.0.1"
    arc_tol_mm: float = 0.02
    max_arc_seg: int = 256


def _new_uuid() -> str:
    """Generate a KiCad-style UUID (8-4-4-4-12 hex)."""
    u = uuid.uuid4()
    return f"{u.hex[:8]}-{u.hex[8:12]}-{u.hex[12:16]}-{u.hex[16:20]}-{u.hex[20:]}"


def _fmt(v: float) -> float:
    """Format float with reasonable precision for KiCad."""
    return round(v, 6)


def _uuid_attr(u: str) -> tuple[str, str]:
    return ("uuid", u) if u else ("tstamp", u.replace("-", "")[:16])


def _locked_attr(locked: bool) -> tuple[str, str] | None:
    return ("locked", "yes") if locked else None


def _xy(x: float, y: float) -> Node:
    return Node("xy", [_fmt(x), _fmt(y)])


def _pts_node(name: str, pts: list[tuple[float, float]]) -> Node:
    children = [Node("xy", [_fmt(x), _fmt(y)]) for x, y in pts]
    return Node(name, children)


def _net_node(net: str | None, net_map: dict[str, int] | None = None) -> Node | None:
    if not net:
        return None
    if net_map is not None and net in net_map:
        return Node("net", [net_map[net], net])
    return Node("net_name", [net])


def write_layer_stackup(board: Board, config: WriteConfig) -> Node:
    """Write the layer stackup section."""
    layers = []
    for ly in board.stackup:
        if ly.kind == "copper":
            typ = "copper"
        elif ly.kind == "dielectric":
            typ = ly.material.lower() if ly.material and ly.material.lower() in ("core", "prepreg") else "core"
        else:
            continue
        atoms = [ly.name, ("type", typ)]
        if ly.thickness_mm > 0:
            atoms.append(("thickness", _fmt(ly.thickness_mm)))
        if ly.epsilon_r is not None:
            atoms.append(("epsilon_r", _fmt(ly.epsilon_r)))
        if ly.material:
            atoms.append(("material", ly.material))
        layers.append(Node("layer", atoms))
    return Node("layers", layers)


def write_setup(board: Board, config: WriteConfig) -> Node:
    """Write the setup section."""
    setup_children = [
        Node("stackup", [write_layer_stackup(board, config)]),
    ]
    return Node("setup", setup_children)


def write_pad(pad: Pad, net_map: dict[str, int] | None, copper_layers: list[str], config: WriteConfig) -> Node:
    """Write a pad."""
    args = [pad.number, Sym(pad.kind), Sym(pad.shape)]
    
    # Position
    args.append(Node("at", [_fmt(pad.x), _fmt(pad.y), _fmt(pad.rot_deg)]))
    
    # Size
    args.append(Node("size", [_fmt(pad.w), _fmt(pad.h)]))
    
    # Layers
    if pad.layers:
        layers = [ly for ly in pad.layers if ly in copper_layers]
        if layers:
            args.append(Node("layers", [Sym(ly) for ly in layers]))
    
    # Net
    net_node = _net_node(pad.net, net_map)
    if net_node:
        args.append(net_node)
    
    # Drill
    if pad.drill is not None and pad.drill > 0:
        args.append(Node("drill", [_fmt(pad.drill)]))
    
    # UUID
    if pad.uuid:
        args.append(Node(*_uuid_attr(pad.uuid)))
    
    # Pin function
    if pad.function:
        args.append(Node("pinfunction", [pad.function]))
    
    return Node("pad", args)


def write_footprint(fp: Footprint, pads: list[Pad], net_map: dict[str, int] | None, 
                    copper_layers: list[str], config: WriteConfig) -> Node:
    """Write a footprint."""
    args = [fp.lib_id]
    
    # Layer
    args.append(Node("layer", [fp.layer]))
    
    # Position
    args.append(Node("at", [_fmt(fp.x), _fmt(fp.y), _fmt(fp.rot_deg)]))
    
    # Properties
    args.append(Node("property", ["Reference", fp.ref, ("at", [0, 0]), ("layer", "F.SilkS")]))
    args.append(Node("property", ["Value", fp.value, ("at", [0, 0]), ("layer", "F.Fab")]))
    
    for key, val in fp.properties.items():
        if key.lower() not in ("reference", "value"):
            args.append(Node("property", [key, val, ("at", [0, 0])]))
    
    # UUID
    if fp.uuid:
        args.append(Node(*_uuid_attr(fp.uuid)))
    
    # Locked
    if fp.locked:
        args.append(Node("locked", []))
    
    # Pads
    for pad in pads:
        args.append(write_pad(pad, net_map, copper_layers, config))
    
    # Courtyard (if any)
    if fp.courtyard:
        for x1, y1, x2, y2 in fp.courtyard:
            args.append(Node("fp_line", [
                Node("start", [_fmt(x1), _fmt(y1)]),
                Node("end", [_fmt(x2), _fmt(y2)]),
                ("layer", f"{fp.layer}.CrtYd"),
            ]))
    
    return Node("footprint", args)


def write_segment(seg: Segment, net_map: dict[str, int] | None, config: WriteConfig) -> Node:
    """Write a track segment."""
    args = [
        Node("start", [_fmt(seg.x1), _fmt(seg.y1)]),
        Node("end", [_fmt(seg.x2), _fmt(seg.y2)]),
        Node("width", [_fmt(seg.width)]),
        Node("layer", [seg.layer]),
    ]
    net_node = _net_node(seg.net, net_map)
    if net_node:
        args.append(net_node)
    if seg.uuid:
        args.append(Node(*_uuid_attr(seg.uuid)))
    if seg.locked:
        args.append(Node("locked", []))
    return Node("segment", args)


def write_arc(arc: Arc, net_map: dict[str, int] | None, config: WriteConfig) -> Node:
    """Write an arc."""
    args = [
        Node("start", [_fmt(arc.start[0]), _fmt(arc.start[1])]),
        Node("mid", [_fmt(arc.mid[0]), _fmt(arc.mid[1])]),
        Node("end", [_fmt(arc.end[0]), _fmt(arc.end[1])]),
        Node("width", [_fmt(arc.width)]),
        Node("layer", [arc.layer]),
    ]
    net_node = _net_node(arc.net, net_map)
    if net_node:
        args.append(net_node)
    if arc.uuid:
        args.append(Node(*_uuid_attr(arc.uuid)))
    if arc.locked:
        args.append(Node("locked", []))
    return Node("arc", args)


def write_via(via: Via, net_map: dict[str, int] | None, config: WriteConfig) -> Node:
    """Write a via."""
    args = [
        Node("at", [_fmt(via.x), _fmt(via.y)]),
        Node("size", [_fmt(via.diameter)]),
        Node("drill", [_fmt(via.drill)]),
        Node("layers", [Sym(via.layers[0]), Sym(via.layers[1])]),
    ]
    net_node = _net_node(via.net, net_map)
    if net_node:
        args.append(net_node)
    if via.uuid:
        args.append(Node(*_uuid_attr(via.uuid)))
    if via.locked:
        args.append(Node("locked", []))
    return Node("via", args)


def write_zone(zone: Zone, net_map: dict[str, int] | None, config: WriteConfig) -> Node:
    """Write a zone (pour or keepout)."""
    args = [
        ("layer", zone.layer),
    ]
    net_node = _net_node(zone.net, net_map)
    if net_node:
        args.append(net_node)
    
    args.append(("hatch", "edge"))
    args.append(("min_thickness", 0.25))
    
    if zone.uuid:
        args.append(Node(*_uuid_attr(zone.uuid)))
    if zone.locked:
        args.append(Node("locked", []))
    
    if zone.keepout:
        ko_children = []
        for k, v in zone.keepout_rules.items():
            ko_children.append(Node(k, [v]))
        if ko_children:
            args.append(Node("keepout", ko_children))
    else:
        # Outline polygon
        if zone.polygon:
            args.append(_pts_node("polygon", zone.polygon))
        # Filled polygons - for simplicity we don't handle fills separately in the parser model
        # The parser stores filled polygons as separate Zone objects with filled=True
    
    return Node("zone", args)


def write_outline(board: Board, config: WriteConfig) -> list[Node]:
    """Write board outline (Edge.Cuts)."""
    if not board.outline:
        return []
    
    nodes = []
    for x1, y1, x2, y2 in board.outline:
        nodes.append(Node("gr_line", [
            Node("start", [_fmt(x1), _fmt(y1)]),
            Node("end", [_fmt(x2), _fmt(y2)]),
            ("layer", "Edge.Cuts"),
        ]))
    return nodes


def write_board(board: Board, config: WriteConfig = WriteConfig()) -> Node:
    """Write a complete kicad_pcb S-expression."""
    # Build net map (name -> index)
    net_names = set()
    for p in board.pads:
        if p.net:
            net_names.add(p.net)
    for s in board.segments:
        if s.net:
            net_names.add(s.net)
    for v in board.vias:
        if v.net:
            net_names.add(v.net)
    net_names = sorted(net_names)
    net_map = {name: i for i, name in enumerate(net_names)}
    
    # Copper layer names
    copper_layers = [ly.name for ly in board.stackup if ly.kind == "copper"]
    
    # Root node
    root_args = [
        ("version", config.version),
        ("generator", config.generator),
        ("generator_version", config.generator_version),
    ]
    
    children = [
        Node("net", [i, name]) for name, i in net_map.items()
    ]
    
    # Root-level layers list (required by parser)
    layer_items = []
    for i, ly in enumerate(board.stackup):
        if ly.kind == "copper":
            layer_items.append(Node("", [i, Sym(ly.name), Sym("signal")]))
        elif ly.kind == "dielectric":
            # Dielectric layers get indices too
            layer_items.append(Node("", [i, Sym(ly.name), Sym("dielectric")]))
    children.append(Node("layers", layer_items))
    
    # Setup
    children.append(write_setup(board, config))
    
    # Footprints and pads
    fp_index = 0
    pads_by_fp: dict[int, list[Pad]] = {}
    for p in board.pads:
        pads_by_fp.setdefault(p.fp_index, []).append(p)
    
    for fp in board.footprints:
        fp_pads = pads_by_fp.get(fp_index, [])
        children.append(write_footprint(fp, fp_pads, net_map, copper_layers, config))
        fp_index += 1
    
    # Tracks
    for seg in board.segments:
        if seg.arc_index is None:  # Only write non-arc segments
            children.append(write_segment(seg, net_map, config))
    
    # Arcs
    for arc in board.arcs:
        children.append(write_arc(arc, net_map, config))
    
    # Vias
    for via in board.vias:
        children.append(write_via(via, net_map, config))
    
    # Zones
    for zone in board.zones:
        children.append(write_zone(zone, net_map, config))
    
    # Outline
    for node in write_outline(board, config):
        children.append(node)
    
    return Node("kicad_pcb", root_args + children)


def write_kicad_pcb(board: Board, filepath: str, config: WriteConfig = WriteConfig()) -> None:
    """Write a Board to a .kicad_pcb file."""
    root = write_board(board, config)
    with open(filepath, "w") as f:
        f.write(dumps(root))