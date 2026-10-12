"""KiCad .kicad_sch / .kicad_sym  ->  light-weight schematic and symbol models.

This extracts what the board model needs (references, values, footprint assignments,
labels, sheets, symbol pin names such as D/S/G). It does NOT build full connectivity from
wires; for authoritative nets export a netlist (`kicad-cli sch export netlist`) and read
that with `netlist.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sexpr import Node


@dataclass
class LibPin:
    number: str
    name: str
    etype: str            # passive, input, output, power_in, ...
    x: float
    y: float
    rot_deg: float
    unit: int = 0


@dataclass
class LibSymbol:
    name: str
    properties: dict = field(default_factory=dict)
    pins: list[LibPin] = field(default_factory=list)
    extends: str | None = None


@dataclass
class SchSymbol:
    lib_id: str
    reference: str
    value: str
    footprint: str
    x: float
    y: float
    rot_deg: float
    unit: int = 1
    dnp: bool = False
    properties: dict = field(default_factory=dict)
    uuid: str = ""


@dataclass
class SchLabel:
    kind: str             # label | global_label | hierarchical_label
    text: str
    x: float
    y: float
    rot_deg: float


@dataclass
class SchSheet:
    name: str
    file: str
    x: float
    y: float


@dataclass
class Schematic:
    metadata: dict = field(default_factory=dict)
    symbols: list[SchSymbol] = field(default_factory=list)
    labels: list[SchLabel] = field(default_factory=list)
    wires: list[list[tuple[float, float]]] = field(default_factory=list)
    junctions: list[tuple[float, float]] = field(default_factory=list)
    no_connects: list[tuple[float, float]] = field(default_factory=list)
    sheets: list[SchSheet] = field(default_factory=list)
    lib_symbols: dict[str, LibSymbol] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def symbol(self, reference: str) -> SchSymbol | None:
        return next((s for s in self.symbols if s.reference == reference), None)

    def pin_names(self, lib_id: str) -> dict[str, str]:
        """Map pin number -> pin name for a library symbol, e.g. {"3": "D", "2": "S", "1": "G"}."""
        ls = self.lib_symbols.get(lib_id)
        return {p.number: p.name for p in ls.pins} if ls else {}


def _props(node: Node) -> dict[str, str]:
    out = {}
    for p in node.children("property"):
        a = p.atoms()
        if len(a) >= 2:
            out[str(a[0])] = str(a[1])
    return out


def _at(node: Node) -> tuple[float, float, float]:
    a = [v for v in node.values("at") if isinstance(v, (int, float))]
    return (float(a[0]) if len(a) > 0 else 0.0, float(a[1]) if len(a) > 1 else 0.0,
            float(a[2]) if len(a) > 2 else 0.0)


def parse_lib_symbol(node: Node) -> LibSymbol:
    a = node.atoms()
    ls = LibSymbol(name=str(a[0]) if a else "", properties=_props(node))
    ext = node.value("extends")
    ls.extends = str(ext) if ext is not None else None

    def add_pins(container: Node, unit: int):
        for pin in container.children("pin"):
            pa = pin.atoms()
            x, y, rot = _at(pin)
            ls.pins.append(LibPin(
                number=str(pin.value("number", "")), name=str(pin.value("name", "")),
                etype=str(pa[0]) if pa else "", x=x, y=y, rot_deg=rot, unit=unit))

    add_pins(node, 0)
    for sub in node.children("symbol"):
        sa = sub.atoms()
        m = re.search(r"_(\d+)_(\d+)$", str(sa[0])) if sa else None
        add_pins(sub, int(m.group(1)) if m else 0)
    return ls


def _instance_reference(sym: Node) -> str | None:
    inst = sym.child("instances")
    if inst is None:
        return None
    for proj in inst.children("project"):
        for path in proj.children("path"):
            r = path.value("reference")
            if r:
                return str(r)
    return None


def parse_schematic(root: Node) -> Schematic:
    if root.name != "kicad_sch":
        raise ValueError(f"not a kicad_sch file (root is {root.name!r})")
    s = Schematic()
    s.metadata.update(version=root.value("version"), generator=root.value("generator"),
                      generator_version=root.value("generator_version"), uuid=root.value("uuid"))
    libs = root.child("lib_symbols")
    if libs is not None:
        for sym in libs.children("symbol"):
            ls = parse_lib_symbol(sym)
            s.lib_symbols[ls.name] = ls
    for sym in root.children("symbol"):
        lib_id = sym.value("lib_id")
        if lib_id is None:
            continue
        props = _props(sym)
        x, y, rot = _at(sym)
        ref = _instance_reference(sym) or props.get("Reference", "")
        u = sym.value("unit", 1)
        s.symbols.append(SchSymbol(
            lib_id=str(lib_id), reference=ref, value=props.get("Value", ""),
            footprint=props.get("Footprint", ""), x=x, y=y, rot_deg=rot,
            unit=int(u) if isinstance(u, (int, float)) else 1,
            dnp=str(sym.value("dnp", "no")) == "yes", properties=props,
            uuid=str(sym.value("uuid", ""))))
    for w in root.children("wire"):
        pts = [tuple(float(v) for v in xy.atoms()[:2]) for xy in (w.child("pts").children("xy") if w.child("pts") else [])]
        if len(pts) >= 2:
            s.wires.append(pts)
    for kind in ("label", "global_label", "hierarchical_label"):
        for lb in root.children(kind):
            a = lb.atoms()
            x, y, rot = _at(lb)
            s.labels.append(SchLabel(kind, str(a[0]) if a else "", x, y, rot))
    for j in root.children("junction"):
        s.junctions.append(j.xy("at"))
    for nc in root.children("no_connect"):
        s.no_connects.append(nc.xy("at"))
    for sh in root.children("sheet"):
        pr = _props(sh)
        x, y = sh.xy("at")
        s.sheets.append(SchSheet(pr.get("Sheetname", pr.get("Sheet name", "")),
                                 pr.get("Sheetfile", pr.get("Sheet file", "")), x, y))
    return s


def parse_symbol_lib(root: Node) -> dict[str, LibSymbol]:
    if root.name != "kicad_symbol_lib":
        raise ValueError(f"not a kicad_symbol_lib file (root is {root.name!r})")
    out = {}
    for sym in root.children("symbol"):
        ls = parse_lib_symbol(sym)
        out[ls.name] = ls
    return out
