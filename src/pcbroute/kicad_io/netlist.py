"""KiCad netlists (.net S-expression and .xml) -> Netlist model.

This is the authoritative schematic connectivity. Produce one with
`kicad-cli sch export netlist --format kicadsexpr` (or `--format kicadxml`).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .sexpr import Node


@dataclass
class NetNode:
    ref: str
    pin: str
    pinfunction: str = ""
    pintype: str = ""


@dataclass
class NetlistNet:
    code: str
    name: str
    nodes: list[NetNode] = field(default_factory=list)


@dataclass
class NetlistComponent:
    ref: str
    value: str = ""
    footprint: str = ""
    libsource: str = ""
    fields: dict = field(default_factory=dict)


@dataclass
class Netlist:
    version: str = ""
    source: str = ""
    components: dict[str, NetlistComponent] = field(default_factory=dict)
    nets: dict[str, NetlistNet] = field(default_factory=dict)

    def pins_on_net(self, name: str) -> list[NetNode]:
        n = self.nets.get(name)
        return n.nodes if n else []

    def net_of(self, ref: str, pin: str) -> str | None:
        for n in self.nets.values():
            for nd in n.nodes:
                if nd.ref == ref and nd.pin == pin:
                    return n.name
        return None


def parse_netlist(root: Node) -> Netlist:
    if root.name != "export":
        raise ValueError(f"not a KiCad netlist (root is {root.name!r})")
    nl = Netlist(version=str(root.value("version", "")))
    design = root.child("design")
    if design is not None:
        nl.source = str(design.value("source", ""))
    comps = root.child("components")
    for c in (comps.children("comp") if comps is not None else []):
        ref = str(c.value("ref", ""))
        ls = c.child("libsource")
        lib = f"{ls.value('lib', '')}:{ls.value('part', '')}" if ls is not None else ""
        fields = {}
        for p in c.children("property"):
            fields[str(p.value("name", ""))] = str(p.value("value", ""))
        fl = c.child("fields")
        for f in (fl.children("field") if fl is not None else []):
            a = f.atoms()
            fields[str(f.value("name", ""))] = str(a[-1]) if a else ""
        nl.components[ref] = NetlistComponent(ref, str(c.value("value", "")),
                                              str(c.value("footprint", "")), lib, fields)
    nets = root.child("nets")
    for n in (nets.children("net") if nets is not None else []):
        net = NetlistNet(code=str(n.value("code", "")), name=str(n.value("name", "")))
        for nd in n.children("node"):
            net.nodes.append(NetNode(str(nd.value("ref", "")), str(nd.value("pin", "")),
                                     str(nd.value("pinfunction", "")), str(nd.value("pintype", ""))))
        nl.nets[net.name] = net
    return nl


def parse_netlist_xml(text: str) -> Netlist:
    root = ET.fromstring(text)
    if root.tag != "export":
        raise ValueError(f"not a KiCad XML netlist (root is <{root.tag}>)")
    nl = Netlist(version=root.get("version", ""))
    src = root.find("design/source")
    nl.source = src.text or "" if src is not None else ""
    for c in root.findall("components/comp"):
        ref = c.get("ref", "")
        ls = c.find("libsource")
        lib = f"{ls.get('lib', '')}:{ls.get('part', '')}" if ls is not None else ""
        fields = {f.get("name", ""): (f.text or "") for f in c.findall("fields/field")}
        nl.components[ref] = NetlistComponent(
            ref, (c.findtext("value") or ""), (c.findtext("footprint") or ""), lib, fields)
    for n in root.findall("nets/net"):
        net = NetlistNet(code=n.get("code", ""), name=n.get("name", ""))
        for nd in n.findall("node"):
            net.nodes.append(NetNode(nd.get("ref", ""), nd.get("pin", ""),
                                     nd.get("pinfunction", ""), nd.get("pintype", "")))
        nl.nets[net.name] = net
    return nl
