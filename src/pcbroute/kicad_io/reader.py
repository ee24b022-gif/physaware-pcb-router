"""One entry point for any KiCad file: detect the type, parse it, return a typed model."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import netlist as nl_mod
from . import pcb as pcb_mod
from . import project as proj_mod
from . import schematic as sch_mod
from .sexpr import Node, ParseError, parse, read_text

# What each kind's `.model` holds:
#   pcb / footprint -> Board           schematic -> Schematic          symbol_lib -> dict[str, LibSymbol]
#   netlist -> Netlist                 project -> ProjectSettings      rules -> list[dict]
#   lib_table -> list[dict]            dsn / ses -> summary dict       project_local -> dict (raw JSON)
#   worksheet / sexpr / json / xml -> None (see `.root` for the parsed tree)
#   legacy -> None (KiCad 5 and older text formats are not parsed)

_HEAD_KIND = {
    "kicad_pcb": "pcb", "kicad_sch": "schematic", "kicad_symbol_lib": "symbol_lib",
    "footprint": "footprint", "module": "footprint", "export": "netlist",
    "pcb": "dsn", "session": "ses", "fp_lib_table": "lib_table", "sym_lib_table": "lib_table",
    "kicad_wks": "worksheet", "drawing_sheet": "worksheet", "version": "rules",
}
_EXT_KIND = {
    ".kicad_pcb": "pcb", ".kicad_sch": "schematic", ".kicad_sym": "symbol_lib", ".kicad_mod": "footprint",
    ".kicad_pro": "project", ".kicad_prl": "project_local", ".kicad_dru": "rules", ".kicad_wks": "worksheet",
    ".dsn": "dsn", ".ses": "ses", ".net": "netlist",
}
_LEGACY_PREFIXES = ("EESchema", "PCBNEW", "Eeschema", "libModules")


@dataclass
class KicadFile:
    path: Path
    kind: str
    version: Any = None
    generator: Any = None
    root: Node | None = None
    model: Any = None
    warnings: list[str] = field(default_factory=list)


def detect_kind(path: Path, text: str) -> str:
    head = text.lstrip()[:400]
    if head.startswith(_LEGACY_PREFIXES):
        return "legacy"
    if head.startswith("{"):
        return {"": "json", ".kicad_pro": "project", ".kicad_prl": "project_local"}.get(path.suffix.lower(), "json")
    if head.startswith("<"):
        return "netlist_xml" if "<export" in head else "xml"
    m = re.match(r"\(\s*([^\s()\"]+)", head)
    if m:
        kind = _HEAD_KIND.get(m.group(1).lower())
        if kind:
            return kind
    return _EXT_KIND.get(path.suffix.lower(), "sexpr" if head.startswith("(") else "unknown")


def read_kicad(path) -> KicadFile:
    """Read any KiCad file. Raises ParseError if the text is not valid S-expression/JSON/XML."""
    p = Path(path)
    text = read_text(p)
    kind = detect_kind(p, text)
    kf = KicadFile(path=p, kind=kind)

    if kind == "legacy":
        kf.warnings.append("KiCad 5 or older text format; open it in a current KiCad and save it to convert")
        return kf
    if kind in ("project", "project_local", "json"):
        data = json.loads(text)
        kf.model = proj_mod.parse_project(data) if kind == "project" else data
        kf.version = (data.get("meta") or {}).get("version")
        return kf
    if kind in ("netlist_xml", "xml"):
        if kind == "netlist_xml":
            kf.model = nl_mod.parse_netlist_xml(text)
            kf.kind = "netlist"
            kf.version = kf.model.version
        else:
            try:
                kf.root = None
                ET.fromstring(text)
            except ET.ParseError as e:
                raise ParseError(str(e)) from e
        return kf
    if kind == "unknown":
        kf.warnings.append("unrecognised file type")
        return kf

    root = parse(text)
    kf.root = root
    kf.version = root.value("version")
    kf.generator = root.value("generator")
    if kind == "pcb":
        kf.model = pcb_mod.parse_board(root)
        kf.warnings += kf.model.warnings
    elif kind == "footprint":
        kf.model = pcb_mod.parse_footprint_file(root)
    elif kind == "schematic":
        kf.model = sch_mod.parse_schematic(root)
        kf.warnings += kf.model.warnings
    elif kind == "symbol_lib":
        kf.model = sch_mod.parse_symbol_lib(root)
    elif kind == "netlist":
        kf.model = nl_mod.parse_netlist(root)
    elif kind == "rules":
        kf.model = proj_mod.parse_rules(root)
    elif kind == "lib_table":
        kf.model = proj_mod.parse_lib_table(root)
    elif kind in ("dsn", "ses"):
        kf.model = proj_mod.summarize_specctra(root)
        kf.version = None
    return kf


@dataclass
class KicadProject:
    name: str
    directory: Path
    settings: proj_mod.ProjectSettings | None = None
    board: Any = None                                   # Board
    schematics: dict[str, Any] = field(default_factory=dict)   # sheet file name -> Schematic
    warnings: list[str] = field(default_factory=list)


def read_project(path) -> KicadProject:
    """Read a KiCad project from a directory or a .kicad_pro file: settings, board, all sheets."""
    p = Path(path)
    if p.is_dir():
        pros = sorted(p.glob("*.kicad_pro"))
        if not pros:
            raise FileNotFoundError(f"no .kicad_pro in {p}")
        p = pros[0]
    d, stem = p.parent, p.stem
    proj = KicadProject(name=stem, directory=d)
    if p.suffix == ".kicad_pro" and p.exists():
        proj.settings = read_kicad(p).model
    pcb_path = d / f"{stem}.kicad_pcb"
    if pcb_path.exists():
        kf = read_kicad(pcb_path)
        proj.board = kf.model
        proj.warnings += kf.warnings
    else:
        proj.warnings.append(f"{pcb_path.name} not found")

    todo = [f"{stem}.kicad_sch"]
    while todo:
        name = todo.pop()
        if name in proj.schematics:
            continue
        sp = d / name
        if not sp.exists():
            proj.warnings.append(f"sheet file {name} not found")
            continue
        kf = read_kicad(sp)
        proj.schematics[name] = kf.model
        todo.extend(sh.file for sh in kf.model.sheets if sh.file)
    return proj
