"""KiCad file reading: any KiCad file in, typed model out.

    from pcbroute.kicad_io import read_kicad, read_project
    board = read_kicad("buck.kicad_pcb").model       # -> pcbroute.board.Board
"""
from .netlist import Netlist, parse_netlist, parse_netlist_xml
from .project import ProjectSettings
from .reader import KicadFile, KicadProject, detect_kind, read_kicad, read_project
from .schematic import Schematic
from .sexpr import Node, ParseError, Sym, parse, parse_file
from .summary import summarize
from .writer import write_kicad_pcb, WriteConfig

__all__ = [
    "KicadFile", "KicadProject", "Netlist", "Node", "ParseError", "ProjectSettings", "Schematic", "Sym",
    "detect_kind", "parse", "parse_file", "parse_netlist", "parse_netlist_xml", "read_kicad", "read_project",
    "summarize",
    "write_kicad_pcb", "WriteConfig",
]
