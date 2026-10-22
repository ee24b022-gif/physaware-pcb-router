"""Export routed designs to KiCad and other formats."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from ..board import Board, Segment, Via
from ..ir import Design, Net
from ..ir.geometry import Pt
from ..routing import RouteResult
from ..kicad_io import read_kicad, write_kicad_pcb, WriteConfig
from ..kicad_io.pcb import parse_board
from ..kicad_io.sexpr import parse_file


def apply_routing_to_board(original_kicad_path: str, design: Design, results: list[RouteResult]) -> Board:
    """Create a new Board with routed tracks and vias added to the original KiCad board."""
    # Parse the original KiCad file to get the parser Board
    kf = read_kicad(original_kicad_path)
    board = kf.model
    
    # Build IR layer ID -> source layer name mapping
    ir_to_source = {ly.id: ly.source_name for ly in design.board.stackup.copper}
    
    # Add routed segments
    for result in results:
        if result.status.value not in ("routed", "partial"):
            continue
        for a, b, layer, width in result.segments:
            # Map IR layer ID to KiCad layer name
            kicad_layer = ir_to_source.get(layer, layer)
            seg = Segment(
                net=result.net_name,
                layer=kicad_layer,
                x1=a.x / 1_000_000.0,
                y1=a.y / 1_000_000.0,
                x2=b.x / 1_000_000.0,
                y2=b.y / 1_000_000.0,
                width=width / 1_000_000.0,
                uuid=_new_uuid(),
                locked=False,
            )
            board.segments.append(seg)
        
        # Add routed vias
        for pos, layer_from, layer_to, dia, drill in result.vias:
            kicad_layer_from = ir_to_source.get(layer_from, layer_from)
            kicad_layer_to = ir_to_source.get(layer_to, layer_to)
            via = Via(
                net=result.net_name,
                x=pos.x / 1_000_000.0,
                y=pos.y / 1_000_000.0,
                diameter=dia / 1_000_000.0,
                drill=drill / 1_000_000.0,
                layers=(kicad_layer_from, kicad_layer_to),
                uuid=_new_uuid(),
                locked=False,
            )
            board.vias.append(via)
    
    return board


def _new_uuid() -> str:
    u = uuid.uuid4()
    return f"{u.hex[:8]}-{u.hex[8:12]}-{u.hex[12:16]}-{u.hex[16:20]}-{u.hex[20:]}"


def export_kicad_pcb(original_kicad_path: str, design: Design, results: list[RouteResult], 
                     filepath: str, config: WriteConfig = WriteConfig()) -> Board:
    """Export a routed design to a .kicad_pcb file."""
    board = apply_routing_to_board(original_kicad_path, design, results)
    write_kicad_pcb(board, filepath, config)
    return board


def validate_routed_board(original_kicad_path: str, design: Design, results: list[RouteResult]) -> list[str]:
    """Validate the routed board and return any issues."""
    from ..geometry import GeometryKernel, KernelConfig
    from ..ir import ir_from_kicad
    from ..kicad_io import read_kicad
    from pathlib import Path
    import tempfile
    import os
    
    board = apply_routing_to_board(original_kicad_path, design, results)
    
    # Write to temp file and re-read
    with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False) as f:
        tmp_path = f.name
    try:
        write_kicad_pcb(board, tmp_path)
        kf = read_kicad(tmp_path)
        d2 = ir_from_kicad(kf.model)
        issues = d2.validate()
        return [f"{i.level}: {i.code}: {i.message}" for i in issues if i.level == "error"]
    finally:
        os.unlink(tmp_path)


def routing_summary(results: list[RouteResult], design: Design) -> dict:
    """Generate a summary of routing results."""
    total_nets = len([n for n in design.nets if len(n.pins) >= 2])
    routed = sum(1 for r in results if r.status.value == "routed")
    partial = sum(1 for r in results if r.status.value == "partial")
    failed = sum(1 for r in results if r.status.value == "failed")
    skipped = sum(1 for r in results if r.status.value == "skipped")
    
    total_segments = sum(len(r.segments) for r in results)
    total_vias = sum(r.via_count for r in results)
    total_length_mm = sum(r.path_length_nm / 1_000_000.0 for r in results)
    
    return {
        "total_routable_nets": total_nets,
        "routed": routed,
        "partial": partial,
        "failed": failed,
        "skipped": skipped,
        "total_segments": total_segments,
        "total_vias": total_vias,
        "total_length_mm": total_length_mm,
        "success_rate": routed / total_nets if total_nets > 0 else 0.0,
    }