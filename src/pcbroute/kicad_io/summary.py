"""Human-readable summaries of anything `read_kicad` returns."""
from __future__ import annotations

import dataclasses
import json
from collections import Counter

from .reader import KicadFile, KicadProject


def _board_summary(b) -> list[str]:
    out = [f"copper layers: {', '.join(b.copper_layers()) or '-'}"]
    if b.stackup:
        out.append("stackup (top to bottom):")
        for ly in b.stackup:
            extra = f", eps_r={ly.epsilon_r}" if ly.epsilon_r else ""
            out.append(f"  {ly.name:<14} {ly.kind:<10} {ly.thickness_mm:g} mm{extra}")
    out.append(f"footprints: {len(b.footprints)}  pads: {len(b.pads)}  segments: {len(b.segments)}  "
               f"vias: {len(b.vias)}  zones: {len(b.zones)}  outline edges: {len(b.outline)}")
    for f in b.footprints[:40]:
        out.append(f"  {f.ref or '?':<6} {f.value:<14} {f.lib_id}  at ({f.x:g}, {f.y:g}) rot {f.rot_deg:g} on {f.layer}")
    if len(b.footprints) > 40:
        out.append(f"  ... {len(b.footprints) - 40} more")
    counts = Counter(p.net for p in b.pads if p.net)
    if counts:
        out.append("nets (pads):")
        for net, n in sorted(counts.items()):
            out.append(f"  {net}: {n}")
    return out


def summarize(kf) -> str:
    if isinstance(kf, KicadProject):
        lines = [f"project {kf.name}  ({kf.directory})"]
        if kf.settings:
            lines.append("net classes: " + ", ".join(sorted(kf.settings.net_classes)))
        if kf.board:
            lines += _board_summary(kf.board)
        for name, sch in kf.schematics.items():
            lines.append(f"sheet {name}: {len(sch.symbols)} symbols, {len(sch.labels)} labels")
        lines += [f"warning: {w}" for w in kf.warnings]
        return "\n".join(lines)

    assert isinstance(kf, KicadFile)
    lines = [f"{kf.path.name}: {kf.kind}" + (f" (version {kf.version})" if kf.version else "")
             + (f", generator {kf.generator}" if kf.generator else "")]
    m = kf.model
    if kf.kind in ("pcb", "footprint"):
        lines += _board_summary(m)
    elif kf.kind == "schematic":
        lines.append(f"symbols: {len(m.symbols)}  labels: {len(m.labels)}  wires: {len(m.wires)}  sheets: {len(m.sheets)}")
        for s in m.symbols[:60]:
            pins = m.pin_names(s.lib_id)
            pinstr = " ".join(f"{k}={v}" for k, v in sorted(pins.items()))
            lines.append(f"  {s.reference:<6} {s.value:<14} {s.lib_id}  fp={s.footprint or '-'}  [{pinstr}]")
        names = sorted({lb.text for lb in m.labels})
        if names:
            lines.append("labels: " + ", ".join(names))
        for sh in m.sheets:
            lines.append(f"  sheet {sh.name} -> {sh.file}")
    elif kf.kind == "symbol_lib":
        for name, ls in m.items():
            lines.append(f"  {name}: {len(ls.pins)} pins")
    elif kf.kind == "netlist":
        lines.append(f"components: {len(m.components)}  nets: {len(m.nets)}")
        for n in m.nets.values():
            lines.append(f"  {n.name}: " + ", ".join(f"{x.ref}.{x.pin}" + (f"({x.pinfunction})" if x.pinfunction else "")
                                                       for x in n.nodes))
    elif kf.kind == "project":
        for c in m.net_classes.values():
            lines.append(f"  class {c.name}: clearance={c.clearance} track={c.track_width} "
                         f"via={c.via_diameter}/{c.via_drill}")
        for pat, cls in m.netclass_patterns:
            lines.append(f"  pattern {pat} -> {cls}")
    elif kf.kind in ("dsn", "ses", "rules", "lib_table"):
        lines.append(json.dumps(m, indent=2, default=str))
    lines += [f"warning: {w}" for w in kf.warnings]
    return "\n".join(lines)


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return obj
