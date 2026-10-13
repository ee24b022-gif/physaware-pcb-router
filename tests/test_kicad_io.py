import json
import math
from pathlib import Path

import pytest

from pcbroute.kicad_io import ParseError, Sym, detect_kind, parse, read_kicad, read_project, summarize
from pcbroute.kicad_io.pcb import parse_board
from pcbroute.kicad_io.__main__ import main

D = Path(__file__).parent / "data"


# ---- S-expression parser ----------------------------------------------------
def test_parse_atoms_and_nesting():
    n = parse('(pad "1" smd rect (at -1.5 2) (size 0.6 1e-1) (net 3 "VIN"))')
    assert n.name == "pad"
    a = n.atoms()
    assert a[0] == "1" and not isinstance(a[0], Sym)       # quoted string
    assert isinstance(a[1], Sym) and a[1] == "smd"          # bare symbol
    assert n.xy("at") == (-1.5, 2.0)
    assert n.values("size") == [0.6, 0.1]
    assert n.child("net").atoms() == [3, "VIN"]


def test_parse_escapes_and_multiple_roots():
    n = parse(r'(a "he said \"hi\"\\ok") (b 1)')
    assert n.name == "__root__"
    assert n.child("a").atoms()[0] == 'he said "hi"\\ok'
    assert n.child("b").atoms() == [1]


def test_parse_errors():
    with pytest.raises(ParseError):
        parse("(a (b)")
    with pytest.raises(ParseError):
        parse("(a))")
    with pytest.raises(ParseError):
        parse('(a "unterminated)')


def test_deep_nesting_does_not_recurse():
    depth = 20000
    n = parse("(" * depth + "x" + ")" * depth)
    assert n is not None


# ---- PCB -> Board -----------------------------------------------------------
@pytest.fixture(scope="module")
def board():
    return read_kicad(D / "buck_mini.kicad_pcb").model


def test_detect_kinds():
    assert detect_kind(Path("x.kicad_pcb"), "(kicad_pcb (version 1))") == "pcb"
    assert detect_kind(Path("x.txt"), "(kicad_sch (version 1))") == "schematic"
    assert detect_kind(Path("x.kicad_pro"), "{}") == "project"
    assert detect_kind(Path("x.sch"), "EESchema Schematic File Version 4") == "legacy"
    assert detect_kind(Path("x.dsn"), "(pcb demo)") == "dsn"


def test_stackup(board):
    assert [l.name for l in board.stackup if l.kind == "copper"] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    d1 = next(l for l in board.stackup if l.name == "dielectric 1")
    assert d1.thickness_mm == 0.2 and d1.epsilon_r == 4.5
    assert board.metadata["stackup_defined"] is True
    assert board.copper_layers() == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]


def test_footprint_rotation_moves_pads(board):
    # Footprint at (100,100) rot 90 (CCW on screen, y down): local (-1,0) -> (0,+1); local (1,0) -> (0,-1)
    p1 = next(p for p in board.pads if p.ref == "Q1" and p.number == "1")
    p5 = next(p for p in board.pads if p.ref == "Q1" and p.number == "5")
    assert (round(p1.x, 6), round(p1.y, 6)) == (100.0, 101.0)
    assert (round(p5.x, 6), round(p5.y, 6)) == (100.0, 99.0)
    x0, y0, x1, y1 = p1.bbox()       # 0.6 x 1.2 pad at 90 deg -> 1.2 wide, 0.6 tall
    assert round(x1 - x0, 6) == 1.2 and round(y1 - y0, 6) == 0.6
    assert p1.net == "SW" and p5.net == "VIN"


def test_pad_layers_and_thru_hole(board):
    c2 = next(p for p in board.pads if p.ref == "C1" and p.number == "2")
    assert c2.kind == "thru_hole" and c2.drill == 0.8
    assert c2.layers == ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
    c1 = next(p for p in board.pads if p.ref == "C1" and p.number == "1")
    assert c1.layers == ("F.Cu",) and c1.layer == "F.Cu"
    assert (c1.x, c1.y) == (101.6, 100.0)


def test_footprints_segments_vias(board):
    assert {f.ref for f in board.footprints} == {"Q1", "C1", "H1"}
    assert board.footprint("Q1").value == "HS_FET"
    seg = next(s for s in board.segments if s.net == "VIN")
    assert (seg.x1, seg.x2, seg.width, seg.layer) == (100.0, 103.0, 0.5, "F.Cu")
    v = board.vias[0]
    assert (v.x, v.y, v.net, v.layers) == (101.0, 101.0, "GND", ("F.Cu", "In1.Cu"))
    assert board.nets() == {"VIN", "SW", "GND"}


def test_arc_discretisation(board):
    arcs = [s for s in board.segments if s.net == "SW"]
    assert len(arcs) >= 4
    pts = [(s.x1, s.y1) for s in arcs] + [(arcs[-1].x2, arcs[-1].y2)]
    for x, y in pts:
        assert math.hypot(x - 108, y - 100) == pytest.approx(2.0, abs=1e-3)
    assert pts[0] == (110, 100) and pts[-1] == (108, 98)
    length = sum(math.hypot(s.x2 - s.x1, s.y2 - s.y1) for s in arcs)
    assert length == pytest.approx(math.pi, rel=0.01)


def test_zones_and_outline(board):
    outline = [z for z in board.zones if z.layer == "In1.Cu" and z.net == "GND"]
    assert {z.filled for z in outline} == {False, True}
    ko = [z for z in board.zones if z.keepout]
    assert {z.layer for z in ko} == {"F.Cu", "B.Cu"}
    assert ko[0].keepout_rules["tracks"] == "not_allowed" and ko[0].net is None
    assert len(board.outline) == 4


def test_kicad9_net_by_name_and_no_stackup():
    root = parse('''(kicad_pcb (version 20241229) (layers (0 "F.Cu" signal) (2 "B.Cu" signal))
        (footprint "X:Y" (layer "F.Cu") (at 0 0) (property "Reference" "R1")
          (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net "SW")))
        (segment (start 0 0) (end 1 0) (width 0.2) (layer "F.Cu") (net "VIN")))''')
    b = parse_board(root)
    assert b.pads[0].net == "SW" and b.segments[0].net == "VIN"
    assert b.metadata["stackup_defined"] is False and b.warnings


def test_legacy_module_keyword():
    root = parse('''(kicad_pcb (version 4) (layers (0 F.Cu signal) (31 B.Cu signal))
        (net 0 "") (net 1 GND)
        (module Lib:Foo (layer F.Cu) (at 5 5)
          (fp_text reference U1 (at 0 0) (layer F.SilkS))
          (pad 1 smd rect (at 1 0) (size 1 1) (layers F.Cu) (net 1 GND))))''')
    b = parse_board(root)
    assert b.footprints[0].ref == "U1" and b.pads[0].net == "GND" and b.pads[0].number == "1"
    assert (b.pads[0].x, b.pads[0].y) == (6.0, 5.0)


# ---- Other file types -------------------------------------------------------
def test_schematic():
    sch = read_kicad(D / "proj" / "demo.kicad_sch").model
    q = sch.symbol("Q1")                      # reference comes from (instances ...), not the "Q?" property
    assert q is not None and q.value == "BSC0902" and q.footprint.endswith("PowerPAK_5x6")
    assert sch.pin_names("Transistor_FET:Q_NMOS_GDS") == {"3": "D", "1": "G", "2": "S"}
    assert {l.text for l in sch.labels} == {"SW", "VIN"}
    assert sch.wires == [[(50.0, 40.0), (60.0, 40.0)]] and sch.junctions == [(60.0, 40.0)]
    assert sch.sheets[0].file == "driver.kicad_sch"


def test_netlist_sexpr_and_xml():
    nl = read_kicad(D / "buck_mini.net").model
    assert nl.components["Q1"].footprint.endswith("PowerPAK_5x6")
    assert nl.net_of("Q1", "5") == "VIN" and nl.net_of("C1", "1") == "VIN"
    assert nl.nets["VIN"].nodes[0].pinfunction == "D"
    xml = '''<export version="D"><components><comp ref="R1"><value>1k</value><footprint>R_0603</footprint></comp></components>
             <nets><net code="1" name="N1"><node ref="R1" pin="1" pinfunction="~"/></net></nets></export>'''
    p = D / "_tmp.xml"
    p.write_text(xml)
    try:
        kf = read_kicad(p)
    finally:
        p.unlink()
    assert kf.kind == "netlist" and kf.model.nets["N1"].nodes[0].ref == "R1"


def test_footprint_file():
    b = read_kicad(D / "R_mini.kicad_mod").model
    assert [p.number for p in b.pads] == ["1", "2"] and b.pads[0].x == -0.825
    assert b.pads[0].ref == "REF**" and b.pads[0].layers == ("F.Cu",)


def test_project_json_and_read_project():
    kf = read_kicad(D / "proj" / "demo.kicad_pro")
    ps = kf.model
    assert ps.net_classes["Power"].track_width == 2.0
    assert ps.netclass_for("VIN").name == "Power" and ps.netclass_for("GND").name == "Default"
    assert ps.design_rules["min_clearance"] == 0.15
    proj = read_project(D / "proj")
    assert proj.board is not None and set(proj.schematics) == {"demo.kicad_sch", "driver.kicad_sch"}
    assert not proj.warnings or all("not found" not in w for w in proj.warnings)


def test_other_sexpr_types():
    lib = read_kicad_text("(fp_lib_table (version 7) (lib (name \"a\") (type \"KiCad\") (uri \"${KIPRJMOD}/a.pretty\")))", "fp-lib-table")
    assert lib.kind == "lib_table" and lib.model[0]["name"] == "a"
    dru = read_kicad_text('(version 1) (rule "hot" (layer outer) (condition "A.NetClass == \'Power\'") (constraint clearance (min 0.3mm)))', "x.kicad_dru")
    assert dru.kind == "rules" and dru.model[0]["name"] == "hot" and dru.model[0]["constraints"][0]["type"] == "clearance"
    dsn = read_kicad_text('(pcb demo (resolution um 10) (unit um) (structure (layer F.Cu (type signal)) (layer B.Cu (type signal))) (network (net VIN) (net GND)))', "x.dsn")
    assert dsn.kind == "dsn" and dsn.model["layers"] == ["F.Cu", "B.Cu"] and dsn.model["nets"] == ["VIN", "GND"]
    sym = read_kicad_text('(kicad_symbol_lib (version 20231120) (symbol "R" (symbol "R_1_1" (pin passive line (at 0 0 0) (length 1) (name "~") (number "1")))))', "x.kicad_sym")
    assert sym.kind == "symbol_lib" and sym.model["R"].pins[0].number == "1"


def test_legacy_not_parsed():
    kf = read_kicad_text("EESchema Schematic File Version 4\n", "old.sch")
    assert kf.kind == "legacy" and kf.warnings


def test_summary_and_cli(capsys):
    text = summarize(read_kicad(D / "buck_mini.kicad_pcb"))
    assert "Q1" in text and "VIN: 2" in text
    assert main([str(D / "buck_mini.kicad_pcb"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["pads"]) == 5
    assert main([str(D / "buck_mini.kicad_pcb"), "--tree", "1"]) == 0
    assert main([str(D / "nonexistent.kicad_pcb")]) == 1


def read_kicad_text(text, name):
    p = D / ("_tmp_" + name)
    p.write_text(text)
    try:
        return read_kicad(p)
    finally:
        p.unlink()


# ---- health checks ----------------------------------------------------------
def test_health_and_alignment_on_synthetic_board():
    from pcbroute.board import Board, Footprint, Pad, Segment, rotate_kicad
    from pcbroute.kicad_io.health import board_health, pad_alignment

    b = Board()
    for ref, rot in (("R1", 0), ("R2", 90), ("R3", 180), ("R4", 270)):
        fx, fy = 10.0 * (1 + int(rot / 90)), 10.0
        b.footprints.append(Footprint(ref, "1k", "R", fx, fy, rot, "F.Cu"))
        for num, lx in (("1", -0.8), ("2", 0.8)):
            dx, dy = rotate_kicad(lx, 0, rot)
            b.pads.append(Pad(ref, num, f"N_{ref}_{num}", "F.Cu", fx + dx, fy + dy, 1.0, 1.0,
                              rot_deg=rot, layers=("F.Cu",)))
            # a track that ends exactly on the pad centre and runs away from it
            b.segments.append(Segment(f"N_{ref}_{num}", "F.Cu", fx + dx, fy + dy, fx + dx, fy + dy + 3.0, 0.2))
    b.footprints.append(Footprint("U9", "x", "X", 0.0, 0.0, 0, "F.Cu"))
    b.pads.append(Pad("C9", "1", "N_C9", "F.Cu", 10.0, 10.0, 1.0, 1.0, layers=("F.Cu",)))   # sits on R1's footprint
    b.pads.append(Pad("C9", "2", None, "F.Cu", 50.0, 50.0, 1.0, 1.0, layers=("F.Cu",)))
    h = board_health(b)
    assert h["footprints_at_origin"] == ["U9"]
    assert h["signal_pads_on_nets_with_no_copper"] == 1
    assert h["overlapping_pad_pairs"] >= 1
    al = pad_alignment(b)
    assert all(hit == tot for hit, tot in al.values()) and set(al) == {0, 90, 180, 270}


def test_health_cli(capsys):
    assert main([str(D / "buck_mini.kicad_pcb"), "--health"]) == 0
    out = capsys.readouterr().out
    assert "overlapping pad pairs" in out and "rot" in out


# ---- IDs, locked flags, arcs, courtyards (needed by the IR) ----------------------
def test_uuid_locked_arcs_courtyard(board):
    assert next(p for p in board.pads if p.ref == "Q1" and p.number == "1").uuid == ""      # fixture pads have no uuid
    assert next(s for s in board.segments if s.net == "VIN").uuid == "s1"
    assert board.vias[0].locked is True and board.vias[0].uuid == "v1"
    assert board.footprint("H1").locked is True and board.footprint("Q1").locked is False
    assert len(board.arcs) == 1 and board.arcs[0].uuid == "a1" and board.arcs[0].start == (110, 100)
    flat = [s for s in board.segments if s.arc_index is not None]
    assert flat and all(s.arc_index == 0 for s in flat)
    cy = board.footprint("Q1").courtyard          # rect (-2,-1.5)..(2,1.5), footprint at (100,100) rot 90
    xs = [v for e in cy for v in (e[0], e[2])]
    ys = [v for e in cy for v in (e[1], e[3])]
    assert (round(min(xs), 6), round(max(xs), 6)) == (98.5, 101.5)      # 3.0 wide after the 90 degree rotation
    assert (round(min(ys), 6), round(max(ys), 6)) == (98.0, 102.0)
    assert len(cy) == 4
    z = [z for z in board.zones if z.net == "GND"]
    assert {zz.zone_index for zz in z} == {0} and z[0].uuid == "z1"


def test_netclass_wildcards():
    from pcbroute.kicad_io.project import parse_project
    ps = parse_project({"net_settings": {"classes": [{"name": "Default", "track_width": 0.25}, {"name": "Power", "track_width": 1.0}],
                                         "netclass_patterns": [{"netclass": "Power", "pattern": "VDD*"}]}})
    assert ps.netclass_for("VDD_1p8V").name == "Power" and ps.netclass_for("SW").name == "Default"
