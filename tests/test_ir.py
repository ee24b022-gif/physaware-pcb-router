import dataclasses
from pathlib import Path

import pytest

from pcbroute.ir import (Design, DiffPair, IR_VERSION, IRVersionError, KeepoutKind, LayerKind, NetKind, PinKind, Pt,
                         Relations, ViaKind, ZoneKind, ir_from_kicad, mm_to_nm)
from pcbroute.ir.geometry import chain_segments, point_in_ring, ring_area2
from pcbroute.ir.inference import infer_diff_pairs, infer_net_kind
from pcbroute.kicad_io import read_kicad
from pcbroute.kicad_io.project import parse_project

D = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def design():
    return ir_from_kicad(read_kicad(D / "buck_mini.kicad_pcb").model, name="buck_mini")


def test_geometry_primitives():
    assert mm_to_nm(0.123456) == 123456 and mm_to_nm(-1.5) == -1_500_000
    assert Pt(12_345, 6_789).snapped(10_000) == Pt(10_000, 10_000)
    M = 1_000_000
    rings, open_chains = chain_segments([((0, 0), (10 * M, 0)), ((10 * M, 0), (10 * M, 5 * M)),
                                         ((10 * M, 5 * M), (0, 5 * M)), ((0, 5 * M), (0, 0)),
                                         ((100 * M, 100 * M), (200 * M, 100 * M))])
    assert len(rings) == 1 and open_chains == 1 and abs(ring_area2(rings[0])) == 2 * 50 * M * M
    assert point_in_ring(5 * M, 2 * M, rings[0]) and not point_in_ring(11 * M, 2 * M, rings[0])


def test_chain_segments_tolerates_gaps_across_bin_boundaries():
    # ends 800 nm apart that straddle a 1000 nm hash-bin boundary must still join; 5000 nm apart must not
    ring, _ = chain_segments([((0, 0), (10_000, 0)), ((10_000 + 300, 0), (10_000, 8_000)), ((10_000, 8_000), (0, 0))] , tol_nm=1000)
    assert len(ring) == 1 and len(ring[0]) == 3
    ring2, open2 = chain_segments([((0, 0), (10_000, 0)), ((15_000, 0), (10_000, 8_000)), ((10_000, 8_000), (0, 0))], tol_nm=1000)
    assert ring2 == [] and open2 >= 1


def test_board_outline_and_stackup(design):
    b = design.board
    xs = [p.x for p in b.outline]; ys = [p.y for p in b.outline]
    assert (max(xs) - min(xs), max(ys) - min(ys)) == (30_000_000, 20_000_000) and not b.outline_inferred
    assert b.stackup.layer_ids() == ("L0", "L1", "L2", "L3")
    assert [l.source_name for l in b.stackup.copper] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    assert [l.side for l in b.stackup.copper] == ["top", "inner", "inner", "bottom"]
    assert b.stackup.gap_nm("L0", "L1") == 200_000 and b.stackup.gap_nm("L1", "L0") == 200_000
    assert b.stackup.gap_nm("L0", "L3") is None
    assert b.stackup.dielectrics[0].epsilon_r == 4.5 and b.stackup.defined_in_source


def test_components_and_pins(design):
    q1 = design.component("Q1")
    assert q1.footprint == "Package_DFN_QFN:PowerPAK_5x6" and q1.value == "HS_FET" and len(q1.pins) == 2
    p5 = q1.pin("5")
    assert p5.id == "Q1.5" and p5.net == "VIN" and p5.layers == ("L0",) and p5.position == Pt(100_000_000, 99_000_000)
    ko = [k for k in q1.keepouts if k.kind == KeepoutKind.COURTYARD]
    assert len(ko) == 1 and len(ko[0].ring) == 4
    xs = [p.x for p in ko[0].ring]; ys = [p.y for p in ko[0].ring]
    assert (max(xs) - min(xs), max(ys) - min(ys)) == (3_000_000, 4_000_000)      # 4x3 mm rect, rotated 90
    c1 = design.component("C1")
    assert c1.pin("2").kind == PinKind.THROUGH and c1.pin("2").layers == ("L0", "L1", "L2", "L3")
    h1 = design.component("H1")
    assert h1.locked and h1.pins[0].kind == PinKind.NPTH and h1.pins[0].net is None
    assert len(design.pins()) == 5


def test_nets(design):
    names = [n.name for n in design.nets]
    assert names == ["GND", "SW", "VIN"] and [n.index for n in design.nets] == [0, 1, 2]
    vin = design.net("VIN")
    assert set(vin.pins) == {"Q1.5", "C1.1"} and vin.kind == NetKind.POWER and vin.layer_mask == ("L0", "L1", "L2", "L3")
    assert design.net("GND").kind == NetKind.GROUND
    assert vin.width_nm == 250_000 and vin.clearance_nm == 200_000      # typical values assumed without project settings
    assert any("assumed" in w for w in design.provenance.warnings)


def test_net_classes_from_project_settings():
    ps = read_kicad(D / "proj" / "demo.kicad_pro").model
    d = ir_from_kicad(read_kicad(D / "buck_mini.kicad_pcb").model, ps)
    assert d.net("VIN").net_class == "Power" and d.net("VIN").width_nm == 2_000_000
    assert d.net("SW").net_class == "Default" and d.net("SW").width_nm == 250_000
    assert d.rules.min_clearance_nm == 150_000 and not d.rules.defaults_assumed
    assert d.rules.net_class("Power").via_diameter_nm == 600_000


def test_copper(design):
    c = design.copper
    assert len(c.segments) == 1 and len(c.arcs) == 1 and len(c.vias) == 1     # arc pieces are not duplicated as segments
    assert c.arcs[0].start == Pt(110_000_000, 100_000_000) and c.arcs[0].source_id == "a1"
    v = c.vias[0]
    assert (v.layer_from, v.layer_to, v.kind, v.locked) == ("L0", "L1", ViaKind.BLIND, True)
    assert c.protected_ids() == frozenset({"via:0"})
    pour = next(z for z in c.zones if z.kind == ZoneKind.POUR)
    assert pour.layer == "L1" and pour.net == "GND" and len(pour.fills) == 1
    ko = [z for z in c.zones if z.kind == ZoneKind.KEEPOUT]
    assert {z.layer for z in ko} == {"L0", "L3"} and ko[0].forbids == ("copper", "tracks")


def test_layer_kind_override_and_inference():
    kb = read_kicad(D / "buck_mini.kicad_pcb").model
    d = ir_from_kicad(kb, layer_kinds={"In1.Cu": LayerKind.GROUND})
    assert d.board.stackup.layer("L1").kind == LayerKind.GROUND
    assert all(l.kind == LayerKind.SIGNAL for l in ir_from_kicad(kb).board.stackup.copper)   # zone covers 33%: not a plane


def test_missing_outline_is_inferred_with_warning():
    kb = read_kicad(D / "buck_mini.kicad_pcb").model
    kb.outline = []
    d = ir_from_kicad(kb)
    assert d.board.outline_inferred and any(i.code == "outline_inferred" for i in d.validate())


def test_validate_clean_and_catches_errors(design):
    assert [i for i in design.validate() if i.level == "error"] == []
    bad_pin = dataclasses.replace(design.components[0].pins[0], net="NOPE")
    comp = dataclasses.replace(design.components[0], pins=(bad_pin,) + design.components[0].pins[1:])
    bad = dataclasses.replace(design, components=(comp,) + design.components[1:],
                              relations=Relations(diff_pairs=(DiffPair("x", "SW", "MISSING"),)),
                              copper=dataclasses.replace(design.copper, protected=("track:999",)))
    codes = {i.code for i in bad.validate()}
    assert {"pin_unknown_net", "diffpair_unknown_net", "protected_unknown"} <= codes


def test_json_round_trip_is_exact(design):
    text = design.to_json()
    back = Design.from_json(text)
    assert back == design and back.ir_version == IR_VERSION
    assert Design.from_json(back.to_json()) == design


def test_version_mismatch_is_rejected(design):
    data = design.to_dict()
    data["ir_version"] = "9.9.9"
    with pytest.raises(IRVersionError):
        Design.from_dict(data)


def test_name_inference():
    assert infer_net_kind("GND") == NetKind.GROUND and infer_net_kind("+3V3") == NetKind.POWER
    assert infer_net_kind("VDD_1p8V") == NetKind.POWER and infer_net_kind("REF_CLK") == NetKind.CLOCK
    assert infer_net_kind("TxOUT17") == NetKind.SIGNAL
    pairs = infer_diff_pairs(["USB_D+", "USB_D-", "TX0_P", "TX0_N", "CLK", "GND", "RESBDN", "+3V3"])
    assert ("USB_D", "USB_D+", "USB_D-") in pairs and ("TX0", "TX0_P", "TX0_N") in pairs and len(pairs) == 2


def test_validate_warns_about_components_outside_outline(design):
    moved = dataclasses.replace(design.components[0], position=Pt(0, 0))
    d = dataclasses.replace(design, components=(moved,) + design.components[1:])
    w = [i for i in d.validate() if i.code == "component_outside_outline"]
    assert len(w) == 1 and w[0].level == "warning" and design.components[0].ref in w[0].message
    assert not [i for i in design.validate() if i.code == "component_outside_outline"]
