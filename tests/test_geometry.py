"""Geometry kernel tests. All coordinates are integer nanometres; helper mm() converts at the test boundary only."""
import dataclasses
import random

import pytest

from pcbroute.geometry import (BBox, GeometryKernel, GridIndex, InflatedObstacle, Item, KernelConfig, LayeredIndex,
                               PolyCore, SegCore, closer_than, convex_hull, disc_polygon, dist_floor, half_width_nm,
                               inflation_nm, pad_geometry, point_in_polygon, ring_locate, segments_intersect, touches)
from pcbroute.ir import (Board, Component, Copper, CopperLayer, Cutout, Design, LayerKind, Net, NetClass, NetKind,
                         PadShape, Pin, PinKind, Polygon, Pt, Rules, Side, Stackup, TrackArc, TrackSeg, Via, ViaKind,
                         ViaType, Zone, ZoneKind)

M = 1_000_000


def mm(v: float) -> int:
    return int(round(v * M))


def P(x: float, y: float) -> Pt:
    return Pt(mm(x), mm(y))


def rect_ring(x0, y0, x1, y1):
    return tuple(P(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


ALL = ("L0", "L1", "L2", "L3")


def make_pin(ref, num, net, layers, x, y, shape, w, h, rot=0.0, kind=PinKind.SMD, drill=None):
    return Pin(id=f"{ref}.{num}", ref=ref, number=num, function="", net=net, layers=tuple(layers), position=P(x, y),
               shape=shape, width_nm=mm(w), height_nm=mm(h), rotation_deg=rot, kind=kind,
               drill_nm=mm(drill) if drill else None)


def make_design(arcs=()):
    kinds = [LayerKind.SIGNAL, LayerKind.GROUND, LayerKind.POWER, LayerKind.SIGNAL]
    sides = ["top", "inner", "inner", "bottom"]
    stack = Stackup(tuple(CopperLayer(f"L{i}", i, kinds[i], 35_000, n, sides[i])
                          for i, n in enumerate(("F.Cu", "In1.Cu", "In2.Cu", "B.Cu"))))
    board = Board(outline=rect_ring(0, 0, 100, 100), stackup=stack, cutouts=(Cutout("cutout:0", rect_ring(70, 92, 80, 98)),))
    pins = [
        make_pin("U1", "1", "A", ["L0"], 20, 20, PadShape.RECT, 1.0, 1.0),
        make_pin("R1", "1", "B", ["L0"], 30, 20, PadShape.RECT, 2.0, 0.5, rot=90.0),
        make_pin("J1", "1", "C", ALL, 40, 20, PadShape.CIRCLE, 1.6, 1.6, kind=PinKind.THROUGH, drill=0.8),
        make_pin("O1", "1", "D", ["L0"], 50, 20, PadShape.OVAL, 2.0, 1.0),
        make_pin("X1", "1", "E", ["L0"], 60, 20, PadShape.RECT, 1.0, 1.0),
        make_pin("W1", "1", "WIDE", ["L0"], 70, 30, PadShape.RECT, 1.0, 1.0),
        make_pin("H1", "1", None, ALL, 10, 10, PadShape.CIRCLE, 3.2, 3.2, kind=PinKind.NPTH, drill=3.2),
    ]
    comps = tuple(Component(p.ref, "fp", "", Side.TOP, p.position, 0.0, (p,)) for p in pins)
    names = ["A", "B", "C", "D", "E", "WIDE", "GND", "PWR", "X", "Y"]
    nets = tuple(Net(n, i, tuple(p.id for p in pins if p.net == n), "Wide" if n == "WIDE" else "Default", 250_000,
                     500_000 if n == "WIDE" else 200_000, ALL, 0.3, "default", NetKind.SIGNAL) for i, n in enumerate(names))
    seg = lambda i, net, layer, a, b: TrackSeg(f"track:{i}", net, layer, P(*a), P(*b), 250_000)  # noqa: E731
    segs = (seg(0, "X", "L0", (10, 40), (50, 40)),
            seg(1, "X", "L0", (5, 50), (95, 90)),          # long diagonal: lives in the index's "big" list
            seg(2, "Y", "L2", (65, 10), (65, 30)))
    via = Via("via:0", "GND", P(80, 20), 600_000, 300_000, "L0", "L3", ViaKind.THROUGH)
    zones = (
        Zone("zone:K1:L0", None, "L0", ZoneKind.KEEPOUT, rect_ring(85, 10, 95, 20), forbids=("vias",)),
        Zone("zone:K2:L0", None, "L0", ZoneKind.KEEPOUT, rect_ring(85, 30, 95, 40), forbids=("tracks",)),
        Zone("zone:P1:L1", "GND", "L1", ZoneKind.POUR, rect_ring(20, 50, 60, 90), (Polygon(rect_ring(20, 50, 60, 90)),)),
        Zone("zone:P2:L2", "PWR", "L2", ZoneKind.POUR, rect_ring(60, 50, 95, 68), (Polygon(rect_ring(60, 50, 95, 68)),)),
        Zone("zone:P3:L2", "PWR", "L2", ZoneKind.POUR, rect_ring(79.5, 40, 80.5, 41), (Polygon(rect_ring(79.5, 40, 80.5, 41)),)),
    )
    rules = Rules(100_000, 200_000, 400_000, 200_000, 100_000,
                  (NetClass("Default", 250_000, 200_000, 600_000, 300_000), NetClass("Wide", 250_000, 500_000)),
                  (ViaType("Default", 600_000, 300_000, "L0", "L3"),), "Default", False)
    return Design(board=board, components=comps, nets=nets, copper=Copper(segs, tuple(arcs), (via,), zones), rules=rules)


@pytest.fixture(scope="module")
def k():
    return GeometryKernel(make_design())


# ------------------------------------------------------------------ exact predicates
def test_clearance_boundary_is_exact_and_uses_euclidean_distance():
    o, q = SegCore((0, 0), (0, 0)), SegCore((3000, 4000), (3000, 4000))
    assert dist_floor(o, q) == 5000
    assert not closer_than(o, 0, q, 0, 5000)       # exactly at the required gap is legal
    assert closer_than(o, 0, q, 0, 5001)           # 1 nm short is a violation
    assert touches(o, 0, q, 0, 5000) and not touches(o, 0, q, 0, 4999)
    assert dist_floor(SegCore((0, 0), (0, 0)), SegCore((1, 1), (1, 1))) == 1    # floor(sqrt 2)


def test_segment_predicates():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert segments_intersect((0, 0), (10, 0), (10, 0), (10, 5))          # touching at an endpoint
    assert not segments_intersect((0, 0), (10, 0), (0, 1), (10, 1))
    a, b = SegCore((0, 0), (1000, 0)), SegCore((0, 1000), (1000, 1000))
    assert dist_floor(a, b) == 1000 and not closer_than(a, 0, b, 0, 1000) and closer_than(a, 0, b, 0, 1001)
    cross_ = SegCore((0, -500), (0, 500))
    assert closer_than(SegCore((-500, 0), (500, 0)), 0, cross_, 0, 1)      # crossing: distance 0


def test_polygon_with_hole():
    outer = ((0, 0), (10_000, 0), (10_000, 10_000), (0, 10_000))
    hole = ((4000, 4000), (6000, 4000), (6000, 6000), (4000, 6000))
    poly = PolyCore(outer, (hole,))
    assert ring_locate((5000, 5000), outer) == 1 and ring_locate((0, 5000), outer) == 0 and ring_locate((-1, 5), outer) == -1
    assert point_in_polygon((1000, 1000), poly) and not point_in_polygon((5000, 5000), poly)
    assert point_in_polygon((4000, 5000), poly)                            # a hole's boundary is still copper
    assert dist_floor(SegCore((5000, 5000), (5000, 5000)), poly) == 1000   # centre of the hole: 1000 from its edge
    assert dist_floor(SegCore((12_000, 5000), (12_000, 5000)), poly) == 2000


def test_banded_point_location_matches_the_plain_scan():
    import math
    rng = random.Random(5)
    # a 600-vertex star (concave) with a keyhole-like notch, plus a hole polygon
    star = tuple((int(50 * M + (30 * M + (rng.random() * 15 * M)) * math.cos(2 * math.pi * i / 600)),
                  int(50 * M + (30 * M + (rng.random() * 15 * M)) * math.sin(2 * math.pi * i / 600))) for i in range(600))
    hole = ((45 * M, 45 * M), (55 * M, 45 * M), (55 * M, 55 * M), (45 * M, 55 * M))
    poly = PolyCore(star, (hole,))
    for _ in range(3000):
        p = (rng.randrange(0, 100 * M), rng.randrange(0, 100 * M))
        want = ring_locate(p, star) >= 0 and not (ring_locate(p, hole) > 0)
        assert point_in_polygon(p, poly) == want
    for v in star[:50] + hole:                                              # vertices and edge midpoints are boundary points
        assert point_in_polygon(v, poly) or ring_locate(v, hole) == 0
    mid = ((hole[0][0] + hole[1][0]) // 2, hole[0][1])
    assert point_in_polygon(mid, poly)


def test_convex_hull_and_disc_polygon_contain_the_disc():
    h = convex_hull([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)])
    assert set(h) == {(0, 0), (4, 0), (4, 4), (0, 4)}
    r = 625_000
    poly = tuple(disc_polygon((0, 0), r))
    for v in poly:
        assert v[0] ** 2 + v[1] ** 2 >= r * r                                # circumscribed: every vertex is outside the circle
    for sx in (-1, 1):
        for sy in (-1, 1):
            for p in ((3 * r // 5, 4 * r // 5), (4 * r // 5, 3 * r // 5), (r, 0), (0, r)):
                assert ring_locate((sx * p[0], sy * p[1]), poly) >= 0        # integer points exactly on the circle are inside


# ------------------------------------------------------------------ spatial index
def test_index_is_separated_by_layer(k):
    pad_box = BBox(mm(19), mm(19), mm(21), mm(21))
    assert [i.obj_id for i in k.items_in_bbox("L0", pad_box, ["pad"])] == ["U1.1"]
    assert k.items_in_bbox("L1", pad_box, ["pad"]) == []                       # SMD pad exists on L0 only
    tht = BBox(mm(39), mm(19), mm(41), mm(21))
    assert {l for l in ALL if any(i.obj_id == "J1.1" for i in k.items_in_bbox(l, tht))} == set(ALL)


def test_point_queries_respect_pad_rotation_and_round_shapes(k):
    # R1 is 2.0 x 0.5 mm rotated 90 degrees: tall and thin
    assert [i.obj_id for i in k.items_at_point("L0", (mm(30), mm(20.9)), ["pad"])] == ["R1.1"]
    assert k.items_at_point("L0", (mm(30.4), mm(20)), ["pad"]) == []
    assert [i.obj_id for i in k.items_at_point("L0", (mm(30.25), mm(20)), ["pad"])] == ["R1.1"]    # on the boundary
    # O1 is an oval (capsule): inside its bounding box corner but outside the rounded end
    assert [i.obj_id for i in k.items_at_point("L0", (mm(50.9), mm(20)), ["pad"])] == ["O1.1"]
    assert k.items_at_point("L0", (mm(50.95), mm(20.45)), ["pad"]) == []
    # a point inside the polygon of a keepout
    assert [i.obj_id for i in k.items_at_point("L0", (mm(90), mm(15)), ["keepout"])] == ["zone:K1:L0"]


def test_segment_queries_and_margin(k):
    hit = k.items_hit_by_segment("L0", (mm(30), mm(35)), (mm(30), mm(45)), kinds=["track"])
    assert [i.obj_id for i in hit] == ["track:0"]
    near = (mm(30), mm(35)), (mm(30), mm(39.8))               # stops 0.2 mm below the track centreline
    assert k.items_hit_by_segment("L0", *near, kinds=["track"]) == []
    assert [i.obj_id for i in k.items_hit_by_segment("L0", *near, margin=mm(0.1), kinds=["track"])] == ["track:0"]
    # the long diagonal is stored as a "big" object and must still be found
    assert [i.obj_id for i in k.items_at_point("L0", (mm(50), mm(70)), ["track"])] == ["track:1"]
    assert k.items_hit_by_segment("L2", (mm(60), mm(20)), (mm(70), mm(20)), kinds=["track"])[0].obj_id == "track:2"
    assert k.items_hit_by_segment("L0", (mm(60), mm(20)), (mm(70), mm(20)), kinds=["track"]) == []   # that track is on L2


def test_add_and_remove_objects(k):
    kk = GeometryKernel(make_design())
    kk.add_track(TrackSeg("track:900", "Y", "L3", P(10, 80), P(20, 80), 250_000))
    assert [i.obj_id for i in kk.items_at_point("L3", (mm(15), mm(80)), ["track"])] == ["track:900"]
    assert kk.remove_object("track:900") == 1
    assert kk.items_at_point("L3", (mm(15), mm(80)), ["track"]) == []
    assert kk.remove_object("track:900") == 0
    assert kk.remove_object("via:0") == 4                       # a through via is indexed on every layer


def test_index_matches_brute_force_on_random_data():
    rng = random.Random(1234)
    idx = LayeredIndex(["L0"], cell_nm=3_000_000)
    items = []
    for i in range(300):
        a = (rng.randrange(0, 60 * M), rng.randrange(0, 60 * M))
        b = (a[0] + rng.randrange(-8 * M, 8 * M), a[1] + rng.randrange(-8 * M, 8 * M))
        items.append(Item(f"t{i}@L0", f"t{i}", "track", "L0", None, SegCore(a, b), rng.randrange(50_000, 400_000)))
    for i in range(40):
        x, y = rng.randrange(0, 55 * M), rng.randrange(0, 55 * M)
        w, h = rng.randrange(M, 5 * M), rng.randrange(M, 5 * M)
        items.append(Item(f"p{i}@L0", f"p{i}", "pad", "L0", None, PolyCore(((x, y), (x + w, y), (x + w, y + h), (x, y + h))), 0))
    items.append(Item("long@L0", "long", "track", "L0", None, SegCore((0, 0), (60 * M, 59 * M)), 100_000))
    for it in items:
        idx.add(it)
    for _ in range(300):
        a = (rng.randrange(0, 60 * M), rng.randrange(0, 60 * M))
        b = (a[0] + rng.randrange(-6 * M, 6 * M), a[1] + rng.randrange(-6 * M, 6 * M))
        m = rng.choice([0, 100_000, 500_000])
        got = {i.key for i in idx.query_segment("L0", a, b, m)}
        want = {it.key for it in items if touches(SegCore(a, b), m, it.core, it.radius)}
        assert got == want


# ------------------------------------------------------------------ clearance expansion
def test_inflation_rounds_half_widths_up():
    assert inflation_nm(250_000, 200_000) == 325_000
    assert half_width_nm(251) == 126 and inflation_nm(251, 0) == 126


def test_pad_inflation_has_rounded_corners_and_exact_boundary(k):
    pad = k.index.items["U1.1@L0"]                               # 1 mm square at (20, 20)
    ob = k.clearance.inflate(pad, 250_000, "B")                   # w/2 + c = 125 000 + 200 000
    assert isinstance(ob, InflatedObstacle) and ob.radius == 325_000
    assert not ob.blocks_point((mm(20.5) + 325_000, mm(20)))      # exactly on the inflated edge
    assert ob.blocks_point((mm(20.5) + 324_999, mm(20)))
    cx, cy = mm(20.5), mm(20.5)                                   # pad corner: 0.195 / 0.26 / 0.325 is a 3-4-5 triangle
    assert not ob.blocks_point((cx + 195_000, cy + 260_000))
    assert ob.blocks_point((cx + 195_000, cy + 259_999))
    assert ob.bbox == pad.bbox.inflate(325_000)


def test_track_inflation_and_blocks_segment(k):
    trk = k.index.items["track:0@L0"]
    ob = k.clearance.inflate(trk, 250_000, "A")                   # 125 000 (track) + 125 000 + 200 000
    assert ob.radius == 450_000
    assert ob.blocks_segment((mm(20), mm(40.449999)), (mm(30), mm(40.449999)))
    assert not ob.blocks_segment((mm(20), mm(40.45)), (mm(30), mm(40.45)))
    assert ob.blocks_segment((mm(30), mm(30)), (mm(30), mm(50)))          # crossing


def test_outline_of_inflated_obstacle_contains_exact_region(k):
    ob = k.clearance.inflate(k.index.items["via:0@L0"], 250_000, "A")     # disc: 300 000 + 125 000 + 200 000 = 625 000
    assert ob.radius == 625_000
    out = ob.outline()
    cx, cy = mm(80), mm(20)
    assert all((x - cx) ** 2 + (y - cy) ** 2 >= ob.radius ** 2 for x, y in out)
    r = ob.radius
    for sx in (-1, 1):
        for sy in (-1, 1):
            for dx, dy in ((3 * r // 5, 4 * r // 5), (4 * r // 5, 3 * r // 5), (r, 0), (0, r)):
                assert ring_locate((cx + sx * dx, cy + sy * dy), out) >= 0


def test_clearance_rules_use_the_larger_net_class(k):
    # W1 (net WIDE, 0.5 mm clearance) at (70, 30): pad edge at x = 69.5
    near = lambda net, gap: k.trace_violations("L0", (mm(69.5) - gap - 125_000, mm(25)), (mm(69.5) - gap - 125_000, mm(35)), 250_000, net)  # noqa: E731
    assert near("A", 500_000) == []                               # needs max(0.2, 0.5) = 0.5 mm, exactly met
    v = near("A", 499_999)
    assert [x.code for x in v] == ["clearance"] and v[0].required_nm == 500_000 and v[0].gap_nm == 499_999
    # the same trace against a normal pad only needs 0.2 mm
    assert k.trace_violations("L0", (mm(19.5) - 325_000, mm(15)), (mm(19.5) - 325_000, mm(25)), 250_000, "B") == []


def test_same_net_is_not_an_obstacle_and_pours_yield(k):
    nets = {o.item.obj_id for o in k.obstacles_for("A", "L0", 250_000)}
    assert "U1.1" not in nets and "track:0" in nets
    assert not any(o.item.kind == "pour" for o in k.obstacles_for("A", "L1", 250_000))
    assert k.trace_violations("L1", (mm(30), mm(60)), (mm(50), mm(60)), 250_000, "A") == []    # crossing another net's pour is fine


def test_keepouts_are_selective_and_carry_no_clearance(k):
    inside_tracks_only = k.trace_violations("L0", (mm(88), mm(35)), (mm(92), mm(35)), 250_000, "A")
    assert [x.code for x in inside_tracks_only] == ["keepout"]
    assert k.trace_violations("L0", (mm(88), mm(15)), (mm(92), mm(15)), 250_000, "A") == []     # that one forbids vias only
    just_clear = k.trace_violations("L0", (mm(88), mm(30) - 125_000), (mm(92), mm(30) - 125_000), 250_000, "A")
    assert just_clear == []                                         # keepout needs only w/2, no net clearance


def test_trace_violations_exact_gap_and_crossing(k):
    y_legal = mm(40) + 125_000 + 125_000 + 200_000                  # copper gap exactly 0.2 mm above track:0
    assert k.trace_violations("L0", (mm(20), y_legal), (mm(30), y_legal), 250_000, "A") == []
    v = k.trace_violations("L0", (mm(20), y_legal - 1), (mm(30), y_legal - 1), 250_000, "A")
    assert [(x.code, x.obj_id, x.gap_nm, x.required_nm) for x in v] == [("clearance", "track:0", 199_999, 200_000)]
    cross_ = k.trace_violations("L0", (mm(30), mm(35)), (mm(30), mm(45)), 250_000, "A")
    assert cross_[0].gap_nm < 0 and cross_[0].obj_id == "track:0"


def test_board_edge_outside_and_hole_clearance(k):
    edge_ok = k.trace_violations("L0", (mm(30), 125_000 + 200_000), (mm(40), 125_000 + 200_000), 250_000, "A")
    assert edge_ok == []
    edge_bad = k.trace_violations("L0", (mm(30), 125_000 + 199_999), (mm(40), 125_000 + 199_999), 250_000, "A")
    assert [x.code for x in edge_bad] == ["board_edge"]
    assert [x.code for x in k.trace_violations("L0", (mm(30), mm(50)), (mm(30), mm(101)), 250_000, "A")][0] == "outside_board"
    assert "in_cutout" in [x.code for x in k.trace_violations("L0", (mm(75), mm(90)), (mm(75), mm(95)), 250_000, "A")]
    # NPTH hole at (10, 10), radius 1.6 mm: copper must stay 0.2 mm away
    y_ok = mm(10) + 1_600_000 + 125_000 + 200_000
    assert k.trace_violations("L0", (mm(5), y_ok), (mm(15), y_ok), 250_000, "A") == []
    assert [x.code for x in k.trace_violations("L0", (mm(5), y_ok - 1), (mm(15), y_ok - 1), 250_000, "A")] == ["hole_clearance"]


def test_point_blocked_for_routers(k):
    assert k.point_blocked("L0", (mm(20), mm(20)), "B", 250_000)           # on another net's pad
    assert not k.point_blocked("L0", (mm(20), mm(20)), "A", 250_000)       # its own pad is a target, not an obstacle
    assert k.point_blocked("L0", (mm(-1), mm(20)), "A", 250_000)           # off the board
    assert not k.point_blocked("L0", (mm(45), mm(30)), "A", 250_000)


# ------------------------------------------------------------------ via legality
def via(k, x, y, net="A", dia=600_000, drill=300_000, a="L0", b="L3", **kw):
    return k.check_via((mm(x), mm(y)), net, dia, drill, a, b, **kw)


def test_legal_through_via_in_open_space(k):
    r = via(k, 45, 45)
    assert r.legal and r.kind == ViaKind.THROUGH and r.layers == ALL and r.violations == () and r.plane == ()


def test_annulus_drill_and_diameter_rules(k):
    assert via(k, 45, 45, dia=350_000, drill=200_000).codes() == {"diameter_too_small", "annulus_too_small"}
    assert via(k, 45, 45, dia=500_000, drill=150_000).codes() == {"drill_too_small"}
    assert via(k, 45, 45, dia=500_000, drill=400_000).codes() == {"annulus_too_small"}
    assert via(k, 45, 45, dia=500_000, drill=300_000).legal                   # annulus exactly 0.1 mm
    assert via(k, 45, 45, dia=500_000, drill=500_000).codes() >= {"drill_ge_diameter"}


def test_layer_span_rules(k):
    assert via(k, 45, 45, a="L0", b="L9").codes() == {"unknown_layer"}
    assert via(k, 45, 45, a="L1", b="L1").codes() == {"span_too_short"}
    assert via(k, 45, 45, a="L0", b="L1").codes() == {"span_not_allowed"}      # blind vias are off by default
    assert via(k, 45, 45, a="L0", b="L2", kind=ViaKind.THROUGH).codes() >= {"span_kind_mismatch"}
    assert via(k, 45, 45, a="L0", b="L2", kind=ViaKind.MICRO).codes() == {"span_not_allowed"}   # microvias are adjacent-only
    assert via(k, 45, 45, a="L3", b="L0").legal                                # order does not matter
    kb = GeometryKernel(make_design(), KernelConfig(allow_blind_buried=True))
    blind, buried = via(kb, 45, 45, a="L0", b="L1"), via(kb, 45, 45, a="L1", b="L2")
    assert blind.legal and blind.kind == ViaKind.BLIND and buried.legal and buried.kind == ViaKind.BURIED
    # a via type listed in the rules allows its own span even with the global switch off
    d = make_design()
    d = dataclasses.replace(d, rules=dataclasses.replace(d.rules, via_types=d.rules.via_types + (
        ViaType("Blind", 500_000, 250_000, "L0", "L1", ViaKind.BLIND),)))
    kv = GeometryKernel(d)
    assert via(kv, 45, 45, a="L0", b="L1").legal and via(kv, 45, 45, a="L0", b="L2").codes() == {"span_not_allowed"}


def test_via_cannot_land_on_another_nets_pad_on_an_intermediate_layer():
    kb = GeometryKernel(make_design(), KernelConfig(allow_blind_buried=True))
    # J1 is a through-hole pad (net C) present on all four layers; a buried L1..L2 via of net A lands on it
    r = via(kb, 40, 20, "A", a="L1", b="L2")
    assert not r.legal and {(v.code, v.layer, v.obj_id) for v in r.errors()} >= {("clearance", "L1", "J1.1"), ("clearance", "L2", "J1.1")}
    # the same pad is not a problem for a via that does not cross its layers... it exists on all of them, so move off it
    assert via(kb, 43, 20, "A", a="L1", b="L2").legal
    # a via of the pad's own net is allowed by clearance, but the two drills may not overlap
    same = via(kb, 40, 20, "C", a="L1", b="L2")
    assert same.codes() == {"hole_to_hole"}


def test_smd_pad_only_blocks_layers_it_exists_on():
    kb = GeometryKernel(make_design(), KernelConfig(allow_blind_buried=True))
    assert via(kb, 20, 20, "B", a="L1", b="L2").legal                          # buried under a top-layer SMD pad: fine
    thru = via(kb, 20, 20, "B")
    assert not thru.legal and {v.layer for v in thru.errors()} == {"L0"}       # through via hits it on L0 only
    assert not via(kb, 20, 20, "B", a="L0", b="L1").legal
    assert via(kb, 20, 20, "A").legal                                          # via-in-pad of its own net


def test_via_checks_every_intermediate_layer():
    kb = GeometryKernel(make_design(), KernelConfig(allow_blind_buried=True))
    # track:2 (net Y) runs on L2 only, at x = 65
    r = via(kb, 65, 20, "A")
    assert not r.legal and {(v.layer, v.obj_id) for v in r.errors()} == {("L2", "track:2")}
    assert via(kb, 65, 20, "A", a="L0", b="L1").legal                          # does not reach L2
    assert not via(kb, 65, 20, "A", a="L1", b="L2").legal
    ok_x = 65 + (125_000 + 300_000 + 200_000) / M                              # exactly clear
    assert via(kb, ok_x, 20, "A").legal
    assert not via(kb, ok_x - 0.000001, 20, "A").legal


def test_keepouts_board_edge_and_cutouts(k):
    assert via(k, 90, 15).codes() == {"keepout"}                               # no-via keepout on L0
    assert via(k, 90, 35).legal                                                # that keepout only forbids tracks
    assert via(k, 0.5, 50).legal and via(k, 0.499999, 50).codes() == {"board_edge"}
    assert via(k, -1, 50).codes() == {"outside_board"} and via(k, 75, 95).codes() == {"in_cutout"}


def test_hole_to_hole_and_mounting_holes(k):
    # existing via:0 (net GND) at (80, 20) with a 0.3 mm drill; same-net vias skip copper clearance but not drill spacing
    assert via(k, 80.5, 20, "GND").legal
    assert via(k, 80.499999, 20, "GND").codes() == {"hole_to_hole"}
    # mounting hole at (10, 10), radius 1.6 mm: via copper keeps 0.2 mm away
    assert via(k, 12.1, 10, "A").legal and via(k, 12.099999, 10, "A").codes() == {"hole_clearance"}


def test_plane_antipads_and_thermal_reliefs():
    kp = GeometryKernel(make_design(), KernelConfig(allow_plane_cuts=False))
    cut = via(kp, 40, 60, "A")                                                 # crosses the GND plane on L1
    assert cut.codes() == {"plane_cut"} and cut.plane[0].kind == "antipad" and cut.plane[0].layer == "L1"
    k2 = GeometryKernel(make_design())
    ok = via(k2, 40, 60, "A")
    assert ok.legal and [(p.layer, p.kind, p.radius_nm) for p in ok.plane] == [("L1", "antipad", 300_000 + 200_000)]
    tie = via(k2, 40, 60, "GND")                                               # same net as the plane: thermal relief
    assert tie.legal and [(p.layer, p.kind, p.spokes) for p in tie.plane] == [("L1", "thermal_relief", 4)]
    pwr = via(k2, 65, 60, "PWR")
    assert pwr.legal and [(p.layer, p.plane_net, p.kind) for p in pwr.plane] == [("L2", "PWR", "thermal_relief")]
    starved = via(k2, 80, 40.5, "PWR")                                         # tiny plane island: no room for spokes
    assert starved.legal and [w.code for w in starved.warnings()] == ["thermal_relief_spokes"] and starved.plane[0].spokes == 0


# ------------------------------------------------------------------ kernel integration
def test_kernel_builds_from_the_ir(k):
    s = k.stats()
    assert s["edge"] == 8 * 4 and s["pour"] == 3 and s["keepout"] == 2 and s["hole"] == 4
    assert s["via"] == 4 and s["track"] == 3
    assert s["pad"] == 5 + 4                                             # five SMD pads on L0, plus the through-hole pad on all four layers
    assert len(k.edge_items) == 8 and len(k.hole_items) == 1


def test_everything_in_the_index_is_integer_nanometres(k):
    for it in k.index.items.values():
        pts = [it.core.a, it.core.b] if isinstance(it.core, SegCore) else list(it.core.ring)
        assert all(type(c) is int for p in pts for c in p) and type(it.radius) is int
        assert all(type(c) is int for c in (it.bbox.x0, it.bbox.y0, it.bbox.x1, it.bbox.y1))
    core, r = pad_geometry(dataclasses.replace(make_pin("Z", "1", "A", ["L0"], 5, 5, PadShape.RECT, 2.0, 1.0), rotation_deg=45.0))
    assert all(type(c) is int for p in core.ring for c in p) and type(r) is int


def test_arcs_are_flattened_conservatively():
    arc = TrackArc("arc:0", "X", "L0", Pt(mm(60), mm(50)), Pt(57_071_068, 42_928_932), Pt(mm(50), mm(40)), 250_000)
    ka = GeometryKernel(make_design(arcs=(arc,)))
    pieces = [i for i in ka.index.items.values() if i.obj_id == "arc:0"]
    assert len(pieces) >= 6 and all(p.radius == 125_000 + 1_000 for p in pieces)
    assert {i.obj_id for i in ka.items_at_point("L0", (mm(56), mm(42)), ["track"])} == {"arc:0"}    # on the circle (6-8-10)
    assert ka.items_at_point("L0", (mm(56.3), mm(41.6)), ["track"]) == []                         # radius 10.5: off the track
    ka.remove_object("arc:0")
    assert ka.items_at_point("L0", (mm(56), mm(42)), ["track"]) == []


def test_kernel_builds_from_a_real_kicad_import():
    from pathlib import Path
    from pcbroute.ir import ir_from_kicad
    from pcbroute.kicad_io import read_kicad
    d = ir_from_kicad(read_kicad(Path(__file__).parent / "data" / "buck_mini.kicad_pcb").model)
    kk = GeometryKernel(d)
    assert kk.stats()["pad"] >= 4 and kk.inside_board((mm(100), mm(100)))
    assert not kk.inside_board((mm(10), mm(10)))
    # Q1 pad 5 (net VIN) at (100, 99): a different net may not touch it, VIN itself may
    assert not kk.is_trace_legal("L0", (mm(99), mm(99)), (mm(101), mm(99)), 250_000, "SW")
    assert kk.is_trace_legal("L0", (mm(99), mm(99)), (mm(101), mm(99)), 250_000, "VIN")
