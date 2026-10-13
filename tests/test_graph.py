import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pcbroute.graph import GraphConfig, build_graph
from pcbroute.ir import DiffPair, LayerKind, Relations, ir_from_kicad
from pcbroute.kicad_io import read_kicad

D = Path(__file__).parent / "data"
NODE_TYPES = {"pad", "component", "net", "region", "layer", "via", "obstacle"}


@pytest.fixture(scope="module")
def design():
    return ir_from_kicad(read_kicad(D / "buck_mini.kicad_pcb").model, name="buck_mini")


@pytest.fixture(scope="module")
def g(design):
    return build_graph(design)


def test_structure_is_consistent(g):
    assert set(g.x) == NODE_TYPES
    assert g.check() == []
    assert {k: v.dtype for k, v in g.x.items()} == {k: np.float32 for k in NODE_TYPES}
    assert all(ei.dtype == np.int64 for ei in g.edge_index.values())


def test_required_edge_groups_present(g):
    rels = {r for (_s, r, _d) in g.edge_index}
    assert {"member_of", "part_of", "on_net", "near", "corridor", "via_site", "stack", "spans", "in_layer",
            "in_region", "overlaps", "paired_with"} <= rels
    for (s, r, d), ei in list(g.edge_index.items()):
        if not r.startswith("rev_") and (d, f"rev_{r}", s) in g.edge_index:
            assert np.array_equal(g.edge_index[(d, f"rev_{r}", s)], ei[::-1])


def test_node_counts_and_membership(design, g):
    assert g.num_nodes("pad") == 5 and g.num_nodes("component") == 3 and g.num_nodes("net") == 3
    assert g.num_nodes("layer") == 4 and g.num_nodes("via") == 1
    gx, gy = g.meta["grid"]
    assert g.num_nodes("region") == 4 * int((g.cat["region"]["layer"] == 0).sum()) and gx * gy >= 384
    net_of_pad = dict(zip(g.node_ids["pad"], g.cat["pad"]["net"]))
    assert net_of_pad["Q1.5"] == design.net("VIN").index and net_of_pad["H1."] == -1
    src, dst = g.edge_index[("pad", "member_of", "net")]
    assert len(src) == 4 and set(dst) <= {0, 1, 2}


def test_obstacles_cover_holes_cutouts_and_locked_copper(g):
    kinds = [g.x["obstacle"][i, g.x_names["obstacle"].index(f"kind={k}")] for k in ("mounting_hole", "locked_via", "keepout_zone")
             for i in range(g.num_nodes("obstacle"))]
    assert sum(kinds) >= 4          # 1 hole, 1 locked via, 2 keepouts
    ids = g.node_ids["obstacle"]
    assert "via:0" in ids and any(i.startswith("hole:") for i in ids) and any(i.startswith("zone:1:") for i in ids)


def test_region_features(g):
    names = g.x_names["region"]
    cap = g.x["region"][:, names.index("capacity_mm")]
    assert (cap >= 0).all() and cap.max() > 0
    blk = g.x["region"][:, names.index("blocked_frac")]
    assert blk.max() > 0 and blk.max() <= 1.0                       # keepouts and the mounting hole block routing
    used = g.x["region"][:, names.index("used_mm")]
    assert used.sum() > 0                                           # the existing track and arc consume capacity
    assert (g.x["region"][:, names.index("inside_frac")] > 0).all()


def test_plane_ownership_and_demand_exclusion(design):
    d = ir_from_kicad(read_kicad(D / "buck_mini.kicad_pcb").model, layer_kinds={"In1.Cu": LayerKind.GROUND})
    g = build_graph(d)
    owner = g.cat["region"]["plane_net"]
    assert (owner == d.net("GND").index).sum() > 0 and (owner[g.cat["region"]["layer"] != 1] == -1).all()
    assert g.x["region"][:, g.x_names["region"].index("plane_flag")].sum() == (owner >= 0).sum()
    assert "ground" in g.meta["plane_nets_excluded_from_demand"]


def test_pair_edges(design):
    d = dataclasses.replace(design, relations=Relations(diff_pairs=(DiffPair("p", "SW", "VIN", 150_000, 200_000),)))
    g = build_graph(d)
    ei = g.edge_index[("net", "paired_with", "net")]
    assert ei.shape == (2, 2) and set(map(tuple, ei.T)) == {(1, 2), (2, 1)}
    assert g.x["net"][:, g.x_names["net"].index("in_diff_pair")].tolist() == [0.0, 1.0, 1.0]


def test_fourier_features_and_determinism(design):
    base = build_graph(design)
    four = build_graph(design, GraphConfig(fourier_bands=3))
    assert four.x["pad"].shape[1] == base.x["pad"].shape[1] + 12 and four.check() == []
    again = build_graph(design)
    assert all(np.array_equal(base.x[t], again.x[t]) for t in base.x)
    assert all(np.array_equal(base.edge_index[e], again.edge_index[e]) for e in base.edge_index)


def test_coordinates_are_normalised(g):
    names = g.x_names["pad"]
    xy = g.x["pad"][:, [names.index("x"), names.index("y")]]
    assert np.abs(xy).max() <= 1.0 + 1e-6
    assert g.norm["scale_nm"] == 15_000_000


def test_networkx_export(g):
    nxg = g.to_networkx()
    assert nxg.number_of_nodes() == sum(g.num_nodes(t) for t in g.x)
    assert nxg.number_of_edges() == sum(g.num_edges(e) for e in g.edge_index)


def test_pyg_export(g):
    try:
        import torch
        import torch_geometric  # noqa: F401
    except Exception as e:                       # missing or broken torch install: skip, do not fail
        pytest.skip(f"torch / torch_geometric unavailable: {type(e).__name__}")
    data = g.to_pyg()
    assert set(data.node_types) == NODE_TYPES
    assert data["pad"].x.shape == tuple(g.x["pad"].shape) and data["pad"].x.dtype == torch.float32
    assert data["pad", "member_of", "net"].edge_index.dtype == torch.int64
    assert data.validate()


def test_off_board_parts_are_flagged_not_dropped(design):
    from pcbroute.ir import Pt
    moved_pins = tuple(dataclasses.replace(p, position=Pt(0, 0)) for p in design.components[0].pins)
    moved = dataclasses.replace(design.components[0], position=Pt(0, 0), pins=moved_pins)
    d = dataclasses.replace(design, components=(moved,) + design.components[1:])
    g = build_graph(d)
    on = g.x["pad"][:, g.x_names["pad"].index("on_board")]
    assert g.num_nodes("pad") == 5 and int((on == 0).sum()) == len(moved_pins)
    assert g.x["component"][0, g.x_names["component"].index("on_board")] == 0
    assert g.x["net"][:, g.x_names["net"].index("offboard_frac")].max() > 0
    assert build_graph(design).x["pad"][:, g.x_names["pad"].index("on_board")].min() == 1
