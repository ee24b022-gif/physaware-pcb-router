"""Tests for multi-net KiCad fixtures."""
import pytest

from pcbroute.kicad_io import read_kicad
from pcbroute.ir import ir_from_kicad
from pathlib import Path


@pytest.fixture(params=[
    "multi_net_open.kicad_pcb",
    "multi_net_obstacle.kicad_pcb", 
    "multi_net_congested.kicad_pcb",
])
def multi_net_fixture(request):
    """Load a multi-net test fixture."""
    p = Path("tests/data") / request.param
    kf = read_kicad(p)
    return ir_from_kicad(kf.model), request.param


def test_fixture_loads_and_has_routable_nets(multi_net_fixture):
    """Verify each fixture loads and has at least 3 routable nets."""
    design, name = multi_net_fixture
    routable = [n for n in design.nets if len(n.pins) >= 2]
    assert len(routable) >= 3, f"{name}: expected >=3 routable nets, got {len(routable)}"


def test_fixture_has_multiple_layers(multi_net_fixture):
    """Verify each fixture has at least 2 copper layers."""
    design, name = multi_net_fixture
    copper_layers = design.board.stackup.layer_ids()
    assert len(copper_layers) >= 2, f"{name}: expected >=2 copper layers, got {copper_layers}"


def test_fixture_parses_valid_kiCad(multi_net_fixture):
    """Verify each fixture produces no validation errors on load."""
    design, name = multi_net_fixture
    errors = [i for i in design.validate() if i.level == "error"]
    assert len(errors) == 0, f"{name}: fixture has validation errors: {errors}"


def test_open_fixture_has_separated_nets():
    """Open board with separated nets."""
    p = Path("tests/data/multi_net_open.kicad_pcb")
    kf = read_kicad(p)
    d = ir_from_kicad(kf.model)
    
    routable = [n for n in d.nets if len(n.pins) >= 2]
    assert len(routable) == 3
    net_names = {n.name for n in routable}
    assert "NET_1" in net_names and "NET_2" in net_names and "GND" in net_names
    
    # Check 2 copper layers
    copper_layers = d.board.stackup.layer_ids()
    assert len(copper_layers) == 2


def test_obstacle_fixture_has_zone():
    """Board with an existing copper obstacle/zone."""
    p = Path("tests/data/multi_net_obstacle.kicad_pcb")
    kf = read_kicad(p)
    d = ir_from_kicad(kf.model)
    
    routable = [n for n in d.nets if len(n.pins) >= 2]
    assert len(routable) == 3
    
    # Check for zone (copper pour)
    assert len(d.copper.zones) >= 1, "Expected at least one zone"
    zone = d.copper.zones[0]
    assert zone.kind.name == "POUR"
    assert zone.net == "GND"
    # IR layer names are L0, L1, etc. (F.Cu maps to L0)
    assert zone.layer == "L0"


def test_congested_fixture_has_existing_tracks():
    """More congested board with existing tracks."""
    p = Path("tests/data/multi_net_congested.kicad_pcb")
    kf = read_kicad(p)
    d = ir_from_kicad(kf.model)
    
    routable = [n for n in d.nets if len(n.pins) >= 2]
    assert len(routable) == 4
    
    # Check for existing tracks
    assert len(d.copper.segments) >= 2, "Expected at least 2 existing segments"
    
    # Check nets
    net_names = {n.name for n in routable}
    assert "NET_1" in net_names
    assert "NET_2" in net_names
    assert "NET_3" in net_names
    assert "GND" in net_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])