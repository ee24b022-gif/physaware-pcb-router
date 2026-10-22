"""Real-board evaluation test for multi-net KiCad fixture."""
import json
import tempfile
import os
import time
from pathlib import Path

import pytest

from pcbroute.ml import predict_net_order
from pcbroute.routing import GridRouter, RouterConfig
from pcbroute.ir import ir_from_kicad
from pcbroute.kicad_io import read_kicad
from pcbroute.export import export_kicad_pcb


@pytest.fixture
def multi_net_board():
    """Load the multi-net test fixture."""
    p = Path("tests/data/multi_net.kicad_pcb")
    kf = read_kicad(p)
    return ir_from_kicad(kf.model)


def test_fixture_loads_and_has_routable_nets(multi_net_board):
    """Verify the fixture loads and has at least 3 routable nets."""
    routable = [n for n in multi_net_board.nets if len(n.pins) >= 2]
    assert len(routable) >= 3, f"Expected >=3 routable nets, got {len(routable)}"
    for n in routable:
        assert len(n.pins) == 2, f"Net {n.name} should have 2 pins"


def test_fixture_has_multiple_layers(multi_net_board):
    """Verify the fixture has at least 2 copper layers."""
    copper_layers = multi_net_board.board.stackup.layer_ids()
    assert len(copper_layers) >= 2, f"Expected >=2 copper layers, got {copper_layers}"


def test_fixture_has_existing_copper(multi_net_board):
    """Verify the fixture has at least one existing copper object."""
    assert len(multi_net_board.copper.segments) > 0, "Expected at least one segment"
    assert len(multi_net_board.copper.zones) > 0, "Expected at least one zone"


def test_fixture_valid_kiCad_syntax(multi_net_board):
    """Verify the fixture produces no validation errors on load."""
    errors = [i for i in multi_net_board.validate() if i.level == "error"]
    assert len(errors) == 0, f"Fixture has validation errors: {errors}"


def test_real_board_evaluation(multi_net_board):
    """End-to-end real-board evaluation with export/reparse validation."""
    config = RouterConfig(
        grid_nm=1_000_000,
        max_steps_per_net=3000,
        max_ripup_iterations=0,
        random_seed=42,
    )
    
    model_path = "/tmp/pcbroute_model.pt"
    
    heuristic_order = predict_net_order(multi_net_board, model_path=None)
    ml_order = predict_net_order(multi_net_board, model_path=model_path)
    checkpoint_loaded = Path(model_path).exists()
    
    assert len(heuristic_order) > 0
    assert len(ml_order) > 0
    
    results = {}
    
    for method, order in [("heuristic", heuristic_order), ("ml", ml_order)]:
        router = GridRouter(multi_net_board, config)
        start = time.perf_counter()
        results_list, metrics = router.route_all(order)
        runtime = time.perf_counter() - start
        
        clean_pass = metrics.nets_failed == 0 and metrics.nets_partial == 0
        
        # Export and reparse
        with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False) as f:
            export_path = f.name
        
        try:
            exported_board = export_kicad_pcb(
                "tests/data/multi_net.kicad_pcb",
                multi_net_board,
                results_list,
                export_path,
            )
            
            # Reparse
            kf2 = read_kicad(export_path)
            d2 = ir_from_kicad(kf2.model)
            
            issues = d2.validate()
            errors = [i for i in issues if i.level == "error"]
            warnings = [i for i in issues if i.level == "warning"]
            
            clearance = len([i for i in errors if i.code == "clearance"])
            shorts = len([i for i in errors if i.code == "short"])
            oob = len([i for i in errors if i.code in ("outside_board", "in_cutout")])
            
            reparse_ok = True
            reparse_errors = []
            
        except Exception as e:
            reparse_ok = False
            reparse_errors = [str(e)]
            clearance = shorts = oob = -1
        
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
        
        results[method] = {
            "routed": metrics.nets_routed,
            "partial": metrics.nets_partial,
            "failed": metrics.nets_failed,
            "clean_pass": clean_pass,
            "wire_length_mm": metrics.total_wire_length_mm,
            "via_count": metrics.total_vias,
            "runtime_s": runtime,
            "reparse_ok": reparse_ok,
            "reparse_errors": reparse_errors,
            "clearance": clearance,
            "shorts": shorts,
            "out_of_board": oob,
        }
    
    # Verify both methods succeeded
    h = results["heuristic"]
    m = results["ml"]
    
    assert h["reparse_ok"], f"Heuristic reparse failed: {h['reparse_errors']}"
    assert m["reparse_ok"], f"ML reparse failed: {m['reparse_errors']}"
    assert h["clearance"] == 0, f"Heuristic has clearance errors: {h['clearance']}"
    assert m["clearance"] == 0, f"ML has clearance errors: {m['clearance']}"
    assert h["shorts"] == 0, f"Heuristic has shorts: {h['shorts']}"
    assert m["shorts"] == 0, f"ML has shorts: {m['shorts']}"
    assert h["out_of_board"] == 0, f"Heuristic has OOB: {h['out_of_board']}"
    assert m["out_of_board"] == 0, f"ML has OOB: {m['out_of_board']}"
    
    # Both should clean-pass on this simple board
    assert h["clean_pass"], f"Heuristic did not clean-pass: routed={h['routed']}, partial={h['partial']}, failed={h['failed']}"
    assert m["clean_pass"], f"ML did not clean-pass: routed={m['routed']}, partial={m['partial']}, failed={m['failed']}"
    
    # All routable nets should be routed
    assert h["routed"] == 4, f"Heuristic routed {h['routed']}/4 nets"
    assert m["routed"] == 4, f"ML routed {m['routed']}/4 nets"
    
    # Orders should differ (this is the key ML test)
    assert heuristic_order != ml_order, "ML and heuristic orders should differ"
    
    # Checkpoint should be loaded
    assert checkpoint_loaded, "Model checkpoint not loaded"


def test_real_board_report_generation(multi_net_board):
    """Test that a JSON report can be generated."""
    config = RouterConfig(grid_nm=1_000_000, max_steps_per_net=3000, random_seed=42)
    model_path = "/tmp/pcbroute_model.pt"
    
    heuristic_order = predict_net_order(multi_net_board, model_path=None)
    ml_order = predict_net_order(multi_net_board, model_path=model_path)
    
    # Run both
    router_h = GridRouter(multi_net_board, config)
    results_h, metrics_h = router_h.route_all(heuristic_order)
    
    router_m = GridRouter(multi_net_board, config)
    results_m, metrics_m = router_m.route_all(ml_order)
    
    # Generate report
    report = {
        "board_path": "tests/data/multi_net.kicad_pcb",
        "board_name": multi_net_board.name,
        "n_routable_nets": len([n for n in multi_net_board.nets if len(n.pins) >= 2]),
        "heuristic_order": heuristic_order,
        "ml_order": ml_order,
        "orders_differ": heuristic_order != ml_order,
        "checkpoint_loaded": Path(model_path).exists(),
        "heuristic": {
            "routed": metrics_h.nets_routed,
            "partial": metrics_h.nets_partial,
            "failed": metrics_h.nets_failed,
            "clean_pass": metrics_h.nets_failed == 0 and metrics_h.nets_partial == 0,
            "wire_length_mm": metrics_h.total_wire_length_mm,
            "via_count": metrics_h.total_vias,
            "runtime_s": 0,  # placeholder
        },
        "ml": {
            "routed": metrics_m.nets_routed,
            "partial": metrics_m.nets_partial,
            "failed": metrics_m.nets_failed,
            "clean_pass": metrics_m.nets_failed == 0 and metrics_m.nets_partial == 0,
            "wire_length_mm": metrics_m.total_wire_length_mm,
            "via_count": metrics_m.total_vias,
            "runtime_s": 0,
        },
    }
    
    # Verify report structure
    assert report["n_routable_nets"] >= 3
    assert report["orders_differ"] is True
    assert report["checkpoint_loaded"] is True
    assert report["heuristic"]["clean_pass"] is True
    assert report["ml"]["clean_pass"] is True
    
    # Write report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(report, f, indent=2)
        report_path = f.name
    
    # Read back and verify
    with open(report_path) as f:
        loaded = json.load(f)
    
    assert loaded["board_path"] == report["board_path"]
    assert loaded["heuristic"]["routed"] == 4
    assert loaded["ml"]["routed"] == 4
    
    os.unlink(report_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])