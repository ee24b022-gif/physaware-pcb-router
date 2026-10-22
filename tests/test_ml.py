"""Test ML module imports and basic functionality."""
import pytest

from pcbroute.ml import predict_net_order
from pcbroute.ir import ir_from_kicad
from pcbroute.kicad_io import read_kicad
from pathlib import Path


def test_ml_import():
    """Test that predict_net_order can be imported."""
    assert predict_net_order is not None


def test_predict_net_order_heuristic_fallback():
    """Test predict_net_order with heuristic fallback (no model)."""
    d = ir_from_kicad(read_kicad(Path("tests/data/buck_mini.kicad_pcb")).model)
    
    # Should work without a model checkpoint (heuristic fallback)
    order = predict_net_order(d, model_path=None)
    assert isinstance(order, list)
    assert all(isinstance(n, str) for n in order)
    # Should include VIN (the only routable net in buck_mini)
    assert "VIN" in order


def test_predict_net_order_with_missing_model():
    """Test predict_net_order with non-existent model path falls back to heuristic."""
    d = ir_from_kicad(read_kicad(Path("tests/data/buck_mini.kicad_pcb")).model)
    
    order = predict_net_order(d, model_path="nonexistent/path/model.pt")
    assert isinstance(order, list)
    assert "VIN" in order