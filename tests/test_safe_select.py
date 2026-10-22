"""Tests for safe candidate selection policy."""
import pytest
from dataclasses import dataclass

from pcbroute.routing.safe_select import (
    CandidateResult,
    CandidateStatus,
    SelectionResult,
    _compare_candidates,
    _selection_reason,
    select_best_ordering,
    route_with_safe_selection,
)
from pcbroute.routing import RouteResult, RouteStatus, RoutingMetrics
from pcbroute.ir import Design, Net
from pcbroute.ir.geometry import Pt


def make_candidate_result(
    method: str,
    clean_pass: bool,
    all_routed: bool,
    wire_length_mm: float = 100.0,
    via_count: int = 0,
    runtime_s: float = 1.0,
) -> CandidateResult:
    """Helper to create a CandidateResult for testing."""
    return CandidateResult(
        method=method,
        order=["NET1", "NET2"],
        results=[],
        metrics={},
        validation_errors=[],
        status=CandidateStatus.CLEAN if clean_pass else CandidateStatus.INVALID,
        clean_pass=clean_pass,
        all_routed=all_routed,
        wire_length_mm=wire_length_mm,
        via_count=via_count,
        runtime_s=runtime_s,
    )


def test_compare_clean_vs_invalid():
    """Clean candidate beats invalid candidate."""
    clean = make_candidate_result("ml", True, True, 100.0)
    invalid = make_candidate_result("heuristic", False, False, 50.0)
    assert _compare_candidates(clean, invalid) == -1  # clean is better
    assert _compare_candidates(invalid, clean) == 1   # invalid is worse


def test_compare_both_clean_all_routed_vs_partial():
    """Clean+all-routed beats clean+partial."""
    a = make_candidate_result("ml", True, True, 100.0)
    b = make_candidate_result("heuristic", True, False, 50.0)
    assert _compare_candidates(a, b) == -1
    assert _compare_candidates(b, a) == 1


def test_compare_both_clean_both_all_routed_wire_length():
    """Shorter wire length wins when both clean and all routed."""
    a = make_candidate_result("ml", True, True, 50.0)
    b = make_candidate_result("heuristic", True, True, 100.0)
    assert _compare_candidates(a, b) == -1
    assert _compare_candidates(b, a) == 1


def test_compare_equal_wire_length_vias():
    """Fewer vias wins when wire length equal."""
    a = make_candidate_result("ml", True, True, 100.0, via_count=0)
    b = make_candidate_result("heuristic", True, True, 100.0, via_count=2)
    assert _compare_candidates(a, b) == -1
    assert _compare_candidates(b, a) == 1


def test_compare_equal_vias_runtime():
    """Lower runtime wins when vias equal."""
    a = make_candidate_result("ml", True, True, 100.0, via_count=0, runtime_s=0.5)
    b = make_candidate_result("heuristic", True, True, 100.0, via_count=0, runtime_s=1.0)
    assert _compare_candidates(a, b) == -1
    assert _compare_candidates(b, a) == 1


def test_selection_reason_clean_vs_invalid():
    """Reason mentions clean pass when one is invalid."""
    clean = make_candidate_result("ml", True, True, 100.0)
    invalid = make_candidate_result("heuristic", False, False, 50.0)
    reason = _selection_reason(clean, invalid, True)
    assert "clean pass vs validation failure" in reason
    assert "ML" in reason


def test_selection_reason_all_routed_vs_partial():
    """Reason mentions all nets routed when one is partial."""
    a = make_candidate_result("ml", True, True, 100.0)
    b = make_candidate_result("heuristic", True, False, 50.0)
    reason = _selection_reason(a, b, True)
    assert "all nets routed vs incomplete" in reason
    assert "ML" in reason


def test_selection_reason_shorter_wire():
    """Reason mentions shorter wire length when lengths differ."""
    a = make_candidate_result("ml", True, True, 50.0)
    b = make_candidate_result("heuristic", True, True, 100.0)
    reason = _selection_reason(a, b, True)
    assert "shorter wire length" in reason
    assert "50.0mm vs 100.0mm" in reason
    assert "ML" in reason


def test_selection_reason_fewer_vias():
    """Reason mentions fewer vias when wire length equal."""
    a = make_candidate_result("ml", True, True, 100.0, via_count=0)
    b = make_candidate_result("heuristic", True, True, 100.0, via_count=2)
    reason = _selection_reason(a, b, True)
    assert "fewer vias" in reason
    assert "0 vs 2" in reason
    assert "ML" in reason


def test_selection_reason_lower_runtime():
    """Reason mentions lower runtime when wire length and vias equal."""
    a = make_candidate_result("ml", True, True, 100.0, via_count=0, runtime_s=0.5)
    b = make_candidate_result("heuristic", True, True, 100.0, via_count=0, runtime_s=1.0)
    reason = _selection_reason(a, b, True)
    assert "lower runtime" in reason
    assert "0.500s vs 1.000s" in reason
    assert "ML" in reason


def test_selection_reason_tie_break():
    """Reason mentions deterministic tie-break when all metrics equal."""
    a = make_candidate_result("ml", True, True, 100.0, via_count=0, runtime_s=1.0)
    b = make_candidate_result("heuristic", True, True, 100.0, via_count=0, runtime_s=1.0)
    reason = _selection_reason(a, b, True)
    assert "deterministic tie-break" in reason
    assert "ML" in reason
    reason_b = _selection_reason(b, a, False)
    assert "deterministic tie-break" in reason_b
    assert "Heuristic" in reason_b


def test_real_board_selection():
    """Integration test with real multi-net fixture."""
    import time
    from pathlib import Path
    from pcbroute.routing import RouterConfig
    from pcbroute.kicad_io import read_kicad
    from pcbroute.ir import ir_from_kicad
    from pcbroute.routing.safe_select import route_with_safe_selection

    board_path = Path("tests/data/multi_net.kicad_pcb")
    config = RouterConfig(grid_nm=1_000_000, max_steps_per_net=3000, random_seed=42)
    model_path = "/tmp/pcbroute_model.pt"

    result = route_with_safe_selection(str(board_path), config, model_path)

    print(f"Selected: {result.selected_method}")
    print(f"Reason: {result.selection_reason}")
    print(f"Fallback: {result.fallback_used} - {result.fallback_reason}")
    print(f"ML: clean={result.ml_candidate.clean_pass}, all_routed={result.ml_candidate.all_routed}, len={result.ml_candidate.wire_length_mm:.1f}")
    print(f"Heuristic: clean={result.heuristic_candidate.clean_pass}, all_routed={result.heuristic_candidate.all_routed}, len={result.heuristic_candidate.wire_length_mm:.1f}")

    # Both should be clean on this simple board
    assert result.ml_candidate.clean_pass
    assert result.heuristic_candidate.clean_pass
    assert result.ml_candidate.all_routed
    assert result.heuristic_candidate.all_routed
    assert len(result.ml_candidate.validation_errors) == 0
    assert len(result.heuristic_candidate.validation_errors) == 0
    assert result.ml_candidate.all_routed
    assert result.heuristic_candidate.all_routed

    # Selected method should be one of the two
    assert result.selected_method in ("ml", "heuristic")
    assert result.selected_order == (result.ml_candidate.order if result.selected_method == "ml" else result.heuristic_candidate.order)
    
    # Check the reason text - winner is determined by runtime (wire lengths are equal)
    assert "lower runtime" in result.selection_reason or "deterministic tie-break" in result.selection_reason
    # When there's a tie-break, "vs" may not be present; otherwise "vs" should be present
    if "tie-break" not in result.selection_reason:
        assert "vs" in result.selection_reason
    # Verify winner is identified
    assert result.selected_method in ("ml", "heuristic")


def test_fallback_when_ml_invalid():
    """Test that heuristic is selected when ML fails."""
    # This is hard to test without a board where ML fails.
    # We verify the fallback logic in _compare_candidates works correctly.
    clean = make_candidate_result("heuristic", True, True, 100.0)
    invalid = make_candidate_result("ml", False, False, 50.0)
    # Heuristic should win
    assert _compare_candidates(clean, invalid) == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])