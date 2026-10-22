"""Safe candidate selection policy for net ordering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from . import GridRouter, RouterConfig, RouteResult, RouteStatus
from ..ml import predict_net_order
from ..export import validate_routed_board
from ..ir import Design, ir_from_kicad
from ..ir.geometry import Pt
from ..kicad_io import read_kicad
from ..export import export_kicad_pcb, routing_summary

import tempfile
import os
import json
from pathlib import Path


class CandidateStatus(Enum):
    CLEAN = "clean"
    PARTIAL = "partial"
    INVALID = "invalid"


@dataclass
class CandidateResult:
    method: str  # "ml" or "heuristic"
    order: list[str]
    results: list[RouteResult]
    metrics: dict
    validation_errors: list[str]
    status: CandidateStatus
    clean_pass: bool
    all_routed: bool
    wire_length_mm: float
    via_count: int
    runtime_s: float


@dataclass
class SelectionResult:
    selected_method: str
    selected_order: list[str]
    selected_results: list[RouteResult]
    selected_metrics: dict
    ml_candidate: CandidateResult
    heuristic_candidate: CandidateResult
    fallback_used: bool
    fallback_reason: Optional[str]
    selection_reason: str


def _validate_candidate(
    design: Design,
    results: list[RouteResult],
    board_path: str,
) -> tuple[list[str], bool, bool, int, int, int]:
    """Validate a routed candidate. Returns (errors, clean_pass, all_routed, clearance, shorts, oob)."""
    errors = []
    clean_pass = True
    all_routed = True
    clearance = 0
    shorts = 0
    oob = 0

    try:
        with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False) as f:
            export_path = f.name

        exported_board = export_kicad_pcb(board_path, design, results, export_path)
        kf2 = read_kicad(export_path)
        d2 = ir_from_kicad(kf2.model)
        issues = d2.validate()
        errs = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        clearance = len([i for i in errs if i.code == "clearance"])
        shorts = len([i for i in errs if i.code == "short"])
        oob = len([i for i in errs if i.code in ("outside_board", "in_cutout")])

        if clearance > 0 or shorts > 0 or oob > 0:
            clean_pass = False
            for e in errs:
                errors.append(f"{e.code}: {e.message}")

        for r in results:
            if r.status != RouteStatus.ROUTED:
                all_routed = False
                clean_pass = False
                break

    except Exception as e:
        errors.append(f"Export/reparse failed: {e}")
        clean_pass = False
        all_routed = False
    finally:
        if 'export_path' in locals() and os.path.exists(export_path):
            os.unlink(export_path)

    return errors, clean_pass, all_routed, clearance, shorts, oob


def _evaluate_candidate(
    design: Design,
    order: list[str],
    method: str,
    config: RouterConfig,
    board_path: str,
    model_path: Optional[str],
) -> CandidateResult:
    """Evaluate a single ordering candidate."""
    router = GridRouter(design, config)
    results, metrics = router.route_all(order)

    validation_errors, clean_pass, all_routed, clearance, shorts, oob = _validate_candidate(
        design, results, board_path
    )

    if clean_pass:
        status = CandidateStatus.CLEAN
    elif all_routed:
        status = CandidateStatus.PARTIAL  # all routed but validation issues
    else:
        status = CandidateStatus.INVALID

    return CandidateResult(
        method=method,
        order=order,
        results=results,
        metrics={
            "routed": metrics.nets_routed,
            "partial": metrics.nets_partial,
            "failed": metrics.nets_failed,
            "wire_length_mm": metrics.total_wire_length_mm,
            "via_count": metrics.total_vias,
            "runtime_s": metrics.total_time_s,
        },
        validation_errors=validation_errors,
        status=status,
        clean_pass=clean_pass,
        all_routed=all_routed,
        wire_length_mm=metrics.total_wire_length_mm,
        via_count=metrics.total_vias,
        runtime_s=metrics.total_time_s,
    )


def _compare_candidates(a: CandidateResult, b: CandidateResult) -> int:
    """Compare two candidates. Returns -1 if a < b (a is better), 1 if a > b, 0 if equal."""
    # Priority 1: clean pass
    if a.clean_pass != b.clean_pass:
        return -1 if a.clean_pass else 1
    # Priority 2: all nets routed
    if a.all_routed != b.all_routed:
        return -1 if a.all_routed else 1
    # Priority 3: wire length
    if a.wire_length_mm != b.wire_length_mm:
        return -1 if a.wire_length_mm < b.wire_length_mm else 1
    # Priority 4: via count
    if a.via_count != b.via_count:
        return -1 if a.via_count < b.via_count else 1
    # Priority 5: runtime
    if a.runtime_s != b.runtime_s:
        return -1 if a.runtime_s < b.runtime_s else 1
    return 0


def _selection_reason(a: CandidateResult, b: CandidateResult, a_is_ml: bool) -> str:
    """Generate the actual deciding criterion for selection."""
    # a is the selected candidate, b is the rejected candidate
    # Priority 1: clean pass
    if a.clean_pass != b.clean_pass:
        return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (clean pass vs validation failure)"
    # Priority 2: all nets routed
    if a.all_routed != b.all_routed:
        return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (all nets routed vs incomplete)"
    # Priority 3: wire length (with epsilon for floating-point)
    eps_wire = 1e-3  # 1um tolerance
    if abs(a.wire_length_mm - b.wire_length_mm) > eps_wire:
        # a is selected, so a should have shorter wire length
        better, worse = (a, b) if a.wire_length_mm < b.wire_length_mm else (b, a)
        return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (shorter wire length: {better.wire_length_mm:.1f}mm vs {worse.wire_length_mm:.1f}mm)"
    # Priority 4: via count
    if a.via_count != b.via_count:
        better, worse = (a, b) if a.via_count < b.via_count else (b, a)
        return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (fewer vias: {better.via_count} vs {worse.via_count})"
    # Priority 5: runtime (with epsilon for floating-point)
    eps = 1e-3  # 1ms tolerance
    if abs(a.runtime_s - b.runtime_s) > eps:
        better, worse = (a, b) if a.runtime_s < b.runtime_s else (b, a)
        return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (lower runtime: {better.runtime_s:.3f}s vs {worse.runtime_s:.3f}s)"
    # All measured metrics equal - deterministic tie-break to ML
    return f"{'ML' if a_is_ml else 'Heuristic'} candidate selected (deterministic tie-break: all metrics equal)"


def select_best_ordering(
    board_path: str,
    design: Design,
    config: RouterConfig,
    model_path: Optional[str] = None,
) -> SelectionResult:
    """Select the best net ordering using safe candidate selection."""
    # Evaluate ML candidate
    ml_order = predict_net_order(design, model_path=model_path)
    ml_candidate = _evaluate_candidate(design, ml_order, "ml", config, board_path, model_path)

    # Evaluate heuristic candidate
    heuristic_order = predict_net_order(design, model_path=None)
    heuristic_candidate = _evaluate_candidate(design, heuristic_order, "heuristic", config, board_path, None)

    # Compare candidates
    cmp = _compare_candidates(ml_candidate, heuristic_candidate)

    if cmp <= 0:
        # ML is better or equal
        selected = ml_candidate
        fallback_used = False
        fallback_reason = None
        selection_reason = _selection_reason(ml_candidate, heuristic_candidate, True)
    else:
        # Heuristic is better
        selected = heuristic_candidate
        fallback_used = True
        fallback_reason = (
            "ML candidate failed validation" if not ml_candidate.clean_pass
            else "ML candidate not all routed" if not ml_candidate.all_routed
            else "Heuristic candidate has shorter wire length"
            if heuristic_candidate.wire_length_mm < ml_candidate.wire_length_mm
            else "Heuristic candidate has fewer vias"
            if heuristic_candidate.via_count < ml_candidate.via_count
            else "Heuristic candidate has lower runtime"
        )
        selection_reason = _selection_reason(heuristic_candidate, ml_candidate, False)

    return SelectionResult(
        selected_method=selected.method,
        selected_order=selected.order,
        selected_results=selected.results,
        selected_metrics=selected.metrics,
        ml_candidate=ml_candidate,
        heuristic_candidate=heuristic_candidate,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        selection_reason=selection_reason,
    )


def route_with_safe_selection(
    board_path: str,
    config: RouterConfig = RouterConfig(),
    model_path: Optional[str] = None,
) -> SelectionResult:
    """Route a board using safe candidate selection."""
    kf = read_kicad(board_path)
    design = ir_from_kicad(kf.model)
    return select_best_ordering(board_path, design, config, model_path)


def main(argv=None) -> int:
    import sys
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print("Usage: pcbroute-route-safe <board.kicad_pcb> [--model PATH]")
        return 1

    board_path = argv[0]
    model_path = None
    if "--model" in argv:
        idx = argv.index("--model")
        if idx + 1 < len(argv):
            model_path = argv[idx + 1]

    config = RouterConfig(grid_nm=1_000_000, max_steps_per_net=10000, random_seed=42)
    result = route_with_safe_selection(board_path, config, model_path)

    print(f"Selected method: {result.selected_method}")
    print(f"Selection reason: {result.selection_reason}")
    if result.fallback_used:
        print(f"Fallback reason: {result.fallback_reason}")
    print(f"ML: clean={result.ml_candidate.clean_pass}, routed={result.ml_candidate.all_routed}, "
          f"len={result.ml_candidate.wire_length_mm:.1f}mm, vias={result.ml_candidate.via_count}")
    print(f"Heuristic: clean={result.heuristic_candidate.clean_pass}, routed={result.heuristic_candidate.all_routed}, "
          f"len={result.heuristic_candidate.wire_length_mm:.1f}mm, vias={result.heuristic_candidate.via_count}")
    print(f"Validation errors (ML): {result.ml_candidate.validation_errors}")
    print(f"Validation errors (Heuristic): {result.heuristic_candidate.validation_errors}")

    # Return non-zero if selected candidate is not clean
    return 0 if result.ml_candidate.clean_pass or result.heuristic_candidate.clean_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())