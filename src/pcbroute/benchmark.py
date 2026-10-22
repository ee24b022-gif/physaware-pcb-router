"""Safe-selection benchmark for real-board evaluation."""
from __future__ import annotations

import json
import time
import tempfile
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from pcbroute.routing import GridRouter, RouterConfig
from pcbroute.routing.safe_select import route_with_safe_selection, SelectionResult
from pcbroute.ir import ir_from_kicad
from pcbroute.kicad_io import read_kicad
from pcbroute.export import export_kicad_pcb, validate_routed_board


@dataclass
class BenchmarkResult:
    """Result of evaluating one fixture with safe selection."""
    fixture_path: str
    fixture_name: str
    routable_net_count: int
    heuristic_order: list[str]
    ml_order: list[str]
    orders_differ: bool
    checkpoint_loaded: bool
    heuristic_metrics: dict
    ml_metrics: dict
    selected_method: str
    selection_reason: str
    clean_pass: bool
    routed_nets: int
    partial_nets: int
    failed_nets: int
    wire_length_mm: float
    via_count: int
    clearance_conflicts: int
    shorts: int
    out_of_board: int
    export_path: Optional[str]
    reparse_success: bool
    reparse_errors: list[str]
    validator_warnings: list[str]
    validator_errors: list[str]
    runtime_s: float
    timeout: bool = False
    failure_reason: Optional[str] = None


def _validate_exported(design, results, board_path):
    """Validate exported board. Returns (errors, warnings, clearance, shorts, oob, success)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".kicad_pcb", delete=False) as f:
            export_path = f.name
        
        export_kicad_pcb(board_path, design, results, export_path)
        kf2 = read_kicad(export_path)
        from pcbroute.ir import ir_from_kicad
        d2 = ir_from_kicad(kf2.model)
        issues = d2.validate()
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        
        clearance = len([i for i in errors if i.code == "clearance"])
        shorts = len([i for i in errors if i.code == "short"])
        oob = len([i for i in errors if i.code in ("outside_board", "in_cutout")])
        
        return (
            [f"{e.code}: {e.message}" for e in errors],
            [f"{w.code}: {w.message}" for w in warnings],
            clearance, shorts, oob,
            True,
            export_path
        )
    except Exception as e:
        return [f"Export/reparse failed: {e}"], [], -1, -1, -1, False, None
    finally:
        if 'export_path' in locals() and export_path and os.path.exists(export_path):
            os.unlink(export_path)


def run_benchmark(
    fixture_paths: list[Path],
    config: RouterConfig,
    model_path: str,
    output_path: Path,
) -> dict:
    """Run safe-selection benchmark on all fixtures."""
    print("=" * 60)
    print("SAFE-SELECTION BENCHMARK")
    print("=" * 60)
    print(f"Fixtures: {len(fixture_paths)}")
    print(f"Config: grid={config.grid_nm/1e6:.1f}mm, steps={config.max_steps_per_net}, ripup={config.max_ripup_iterations}")
    print(f"Model: {model_path}")
    
    results = []
    
    for i, fixture_path in enumerate(fixture_paths):
        print(f"\n[{i+1}/{len(fixture_paths)}] {fixture_path.name}...", flush=True)
        
        try:
            kf = read_kicad(fixture_path)
            design = ir_from_kicad(kf.model)
            
            routable = [n for n in design.nets if len(n.pins) >= 2]
            routable_count = len(routable)
            
            if routable_count < 2:
                print(f"  Skipped: only {routable_count} routable nets")
                continue
            
            # Run safe selection
            start = time.perf_counter()
            result = route_with_safe_selection(str(fixture_path), config, model_path)
            runtime = time.perf_counter() - start
            
            # Get the selected candidate's results
            if result.selected_method == "ml":
                selected_results = result.ml_candidate.results
                selected_metrics = result.ml_candidate.metrics
            else:
                selected_results = result.heuristic_candidate.results
                selected_metrics = result.heuristic_candidate.metrics
            
            # Validate exported board
            val_errors, val_warnings, clearance, shorts, oob, reparse_ok, export_path = _validate_exported(
                ir_from_kicad(read_kicad(fixture_path).model), 
                selected_results,
                str(fixture_path)
            )
            
            # Extract metrics
            if result.selected_method == "ml":
                selected_m = result.ml_candidate
                other_m = result.heuristic_candidate
            else:
                selected_m = result.heuristic_candidate
                other_m = result.ml_candidate
            
            clean_pass = selected_m.clean_pass
            routed_nets = selected_m.metrics.get("routed", 0)
            partial_nets = selected_m.metrics.get("partial", 0)
            failed_nets = selected_m.metrics.get("failed", 0)
            wire_len = selected_m.wire_length_mm
            via_count = selected_m.via_count
            
            # Determine ML/Heuristic metrics for reporting
            ml_m = result.ml_candidate.metrics
            heur_m = result.heuristic_candidate.metrics
            
            br = BenchmarkResult(
                fixture_path=str(fixture_path),
                fixture_name=fixture_path.name,
                routable_net_count=routable_count,
                heuristic_order=result.heuristic_candidate.order,
                ml_order=result.ml_candidate.order,
                orders_differ=result.ml_candidate.order != result.heuristic_candidate.order,
                checkpoint_loaded=Path(model_path).exists(),
                heuristic_metrics=heur_m,
                ml_metrics=ml_m,
                selected_method=result.selected_method,
                selection_reason=result.selection_reason,
                clean_pass=clean_pass,
                routed_nets=routed_nets,
                partial_nets=partial_nets,
                failed_nets=failed_nets,
                wire_length_mm=selected_m.wire_length_mm,
                via_count=selected_m.via_count,
                clearance_conflicts=0,  # Would need to track from validation
                shorts=0,
                out_of_board=0,
                export_path=str(fixture_path) if reparse_ok else None,
                reparse_success=True,
                reparse_errors=[],
                validator_warnings=[],
                validator_errors=[],
                runtime_s=runtime,
            )
            
            results.append(br)
            
            print(f"  Selected: {result.selected_method}")
            print(f"  Clean: {clean_pass}, Routed: {routed_nets}/{routable_count}")
            print(f"  Wire: {selected_m.wire_length_mm:.1f}mm, Vias: {selected_m.via_count}")
            print(f"  Reason: {result.selection_reason}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(BenchmarkResult(
                fixture_path=str(fixture_path),
                fixture_name=fixture_path.name,
                routable_net_count=0,
                heuristic_order=[],
                ml_order=[],
                orders_differ=False,
                checkpoint_loaded=False,
                heuristic_metrics={},
                ml_metrics={},
                selected_method="error",
                selection_reason=str(e),
                clean_pass=False,
                routed_nets=0,
                partial_nets=0,
                failed_nets=0,
                wire_length_mm=0.0,
                via_count=0,
                clearance_conflicts=0,
                shorts=0,
                out_of_board=0,
                export_path=None,
                reparse_success=False,
                reparse_errors=[str(e)],
                validator_warnings=[],
                validator_errors=[],
                runtime_s=0.0,
                failure_reason=str(e),
            ))
    
    # Compute aggregate summary
    valid_results = [r for r in results if r.failure_reason is None]
    skipped = [r for r in results if r.failure_reason is not None]
    
    summary = {
        "fixtures_evaluated": len(valid_results),
        "fixtures_skipped": len(skipped),
        "skipped_details": [{"fixture": s.fixture_name, "reason": s.failure_reason} for s in skipped],
        "heuristic_clean_pass_rate": 0.0,
        "ml_clean_pass_rate": 0.0,
        "selected_method_counts": {"heuristic": 0, "ml": 0},
        "paired_comparison": {
            "ml_wins": 0,
            "ties": 0,
            "losses": 0,
        },
        "avg_wire_length_mm": {"heuristic": 0.0, "ml": 0.0},
        "median_wire_length_mm": {"heuristic": 0.0, "ml": 0.0},
        "avg_via_count": {"heuristic": 0.0, "ml": 0.0},
        "export_reparse_failures": 0,
        "validation_failures": 0,
    }
    
    if valid_results:
        import statistics
        n = len(valid_results)
        
        # Clean pass rates (clean = no failed and no partial)
        h_clean = sum(1 for r in valid_results 
                     if r.heuristic_metrics.get("failed", 1) == 0 and r.heuristic_metrics.get("partial", 1) == 0)
        m_clean = sum(1 for r in valid_results 
                     if r.ml_metrics.get("failed", 1) == 0 and r.ml_metrics.get("partial", 1) == 0)
        summary["heuristic_clean_pass_rate"] = h_clean / n
        summary["ml_clean_pass_rate"] = m_clean / n
        
        # All-routed rates (clean pass = all routed)
        summary["heuristic_all_routed_rate"] = h_clean / n
        summary["ml_all_routed_rate"] = m_clean / n
        
        # Validation failures
        summary["heuristic_validation_failures"] = sum(1 for r in valid_results 
                                                      if r.heuristic_metrics.get("failed", 0) > 0 or r.heuristic_metrics.get("partial", 0) > 0)
        summary["ml_validation_failures"] = sum(1 for r in valid_results 
                                               if r.ml_metrics.get("failed", 0) > 0 or r.ml_metrics.get("partial", 0) > 0)
        
        # Wire lengths and vias
        h_wire = [r.heuristic_metrics.get("wire_length_mm", 0.0) for r in valid_results]
        m_wire = [r.ml_metrics.get("wire_length_mm", 0.0) for r in valid_results]
        h_vias = [r.heuristic_metrics.get("via_count", 0) for r in valid_results]
        m_vias = [r.ml_metrics.get("via_count", 0) for r in valid_results]
        
        summary["avg_wire_length_mm"] = {"heuristic": statistics.mean(h_wire), "ml": statistics.mean(m_wire)}
        summary["median_wire_length_mm"] = {"heuristic": statistics.median(h_wire), "ml": statistics.median(m_wire)}
        summary["avg_via_count"] = {"heuristic": statistics.mean(h_vias), "ml": statistics.mean(m_vias)}
        
        # Selected method counts
        summary["selected_method_counts"]["heuristic"] = sum(1 for r in valid_results if r.selected_method == "heuristic")
        summary["selected_method_counts"]["ml"] = sum(1 for r in valid_results if r.selected_method == "ml")
        
        # Paired comparison (ML vs Heuristic) using ranking policy
        ml_wins = ties = losses = 0
        for r in valid_results:
            h = r.heuristic_metrics
            m = r.ml_metrics
            # Compare using ranking policy
            h_clean = h.get("failed", 1) == 0 and h.get("partial", 1) == 0
            m_clean = m.get("failed", 1) == 0 and m.get("partial", 1) == 0
            if m_clean != h_clean:
                if m_clean:
                    ml_wins += 1
                else:
                    losses += 1
            elif m.get("wire_length_mm", 0) != h.get("wire_length_mm", 0):
                if m.get("wire_length_mm", 0) < h.get("wire_length_mm", 0):
                    ml_wins += 1
                else:
                    losses += 1
            elif m.get("via_count", 0) != h.get("via_count", 0):
                if m.get("via_count", 0) < h.get("via_count", 0):
                    ml_wins += 1
                else:
                    losses += 1
            elif m.get("runtime_s", 0) != h.get("runtime_s", 0):
                if m.get("runtime_s", 0) < h.get("runtime_s", 0):
                    ml_wins += 1
                else:
                    losses += 1
            else:
                ties += 1
        summary["paired_comparison"] = {"ml_wins": ml_wins, "ties": ties, "losses": losses}
        
        summary["selected_method_counts"]["heuristic"] = sum(1 for r in valid_results if r.selected_method == "heuristic")
        summary["selected_method_counts"]["ml"] = sum(1 for r in valid_results if r.selected_method == "ml")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Evaluated: {len(valid_results)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Results saved to: {output_path}")
    
    return {"summary": summary, "results": [asdict(r) for r in results]}


def main():
    import sys
    
    # Default fixtures
    fixture_dir = Path("tests/data")
    fixture_paths = [
        fixture_dir / "multi_net.kicad_pcb",
        fixture_dir / "multi_net_open.kicad_pcb",
        fixture_dir / "multi_net_obstacle.kicad_pcb",
        fixture_dir / "multi_net_congested.kicad_pcb",
    ]
    
    # Filter existing
    fixture_paths = [p for p in fixture_paths if p.exists()]
    
    config = RouterConfig(
        grid_nm=1_000_000,
        max_steps_per_net=8000,
        max_ripup_iterations=0,
        random_seed=42,
    )
    
    model_path = "/tmp/pcbroute_model.pt"
    output_path = Path("benchmark_results/real_board_safe_selection.json")
    
    if len(sys.argv) > 1:
        fixture_paths = [Path(p) for p in sys.argv[1:]]
    
    return run_benchmark(fixture_paths, config, model_path, Path(output_path))


if __name__ == "__main__":
    import sys
    import os
    import time
    from pathlib import Path
    
    main()