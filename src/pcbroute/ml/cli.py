"""python -m pcbroute.ml.cli train [--data DIR] [--output PATH] [--epochs N] [--boards N]

python -m pcbroute.ml.cli evaluate INPUT_KICAD_PCB [--model PATH]

python -m pcbroute.ml.cli train-congestion [--data DIR] [--output PATH] [--epochs N] [--boards N]

python -m pcbroute.ml.cli predict-congestion INPUT_KICAD_PCB [--model PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..ir import ir_from_kicad
from ..kicad_io import read_kicad
from . import (
    generate_synthetic_training_data,
    train_net_ordering_model,
    train_congestion_model,
    predict_net_order,
    predict_congestion,
    evaluate_ordering,
)


def _cmd_train(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcbroute-train", description="Train net ordering model on synthetic data.")
    ap.add_argument("--data", default="ml_training_data", help="Training data output directory")
    ap.add_argument("--output", default="models/router.pt", help="Model checkpoint output path")
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20 for smoke test)")
    ap.add_argument("--boards", type=int, default=3, help="Number of synthetic boards (default: 3 for smoke test)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args(argv)

    try:
        print(f"Generating {args.boards} synthetic boards...")
        generate_synthetic_training_data(
            num_boards=args.boards,
            seed=args.seed,
            output_dir=args.data,
        )
        print(f"Training for {args.epochs} epochs...")
        result = train_net_ordering_model(
            data_dir=args.data,
            output_path=args.output,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(f"Train loss: {result['final_train_loss']:.4f}")
        print(f"Val loss: {result['final_val_loss']:.4f}")
        print(f"Model saved to: {result['model_path']}")
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_train_congestion(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcbroute-train-congestion", description="Train congestion prediction model on synthetic data.")
    ap.add_argument("--data", default="ml_training_data", help="Training data output directory")
    ap.add_argument("--output", default="models/congestion.pt", help="Model checkpoint output path")
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20 for smoke test)")
    ap.add_argument("--boards", type=int, default=3, help="Number of synthetic boards (default: 3 for smoke test)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args(argv)

    try:
        print(f"Generating {args.boards} synthetic boards...")
        generate_synthetic_training_data(
            num_boards=args.boards,
            seed=args.seed,
            output_dir=args.data,
        )
        print(f"Training congestion model for {args.epochs} epochs...")
        result = train_congestion_model(
            data_dir=args.data,
            output_path=args.output,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(f"Train loss: {result['final_train_loss']:.4f}")
        print(f"Val loss: {result['final_val_loss']:.4f}")
        print(f"Model saved to: {result['model_path']}")
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_evaluate(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcbroute-evaluate", description="Evaluate ML vs heuristic net ordering.")
    ap.add_argument("input", help="Input .kicad_pcb file")
    ap.add_argument("--model", default="models/router.pt", help="Model checkpoint path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic routing")
    args = ap.parse_args(argv)

    try:
        p = Path(args.input)
        if not p.exists():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 1

        design = ir_from_kicad(read_kicad(p).model)
        print(f"Loaded design: {design.name}")

        # Get heuristic order (HPWL descending)
        heuristic_order = predict_net_order(design, model_path=None)
        print(f"Heuristic order: {heuristic_order}")

        # Get ML order
        ml_order = predict_net_order(design, model_path=args.model)
        print(f"ML order: {ml_order}")

        # Evaluate
        result = evaluate_ordering(design, heuristic_order, ml_order)

        print("\n=== Heuristic ===")
        for k, v in result["heuristic"].items():
            print(f"  {k}: {v}")

        print("\n=== ML ===")
        for k, v in result["ml"].items():
            print(f"  {k}: {v}")

        print("\n=== Improvement (ML - Heuristic) ===")
        for k, v in result["improvement"].items():
            print(f"  {k}: {v:+.3f}" if isinstance(v, float) else f"  {k}: {v:+d}")

        # Success if ML is not worse
        if result["improvement"]["routed_delta"] >= 0:
            print("\nML ordering not worse than heuristic.")
            return 0
        else:
            print("\nML ordering worse than heuristic.")
            return 1

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_predict_congestion(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcbroute-predict-congestion", description="Predict congestion for a board.")
    ap.add_argument("input", help="Input .kicad_pcb file")
    ap.add_argument("--model", default="models/congestion.pt", help="Model checkpoint path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args(argv)

    try:
        p = Path(args.input)
        if not p.exists():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 1

        design = ir_from_kicad(read_kicad(p).model)
        print(f"Loaded design: {design.name}")

        congestion = predict_congestion(design, model_path=args.model)
        print(f"Congestion predictions:")
        for net, score in sorted(congestion.items(), key=lambda x: -x[1]):
            print(f"  {net}: {score:.3f}")

        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Unified entry point for `python -m pcbroute.ml.cli` with subcommands."""
    ap = argparse.ArgumentParser(prog="python -m pcbroute.ml.cli", description="ML commands for PCB routing.")
    sub = ap.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_ap = sub.add_parser("train", help="Train net ordering model")
    train_ap.add_argument("--data", default="ml_training_data", help="Training data output directory")
    train_ap.add_argument("--output", default="models/router.pt", help="Model checkpoint output path")
    train_ap.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20 for smoke test)")
    train_ap.add_argument("--boards", type=int, default=3, help="Number of synthetic boards (default: 3 for smoke test)")
    train_ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Train congestion subcommand
    train_cong_ap = sub.add_parser("train-congestion", help="Train congestion prediction model")
    train_cong_ap.add_argument("--data", default="ml_training_data", help="Training data output directory")
    train_cong_ap.add_argument("--output", default="models/congestion.pt", help="Model checkpoint output path")
    train_cong_ap.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20 for smoke test)")
    train_cong_ap.add_argument("--boards", type=int, default=3, help="Number of synthetic boards (default: 3 for smoke test)")
    train_cong_ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluate subcommand
    eval_ap = sub.add_parser("evaluate", help="Evaluate ML vs heuristic ordering")
    eval_ap.add_argument("input", help="Input .kicad_pcb file")
    eval_ap.add_argument("--model", default="models/router.pt", help="Model checkpoint path")
    eval_ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic routing")

    # Predict congestion subcommand
    predict_cong_ap = sub.add_parser("predict-congestion", help="Predict congestion for a board")
    predict_cong_ap.add_argument("input", help="Input .kicad_pcb file")
    predict_cong_ap.add_argument("--model", default="models/congestion.pt", help="Model checkpoint path")
    predict_cong_ap.add_argument("--seed", type=int, default=42, help="Random seed")

    args = ap.parse_args(argv)

    if args.command == "train":
        return _cmd_train(argv)
    elif args.command == "train-congestion":
        return _cmd_train_congestion(argv)
    elif args.command == "evaluate":
        return _cmd_evaluate(argv)
    elif args.command == "predict-congestion":
        return _cmd_predict_congestion(argv)
    return 1


# Console script entry points
def train_main(argv=None) -> int:
    """Entry point for `pcbroute-train` console script."""
    return _cmd_train(argv)


def train_congestion_main(argv=None) -> int:
    """Entry point for `pcbroute-train-congestion` console script."""
    return _cmd_train_congestion(argv)


def evaluate_main(argv=None) -> int:
    """Entry point for `pcbroute-evaluate` console script."""
    return _cmd_evaluate(argv)


def predict_congestion_main(argv=None) -> int:
    """Entry point for `pcbroute-predict-congestion` console script."""
    return _cmd_predict_congestion(argv)


if __name__ == "__main__":
    raise SystemExit(main())