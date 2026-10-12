"""python -m pcbroute.kicad_io FILE_OR_PROJECT_DIR [--json] [--tree DEPTH]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .reader import read_kicad, read_project
from .summary import summarize, to_jsonable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pcbroute-inspect", description="Inspect any KiCad file or project.")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true", help="dump the parsed model as JSON")
    ap.add_argument("--tree", type=int, metavar="DEPTH", help="print the raw S-expression tree to this depth")
    args = ap.parse_args(argv)
    p = Path(args.path)
    try:
        if p.is_dir() or p.suffix == ".kicad_pro" and not args.tree:
            obj = read_project(p)
            print(json.dumps(to_jsonable(obj), indent=1) if args.json else summarize(obj))
            return 0
        kf = read_kicad(p)
    except Exception as e:  # report parse problems cleanly on the command line
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.tree is not None:
        if kf.root is None:
            print("no S-expression tree for this file type", file=sys.stderr)
            return 1
        print(json.dumps(kf.root.to_list(args.tree), indent=1, default=str))
    elif args.json:
        print(json.dumps(to_jsonable(kf.model), indent=1, default=str))
    else:
        print(summarize(kf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
