# physaware-pcb-router

Physics-aware PCB routing research code for switching power stages (buck converter hot loops).

**Status:** early development. **Keep this repository private** until a patent filing decision has been made.

## Layout

| Path | Purpose |
|---|---|
| `src/pcbroute/board.py` | Board data model (pads, segments, vias, zones, stack-up) |
| `src/pcbroute/kicad_io/` | KiCad `.kicad_pcb` reading and DSN/SES exchange |
| `src/pcbroute/fasthenry/` | Hot-loop identification and FastHenry `.inp` generation |
| `src/pcbroute/grid/` | Rasterization of boards onto routing grids |
| `src/pcbroute/router/` | Classical router (A*, rip-up) and later RL agent |
| `src/pcbroute/surrogate/` | Learned inductance estimator |
| `experiments/gate0/` | Headroom test: Freerouting vs heuristic vs expert layouts |
| `docs/` | Gate specs, invention log, prior-art search log |

## Reading KiCad files

```bash
python -m pcbroute.kicad_io path/to/file.kicad_pcb      # or .kicad_sch, .net, .kicad_mod, .kicad_pro, .dsn ...
python -m pcbroute.kicad_io path/to/project_dir          # board + all schematic sheets + net classes
python -m pcbroute.kicad_io file.kicad_pcb --json        # full parsed model
python -m pcbroute.kicad_io file.kicad_sch --tree 3      # raw S-expression tree
```

```python
from pcbroute.kicad_io import read_kicad, read_project
board = read_kicad("buck.kicad_pcb").model     # pcbroute.board.Board
proj = read_project("my_project_dir")           # .settings, .board, .schematics
```

For schematic connectivity, export a netlist with `kicad-cli sch export netlist` and read the `.net` file.

## Roadmap

1. **Gate 0**: headroom test. Does a routing gap exist between a prioritized heuristic and an expert layout?
2. **Gate 1**: surrogate estimator (accuracy and ranking against FastHenry)
3. **Gate 2**: router PoC in simulation vs Freerouting and ablations
4. **Gate 3**: fabricated boards and measurements

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
