# physaware-pcb-router

Physics-aware PCB routing research code for switching power stages (buck converter hot loops).

**Status:** early development. **Keep this repository private** until a patent filing decision has been made.

## Layout

See `docs/architecture.md` for the package map, dependency rules, IR design decisions, and graph schema.

```python
from pcbroute.kicad_io import read_kicad
from pcbroute.ir import ir_from_kicad
from pcbroute.graph import build_graph

design = ir_from_kicad(read_kicad("board.kicad_pcb").model)    # Stage A: canonical IR
assert not [i for i in design.validate() if i.level == "error"]
graph = build_graph(design)                                    # Stage C: heterogeneous graph (numpy)
data = graph.to_pyg()                                          # torch_geometric HeteroData (needs the ml extra)
```

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
