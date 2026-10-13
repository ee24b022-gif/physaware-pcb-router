# Architecture: compiler for PCB intent and geometry

```
src/pcbroute/
  kicad_io/      CAD adapter: KiCad files -> parser model (KiCad-shaped, lossy where KiCad is)
  board.py       parser model (pads, segments, ...) used only by kicad_io and ir.from_kicad
  ir/            STAGE A  canonical versioned IR (integer nm, immutable, JSON round-trip, validator)
  graph/         STAGE C  IR -> heterogeneous graph (numpy core, PyG / NetworkX exporters)
  geometry/      geometry kernel (month 1, next): spatial index, clearance expansion, via legality
  routing/       STAGE E  deterministic detailed router: A*, negotiated congestion, rip-up/reroute
  ml/            STAGE D  envs/ (Gym, action masks), models/ (GNN, policy), training/  [needs the ml extra]
  validate/      independent DRC and physics checks; must not import routing/
  export/        IR -> KiCad / DSN-SES, round-trip DRC
  fasthenry/ surrogate/   physics-aware extension (Gate 0/1 work)
  cli/
tests/           one test module per package; tests/data holds small hand-written boards only
benchmarks/      (later) PCBWorld / PCBench / synthetic generator harness
native/          (later) C++ or Rust geometry kernel behind the same Python API
```

Dependency rule (enforced by review, then by an import test): `kicad_io -> ir -> graph -> ml`, `ir -> routing`,
`ir -> validate`, `ir -> export`. `ir` imports nothing from the other packages except `ir.from_kicad`, the one
file that knows the parser model. `validate` never imports `routing`, so a router bug cannot hide a DRC bug.

## IR design decisions

| Decision | Choice | Why |
|---|---|---|
| Units | integer nanometres | KiCad files are exact at 1 nm; hashing, equality and snapping are float-noise free |
| Mutability | frozen dataclasses | routing produces a new `Design`; intent patches cannot edit in place |
| Layer ids | `L0` (top) .. `Ln`, with `source_name` kept | the IR does not depend on KiCad naming; export maps back |
| Object ids | `track:n`, `arc:n`, `via:n`, `zone:n:Lk`, `REF.PIN`, plus `source_id` (CAD uuid) | stable handles for "protect" patches and round-trip |
| Arcs | kept as `TrackArc`; flattened pieces from the parser are dropped | no duplicate copper, no export approximation |
| Versioning | `IR_VERSION`, strict check on load | a migration is required before the version changes |
| Unknowns | `None`, never invented (dielectric eps_r, gaps) | the LLM and physics layers must not run on made-up values |

## Things the file cannot say (filled by heuristics, always flagged in `provenance.warnings`)

- layer kind (plane vs signal): zone covers at least half the board and the inner layer holds almost no tracks
- net kind and criticality: name patterns (GND, VDD, CLK...), diff pairs by P/N, +/- suffix
- net layer mask: all layers
- net-class values when no `.kicad_pro` is given: 0.25 mm track, 0.2 mm clearance
- differential pairs: inferred from names, marked `source="inferred"` until a user confirms

## Graph schema (Stage C)

Node types and what each row is: `pad` (one pin), `component`, `net`, `region` (one routing cell on one layer),
`layer` (one copper layer), `via` (existing via), `obstacle` (locked track/via, plane zone, keepout, mounting hole, cutout).

Edge groups: electrical membership, spatial proximity, corridor adjacency, layer transition, pair relationships,
and geometry coupling between pads/vias/obstacles and regions. Reverse edges are named `rev_<relation>`.

Region features: density, blocked fraction, free fraction, capacity (mm of wire), used length, overflow ratio,
plane flag and owner net, RUDY demand and demand per routable layer. Coordinates are normalised by board centre and scale.
`HeteroGraph.check()` verifies shapes, index ranges and finiteness.


## Geometry kernel (`geometry/`)

`GeometryKernel(design)` indexes the IR once and answers every spatial question with exact integer arithmetic.

| Module | Responsibility |
|---|---|
| `shapes.py` | cores (segment or polygon region) and exact squared distances as fractions; "closer than R" is the integer test `num < R^2 * den` |
| `spatial_index.py` | hashed grid per layer (`L0..Ln`), insert/remove for rip-up, exact bbox, point, and segment queries |
| `clearance.py` | `w/2 + c` inflation as an exact predicate (`InflatedObstacle`), per-pair clearance resolution, outer polygon outlines |
| `via_legality.py` | span rules, drill and annulus, board edge, every spanned layer, antipads and thermal reliefs on plane layers |
| `kernel.py` | builds the index from the IR; `trace_violations`, `obstacles_for`, `point_blocked`, `check_via`, `add_*`, `remove_object` |

Rules the kernel applies:
- Touching at exactly the required gap is legal; one nanometre less is a violation.
- Half widths round up, so inflation is conservative by at most 0.5 nm.
- Same-net pads, tracks and vias are not obstacles. Pours never block (they are refilled); keepouts block selectively.
- Rounded rectangles are treated as rectangles; arcs are flattened once at build time and carry an extra radius equal to
  the flattening tolerance, so both are conservative.
- Required clearance between two nets is the larger of the two net classes and the board minimum. Board edge, mounting hole
  and keepout rules come from `KernelConfig` (default: the board minimum, or zero for keepouts).

Validation: every predicate is checked against Shapely on random shapes (`tests/test_geometry_oracle.py`) and on a real
board (18 thousand pairs, zero decision mismatches, distance error below one picometre).
