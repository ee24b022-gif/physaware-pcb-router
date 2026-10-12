# KiCad reader: validation notes

## Real-file test (KiCad 10, file format version 20260206, 4-layer board)

- Parses with no errors and no warnings in about 1 s.
- Item counts match the raw file exactly: 259 footprints, 1147 pads, 1158 vias, 2099 segments, 14 zones, 8 outline edges.
- Stackup parsed correctly, including the 0.1 mm top dielectric.
- KiCad 10 references nets by name only (no net table); handled.

## Pad geometry: what is and is not proven

Assumptions in `kicad_io/pcb.py`:

1. Pad position is stored in the footprint's unrotated frame; board position = footprint origin + `rotate_kicad(local, footprint angle)`.
2. Pad angle in a `.kicad_pcb` is absolute (it already includes the footprint angle).

Evidence so far (`pcbroute-inspect FILE --health`):

- Raw file: a footprint at rot 90 stores its 2 pads at symmetric local positions with pad angle 90, consistent with both assumptions.
- Track ends near same-net pads that land on the pad, rot 90 footprints: 11/19 with my rotation direction versus 2/19 with the opposite direction.
- Rot 270: 7/9 versus 6/7, inconclusive (sample too small). Rot 180 cannot discriminate direction.
- The test board is partly placed and routed (12 footprints at the origin, ~60% of signal pads on nets with no copper), so most alignment checks have little data.

Status: direction is supported for 90 degrees, not yet for 270 or for arbitrary angles. Settle it with a purpose-built board: 0603 resistors at 0/90/180/270 and a few non-multiples (e.g. 45), each with short tracks drawn to both pad centres; then `pad_alignment` should return 100% everywhere.

## Not tested yet

- Footprints on the bottom side (mirroring); the test board has only one.
- KiCad 5/6/7 files from real projects (only small hand-written snippets).
- Zone fill polygons with holes (stored as one large keyholed polygon in this file, 29k vertices).
