# Gate 0: headroom test

**Question:** is there enough loop-inductance gap between a prioritized heuristic layout (B) and an expert layout (C) to justify a learned router?

## Testbench

- Buck stage: 12 V in, 3.3 V out, 10 A, 1 MHz
- Discrete high- and low-side N-MOSFETs, 5x6 mm QFN / PowerPAK
- Input caps: two 10 uF 1206 MLCCs
- Stack-up: 4-layer FR4 (Top, GND plane, Power plane, Bottom), 1 oz copper, 0.2 mm Top-to-GND dielectric

## Baselines (same fixed placement)

- **A**: Freerouting default, unconstrained
- **B**: Freerouting with hot-loop nets prioritized / pre-routed and locked
- **C**: expert layout (via arrays, L2 keep-out in the loop ROI, aligned return vias, shaped pours)

## Measurement

- FastHenry, port across Q_L D-S, 100 MHz (also 200, 300 MHz)
- Validate with a parallel-plate case and a mesh convergence check
- Cross-check C against a second solver

## Decision rule

Take B and apply C's features one at a time (feature transplant). If a cheap rule closes most of the B-to-C gap and B+ lands within about 10% of C, headroom for ML is thin on this board. Run 3-5 variants before the final call.

## Results

(fill in)
