"""Independent cross-check of the exact integer predicates against Shapely (floating point). Skipped without Shapely."""
import math
import random

import pytest

shapely_geometry = pytest.importorskip("shapely.geometry")

from pcbroute.geometry import PolyCore, SegCore, closer_than
from pcbroute.geometry.shapes import d2

M = 1_000_000


def to_shapely(core):
    if isinstance(core, SegCore):
        return shapely_geometry.Point(core.a) if core.a == core.b else shapely_geometry.LineString([core.a, core.b])
    return shapely_geometry.Polygon(core.ring, [list(h) for h in core.holes])


def random_core(rng):
    kind = rng.choice(["seg", "seg", "point", "rect", "rotrect"])
    cx, cy = rng.randrange(0, 20 * M), rng.randrange(0, 20 * M)
    if kind == "point":
        return SegCore((cx, cy), (cx, cy))
    if kind == "seg":
        return SegCore((cx, cy), (cx + rng.randrange(-5 * M, 5 * M), cy + rng.randrange(-5 * M, 5 * M)))
    w, h = rng.randrange(M // 10, 3 * M), rng.randrange(M // 10, 3 * M)
    ang = 0.0 if kind == "rect" else rng.uniform(0, math.tau)
    c, s = math.cos(ang), math.sin(ang)
    ring = tuple((cx + round(x * c - y * s), cy + round(x * s + y * c))
                 for x, y in ((-w // 2, -h // 2), (w // 2, -h // 2), (w // 2, h // 2), (-w // 2, h // 2)))
    return PolyCore(ring)


def test_distances_and_decisions_match_shapely():
    rng = random.Random(99)
    checked = 0
    for _ in range(4000):
        a, b = random_core(rng), random_core(rng)
        exact = (lambda f: (f[0] / f[1]) ** 0.5)(d2(a, b))
        ref = to_shapely(a).distance(to_shapely(b))
        assert abs(exact - ref) < 1e-4, (a, b, exact, ref)
        for gap in (int(ref) - 1, int(ref) + 1, int(ref) + 1000):
            if gap > 0 and abs(ref - gap) > 1e-3:             # skip exact ties, where the float oracle is the noisy one
                assert closer_than(a, 0, b, 0, gap) == (ref < gap)
                checked += 1
    assert checked > 5000
