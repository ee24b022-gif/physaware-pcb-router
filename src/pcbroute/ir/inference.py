"""Heuristics that fill IR fields the CAD file cannot state. Every result is marked as inferred.

These are placeholders for the Stage B intent layer: a human or schema-validated patch overrides them.
"""
from __future__ import annotations

import re

from .nets import NetKind

_GND = re.compile(r"^(?:[A-Z]?GND[A-Z0-9_]*|VSS[A-Z0-9_]*|0V|PGND|AGND|DGND)$", re.I)
_PWR = re.compile(r"^(?:\+?\d+V\d*|\+\d+(?:\.\d+)?V|V(?:CC|DD|IN|BUS|BAT|OUT|REF)[A-Z0-9_]*|VDD[A-Z0-9_]*|VCC[A-Z0-9_]*|\+3V3|\+5V|\+12V)$", re.I)
_CLK = re.compile(r"(?:^|[_/])(?:CLK|CLOCK|REF_?CLK|XTAL|OSC)(?:[_/]|\d|$)", re.I)
_DIFF = re.compile(r"^(?P<base>.*?)(?P<sep>[_\-]?)(?P<pol>[PN]|\+|\-)(?P<tail>\d*)$")


def infer_net_kind(name: str) -> NetKind:
    n = name.split("/")[-1]
    if _GND.match(n):
        return NetKind.GROUND
    if _PWR.match(n):
        return NetKind.POWER
    if _CLK.search(n):
        return NetKind.CLOCK
    return NetKind.SIGNAL


def infer_criticality(kind: NetKind, in_diff_pair: bool, width_ratio: float) -> float:
    """0..1 routing priority: clocks and differential pairs first, then power, then plain signals."""
    if in_diff_pair or kind == NetKind.CLOCK:
        return 0.9
    if kind == NetKind.POWER or width_ratio >= 2.0:
        return 0.7
    if kind == NetKind.GROUND:
        return 0.4
    return 0.3


def infer_diff_pairs(names: list[str]) -> list[tuple[str, str, str]]:
    """Find (name, positive, negative) from names ending in P/N, +/-, e.g. USB_D+ / USB_D-, TX0_P / TX0_N."""
    cand: dict[tuple[str, str, str], dict[str, str]] = {}
    for n in names:
        m = _DIFF.match(n)
        if not m or not m.group("base"):
            continue
        pol = m.group("pol")
        key = (m.group("base"), m.group("sep"), m.group("tail"))
        side = "p" if pol in ("P", "+") else "n"
        cand.setdefault(key, {})[side] = n
    out = []
    for (base, _sep, tail), d in cand.items():
        if "p" in d and "n" in d:
            out.append((f"{base}{tail}".strip("_-") or base, d["p"], d["n"]))
    return sorted(out)
