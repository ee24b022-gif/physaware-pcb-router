"""Project settings (.kicad_pro JSON), custom rules (.kicad_dru), library tables, Specctra summaries."""
from __future__ import annotations

from dataclasses import dataclass, field

from .sexpr import Node


@dataclass
class NetClass:
    name: str
    clearance: float | None = None
    track_width: float | None = None
    via_diameter: float | None = None
    via_drill: float | None = None
    diff_pair_width: float | None = None
    diff_pair_gap: float | None = None


@dataclass
class ProjectSettings:
    net_classes: dict[str, NetClass] = field(default_factory=dict)
    netclass_patterns: list[tuple[str, str]] = field(default_factory=list)   # (pattern, netclass)
    design_rules: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def netclass_for(self, net: str) -> NetClass | None:
        """Exact-name match against patterns, then the Default class."""
        for pattern, cls in self.netclass_patterns:
            if pattern == net and cls in self.net_classes:
                return self.net_classes[cls]
        return self.net_classes.get("Default")


def _f(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_project(data: dict) -> ProjectSettings:
    ps = ProjectSettings(raw=data)
    ns = data.get("net_settings") or {}
    for c in ns.get("classes") or []:
        name = str(c.get("name", ""))
        ps.net_classes[name] = NetClass(
            name=name, clearance=_f(c.get("clearance")), track_width=_f(c.get("track_width")),
            via_diameter=_f(c.get("via_diameter")), via_drill=_f(c.get("via_drill")),
            diff_pair_width=_f(c.get("diff_pair_width")), diff_pair_gap=_f(c.get("diff_pair_gap")))
    for p in ns.get("netclass_patterns") or []:
        ps.netclass_patterns.append((str(p.get("pattern", "")), str(p.get("netclass", ""))))
    for net, cls in (ns.get("netclass_assignments") or {}).items():
        names = cls if isinstance(cls, list) else [cls]
        for c in names:
            ps.netclass_patterns.append((str(net), str(c)))
    ps.design_rules = ((data.get("board") or {}).get("design_settings") or {}).get("rules") or {}
    return ps


def parse_rules(root: Node) -> list[dict]:
    """Custom DRC rules (.kicad_dru) as plain dicts."""
    out = []
    for r in root.children("rule"):
        a = r.atoms()
        rule = {"name": str(a[0]) if a else "", "layer": r.value("layer"),
                "condition": r.value("condition"), "constraints": []}
        for c in r.children("constraint"):
            ca = c.atoms()
            item = {"type": str(ca[0]) if ca else ""}
            for sub in c.children():
                sa = sub.atoms()
                item[sub.name] = sa[0] if len(sa) == 1 else sa
            rule["constraints"].append(item)
        out.append(rule)
    return out


def parse_lib_table(root: Node) -> list[dict]:
    return [{k: c.value(k, "") for k in ("name", "type", "uri", "options", "descr")}
            for c in root.children("lib")]


def summarize_specctra(root: Node) -> dict:
    """Top-level facts from a Specctra .dsn or .ses (a full DSN model comes later)."""
    out: dict = {"kind": root.name}
    res = root.child("resolution")
    if res is not None:
        out["resolution"] = [str(a) if not isinstance(a, (int, float)) else a for a in res.atoms()]
    unit = root.child("unit")
    if unit is not None and unit.atoms():
        out["unit"] = str(unit.atoms()[0])
    st = root.child("structure")
    if st is not None:
        out["layers"] = [str(l.atoms()[0]) for l in st.children("layer") if l.atoms()]
    net = root.child("network")
    if net is not None:
        out["nets"] = [str(n.atoms()[0]) for n in net.children("net") if n.atoms()]
    pl = root.child("placement")
    if pl is not None:
        out["components"] = sum(len(list(c.children("place"))) for c in pl.children("component"))
    routes = root.child("routes")
    if routes is not None:
        nn = routes.child("network_out")
        nets = list(nn.children("net")) if nn is not None else []
        out["routed_nets"] = len(nets)
        out["wires"] = sum(len(list(n.children("wire"))) for n in nets)
        out["vias"] = sum(len(list(n.children("via"))) for n in nets)
    return out
