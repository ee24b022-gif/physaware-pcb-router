"""Framework-agnostic heterogeneous graph container (numpy), with PyTorch Geometric and NetworkX exporters."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EdgeType = tuple[str, str, str]      # (source node type, relation, destination node type)


@dataclass
class HeteroGraph:
    node_ids: dict[str, list[str]]                     # per node type: source ids, row i <-> node i
    x: dict[str, np.ndarray]                           # float32 [N, F]
    x_names: dict[str, list[str]]                      # feature column names
    cat: dict[str, dict[str, np.ndarray]]              # int64 [N] categorical ids (for embedding lookups)
    edge_index: dict[EdgeType, np.ndarray]             # int64 [2, E]
    edge_attr: dict[EdgeType, np.ndarray]              # float32 [E, Fe]  (Fe may be 0)
    edge_attr_names: dict[EdgeType, list[str]]
    norm: dict[str, float] = field(default_factory=dict)   # center_x_nm, center_y_nm, scale_nm
    meta: dict = field(default_factory=dict)

    def num_nodes(self, t: str) -> int:
        return self.x[t].shape[0]

    def num_edges(self, et: EdgeType) -> int:
        return self.edge_index[et].shape[1]

    def check(self) -> list[str]:
        """Structural problems; an empty list means the graph is internally consistent."""
        bad: list[str] = []
        for t, x in self.x.items():
            if x.shape[0] != len(self.node_ids[t]):
                bad.append(f"{t}: {x.shape[0]} rows but {len(self.node_ids[t])} ids")
            if x.shape[1] != len(self.x_names[t]):
                bad.append(f"{t}: {x.shape[1]} columns but {len(self.x_names[t])} names")
            if not np.isfinite(x).all():
                bad.append(f"{t}: non-finite feature values")
            for name, c in self.cat.get(t, {}).items():
                if c.shape[0] != x.shape[0]:
                    bad.append(f"{t}.{name}: wrong length")
        for et, ei in self.edge_index.items():
            s, r, d = et
            if ei.shape[0] != 2:
                bad.append(f"{et}: edge_index must be [2, E]")
                continue
            if ei.size and (ei[0].max() >= self.num_nodes(s) or ei[0].min() < 0):
                bad.append(f"{et}: source index out of range")
            if ei.size and (ei[1].max() >= self.num_nodes(d) or ei[1].min() < 0):
                bad.append(f"{et}: destination index out of range")
            a = self.edge_attr[et]
            if a.shape[0] != ei.shape[1] or a.shape[1] != len(self.edge_attr_names[et]):
                bad.append(f"{et}: edge_attr shape {a.shape} does not match {ei.shape[1]} edges / names")
            if not np.isfinite(a).all():
                bad.append(f"{et}: non-finite edge attributes")
        return bad

    def summary(self) -> str:
        lines = ["nodes:"]
        lines += [f"  {t:<10} {self.num_nodes(t):>6}  x {self.x[t].shape[1]} features" for t in self.x]
        lines.append("edges:")
        lines += [f"  {s}-[{r}]->{d}: {self.num_edges((s, r, d))}" for (s, r, d) in self.edge_index]
        return "\n".join(lines)

    def to_networkx(self):
        import networkx as nx
        g = nx.MultiDiGraph()
        for t, x in self.x.items():
            for i, nid in enumerate(self.node_ids[t]):
                g.add_node((t, i), ntype=t, id=nid, x=x[i])
        for (s, r, d), ei in self.edge_index.items():
            a = self.edge_attr[(s, r, d)]
            for e in range(ei.shape[1]):
                g.add_edge((s, int(ei[0, e])), (d, int(ei[1, e])), etype=r, attr=a[e])
        return g

    def to_pyg(self):
        """torch_geometric.data.HeteroData. Needs torch and torch_geometric (the `ml` extra)."""
        import torch
        from torch_geometric.data import HeteroData
        data = HeteroData()
        for t, x in self.x.items():
            data[t].x = torch.from_numpy(x)
            data[t].node_id = self.node_ids[t]
            data[t].x_names = self.x_names[t]
            for name, c in self.cat.get(t, {}).items():
                data[t][name] = torch.from_numpy(c)
        for et, ei in self.edge_index.items():
            data[et].edge_index = torch.from_numpy(ei)
            data[et].edge_attr = torch.from_numpy(self.edge_attr[et])
        data.norm = self.norm
        return data
