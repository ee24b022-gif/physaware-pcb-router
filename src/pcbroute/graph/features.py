"""Feature assembly helpers: named columns, one-hots, Fourier features."""
from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np


class Cols:
    """Collects named float columns so feature names can never drift out of sync with the matrix."""

    def __init__(self, n: int):
        self.n = n
        self._cols: list[np.ndarray] = []
        self.names: list[str] = []

    def add(self, name: str, values) -> "Cols":
        a = np.asarray(values, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"column {name}: expected {self.n} rows, got {a.shape[0]}")
        self._cols.append(a)
        self.names.append(name)
        return self

    def onehot(self, prefix: str, idx: Sequence[int], labels: Sequence[str]) -> "Cols":
        idx = np.asarray(idx, dtype=np.int64)
        for k, lab in enumerate(labels):
            self.add(f"{prefix}={lab}", (idx == k).astype(np.float32))
        return self

    def multihot(self, prefix: str, mask: np.ndarray, labels: Sequence[str]) -> "Cols":
        for k, lab in enumerate(labels):
            self.add(f"{prefix}[{lab}]", mask[:, k] if mask.size else np.zeros(self.n))
        return self

    def fourier(self, prefix: str, xy: np.ndarray, bands: int) -> "Cols":
        for b in range(bands):
            f = (2.0 ** b) * np.pi
            for ax, nm in enumerate("xy"):
                self.add(f"{prefix}_sin{b}{nm}", np.sin(f * xy[:, ax]))
                self.add(f"{prefix}_cos{b}{nm}", np.cos(f * xy[:, ax]))
        return self

    def build(self) -> tuple[np.ndarray, list[str]]:
        if not self._cols:
            return np.zeros((self.n, 0), dtype=np.float32), []
        return np.stack(self._cols, axis=1).astype(np.float32), list(self.names)


def enum_index(value: Enum, members: Sequence[Enum]) -> int:
    return list(members).index(value)


def edge_arrays(src: list[int], dst: list[int], attrs: list[Sequence[float]], names: list[str]):
    ei = np.array([src, dst], dtype=np.int64).reshape(2, -1)
    a = np.array(attrs, dtype=np.float32).reshape(len(src), len(names)) if names else np.zeros((len(src), 0), np.float32)
    return ei, a, list(names)
