"""Minimal dependency-free S-expression reader for KiCad file formats.

Handles .kicad_pcb, .kicad_sch, .kicad_mod, .kicad_sym, .kicad_dru, .net, lib tables,
and Specctra .dsn/.ses. The parser is iterative, so deeply nested files are fine.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

__all__ = ["Sym", "Node", "ParseError", "parse", "parse_file", "read_text"]


class ParseError(ValueError):
    pass


class Sym(str):
    """A bare token (unquoted, non-numeric) such as `smd` or `F.Cu`."""
    __slots__ = ()


_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', re.S)
_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_UNESC = re.compile(r"\\(.)", re.S)
_ESC = {"n": "\n", "t": "\t", "r": "\r"}


def _atom(tok: str):
    if tok[0] == '"':
        return _UNESC.sub(lambda m: _ESC.get(m.group(1), m.group(1)), tok[1:-1])
    if _INT.match(tok):
        return int(tok)
    if _FLOAT.match(tok):
        return float(tok)
    return Sym(tok)


class Node:
    """One parenthesised list: `name` is the head symbol, `args` the remaining items."""

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return f"Node({self.name!r}, {len(self.args)} args)"

    def atoms(self) -> list:
        return [a for a in self.args if not isinstance(a, Node)]

    def children(self, name: str | None = None) -> Iterator["Node"]:
        for a in self.args:
            if isinstance(a, Node) and (name is None or a.name == name):
                yield a

    def child(self, name: str) -> "Node | None":
        return next(self.children(name), None)

    def value(self, name: str, default=None):
        """First atom of the first child called `name`, e.g. (layer "F.Cu") -> "F.Cu"."""
        c = self.child(name)
        if c is None:
            return default
        a = c.atoms()
        return a[0] if a else default

    def values(self, name: str) -> list:
        c = self.child(name)
        return c.atoms() if c is not None else []

    def xy(self, name: str, default=(0.0, 0.0)) -> tuple[float, float]:
        nums = [x for x in self.values(name) if isinstance(x, (int, float))]
        return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else default

    def walk(self) -> Iterator["Node"]:
        stack = [self]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed([a for a in n.args if isinstance(a, Node)]))

    def to_list(self, depth: int | None = None):
        """Nested-list view for debugging; depth limits how far to expand."""
        out: list = [self.name] if self.name else []
        for a in self.args:
            if isinstance(a, Node):
                out.append(a.to_list(None if depth is None else depth - 1) if (depth is None or depth > 0) else "...")
            else:
                out.append(a)
        return out


def _make_node(items: list) -> Node:
    if items and isinstance(items[0], Sym):
        return Node(str(items[0]), items[1:])
    return Node("", items)


def parse(text: str) -> Node:
    """Parse text into a Node. Multiple top-level forms are wrapped in a `__root__` node."""
    stack: list[list] = []
    top: list = []
    pos = 0
    for m in _TOKEN.finditer(text):
        gap = text[pos:m.start()].strip()
        if gap:
            line = text.count("\n", 0, pos) + 1
            raise ParseError(f"unparseable text near line {line}: {gap[:30]!r}")
        pos = m.end()
        tok = m.group(0)
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if not stack:
                line = text.count("\n", 0, m.start()) + 1
                raise ParseError(f"unexpected ')' at line {line}")
            node = _make_node(stack.pop())
            (stack[-1] if stack else top).append(node)
        else:
            (stack[-1] if stack else top).append(_atom(tok))
    if text[pos:].strip():
        raise ParseError(f"unparseable trailing text: {text[pos:].strip()[:30]!r}")
    if stack:
        raise ParseError("unbalanced parentheses: missing ')'")
    if len(top) == 1 and isinstance(top[0], Node):
        return top[0]
    return Node("__root__", top)


def read_text(path) -> str:
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("utf-8", errors="replace")


def parse_file(path) -> Node:
    return parse(read_text(path))
