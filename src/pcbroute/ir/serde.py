"""Lossless JSON round-trip for IR dataclasses (typed, so enums, tuples and Optionals survive)."""
from __future__ import annotations

import dataclasses
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints


def to_jsonable(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: to_jsonable(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (list, tuple)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, (set, frozenset)):
        return sorted(to_jsonable(x) for x in o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    return o


def from_jsonable(tp: Any, data: Any) -> Any:
    if data is None or tp is Any:
        return data
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return from_jsonable(inner[0], data) if len(inner) == 1 else data
    if dataclasses.is_dataclass(tp):
        hints = get_type_hints(tp)
        kwargs = {f.name: from_jsonable(hints[f.name], data[f.name])
                  for f in dataclasses.fields(tp) if f.name in data}
        return tp(**kwargs)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(data)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_jsonable(args[0], x) for x in data)
        return tuple(from_jsonable(a, x) for a, x in zip(args, data))
    if origin is list:
        return [from_jsonable(args[0], x) for x in data]
    if origin in (set, frozenset):
        return origin(from_jsonable(args[0], x) for x in data)
    if origin is dict:
        return {k: from_jsonable(args[1], v) for k, v in data.items()}
    if tp is float and isinstance(data, int):
        return float(data)
    return data
