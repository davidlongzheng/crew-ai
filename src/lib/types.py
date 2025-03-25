from typing import Any, TypeAlias

StrMap: TypeAlias = dict[str, Any]
TaskIdxs: TypeAlias = list[tuple[int, int]]


def req[T](x: T | None) -> T:
    assert x is not None
    return x
