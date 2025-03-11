from typing import Any

StrMap = dict[str, Any]


def req[T](x: T | None) -> T:
    assert x is not None
    return x
