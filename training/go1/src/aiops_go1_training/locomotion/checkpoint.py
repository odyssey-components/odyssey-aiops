"""Portable numeric checkpoints; loading never executes pickled code."""

import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def write_checkpoint(path: str | Path, arrays: Mapping[str, ArrayLike]) -> None:
    path = Path(path)
    keys = cast(Iterable[object], arrays)
    if not arrays or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("checkpoint requires named arrays")
    values = {k: np.asarray(v) for k, v in arrays.items()}
    if "allow_pickle" in values:
        raise ValueError("checkpoint key uses a reserved NumPy keyword")
    if any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("checkpoint has invalid numerical values")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, allow_pickle=False, **values)  # type: ignore[reportArgumentType]  # NumPy stubs treat dynamic array keys as the allow_pickle option.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: str | Path) -> dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    if not result or any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("invalid checkpoint")
    return result
