"""Flat, exclusive-create TOML artifact writer."""

import json
from pathlib import Path
from typing import cast


def write_toml(path: Path, values: dict[str, object]) -> None:
    # All values are explicitly flat scalar/string/number-array records.
    with path.open("x") as stream:
        for name, value in values.items():
            if not isinstance(value, (str, int, float, bool, list, tuple)):
                raise TypeError("unsupported TOML contract value")
            encoded = json.dumps(
                list(cast(tuple[object, ...], value)) if isinstance(value, tuple) else value,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write(f"{name} = {encoded}\n")
