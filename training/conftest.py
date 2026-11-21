"""Locate the workspace used for temporary training test data."""

from pathlib import Path


def workspace_root() -> Path:
    product = Path(__file__).resolve().parents[1]
    parent = product.parent.parent
    if product.parent.name == "repositories" and (parent / "go.work").is_file():
        return parent
    return product
