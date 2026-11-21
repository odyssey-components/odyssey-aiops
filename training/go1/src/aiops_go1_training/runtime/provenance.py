"""Fingerprint the pinned Go1 model independently of training weights."""

import hashlib
import subprocess
from pathlib import Path

from .local import COMMIT


def model_digest(directory: str | Path) -> str:
    directory = Path(directory)
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj", ".msh"}
    )
    if not files:
        raise ValueError("model files missing")
    sha = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode()
        data = path.read_bytes()
        sha.update(len(name).to_bytes(8, "big"))
        sha.update(name)
        sha.update(len(data).to_bytes(8, "big"))
        sha.update(data)
    return sha.hexdigest()


def require_source(source: Path) -> str:
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if (
        commit != COMMIT
        or subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "unitree_go1"], check=False
        ).returncode
    ):
        raise ValueError("Go1 source must match the pinned, unmodified model")
    return model_digest(source / "unitree_go1")
