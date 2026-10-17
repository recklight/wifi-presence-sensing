"""Run manifests: the provenance written next to every result.

Every command that writes results also writes ``manifest.json`` beside them,
so each number can be traced back to what produced it.  The manifest records

* the complete configuration that produced them,
* the SHA-256 digest, size and path of every input file,
* the master seed,
* the package version, Python and library versions, and the git commit of the
  working tree when there is one,
* a UTC time stamp,
* anything command-specific (e.g. how many packets were skipped, what
  fraction of noise values or table cells were replaced).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.NamedTemporaryFile(delete=False) as handle:
    ...     _ = handle.write(b"abc")
    >>> file_sha256(handle.name)[:12]
    'ba7816bf8f01'
    >>> os.unlink(handle.name)
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """SHA-256 of an array's contents (dtype and shape included)."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(directory: str | Path | None = None) -> str | None:
    """Commit hash of the working tree, or ``None`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if directory is None else str(directory),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "scipy", "sklearn", "pandas", "matplotlib"):
        try:
            imported = __import__(module)
            versions[module] = str(getattr(imported, "__version__", "unknown"))
        except ImportError:  # pragma: no cover - all are hard dependencies
            versions[module] = "missing"
    return versions


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_manifest(
    *,
    command: str,
    config: Mapping[str, Any],
    inputs: Iterable[str | Path] = (),
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a run manifest.

    Parameters
    ----------
    command : str
        The command that produced the results.
    config : mapping
        Full configuration (``Config.to_dict()``).
    inputs : iterable of path, optional
        Input files to fingerprint.
    seed : int, optional
        Master seed.
    extra : mapping, optional
        Command-specific details.

    Returns
    -------
    dict
        JSON-ready manifest.
    """
    from .. import __version__

    files = []
    for item in inputs:
        path = Path(item)
        entry: dict[str, Any] = {"path": path.as_posix()}
        if path.is_file():
            entry.update(sha256=file_sha256(path), bytes=path.stat().st_size)
        else:
            entry["missing"] = True
        files.append(entry)
    return _jsonable(
        {
            "command": command,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": __version__,
            "git_commit": git_commit(Path.cwd()),
            "seed": seed,
            "versions": _library_versions(),
            "inputs": files,
            "config": dict(config),
            "details": dict(extra or {}),
        }
    )


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest as indented JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _LOGGER.info("manifest written to %s", destination)
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    """Write any JSON-able payload (NumPy values allowed)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def run_directory(root: str | Path, label: str) -> Path:
    """Create ``<root>/<label>_<UTC timestamp>`` (suffixed if it already exists)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{label}_{stamp}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    candidate.mkdir(parents=True)
    return candidate
