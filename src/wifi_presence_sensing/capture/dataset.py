"""Dataset manifests: what each capture is, declared explicitly.

Labels, sessions and positions live in the manifest, not in file names, so
renaming a capture never changes what it stands for.

A manifest is a small YAML file listing every recording with its label, the
session it belongs to (used as the cross-validation group) and, for
localization, the ground-truth position.  ``path``, ``label`` and ``session``
are required; relative paths resolve against the manifest's folder or
``data_root``:

.. code-block:: yaml

    name: occupancy-office
    description: empty vs occupied office, three sessions
    recordings:
      - path: session1/empty.dat
        label: empty
        session: session1
      - path: session1/occupied.dat
        label: occupied
        session: session1
        position: [3.0, 2.5]
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)


class ManifestError(ValueError):
    """Raised when a dataset manifest is missing a field or is inconsistent."""


@dataclass(frozen=True)
class RecordingEntry:
    """One capture file and what it represents.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path of the capture.
    label : str
        Class label (e.g. ``"empty"`` / ``"occupied"``).
    session : str
        Recording session; recordings of one session share conditions and are
        kept on the same side of every cross-validation split.
    position : tuple of float or None
        Ground-truth ``(x, y)`` in metres, for localization.
    """

    path: Path
    label: str
    session: str
    position: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        """A short identifier: the file stem prefixed by its session."""
        return f"{self.session}/{self.path.stem}"


@dataclass(frozen=True)
class DatasetManifest:
    """A named list of recordings.

    Attributes
    ----------
    name : str
        Dataset name.
    recordings : tuple of RecordingEntry
        The recordings.
    description : str
        Free text.
    """

    name: str
    recordings: tuple[RecordingEntry, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.recordings:
            raise ManifestError(f"dataset {self.name!r} lists no recordings")
        names = Counter(r.name for r in self.recordings)
        duplicates = [n for n, c in names.items() if c > 1]
        if duplicates:
            raise ManifestError(f"dataset {self.name!r} lists recordings twice: {duplicates}")

    @property
    def labels(self) -> list[str]:
        """Sorted distinct labels."""
        return sorted({r.label for r in self.recordings})

    @property
    def has_positions(self) -> bool:
        """True when every recording carries a ground-truth position."""
        return all(r.position is not None for r in self.recordings)

    @classmethod
    def from_yaml(cls, path: str | Path, *, data_root: str | Path | None = None) -> DatasetManifest:
        """Load a manifest.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML file.
        data_root : str or pathlib.Path, optional
            Base folder for relative recording paths.  Defaults to the folder
            holding the manifest.

        Returns
        -------
        DatasetManifest
            The manifest.

        Raises
        ------
        ManifestError
            If a required field is missing or malformed.
        """
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        base = Path(data_root) if data_root is not None else manifest_path.parent
        return cls.from_dict(raw, base=base)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base: Path) -> DatasetManifest:
        """Build a manifest from a parsed YAML mapping."""
        if not isinstance(raw, dict):
            raise ManifestError("a manifest must be a mapping with 'name' and 'recordings'")
        unknown = set(raw) - {"name", "description", "recordings"}
        if unknown:
            raise ManifestError(f"unknown manifest key(s): {sorted(unknown)}")
        entries = raw.get("recordings")
        if not isinstance(entries, list):
            raise ManifestError("'recordings' must be a list")
        recordings = tuple(_parse_entry(item, base, i) for i, item in enumerate(entries))
        return cls(
            name=str(raw.get("name", "dataset")),
            recordings=recordings,
            description=str(raw.get("description", "")),
        )

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        """Serialize to a YAML-ready mapping."""
        items = []
        for r in self.recordings:
            path = r.path
            if relative_to is not None:
                try:
                    path = r.path.relative_to(relative_to)
                except ValueError:
                    path = r.path
            entry: dict[str, Any] = {
                "path": path.as_posix(),
                "label": r.label,
                "session": r.session,
            }
            if r.position is not None:
                entry["position"] = [float(r.position[0]), float(r.position[1])]
            items.append(entry)
        return {"name": self.name, "description": self.description, "recordings": items}

    def to_yaml(self, path: str | Path) -> Path:
        """Write the manifest, with recording paths relative to its folder."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(relative_to=destination.parent.resolve())
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        return destination


def _parse_entry(item: Any, base: Path, index: int) -> RecordingEntry:
    if not isinstance(item, dict):
        raise ManifestError(f"recording #{index} must be a mapping")
    unknown = set(item) - {"path", "label", "session", "position"}
    if unknown:
        raise ManifestError(f"recording #{index}: unknown key(s) {sorted(unknown)}")
    for key in ("path", "label", "session"):
        if key not in item:
            raise ManifestError(f"recording #{index} is missing {key!r}")
    path = Path(str(item["path"]))
    if not path.is_absolute():
        path = (base / path).resolve()
    position = item.get("position")
    if position is not None:
        if not isinstance(position, list | tuple) or len(position) != 2:
            raise ManifestError(f"recording #{index}: position must be [x, y] in metres")
        position = (float(position[0]), float(position[1]))
    return RecordingEntry(
        path=path,
        label=str(item["label"]),
        session=str(item["session"]),
        position=position,
    )
