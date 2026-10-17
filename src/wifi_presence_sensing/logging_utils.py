"""Logging setup shared by the CLI and the example scripts.

The package never prints progress itself.  Every diagnostic goes through
:mod:`logging`, so a program that calls these functions decides where the
messages land and at what level.  Results (feature tables,
metrics, figures, manifests) are written to files, and only the CLI echoes a
short summary to the terminal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

#: Format used for both the console and the optional log file.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int = "INFO", log_file: Path | None = None) -> None:
    """Attach a console handler, and optionally a file handler, to the root logger.

    Every handler already on the root logger is removed first, so a repeated
    call in a notebook or a test session does not duplicate every message.

    Parameters
    ----------
    level : str or int, optional
        Threshold for both handlers, as a name (``"INFO"``) or a numeric level.
    log_file : pathlib.Path, optional
        When given, the parent directory is created and every record is also
        appended to this file.

    Raises
    ------
    ValueError
        If ``level`` is a string that is not a known logging level.

    Examples
    --------
    >>> configure_logging("WARNING")
    >>> logging.getLogger("wifi_presence_sensing").getEffectiveLevel() == logging.WARNING
    True
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(
                f"unknown logging level {level!r}; "
                "expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
    else:
        resolved = int(level)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(resolved)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved)
        root.addHandler(file_handler)

    root.setLevel(resolved)
    _LOGGER.debug("logging configured at level %s", logging.getLevelName(resolved))
