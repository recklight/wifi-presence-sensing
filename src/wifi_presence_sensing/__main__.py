"""Run the command-line interface with ``python -m wifi_presence_sensing``.

Same as the ``wifi-presence-sensing`` console script, and it also runs from a
checkout without installing (``PYTHONPATH=src``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess test
    main()
