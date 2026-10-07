"""REAPER project generation.

Built-in defaults ship with the package in ``templates/reaper/``: the REAPER template, the
rockband_*.png note colour maps and the RBN2 *.txt note-name (key) maps. A song or the user config
can point ``[reaper] template / color_maps / note_names`` elsewhere, or set one to "none".
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

DEFAULT_TEMPLATE = "Reaper Template.rpp"


def bundled(*parts: str) -> Path:
    """Path to a file/folder in the package's built-in REAPER data."""
    return Path(str(resources.files("yarginator.templates").joinpath("reaper", *parts)))


def resolve(value: str | None, default: Path, base: Path) -> Path | None:
    """Config value -> path: unset = built-in ``default``; "none"/"" = off; else relative to ``base``."""
    if value is None:
        return default
    if value.strip().lower() in ("", "none", "off"):
        return None
    return base / value
