"""Lighting generator interface, registry, and preset loading.

A lighting generator turns the chart (sections, tempo, optionally stem energy) into VENUE lighting
events. Presets are TOML files: built-ins live in ``lighting/presets/``; a song can point
``[lighting] preset`` at its own file to tune the show for a specific rig.
"""
from __future__ import annotations

import re
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

from ..core import instruments
from ..core.chart import Chart, Track
from ..parts.base import PartContext

_SECTION_RE = re.compile(r"^\[(?:section|prc)[ _](.+)\]$")


@dataclass
class LightingContext:
    part: PartContext                  # chart, stems, cache, song length helpers
    preset: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    energy_stem: str | None = None

    @property
    def chart(self) -> Chart:
        return self.part.chart

    def sections(self) -> list[tuple[int, str]]:
        """``(tick, name)`` from EVENTS ``[section x]`` / ``[prc_x]`` markers."""
        ev = self.chart.tracks.get(instruments.EVENTS)
        out = []
        for tick, txt in ev.texts() if ev else []:
            m = _SECTION_RE.match(txt.strip())
            if m:
                out.append((tick, m.group(1)))
        return out


class LightingGenerator(ABC):
    key: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def generate(self, ctx: LightingContext) -> Track:
        """Return the full VENUE track (keep non-lighting events from the existing one)."""


LIGHTING: dict[str, type[LightingGenerator]] = {}


def register_lighting(cls: type[LightingGenerator]) -> type[LightingGenerator]:
    LIGHTING[cls.key] = cls
    return cls


def get_lighting(key: str) -> LightingGenerator:
    try:
        return LIGHTING[key]()
    except KeyError:
        raise KeyError(f"no lighting generator {key!r}; known: {', '.join(sorted(LIGHTING))}") from None


def builtin_presets() -> list[str]:
    return sorted(p.name.removesuffix(".toml") for p in resources.files("yarginator.lighting.presets").iterdir()
                  if p.name.endswith(".toml"))


def load_preset(name_or_path: str, base: Path | None = None) -> dict[str, Any]:
    p = Path(name_or_path)
    if base is not None and not p.is_absolute():
        p = base / p
    if p.suffix == ".toml" and p.exists():
        return tomllib.loads(p.read_text("utf-8-sig"))
    res = resources.files("yarginator.lighting.presets").joinpath(f"{name_or_path}.toml")
    if not res.is_file():
        raise FileNotFoundError(f"lighting preset {name_or_path!r} not found (built-ins: {', '.join(builtin_presets())})")
    return tomllib.loads(res.read_text("utf-8"))
