"""The part-generator interface and registry.

To add or replace how a part is charted:

    @register
    class MyDrums(PartCharter):
        key = "drums-v2"                  # name used in song.toml / --parts
        track_names = (instruments.DRUMS,)
        stem_keys = ("drums",)

        def generate(self, ctx):
            track = ctx.existing(instruments.DRUMS)   # copy of what's there (or empty)
            ...
            return PartResult([track], review=[...], rows=[...])

then select it per song with ``[parts.drums] generator = "drums-v2"``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .. import audio
from ..core import instruments
from ..core.chart import Chart, Track
from ..feedback import ReviewItem


@dataclass
class PartContext:
    chart: Chart                                # the chart being built; read freely, but return tracks instead of mutating
    stems: dict[str, Path] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    stem_override: str | None = None            # [parts.x] stem = "..."
    cache_dir: Path | None = None
    audio_offset_s: float = 0.0
    root: Path | None = None                    # the song folder (for relative paths in options)

    @property
    def tempo(self):
        return self.chart.tempo

    def stem(self, *keys: str) -> Path | None:
        """First available stem among ``keys`` (the per-part override wins)."""
        for k in ((self.stem_override,) if self.stem_override else ()) + keys:
            if k in self.stems:
                return self.stems[k]
        return None

    def existing(self, name: str) -> Track:
        """A copy of the chart's track ``name`` (empty if it doesn't exist yet)."""
        tr = self.chart.tracks.get(name)
        return tr.copy() if tr else Track(name)

    def song_end_tick(self) -> int:
        """[end] event if charted, else the longest stem, else the last event plus a bar."""
        events = self.chart.tracks.get(instruments.EVENTS)
        if events:
            ends = [t for t, txt in events.texts() if txt.strip("[]") == "end"]
            if ends:
                return ends[0]
        lengths = [d for d in (audio.duration(p) for p in self.stems.values()) if d]
        if lengths:
            return self.tempo.s2t(max(lengths))
        return self.chart.end_tick() + self.tempo.bar_ticks(self.tempo.time_sigs[-1])


@dataclass
class PartResult:
    tracks: list[Track]
    review: list[ReviewItem] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)   # written to reports/<part>.csv
    summary: str = ""


class PartCharter(ABC):
    key: ClassVar[str]
    track_names: ClassVar[tuple[str, ...]] = ()
    stem_keys: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""
    implemented: ClassVar[bool] = True
    # Idempotent generators (derived purely from tempo/structure) run every time. Others are skipped
    # when their tracks already have notes, unless asked for by name or with --force.
    always_run: ClassVar[bool] = False

    @abstractmethod
    def generate(self, ctx: PartContext) -> PartResult: ...

    def is_charted(self, track: Track) -> bool:
        """Whether ``track`` already holds this part's playable content (markers like phrases,
        overdrive or solos don't count). Used to skip parts you've already charted or edited."""
        return track.has_notes()


REGISTRY: dict[str, type[PartCharter]] = {}


def register(cls: type[PartCharter]) -> type[PartCharter]:
    REGISTRY[cls.key] = cls
    return cls


def get_charter(key: str) -> PartCharter:
    try:
        return REGISTRY[key]()
    except KeyError:
        raise KeyError(f"no part generator {key!r}; known: {', '.join(sorted(REGISTRY))}") from None
