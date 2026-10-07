"""Per-song configuration (``song.toml``).

Only the keys the core needs are typed. Everything else is still available through ``raw`` and
each part's ``options`` dict, so a new generator can take new settings without touching this file.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SONG_FILE = "song.toml"


@dataclass
class SongMeta:
    name: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: str = ""
    charter: str = ""


@dataclass
class TempoConfig:
    mode: str = "click"           # click | beats | chart
    bpm: float | None = None      # click mode: locked tempo (None = detect)
    first_downbeat_s: float | None = None  # click mode: where bar 2 starts in the audio (None = detect)
    beats_per_bar: int = 4
    stem: str = "song"            # stem used for detection (falls back to drums); the full mix finds the
                                  # true song start even when drums enter late
    bpm_range: tuple[float, float] = (80, 180)  # detection folds half/double-time results into this


@dataclass
class PartConfig:
    enabled: bool = True
    stem: str | None = None       # override which stem feeds this part
    generator: str | None = None  # override which registered generator charts this part
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LightingConfig:
    generator: str = "default"
    preset: str = "default"       # built-in preset name, or a path relative to the song folder
    stem: str | None = None       # stem used for energy analysis when the chart has no sections
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReaperConfig:
    # Unset = the built-in copy shipped with yarginator (templates/reaper/); "none" = don't use one.
    template: str | None = None    # template .RPP
    # stem key or chart track name -> name of the template track it should land on
    tracks: dict[str, str] = field(default_factory=dict)
    color_maps: str | None = None  # folder with rockband_*.png MIDI note colour maps
    note_names: str | None = None  # folder with "RBN2 *.txt" MIDI note-name (key) maps
    review_markers: bool = False   # also put reports/review.csv spots in the project as markers
    preview_fx: bool = True        # keys/vocal/harmony tracks get a note filter + synth; guitar/rhythm/bass no FX
    preview_fx_on: bool = False    # ...switched on (False: added bypassed, toggle in REAPER)


@dataclass
class SongConfig:
    song: SongMeta = field(default_factory=SongMeta)
    chart_file: str = "notes.mid"
    instruments: list[str] | None = None  # what this song has; None = everything
    audio_offset_s: float = 0.0   # analysis only: shift stems later(+)/earlier(-) relative to the chart
    stems: dict[str, str] = field(default_factory=dict)
    tempo: TempoConfig = field(default_factory=TempoConfig)
    parts: dict[str, PartConfig] = field(default_factory=dict)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    reaper: ReaperConfig = field(default_factory=ReaperConfig)
    song_ini: dict[str, Any] = field(default_factory=dict)  # extra song.ini keys, written verbatim
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SongConfig":
        parts = {}
        for key, p in d.get("parts", {}).items():
            p = dict(p)
            parts[key] = PartConfig(enabled=p.pop("enabled", True), stem=p.pop("stem", None),
                                    generator=p.pop("generator", None), options=p)
        light = dict(d.get("lighting", {}))
        return cls(
            song=SongMeta(**{k: str(v) for k, v in d.get("song", {}).items()}),
            chart_file=d.get("chart", {}).get("file", "notes.mid"),
            instruments=d.get("chart", {}).get("instruments"),
            audio_offset_s=float(d.get("audio", {}).get("offset_s", 0.0)),
            stems=dict(d.get("stems", {})),
            tempo=TempoConfig(**d.get("tempo", {})),
            parts=parts,
            lighting=LightingConfig(generator=light.pop("generator", "default"),
                                    preset=light.pop("preset", "default"),
                                    stem=light.pop("stem", None), options=light),
            reaper=ReaperConfig(**d.get("reaper", {})),
            song_ini=dict(d.get("song_ini", {})),
            raw=d,
        )


def user_config_path() -> Path:
    """Personal defaults shared by every song (template, colour maps, charter...)."""
    return Path(os.environ.get("YARGINATOR_CONFIG", Path.home() / ".yarginator.toml"))


def _read_toml(path: Path) -> dict[str, Any]:
    # utf-8-sig: Windows editors (and PowerShell 5) like to add a BOM, which tomllib rejects
    return tomllib.loads(path.read_text("utf-8-sig")) if path.exists() else {}


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``over`` onto ``base``; empty strings in ``over`` don't erase a base value."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        elif v != "" or k not in out:
            out[k] = v
    return out


def load_config(path: str | Path) -> SongConfig:
    """song.toml layered over the user config."""
    return SongConfig.from_dict(merge(_read_toml(user_config_path()), _read_toml(Path(path))))
