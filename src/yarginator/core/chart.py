"""In-memory chart: a tempo map plus named tracks of raw MIDI events.

Tracks hold ``(absolute_tick, mido message)`` pairs and nothing more, so loading and saving a
notes.mid is lossless. Generators that don't understand an event leave it alone, which is what lets
us work on top of existing or hand-edited charts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mido

from .timing import TempoChange, TempoMap, TimeSig

log = logging.getLogger(__name__)

Msg = mido.Message | mido.MetaMessage


@dataclass(frozen=True)
class Note:
    tick: int
    end: int
    pitch: int
    velocity: int = 100
    channel: int = 0

    @property
    def length(self) -> int:
        return self.end - self.tick


def is_note_on(m: Msg) -> bool:
    return m.type == "note_on" and m.velocity > 0


def is_note_off(m: Msg) -> bool:
    return m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)


def _order(m: Msg) -> int:
    """Same-tick ordering: note-offs first (so back-to-back notes don't overlap), then meta
    events (lyrics/text), then note-ons and everything else."""
    if is_note_off(m):
        return 0
    return 1 if m.is_meta else 2


class Track:
    def __init__(self, name: str, events: Iterable[tuple[int, Msg]] = ()):
        self.name = name
        self.events: list[tuple[int, Msg]] = list(events)

    def __repr__(self) -> str:
        return f"Track({self.name!r}, {len(self.events)} events)"

    def copy(self) -> "Track":
        return Track(self.name, self.events)

    # --- adding -----------------------------------------------------------------------------
    def add(self, tick: int, msg: Msg) -> "Track":
        self.events.append((int(tick), msg))
        return self

    def add_note(self, tick: int, length: int, pitch: int, velocity: int = 100, channel: int = 0) -> "Track":
        self.add(tick, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))
        return self.add(tick + max(1, int(length)), mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    def add_text(self, tick: int, text: str) -> "Track":
        return self.add(tick, mido.MetaMessage("text", text=text))

    def add_lyric(self, tick: int, text: str) -> "Track":
        return self.add(tick, mido.MetaMessage("lyrics", text=text))

    # --- querying ---------------------------------------------------------------------------
    def sorted_events(self) -> list[tuple[int, Msg]]:
        return sorted(self.events, key=lambda e: (e[0], _order(e[1])))

    def notes(self, pitches: Iterable[int] | None = None) -> list[Note]:
        wanted = set(pitches) if pitches is not None else None
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        out = []
        for tick, m in self.sorted_events():
            if m.type not in ("note_on", "note_off") or (wanted is not None and m.note not in wanted):
                continue
            key = (m.channel, m.note)
            if is_note_on(m):
                open_notes.setdefault(key, []).append((tick, m.velocity))
            elif open_notes.get(key):
                start, vel = open_notes[key].pop(0)
                out.append(Note(start, tick, m.note, vel, m.channel))
        return sorted(out, key=lambda n: (n.tick, n.pitch))

    def texts(self, kind: str = "text") -> list[tuple[int, str]]:
        return [(t, m.text) for t, m in self.sorted_events() if m.type == kind]

    def lyrics(self) -> list[tuple[int, str]]:
        return self.texts("lyrics")

    def has_notes(self) -> bool:
        return any(is_note_on(m) for _, m in self.events)

    def end_tick(self) -> int:
        return max((t for t, _ in self.events), default=0)

    # --- removing ---------------------------------------------------------------------------
    def remove(self, pred: Callable[[int, Msg], bool]) -> "Track":
        self.events = [(t, m) for t, m in self.events if not pred(t, m)]
        return self

    def remove_notes(self, pitches: Iterable[int] | None = None) -> "Track":
        wanted = set(pitches) if pitches is not None else None
        return self.remove(lambda t, m: m.type in ("note_on", "note_off")
                           and (wanted is None or m.note in wanted))


class Chart:
    def __init__(self, tempo: TempoMap | None = None, conductor_extras: Iterable[tuple[int, Msg]] = ()):
        self.tempo = tempo or TempoMap()
        self.tracks: dict[str, Track] = {}
        # Non-tempo events from the conductor track (track 0), e.g. its track name. Kept verbatim.
        self.conductor_extras: list[tuple[int, Msg]] = list(conductor_extras)

    @property
    def ppq(self) -> int:
        return self.tempo.ppq

    def track(self, name: str, create: bool = True) -> Track | None:
        if name not in self.tracks and create:
            self.tracks[name] = Track(name)
        return self.tracks.get(name)

    def set_track(self, track: Track) -> None:
        """Insert or replace a track (keeping its position if it already existed)."""
        self.tracks[track.name] = track

    def end_tick(self) -> int:
        return max((tr.end_tick() for tr in self.tracks.values()), default=0)

    def has_events(self) -> bool:
        return any(tr.events for tr in self.tracks.values())

    def retime(self, new_tempo: TempoMap, only: set[str] | None = None) -> None:
        """Swap the tempo map, keeping events at the same position in *seconds*: every track, or
        only the tracks named in ``only`` (the rest keep their ticks, i.e. their place on the grid)."""
        old = self.tempo
        for name, tr in self.tracks.items():
            if only is None or name in only:
                tr.events = [(new_tempo.s2t(old.t2s(t)), m) for t, m in tr.events]
        if only is None:
            self.conductor_extras = [(new_tempo.s2t(old.t2s(t)), m) for t, m in self.conductor_extras]
        self.tempo = new_tempo

    # --- file I/O ---------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Chart":
        try:
            mf = mido.MidiFile(str(path))
        except (ValueError, OSError) as e:  # some editors write data bytes > 127; clip them rather than give up
            if "data byte" not in str(e):
                raise
            log.warning("%s: %s; loading with out-of-range data bytes clipped to 127", path, e)
            mf = mido.MidiFile(str(path), clip=True)
        conductor, *rest = mf.tracks
        tempos, sigs, extras = [], [], []
        for tick, m in _absolute(conductor):
            if m.type == "set_tempo":
                tempos.append(TempoChange(tick, m.tempo))
            elif m.type == "time_signature":
                sigs.append(TimeSig(tick, m.numerator, m.denominator))
            elif m.type != "end_of_track":
                extras.append((tick, m))
        chart = cls(TempoMap(tempos, sigs, mf.ticks_per_beat), extras)
        for i, tr in enumerate(rest, 1):
            name = next((m.name for m in tr if m.type == "track_name"), f"TRACK {i}")
            if name in chart.tracks:
                log.warning("duplicate track %r in %s; keeping the last one", name, path)
            chart.tracks[name] = Track(name, [(t, m) for t, m in _absolute(tr)
                                              if m.type not in ("track_name", "end_of_track")])
        return chart

    def save(self, path: str | Path) -> None:
        mf = mido.MidiFile(type=1, ticks_per_beat=self.ppq)
        cond = [(t.tick, mido.MetaMessage("set_tempo", tempo=t.us_per_beat)) for t in self.tempo.tempos]
        cond += [(s.tick, mido.MetaMessage("time_signature", numerator=s.numerator, denominator=s.denominator))
                 for s in self.tempo.time_sigs]
        cond += self.conductor_extras
        mf.tracks.append(_to_midi_track(sorted(cond, key=lambda e: e[0])))
        for tr in self.tracks.values():
            mf.tracks.append(_to_midi_track(tr.sorted_events(), tr.name))
        mf.save(str(path))


def _absolute(track: mido.MidiTrack):
    tick = 0
    for m in track:
        tick += m.time
        yield tick, m


def _to_midi_track(events: list[tuple[int, Msg]], name: str | None = None) -> mido.MidiTrack:
    out = mido.MidiTrack()
    if name is not None:
        out.append(mido.MetaMessage("track_name", name=name, time=0))
    last = 0
    for tick, m in events:
        out.append(m.copy(time=tick - last))
        last = tick
    out.append(mido.MetaMessage("end_of_track", time=0))
    return out


def grid_shift_s(old: TempoMap, new: TempoMap) -> float | None:
    """If ``new`` is ``old`` slid in time (same tempos and time signatures after the offset measure),
    how far it moved in seconds; None if it's a different grid."""
    if (old.ppq != new.ppq or len(old.tempos) != len(new.tempos) or len(old.tempos) < 2
            or [(t.tick, t.us_per_beat) for t in old.tempos[1:]] != [(t.tick, t.us_per_beat) for t in new.tempos[1:]]
            or [(s.tick, s.numerator, s.denominator) for s in old.time_sigs]
            != [(s.tick, s.numerator, s.denominator) for s in new.time_sigs]):
        return None
    tick = old.tempos[1].tick
    return new.t2s(tick) - old.t2s(tick)
