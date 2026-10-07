"""Tempo maps: tick <-> seconds conversion, plus constructors for the ways a song gets mapped.

The audio is never moved. Instead the tempo map bends to fit it, so a music video synced to the
original audio stays in sync:

- ``click_tracked``: modern, click-tracked songs. An "offset measure" absorbs whatever comes
  before the first real downbeat; from then on the tempo is locked.
- ``from_beats``: older or live songs. One tempo change per beat, taken from beat detection or a
  hand-made list of beat times.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterator

DEFAULT_PPQ = 480
DEFAULT_US_PER_BEAT = 500_000  # 120 BPM


def bpm_to_us(bpm: float) -> int:
    return int(round(60_000_000 / bpm))


@dataclass(frozen=True)
class TempoChange:
    tick: int
    us_per_beat: int

    @property
    def bpm(self) -> float:
        return 60_000_000 / self.us_per_beat


@dataclass(frozen=True)
class TimeSig:
    tick: int
    numerator: int = 4
    denominator: int = 4


class TempoMap:
    def __init__(self, tempos=(), time_sigs=(), ppq: int = DEFAULT_PPQ):
        self.ppq = ppq
        self.tempos: list[TempoChange] = sorted(tempos, key=lambda t: t.tick)
        if not self.tempos or self.tempos[0].tick != 0:
            self.tempos.insert(0, TempoChange(0, DEFAULT_US_PER_BEAT))
        self.time_sigs: list[TimeSig] = sorted(time_sigs, key=lambda t: t.tick)
        if not self.time_sigs or self.time_sigs[0].tick != 0:
            self.time_sigs.insert(0, TimeSig(0))
        self._ticks = [t.tick for t in self.tempos]
        self._secs = [0.0]
        for prev, cur in zip(self.tempos, self.tempos[1:]):
            self._secs.append(self._secs[-1] + self._span(cur.tick - prev.tick, prev.us_per_beat))

    def _span(self, ticks: float, us_per_beat: int) -> float:
        return ticks * us_per_beat / 1e6 / self.ppq

    # --- conversion -------------------------------------------------------------------------
    def t2s(self, tick: float) -> float:
        i = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        t = self.tempos[i]
        return self._secs[i] + self._span(tick - t.tick, t.us_per_beat)

    def s2t(self, sec: float) -> int:
        i = max(0, bisect.bisect_right(self._secs, sec) - 1)
        t = self.tempos[i]
        return int(round(t.tick + (sec - self._secs[i]) * 1e6 * self.ppq / t.us_per_beat))

    # --- grid -------------------------------------------------------------------------------
    def time_sig_at(self, tick: int) -> TimeSig:
        i = bisect.bisect_right([s.tick for s in self.time_sigs], tick) - 1
        return self.time_sigs[max(0, i)]

    def beat_ticks(self, ts: TimeSig) -> int:
        return self.ppq * 4 // ts.denominator

    def bar_ticks(self, ts: TimeSig) -> int:
        return ts.numerator * self.beat_ticks(ts)

    def beats(self, end_tick: int) -> Iterator[tuple[int, bool]]:
        """Yield ``(tick, is_downbeat)`` for every beat before ``end_tick``."""
        for n, ts in enumerate(self.time_sigs):
            stop = min(self.time_sigs[n + 1].tick if n + 1 < len(self.time_sigs) else end_tick, end_tick)
            for i, tick in enumerate(range(ts.tick, stop, self.beat_ticks(ts))):
                yield tick, i % ts.numerator == 0

    def downbeats(self, end_tick: int) -> list[int]:
        return [t for t, down in self.beats(end_tick) if down]

    # --- constructors -----------------------------------------------------------------------
    @classmethod
    def constant(cls, bpm: float, numerator: int = 4, ppq: int = DEFAULT_PPQ) -> "TempoMap":
        return cls([TempoChange(0, bpm_to_us(bpm))], [TimeSig(0, numerator)], ppq)

    @classmethod
    def click_tracked(cls, bpm: float, first_downbeat_s: float, numerator: int = 4,
                      ppq: int = DEFAULT_PPQ) -> "TempoMap":
        """Measure 1 is an offset measure stretched so the locked tempo's grid lands exactly on
        ``first_downbeat_s``. Whole bars of silence before the downbeat stay at the locked tempo;
        only the remainder (kept between half and one-and-a-half bars) goes in the offset measure.
        """
        bar_s = numerator * 60 / bpm
        if first_downbeat_s <= 1e-3:
            return cls.constant(bpm, numerator, ppq)
        full_bars, rem = divmod(first_downbeat_s, bar_s)
        if rem < bar_s / 2 and full_bars >= 1:
            rem += bar_s
        offset_bpm = numerator * 60 / rem
        bar = numerator * ppq
        return cls([TempoChange(0, bpm_to_us(offset_bpm)), TempoChange(bar, bpm_to_us(bpm))],
                   [TimeSig(0, numerator)], ppq)

    @classmethod
    def from_beats(cls, beat_times, numerator: int = 4, ppq: int = DEFAULT_PPQ) -> "TempoMap":
        """Variable tempo with one tempo change per beat. ``beat_times[0]`` is treated as a downbeat;
        anything before it becomes an offset measure."""
        beats = sorted(float(b) for b in beat_times)
        if len(beats) < 2:
            raise ValueError("need at least two beats")
        tempos, tick = [], 0
        if beats[0] > 1e-3:
            tempos.append(TempoChange(0, int(round(beats[0] * 1e6 / numerator))))
            tick = numerator * ppq
        for a, b in zip(beats, beats[1:]):
            tempos.append(TempoChange(tick, int(round((b - a) * 1e6))))
            tick += ppq
        return cls(tempos, [TimeSig(0, numerator)], ppq)

    def first_downbeat_tick(self) -> int:
        """Tick of the first downbeat after the offset measure (the second bar line)."""
        ts = self.time_sigs[0]
        return ts.tick + self.bar_ticks(ts)
