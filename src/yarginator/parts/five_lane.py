"""5-lane charting (bass now; guitar / rhythm / 5-lane keys later): frets from pitch contour,
sustains, difficulty reductions, and writing the notes.

Frets follow melodic shape rather than absolute pitch: inside a window of a few beats the distinct
pitches are ranked onto the five frets, so the same pitch keeps its fret, higher notes sit on higher
frets, and a riff that moves gets moving frets. Adjacent notes with different pitches never share a
fret if they can avoid it.

Reductions (rough Rock Band conventions, all tunable):
  Expert: everything, ghost notes included
  Hard:   no ghost notes; fast 16th/triplet runs thinned to 8ths
  Medium: notes on 8ths only, at least an 8th apart, 4 frets (G R Y B)
  Easy:   notes on beats only, at least a beat apart, 3 frets (G R Y)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.chart import Track
from ..core.instruments import FIVE_LANE_BASE, Difficulty

DEFAULTS = {
    "lane_window_beats": 8.0,    # how far around each note to look when ranking pitches onto frets
    "sustain_min_beats": 0.75,   # notes held at least this long become sustains
    "sustain_gap_beats": 0.25,   # a sustain stops this far before the next note
    "ghosts": "expert",          # expert: ghost notes on Expert only; all; none
}


@dataclass
class LaneNote:
    tick: int
    end: int          # where the played note ends (ticks); used for sustains
    lane: int
    pitch: float | None
    ghost: bool = False
    sustain: int = 0  # sustain length in ticks (0 = a normal note)


def assign_lanes(ticks: list[int], pitches: list[float | None], ppq: int, window_beats: float = 8.0,
                 lanes: int = 5) -> list[int]:
    """Rank the distinct pitches of each fixed window (``window_beats`` long, aligned to tick 0 so it
    follows the bar lines) onto ``lanes`` frets. Fixed windows guarantee that a pitch keeps its fret
    throughout a phrase; a sliding window gives each note slightly different context and makes
    repeated notes wander between frets."""
    filled = _fill(pitches)
    span = window_beats * ppq
    seg_of = [int(t // span) for t in ticks]
    ranks: dict[int, list[int]] = {}
    for s, p in zip(seg_of, filled):
        if p is not None:
            ranks.setdefault(s, set()).add(round(p))  # type: ignore[arg-type]
    ranks = {s: sorted(v) for s, v in ranks.items()}
    out = []
    for s, p in zip(seg_of, filled):
        if p is None:
            out.append(0)
            continue
        window = ranks[s]
        r = window.index(round(p))
        out.append(r if len(window) <= lanes else min(lanes - 1, r * lanes // len(window)))
    return separate_neighbours(out, filled, lanes, seg_of)


def separate_neighbours(lanes: list[int], pitches: list[float | None], n_lanes: int,
                        segments: list[int] | None = None) -> list[int]:
    """Back-to-back notes with different pitches shouldn't share a fret: nudge the later one in the
    direction the pitch moved (or the other way if it's against the edge). Not across a window
    boundary: that's a new phrase, and nudging there would split a repeated pitch across two frets."""
    out = list(lanes)
    for k in range(1, len(out)):
        a, b = pitches[k - 1], pitches[k]
        if a is None or b is None or round(a) == round(b) or out[k] != out[k - 1]:
            continue
        if segments is not None and segments[k] != segments[k - 1]:
            continue
        step = 1 if b > a else -1
        cand = out[k] + step
        if not 0 <= cand < n_lanes:
            cand = out[k] - step
        if not 0 <= cand < n_lanes:
            continue
        # don't create a contour inversion with the next note (a long run squeezed onto five frets
        # has to repeat frets; zig-zagging would show the line going the wrong way)
        if k + 1 < len(out) and (segments is None or segments[k + 1] == segments[k]):
            c = pitches[k + 1]
            if c is not None and round(c) != round(b):
                if (c > b and out[k + 1] <= cand) or (c < b and out[k + 1] >= cand):
                    continue
        out[k] = cand
    return out


def _fill(pitches: list[float | None]) -> list[float | None]:
    """Unpitched (ghost) notes borrow the previous pitch, so they sit on the previous fret."""
    out, last = [], next((p for p in pitches if p is not None), None)
    for p in pitches:
        last = p if p is not None else last
        out.append(last)
    return out


def add_sustains(notes: list[LaneNote], ppq: int, min_beats: float, gap_beats: float) -> list[LaneNote]:
    out = []
    for k, n in enumerate(notes):
        nxt = notes[k + 1].tick if k + 1 < len(notes) else None
        end = n.end if nxt is None else min(n.end, nxt - int(gap_beats * ppq))
        end = int(round(end / (ppq / 4)) * (ppq / 4))  # sustain ends on a 16th
        length = end - n.tick
        held = (n.end - n.tick) >= min_beats * ppq and length >= ppq // 2 and not n.ghost  # ghosts are muted
        out.append(replace(n, sustain=length if held else 0))
    return out


def reduce(notes: list[LaneNote], difficulty: Difficulty, ppq: int, ghosts: str = "expert",
           window_beats: float = 8.0) -> list[LaneNote]:
    keep_ghosts = ghosts == "all" or (ghosts == "expert" and difficulty == Difficulty.EXPERT)
    src = [n for n in notes if keep_ghosts or not n.ghost]
    if difficulty == Difficulty.EXPERT:
        return src
    out: list[LaneNote] = []
    if difficulty == Difficulty.HARD:
        for n in src:
            on8 = n.tick % (ppq // 2) == 0
            if on8 or not out or n.tick - out[-1].tick >= ppq // 2:
                out.append(n)
        return out
    grid, spacing, lanes = (ppq // 2, ppq // 2, 4) if difficulty == Difficulty.MEDIUM else (ppq, ppq, 3)
    for n in src:
        if n.tick % grid == 0 and (not out or n.tick - out[-1].tick >= spacing):
            out.append(n)
    squeezed = [round(n.lane * (lanes - 1) / 4) for n in out]
    squeezed = separate_neighbours(squeezed, [n.pitch for n in out], lanes,
                                   [int(n.tick // (window_beats * ppq)) for n in out])
    return [replace(n, lane=lane) for n, lane in zip(out, squeezed)]


def write(track: Track, by_difficulty: dict[Difficulty, list[LaneNote]], ppq: int) -> Track:
    """Replace the track's playable notes (every difficulty's fret range) with these."""
    for d in Difficulty:
        base = FIVE_LANE_BASE[d]
        track.remove_notes(range(base, base + 7))  # frets + forced HOPO/strum markers
    for d, notes in by_difficulty.items():
        base = FIVE_LANE_BASE[d]
        for n in notes:
            track.add_note(n.tick, n.sustain or ppq // 8, base + n.lane)
    return track
