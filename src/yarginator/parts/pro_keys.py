"""Pro keys (PART REAL_KEYS_X/H/M/E) from a keys stem, following the C3 Customs Book rules.

1. Notes: Basic Pitch on the stem, or ``midi = "file.mid"`` to chart from a MIDI transcription.
2. Right hand: the part follows the top voice. A note is kept if it is within ``rh_span`` semitones
   of its chord's top note and of the running top-voice register; the rest is left hand.
3. Grid: the part's feel is measured and taken out, then onsets snap to 16ths (triplets only where
   they clearly fit), as for bass.
4. Two-octave keyboard: the whole part is moved by octaves to best fit C2-C4 (MIDI 48-72).
5. Range shifts (Expert/Hard): only ten white keys show at a time. A per-bar search picks one of
   the six ranges for each bar: preferring the primary ranges (A2-C4, F2-A3, C2-E3), few shifts,
   and as few notes as possible needing to be moved an octave. Notes outside their bar's range are
   moved into it by octaves. A shift note goes about a bar before the first note that needs it
   (never before a note the new range can't show), and one goes at the very start.
6. Expert: chords up to 4 notes within an octave. Sustains end a 16th before the next note
   (simple moves) or an 8th (chord changes).
7. Hard: no grace notes, fast 16ths thinned, chords up to 3 notes within 11 semitones, jumps
   over 11 semitones moved an octave or dropped, shifts re-planned for what's left.
   Medium: a quarter note between gems, chords of 2 within 9 semitones, jumps up to 9, one fixed
   range (no shifts), sustains end a quarter before the next note.
   Easy: about a half note between gems, single notes, jumps up to 7, Medium's range.
8. ``[idle]``/``[play]`` animation events on Expert and a copy of Expert in PART KEYS_ANIM_RH (so the
   in-game right hand animates); ``animation = false`` turns both off.

Not generated (yet): trill / glissando markers, overdrive, solos, left-hand animation.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from ..core import instruments
from ..core.chart import Track
from ..core.instruments import Difficulty
from ..feedback import ReviewItem
from .base import PartCharter, PartContext, PartResult, register
from .quantize import bar_grids, feel_curve, snap

KEY_LO, KEY_HI = 48, 72                     # C2..C4 in Rock Band naming
RANGES = {0: (48, 64), 2: (50, 65), 4: (52, 67), 5: (53, 69), 7: (55, 71), 9: (57, 72)}
PRIMARY = (9, 5, 0)                         # A2-C4, F2-A3, C2-E3
ANIM_RH = "PART KEYS_ANIM_RH"
ANIM_EVENTS = ("[idle]", "[idle_realtime]", "[idle_intense]", "[play]", "[mellow]", "[intense]", "[play_solo]")

DEFAULTS = {
    "midi": None,               # chart from this MIDI file instead of transcribing the stem
    "midi_tracks": None,        # only these track names from it
    "onset_threshold": 0.5,     # Basic Pitch: higher = fewer, surer notes
    "frame_threshold": 0.3,
    "min_amp": 0.2,             # drop transcribed notes quieter than this (0-1)
    "min_note_s": 0.06,
    "cluster_s": 0.04,          # onsets this close together are one chord
    "rh_span": 12,              # right hand: within this many semitones of the top voice
    "register_bars": 4,         # how far around each chord to read the top-voice register
    "grid": "16th",
    "triplets": "8th-triplet",
    "timing_offset_ms": "auto",
    "shift_cost": 6.0,          # range planning: cost of a shift ...
    "nonprimary_cost": 1.0,     # ... of each bar in a non-primary range ...
    "outside_cost": 1.0,        # ... and of each note that has to move an octave to fit
    "shift_lead_bars": 1.0,     # place a shift this far ahead of the first note that needs it
    "sustain_min_beats": 0.5,   # notes held at least this long sustain
    "animation": True,
}

# per-difficulty limits from the book
LIMITS = {
    Difficulty.EXPERT: dict(chord=4, span=12, jump=None, spacing=0, shifts=True, gap_simple=0.25, gap_other=0.5),
    Difficulty.HARD: dict(chord=3, span=11, jump=11, spacing=0, shifts=True, gap_simple=0.25, gap_other=0.5),
    Difficulty.MEDIUM: dict(chord=2, span=9, jump=9, spacing=1.0, shifts=False, gap_simple=1.0, gap_other=1.0),
    Difficulty.EASY: dict(chord=1, span=0, jump=7, spacing=2.0, shifts=False, gap_simple=1.0, gap_other=1.0),
}


def rb_name(m: int) -> str:
    """Rock Band / RBN naming: MIDI 48 = C2, 72 = C4."""
    return f"{'C C# D D# E F F# G G# A A# B'.split()[m % 12]}{m // 12 - 2}"


@dataclass
class Chord:
    tick: int
    pitches: list[int]                 # charted pitches, high to low
    ends: list[int]                    # where each played note ends (ticks)
    lengths: list[int] = field(default_factory=list)   # written note lengths, after sustain rules

    @property
    def top(self) -> int:
        return self.pitches[0]


# --- 1-2: notes and right hand -------------------------------------------------------------------
def right_hand(notes: np.ndarray, cluster_s: float, span: int, register_s: float) -> tuple[np.ndarray, int]:
    """Keep notes within ``span`` of their chord's top and of the running top-voice register."""
    if len(notes) == 0:
        return notes, 0
    starts = notes[:, 0]
    cluster = np.zeros(len(notes), dtype=int)
    for i in range(1, len(notes)):
        cluster[i] = cluster[i - 1] + (starts[i] - starts[i - 1] > cluster_s)
    tops = np.array([notes[cluster == c, 2].max() for c in range(cluster[-1] + 1)])
    ctimes = np.array([starts[cluster == c].min() for c in range(cluster[-1] + 1)])
    reg = np.array([np.median(tops[np.abs(ctimes - t) <= register_s]) for t in ctimes])
    keep = (notes[:, 2] >= tops[cluster] - span) & (notes[:, 2] >= reg[cluster] - span)
    return notes[keep], int((~keep).sum())


def best_octave_shift(pitches, lo: int = KEY_LO, hi: int = KEY_HI) -> int:
    """The whole-part octave move (in semitones) that puts the most notes on the two-octave board."""
    p = np.asarray(pitches)
    options = [12 * k for k in range(-4, 5)]
    return max(options, key=lambda s: (np.sum((p + s >= lo) & (p + s <= hi)), -abs(s)))


def wrap_into(p: int, lo: int, hi: int) -> int | None:
    """Move ``p`` by octaves into [lo, hi], as close to where it was as possible."""
    cands = [p + 12 * k for k in range(-5, 6) if lo <= p + 12 * k <= hi]
    return min(cands, key=lambda q: abs(q - p)) if cands else None


def limit_chord(pitches: list[int], max_notes: int, span: int) -> list[int]:
    """Top note first, then the highest others within ``span`` of it, up to ``max_notes``."""
    ps = sorted(set(pitches), reverse=True)
    out = [ps[0]]
    for q in ps[1:]:
        if len(out) >= max_notes:
            break
        if ps[0] - q <= span:
            out.append(q)
    return out


# --- 5: range planning ---------------------------------------------------------------------------
def plan_ranges(bar_pitches: list[list[int]], shift_cost: float, nonprimary_cost: float,
                outside_cost: float, allowed=None) -> list[int]:
    """Viterbi over bars: one range per bar, minimising octave moves + shifts + non-primary bars.
    Empty bars cost nothing, so a range carries through rests until it's needed."""
    keys = list(allowed or RANGES)
    n = len(bar_pitches)
    if n == 0:
        return []
    cost = np.zeros((n, len(keys)))
    for b, ps in enumerate(bar_pitches):
        for j, r in enumerate(keys):
            lo, hi = RANGES[r]
            cost[b, j] = outside_cost * sum(1 for p in ps if not lo <= p <= hi)
            if ps and r not in PRIMARY:
                cost[b, j] += nonprimary_cost
    total = cost[0].copy()
    back = np.zeros((n, len(keys)), dtype=int)
    for b in range(1, n):
        trans = total[:, None] + shift_cost * (1 - np.eye(len(keys)))
        back[b] = np.argmin(trans, axis=0)
        total = trans[back[b], np.arange(len(keys))] + cost[b]
    path = [int(np.argmin(total))]
    for b in range(n - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    return [keys[j] for j in reversed(path)]


def place_shifts(chords: list[Chord], range_of_bar: list[int], bar_of, bar_ticks: int, lead_bars: float,
                 ppq: int) -> list[tuple[int, int]]:
    """``(tick, range_note)`` list: one at tick 0, then one per range change, about ``lead_bars``
    before the first note in the new range but never before a note the new range can't show."""
    if not chords:
        return [(0, range_of_bar[0] if range_of_bar else PRIMARY[0])]
    shifts = [(0, range_of_bar[bar_of(chords[0].tick)])]
    current = shifts[0][1]
    for k, c in enumerate(chords):
        r = range_of_bar[bar_of(c.tick)]
        if r == current:
            continue
        lo, hi = RANGES[r]
        earliest = shifts[-1][0] + ppq
        for prev in chords[:k]:
            if any(not lo <= p <= hi for p in prev.pitches):
                earliest = max(earliest, prev.tick + ppq // 4)
        at = max(earliest, c.tick - int(lead_bars * bar_ticks))
        at = min(c.tick, (at + ppq - 1) // ppq * ppq if at % ppq else at)  # on a beat, not after the note
        shifts.append((at, r))
        current = r
    return shifts


# --- 6-7: sustains and reductions ----------------------------------------------------------------
def apply_sustains(chords: list[Chord], ppq: int, min_beats: float, gap_simple: float, gap_other: float) -> None:
    for k, c in enumerate(chords):
        nxt = chords[k + 1] if k + 1 < len(chords) else None
        lengths = []
        for p, e in zip(c.pitches, c.ends):
            end = e
            if nxt is not None:
                simple = (len(c.pitches) == 1 and len(nxt.pitches) == 1) or p in nxt.pitches
                end = min(end, nxt.tick - int((gap_simple if simple else gap_other) * ppq))
            end = int(round(end / (ppq / 4)) * (ppq / 4))
            lengths.append(end - c.tick if end - c.tick >= min_beats * ppq else ppq // 8)
        c.lengths = lengths


def thin(chords: list[Chord], spacing_ticks: int, ppq: int) -> list[Chord]:
    """Keep chords at least ``spacing_ticks`` apart, preferring ones on strong beats."""
    if spacing_ticks <= 0:
        return list(chords)

    def strength(c):
        return 3 if c.tick % (2 * ppq) == 0 else 2 if c.tick % ppq == 0 else 1 if c.tick % (ppq // 2) == 0 else 0
    kept: list[Chord] = []
    for level in (3, 2, 1, 0):
        for c in chords:
            if strength(c) != level:
                continue
            if all(abs(c.tick - k.tick) >= spacing_ticks for k in kept):
                kept.append(c)
    return sorted(kept, key=lambda c: c.tick)


def reduce(expert: list[Chord], d: Difficulty, ppq: int) -> list[Chord]:
    lim = LIMITS[d]
    src = [Chord(c.tick, list(c.pitches), list(c.ends)) for c in expert]
    if d == Difficulty.HARD:
        # grace notes: a short single note right before the next one
        src = [c for k, c in enumerate(src)
               if not (len(c.pitches) == 1 and k + 1 < len(src) and src[k + 1].tick - c.tick <= ppq // 4
                       and max(c.ends) - c.tick <= ppq // 8)]
        out = []
        for c in src:  # thin fast 16ths to 8ths
            if c.tick % (ppq // 2) == 0 or not out or c.tick - out[-1].tick >= ppq // 2:
                out.append(c)
        src = out
    else:
        src = thin(src, int(lim["spacing"] * ppq), ppq)
    for c in src:
        c.pitches = limit_chord(c.pitches, lim["chord"], lim["span"])
        c.ends = [max(c.ends)] * len(c.pitches)
    # interval jumps between consecutive tops: move the later note an octave towards the earlier, or drop
    if lim["jump"]:
        out = []
        for c in src:
            if out and abs(c.top - out[-1].top) > lim["jump"]:
                shift = -12 if c.top > out[-1].top else 12
                moved = [p + shift for p in c.pitches]
                if abs(moved[0] - out[-1].top) <= lim["jump"] and all(KEY_LO <= p <= KEY_HI for p in moved):
                    c.pitches = moved
                else:
                    continue
            out.append(c)
        src = out
    return src


# --- the part --------------------------------------------------------------------------------------
@register
class ProKeys(PartCharter):
    key = "prokeys"
    track_names = tuple(instruments.PRO_KEYS.values())
    stem_keys = ("keys",)
    description = "Pro keys from the keys stem (Basic Pitch) or a MIDI: right hand, range shifts, 4 difficulties"

    def is_charted(self, track) -> bool:
        return bool(track.notes(range(KEY_LO, KEY_HI + 1)))

    def generate(self, ctx: PartContext) -> PartResult:
        o = {**DEFAULTS, **ctx.options}
        tm, ppq = ctx.tempo, ctx.chart.ppq
        notes, source = self._notes(ctx, o)
        if notes is None:
            return PartResult([], summary=source)
        notes = notes[(notes[:, 3] >= o["min_amp"]) & (notes[:, 1] - notes[:, 0] >= o["min_note_s"])]
        if len(notes) == 0:
            return PartResult([], summary=f"no notes from {source}")

        bpm = 60_000_000 / tm.tempos[-1].us_per_beat
        bar_s = 4 * 60 / bpm
        notes, n_left = right_hand(notes, o["cluster_s"], int(o["rh_span"]), o["register_bars"] * bar_s / 2)

        # grid
        tom = o["timing_offset_ms"]
        lean_at = feel_curve(tm, list(notes[:, 0])) if tom == "auto" else (lambda t, v=float(tom) / 1000: v)
        # one subdivision per bar (straight 16ths, or triplets where the bar as a whole is in triplets)
        cands = (o["grid"],) + ((o["triplets"],) if o["triplets"] else ())
        grids = bar_grids(tm, list(notes[:, 0]), lean_at, cands)
        bar_t = tm.bar_ticks(tm.time_sig_at(0))
        placed: dict[int, dict[int, int]] = {}
        for s, e, p, _ in notes:
            lean = lean_at(s)
            grid = grids.get(int(tm.s2t(s - lean) // bar_t), o["grid"])
            tick = snap(tm, s - lean, grid, None).tick
            end = max(tick + ppq // 8, tm.s2t(e - lean))
            placed.setdefault(tick, {})
            placed[tick][int(p)] = max(placed[tick].get(int(p), 0), end)

        # octave fit + chords (expert limits)
        shift = best_octave_shift([p for ch in placed.values() for p in ch])
        chords = []
        for tick in sorted(placed):
            ps = limit_chord([p + shift for p in placed[tick]], 4, 12)
            chords.append(Chord(tick, ps, [placed[tick][p - shift] for p in ps]))

        downbeats = tm.downbeats(chords[-1].tick + 8 * ppq)
        bar_ticks = tm.bar_ticks(tm.time_sig_at(0))

        def bar_of(tick):
            return max(0, bisect.bisect_right(downbeats, tick) - 1)

        review, rows = [], []
        tracks, by_diff, shifts_by = {}, {}, {}
        moved_total = 0
        fixed_range = None
        for d in reversed(Difficulty):  # expert first: the others reduce from it
            base = chords if d == Difficulty.EXPERT else reduce(by_diff[Difficulty.EXPERT], d, ppq)
            lim = LIMITS[d]
            n_bars = bar_of(base[-1].tick) + 1 if base else 1
            bar_pitches = [[] for _ in range(n_bars)]
            for c in base:
                bar_pitches[bar_of(c.tick)].extend(c.pitches)
            if lim["shifts"]:
                plan = plan_ranges(bar_pitches, o["shift_cost"], o["nonprimary_cost"], o["outside_cost"])
            else:  # one range for the whole part (Medium/Easy share it)
                if fixed_range is None:
                    allp = [p for ps in bar_pitches for p in ps]
                    fixed_range = min(PRIMARY, key=lambda r: sum(1 for p in allp
                                                                 if not RANGES[r][0] <= p <= RANGES[r][1]))
                plan = [fixed_range] * n_bars
            moved = 0
            for c in base:  # wrap into the bar's range
                lo, hi = RANGES[plan[bar_of(c.tick)]]
                new = []
                for p in c.pitches:
                    q = p if lo <= p <= hi else wrap_into(p, lo, hi)
                    moved += q != p
                    if q is not None and q not in new:
                        new.append(q)
                new = limit_chord(new, lim["chord"], lim["span"]) if new else new
                c.ends = [max(c.ends)] * len(new)
                c.pitches = new
            base = [c for c in base if c.pitches]
            apply_sustains(base, ppq, o["sustain_min_beats"], lim["gap_simple"], lim["gap_other"])
            shifts = (place_shifts(base, plan, bar_of, bar_ticks, o["shift_lead_bars"], ppq) if lim["shifts"]
                      else [(0, fixed_range)])
            by_diff[d], shifts_by[d] = base, shifts
            if d == Difficulty.EXPERT:
                moved_total = moved

        # write
        for d, name in instruments.PRO_KEYS.items():
            tr = ctx.existing(name)
            tr.remove_notes(list(range(0, 10)) + list(range(KEY_LO, KEY_HI + 1)))
            for tick, r in shifts_by[d]:
                tr.add_note(tick, ppq // 4, r)
            for c in by_diff[d]:
                for p, ln in zip(c.pitches, c.lengths):
                    tr.add_note(c.tick, ln, p)
            tracks[name] = tr
        out = list(tracks.values())
        if o["animation"]:
            self._animate(tracks[instruments.PRO_KEYS[Difficulty.EXPERT]], by_diff[Difficulty.EXPERT], ppq, bar_ticks)
            anim = ctx.existing(ANIM_RH)
            anim.remove_notes(range(KEY_LO, KEY_HI + 1))
            for c in by_diff[Difficulty.EXPERT]:
                for p, ln in zip(c.pitches, c.lengths):
                    anim.add_note(c.tick, ln, p)
            out.append(anim)

        for c in by_diff[Difficulty.EXPERT]:
            rows.append({"time_s": round(tm.t2s(c.tick), 3), "tick": c.tick, "notes": " ".join(rb_name(p) for p in c.pitches),
                         "range": next(rb_name(RANGES[r][0]) + "-" + rb_name(RANGES[r][1])
                                       for t, r in reversed(shifts_by[Difficulty.EXPERT]) if t <= c.tick),
                         "sustain_beats": round(max(c.lengths) / ppq, 2) if max(c.lengths) > ppq // 8 else 0})
        counts = " ".join(f"{d.name[0]}{len(by_diff[d])}" for d in reversed(Difficulty))
        n_shifts = len(shifts_by[Difficulty.EXPERT]) - 1
        if n_shifts > 12:
            review.append(ReviewItem(0, self.key, f"{n_shifts} range shifts on Expert; consider a higher shift_cost", "info"))
        summary = (f"{source}: {counts} chords | {n_left} left-hand notes dropped | octave {shift:+d} | "
                   f"{n_shifts} range shifts (X), medium/easy in {rb_name(RANGES[fixed_range][0])}-"
                   f"{rb_name(RANGES[fixed_range][1])} | {moved_total} notes moved an octave to fit (X)")
        return PartResult(out, review, rows, summary)

    def _notes(self, ctx: PartContext, o) -> tuple[np.ndarray | None, str]:
        from ..audio.transcribe import basic_pitch_notes, midi_notes
        offset = np.array([ctx.audio_offset_s, ctx.audio_offset_s, 0, 0])
        if o.get("midi"):
            from pathlib import Path
            p = Path(o["midi"])
            if not p.is_absolute() and ctx.root is not None:
                p = ctx.root / p
            if not p.exists():
                return None, f"skipped: midi file {p} not found"
            return midi_notes(p, o.get("midi_tracks")) + offset, f"MIDI {p.name}"
        stem = ctx.stem(*self.stem_keys)
        if stem is None:
            return None, f"skipped: no stem among {self.stem_keys} and no midi option"
        n = basic_pitch_notes(stem, ctx.cache_dir, float(o["onset_threshold"]), float(o["frame_threshold"]),
                              float(o["min_note_s"]) * 1000)
        return n + offset, "Basic Pitch"

    @staticmethod
    def _animate(track: Track, chords: list[Chord], ppq: int, bar_ticks: int) -> None:
        """[idle] before the part starts and in rests of two bars or more, [play] an 8th before
        notes come back, [idle_realtime] at the end."""
        track.remove(lambda t, m: m.type == "text" and m.text.strip() in ANIM_EVENTS)
        if not chords:
            return
        eighth = ppq // 2
        track.add_text(0, "[idle]")
        track.add_text(max(0, chords[0].tick - eighth), "[play]")
        for a, b in zip(chords, chords[1:]):
            end = a.tick + max(a.lengths)
            if b.tick - end >= 2 * bar_ticks:
                track.add_text(end + eighth, "[idle]")
                track.add_text(b.tick - eighth, "[play]")
        last = chords[-1]
        track.add_text(last.tick + max(last.lengths) + eighth, "[idle_realtime]")
