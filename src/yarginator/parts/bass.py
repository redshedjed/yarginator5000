"""Bass (5-lane) from the bass stem.

    [parts.bass]
    ghosts = "expert"            # ghost/dead notes: on Expert only (default), "all", or "none"
    grid = "16th"                # straight grid; triplets used only where they fit clearly better
    triplets = "8th-triplet"     # or "16th-triplet", or "" to never use triplets
    sustain_min_beats = 0.75
    timing_offset_ms = "auto"    # measure how far ahead/behind the beat the part sits and correct for
                                 # it before snapping; or a fixed number (+ = behind), or 0 to disable
    lowest_note = "auto"         # any string count / tuning: "auto" reads the range from the stem;
                                 # or a note: "E1" 4-string, "D1" drop D, "B0" 5/6-string, "A0" drop A
    highest_note = "auto"        # top of the pitch search: auto = G4, widened to C5 if the stem plays
                                 # up there (6-string high register); or a note / MIDI number
    # plus any note_events.DEFAULTS / five_lane.DEFAULTS key

Kick-drum bleed in a separated bass stem is detected against the drum stem and dropped. Notes far
off the grid, notes that collide when snapped, and uncertain octave jumps go to the review list.
"""
from __future__ import annotations

import numpy as np

from ..audio.analysis import onsets, pitch_features
from ..core import instruments
from ..core.instruments import Difficulty
from ..feedback import ReviewItem
from . import five_lane as fl
from . import note_events as ne
from .base import PartCharter, PartContext, PartResult, register
from .quantize import feel_curve, snap


_NAMES = "C C# D D# E F F# G G# A A# B".split()


def to_midi(value) -> int | None:
    """``"B0"`` / ``"Bb0"`` / ``23`` -> 23; ``"auto"`` / None -> None."""
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = value.strip().replace("♯", "#").replace("♭", "b")
    letter, rest = s[0].upper(), s[1:]
    acc = 1 if rest[:1] == "#" else -1 if rest[:1] == "b" else 0
    octave = int(rest[1:] if acc else rest)
    return 12 * (octave + 1) + _NAMES.index(letter) + acc


def midi_name(m: float) -> str:
    m = int(round(m))
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


@register
class Bass(PartCharter):
    key = "bass"
    track_names = (instruments.BASS,)
    stem_keys = ("bass",)
    description = "5-lane bass from the bass stem: transcribe, snap to grid, frets by contour, 4 difficulties"
    # Default pitch search B0-G4 covers 4/5/6-string basses in standard tuning and drop D/C#/C. It is
    # widened (A0 / C5) only when the stem piles up confident notes at an edge, because a wider search
    # invites subharmonic errors: from A0, YIN reads an A#1-F2 groove as A#0 (the note whose harmonics
    # contain both) and the fifth disappears. The 93 ms window fits two periods of A0 but doesn't smear
    # fast runs (a 186 ms window spans a whole 16th-note lick).
    search = ("B0", "G4")
    widened = ("A0", "C5")
    frame_length = 2048

    def is_charted(self, track) -> bool:
        base = instruments.FIVE_LANE_BASE[Difficulty.EXPERT]
        return bool(track.notes(range(base, base + 5)))

    def generate(self, ctx: PartContext) -> PartResult:
        stem = ctx.stem(*self.stem_keys)
        if stem is None:
            return PartResult([], summary=f"skipped: no stem among {self.stem_keys}")
        o = {**ne.DEFAULTS, **fl.DEFAULTS, **ctx.options}
        tm, ppq = ctx.tempo, ctx.chart.ppq
        off = ctx.audio_offset_s

        lowest = to_midi(o.get("lowest_note", "auto"))
        highest = to_midi(o.get("highest_note", "auto"))
        lo = min(lowest - 2, to_midi(self.search[0])) if lowest is not None else to_midi(self.search[0])
        hi = highest if highest is not None else to_midi(self.search[1])
        feats = pitch_features(stem, ctx.cache_dir, midi_name(lo), midi_name(hi), frame_length=self.frame_length)
        # auto: widen an edge only if the stem piles up confident notes against it
        new_lo = to_midi(self.widened[0]) if lowest is None and ne.crowds_edge(feats, lo, "low") else lo
        new_hi = to_midi(self.widened[1]) if highest is None and ne.crowds_edge(feats, hi, "high") else hi
        if (new_lo, new_hi) != (lo, hi):
            lo, hi = new_lo, new_hi
            feats = pitch_features(stem, ctx.cache_dir, midi_name(lo), midi_name(hi), frame_length=self.frame_length)
        feats.t = feats.t + off
        if lowest is None:
            lowest = ne.estimate_lowest(feats)
        o["lowest_note"], o["search_ceiling"] = lowest, hi
        # standard attacks (trusted timing) + sensitive low-end candidates (only with evidence)
        attacks = onsets(stem, ctx.cache_dir, backtrack=True, hop=64) + off
        extra = onsets(stem, ctx.cache_dir, backtrack=True, hop=64, fmax=float(o.get("attack_fmax", 1500)),
                       delta=float(o.get("attack_delta", 0.03)), wait_s=0.04) + off
        events = ne.transcribe(feats, attacks, extra, **{k: o[k] for k in ne.DEFAULTS})
        drums = ctx.stem("drums")
        n_bleed = ne.mark_bleed(events, onsets(drums, ctx.cache_dir, backtrack=True, hop=64) + off,
                                o["bleed_window_s"], o["bleed_max_db"]) if drums is not None else 0
        events = [e for e in events if e.kind != ne.BLEED]
        if not events:
            return PartResult([], summary="no bass notes found")

        # snap to the grid, after taking out the part's consistent lean (laid-back feel, stem smear);
        # when two notes land on one tick keep the stronger (pitched > ghost, louder)
        tom = o.get("timing_offset_ms", "auto")
        if tom == "auto":
            lean_at = feel_curve(tm, [e.start for e in events if e.kind == ne.PITCHED],
                                 float(o.get("feel_window_bars", 4)))
        else:
            lean_at = lambda t, v=float(tom) / 1000: v  # noqa: E731
        review, rows, placed = [], [], {}
        flagged_bars: set[int] = set()
        for e in events:
            lean = lean_at(e.start)
            bar = tm.s2t(e.start) // (4 * ppq)
            if abs(lean) >= 0.06 and not any(abs(bar - fb) <= 8 for fb in flagged_bars):  # once per stretch
                flagged_bars.add(bar)
                review.append(ReviewItem(e.start, self.key, f"bass sits {lean * 1000:+.0f} ms off the grid here "
                                         "(recording latency?); notes were shifted before snapping", "info"))
            s = snap(tm, e.start - lean, o.get("grid", "16th"), o.get("triplets", "8th-triplet") or None)
            if s.error_ms > 45:
                review.append(ReviewItem(e.start, self.key, f"note {s.error_ms:.0f} ms off the {s.grid} grid"))
            prev = placed.get(s.tick)
            if prev is not None:
                if (prev[0].kind == ne.PITCHED, prev[0].level) >= (e.kind == ne.PITCHED, e.level):
                    continue
                review.append(ReviewItem(e.start, self.key, "two notes snapped to one grid line; kept the stronger"))
            placed[s.tick] = (e, s)
        ordered = [placed[t] for t in sorted(placed)]
        ticks = [s.tick for _, s in ordered]
        pitches = [e.pitch for e, _ in ordered]
        lanes = fl.assign_lanes(ticks, pitches, ppq, float(o["lane_window_beats"]))
        # ghost notes live between the beats; an unpitched hit right on a beat is more likely a real
        # note the pitch tracker missed, so it counts as a note (and shows on lower difficulties)
        for e, s in ordered:
            if e.kind == ne.GHOST and s.tick % ppq == 0 and s.error_ms <= 25:
                e.kind = ne.PITCHED
                e.notes.append("unpitched on the beat")
        notes = [fl.LaneNote(s.tick, max(s.tick + 1, tm.s2t(e.end)), lane, e.pitch, e.kind == ne.GHOST)
                 for (e, s), lane in zip(ordered, lanes)]
        notes = fl.add_sustains(notes, ppq, float(o["sustain_min_beats"]), float(o["sustain_gap_beats"]))
        by_diff = {d: fl.reduce(notes, d, ppq, o["ghosts"], float(o["lane_window_beats"])) for d in Difficulty}

        # octave jumps of exactly 12 between neighbours are often pitch-tracker errors: flag them
        for (a, _), (b, _) in zip(ordered, ordered[1:]):
            if a.pitch is not None and b.pitch is not None and abs(round(b.pitch) - round(a.pitch)) == 12:
                review.append(ReviewItem(b.start, self.key, "octave jump: real, or a pitch-tracking slip?", "info"))

        track = fl.write(ctx.existing(instruments.BASS), by_diff, ppq)
        names = "G R Y B O".split()
        on_ticks = {d: {x.tick for x in by_diff[d]} for d in Difficulty}
        for (e, s), n in zip(ordered, notes):
            rows.append({"time_s": round(e.start, 3), "tick": s.tick, "grid": s.grid, "grid_error_ms": round(s.error_ms, 1),
                         "kind": e.kind, "midi_pitch": round(e.pitch, 1) if e.pitch is not None else "",
                         "fret": names[n.lane], "sustain_beats": round(n.sustain / ppq, 2),
                         "on": "".join(d.name[0] for d in reversed(Difficulty) if n.tick in on_ticks[d]),
                         "notes": " ".join(e.notes)})
        counts = " ".join(f"{d.name[0]}{len(by_diff[d])}" for d in reversed(Difficulty))
        ghosts = sum(1 for n in notes if n.ghost)
        per_bar = getattr(lean_at, "per_bar", None)
        feel = (f"feel {min(per_bar) * 1000:+.0f}..{max(per_bar) * 1000:+.0f} ms" if per_bar
                else f"feel {lean_at(0) * 1000:+.0f} ms")
        played = [round(e.pitch) for e, _ in ordered if e.pitch is not None]
        rng = (f"range {midi_name(min(played))}-{midi_name(max(played))} (floor {midi_name(round(lowest))}, "
               f"search {midi_name(lo)}-{midi_name(hi)})" if played else "")
        summary = (f"{len(notes)} notes ({ghosts} ghost) | {counts} | {sum(1 for n in notes if n.sustain)} sustains"
                   f" | {rng} | {feel} | {n_bleed} bleed dropped | {len(review)} flagged")
        return PartResult([track], review, rows, summary)
