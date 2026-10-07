"""Vocals from scratch: find the vocal events in the stem (see vocal_events), turn sung ones into
pitched notes and unpitched ones (growls, screams, spoken) into "#" notes, drop noise (coughs,
breaths), give each note a lyric from song.toml, and group them into phrases.

    [parts.vocals]
    generator = "vocals-scratch"
    lyrics = "Do*4 Re*3 Mi*6"      # syllables in order; "word*N" repeats; split words like "Hel- lo"
    lyric_mode = "fit"             # fit:   detect every possible syllable boundary, then undo the
                                   #        weakest ones / drop the least convincing events until the
                                   #        note count matches the syllables; every change is flagged
                                   # cycle: wrap the syllables around
                                   # once:  extra notes get "#" placeholders
    unpitched = "keep"             # keep growls/screams/spoken as "#" notes; "drop" for clean singing
    note_start = "vowel"           # notes start where the pitch sounds; "attack" = start of the syllable
    start_delay_ms = 0             # extra nudge later for every note
    phrase_gap_beats = 1.5         # a rest at least this long starts a new phrase
    # detection / classification thresholds: see vocal_events.DEFAULTS (any of them can go here)

Placing timed notes is the slow part of charting vocals by hand; the lyric text is quick to fix in
REAPER. Fit decisions and unpitched notes are flagged in reports/review.csv (and as REAPER markers);
reports/vocals.csv lists every detected event, including the dropped ones and why.
"""
from __future__ import annotations

import re

import numpy as np

from ..audio.analysis import onsets, vocal_features
from ..core import instruments
from ..feedback import ReviewItem
from . import vocal_events as ve
from .base import PartCharter, PartContext, PartResult, register
from .vocals import HIGH, LOW, VocalPitchCharter

_REPEAT = re.compile(r"^(.+?)\*(\d+)$")


def syllables(text: str) -> list[str]:
    """``"Do*2 Re"`` -> ``["Do", "Do", "Re"]``"""
    out = []
    for w in text.split():
        m = _REPEAT.match(w)
        out += [m.group(1)] * int(m.group(2)) if m else [w]
    return out


def snap_attacks(events: list[ve.VocalEvent], attacks, window: float) -> None:
    """Pitch tracking calls a note voiced only once it's steady, which is late on soft attacks; pull
    each start back to the latest attack within ``window`` before it (never into the previous event).
    Events that already start at an attached consonant are left alone."""
    prev_end = float("-inf")
    for e in events:
        if "consonant" not in e.notes:
            cand = attacks[(attacks >= max(e.start - window, prev_end)) & (attacks <= e.start + 0.02)]
            if len(cand):
                e.start = float(min(cand[-1], e.start))
        prev_end = e.end


MIN_NOTE_S = 0.06


def vowel_onset(f, start: float, end: float) -> float:
    """First confidently pitched moment of a note (its vowel), never leaving it shorter than
    MIN_NOTE_S; the original start if the note never settles on a pitch."""
    i, j = np.searchsorted(f.t, start), np.searchsorted(f.t, end - MIN_NOTE_S)
    v = np.nonzero((f.voiced[i:j] > 0.5) & np.isfinite(f.midi[i:j]))[0]
    return float(f.t[i + v[0]]) if len(v) else start


def _clamp(pitch: float) -> int:
    p = int(round(pitch))
    while p > HIGH:
        p -= 12
    while p < LOW:
        p += 12
    return p


@register
class VocalScratchCharter(PartCharter):
    key = "vocals-scratch"
    track_names = (instruments.VOCALS,)
    stem_keys = ("vocals",)
    description = "Vocal notes + lyrics + phrases from the stem and song.toml lyric text; handles growls and noise"

    def is_charted(self, track) -> bool:
        return VocalPitchCharter.is_charted(self, track) or bool(track.lyrics())

    def generate(self, ctx: PartContext) -> PartResult:
        name = self.track_names[0]
        stem = ctx.stem(*self.stem_keys)
        if stem is None:
            return PartResult([], summary=f"skipped: no stem among {self.stem_keys}")
        opts = ctx.options
        words = syllables(str(opts.get("lyrics", "")))
        mode = opts.get("lyric_mode", "fit" if words else "once")
        feats = vocal_features(stem, ctx.cache_dir)
        feats.t = feats.t + ctx.audio_offset_s
        event_opts = {k: opts[k] for k in ve.DEFAULTS if k in opts}
        if words and mode == "fit":
            # over-split on purpose (every re-attack is a candidate); fit_count then undoes the weakest
            # splits until the notes match the lyric count
            event_opts.setdefault("split_dip_db", float(opts.get("fit_split_dip_db", 0.0)))
        events = ve.detect(feats, **event_opts)
        snap_attacks(events, onsets(stem, ctx.cache_dir, backtrack=True, hop=64) + ctx.audio_offset_s,
                     float(opts.get("attack_window_s", 0.15)))
        noise = [e for e in events if e.kind == ve.NOISE]
        kept = [e for e in events if e.kind != ve.NOISE]
        if not kept:
            return PartResult([], summary="no vocal events found")

        review, fit_log = [], []
        if words and mode == "fit":
            if len(kept) > len(words):
                kept, fit_log = ve.fit_count(kept, len(words))
                for line in fit_log:
                    t = float(re.search(r"(\d+\.\d+)s", line).group(1))
                    review.append(ReviewItem(t, self.key, f"fit: {line}"))
            elif len(kept) < len(words):
                review.append(ReviewItem(kept[-1].end, self.key,
                                         f"only {len(kept)} notes for {len(words)} syllables; last "
                                         f"{len(words) - len(kept)} syllable(s) unplaced", "error"))

        # Where each note starts. Detection finds the start of the syllable (consonant / attack), but a
        # sung note is heard at its vowel, so by default notes start where the pitch actually sounds.
        # start_delay_ms nudges every note later on top of that.
        start_mode = opts.get("note_start", "vowel")
        delay = float(opts.get("start_delay_ms", 0)) / 1000
        for e in kept:
            if start_mode == "vowel" and e.kind == ve.SUNG:
                e.start = vowel_onset(feats, e.start, e.end)
            e.start = min(e.start + delay, e.end - MIN_NOTE_S)

        tm, ppq = ctx.tempo, ctx.chart.ppq
        track = ctx.existing(name)
        track.remove_notes(list(range(LOW, HIGH + 1)) + [instruments.VOCAL_PHRASE, instruments.VOCAL_PHRASE_2])
        track.remove(lambda t, m: m.type == "lyrics" or (m.type == "text" and not m.text.startswith("[")))

        ticks = [(tm.s2t(e.start), tm.s2t(e.end)) for e in kept]
        sung_pitches = [e.pitch for e in kept if e.kind == ve.SUNG and e.pitch is not None]
        last_pitch = sung_pitches[0] if sung_pitches else 60.0
        rows = []
        for k, (e, (a, b)) in enumerate(zip(kept, ticks)):
            if k + 1 < len(ticks):
                b = min(b, ticks[k + 1][0] - ppq // 32)
            if e.kind == ve.SUNG and e.pitch is not None:
                last_pitch = e.pitch
            pitch = _clamp(e.pitch if e.kind == ve.SUNG and e.pitch is not None else last_pitch)
            if words and (mode in ("fit", "once") and k < len(words) or mode == "cycle"):
                lyric = words[k % len(words)]
            else:
                lyric = "#" if not words else "?"
            if e.kind == ve.UNPITCHED:
                lyric += "#"  # Rock Band: unpitched (talky) note
                review.append(ReviewItem(e.start, self.key, f"unpitched '{lyric}' ({e.why})", "info"))
            elif e.spread > 1.0:
                review.append(ReviewItem(e.start, self.key, f"'{lyric}' pitch {pitch} wobbles {e.spread:.1f} st"))
            track.add_note(a, max(b - a, ppq // 16), pitch)
            track.add_lyric(a, lyric)
            rows.append(self._row(e, lyric, pitch))
        rows += [self._row(e, "", None) for e in noise]
        rows.sort(key=lambda r: r["time_s"])

        # phrases: split at rests of phrase_gap_beats or more
        gap = float(opts.get("phrase_gap_beats", 1.5)) * ppq
        phrases = [[0]]
        for k in range(1, len(ticks)):
            if ticks[k][0] - ticks[k - 1][1] >= gap:
                phrases.append([])
            phrases[-1].append(k)
        for ph in phrases:
            a = max(0, ticks[ph[0]][0] - ppq // 16)
            track.add_note(a, ticks[ph[-1]][1] + ppq // 16 - a, instruments.VOCAL_PHRASE)

        kinds = {kd: sum(1 for e in kept if e.kind == kd) for kd in (ve.SUNG, ve.UNPITCHED)}
        summary = (f"{len(kept)} notes ({kinds[ve.SUNG]} sung, {kinds[ve.UNPITCHED]} unpitched) in {len(phrases)} "
                   f"phrases | {len(noise)} noise dropped | fit: {len(fit_log)} changes | {len(review)} flagged")
        return PartResult([track], review, rows, summary)

    @staticmethod
    def _row(e: ve.VocalEvent, lyric: str, pitch: int | None) -> dict:
        return {"time_s": round(e.start, 3), "length_s": round(e.length, 3), "kind": e.kind,
                "lyric": lyric, "midi_pitch": pitch if pitch is not None else "",
                "voiced": round(e.voiced, 2), "level_db": round(e.level, 1), "spread": round(e.spread, 2),
                "flatness": round(e.flatness, 3), "why": e.why, "notes": " ".join(e.notes),
                "status": "dropped" if e.kind == ve.NOISE else "note"}


def _harmony_scratch(n: int, stems: tuple[str, ...]) -> type[VocalScratchCharter]:
    """``harm<n>-scratch``: the same from-scratch charting into HARM<n>, e.g. for harmonies-only songs:
    [parts.harm1] generator = "harm1-scratch", lyrics = "..."."""
    return register(type(f"Harmony{n}ScratchCharter", (VocalScratchCharter,), {
        "key": f"harm{n}-scratch", "track_names": (instruments.HARM[n],), "stem_keys": stems,
        "description": f"{instruments.HARM[n]} notes + lyrics + phrases from a stem and lyric text",
    }))


Harmony1ScratchCharter = _harmony_scratch(1, ("harm1", "vocals"))
Harmony2ScratchCharter = _harmony_scratch(2, ("harm2",))
Harmony3ScratchCharter = _harmony_scratch(3, ("harm3", "harm2"))
