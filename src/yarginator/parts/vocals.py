"""Vocal pitches from a stem, for lyrics that are already timed in the chart.

- Keeps lyrics, phrase markers, percussion and anything else in the track; only replaces notes in
  the vocal range.
- One note per lyric: starts on the lyric tick, ends where the stem's energy falls off (never
  touching the next syllable).
- Pitch = pyin when it's confident; otherwise a harmonic-sum (CQT) estimate, which copes better
  with distorted/processed vocals. CQT pitches are octave-corrected against their neighbours.
"""
from __future__ import annotations

import numpy as np

from ..audio.analysis import LOW_MIDI, VocalFeatures, vocal_features
from ..core import instruments
from ..feedback import ReviewItem
from .base import PartCharter, PartContext, PartResult, register

LOW, HIGH = instruments.VOCAL_RANGE


def estimate(f: VocalFeatures, s: float, nxt: float, voiced_threshold: float = 0.5):
    """-> (pitch | None, source, confidence 0-1, end_seconds)"""
    lo, hi = s - 0.03, min(max(nxt - 0.03, s + 0.12), s + 0.8)
    i, j = np.searchsorted(f.t, lo), np.searchsorted(f.t, hi)
    if j - i < 3:
        return None, "none", 0.0, s + 0.1
    seg_db = f.db[i:j]
    loud = np.where(seg_db >= seg_db.max() - 20)[0]
    end = f.t[min(i + (loud.max() if len(loud) else 0) + 1, len(f.t) - 1)]
    good = (f.voiced[i:j] > voiced_threshold) & np.isfinite(f.midi[i:j])
    if good.sum() >= 4:
        return float(np.median(f.midi[i:j][good])), "pyin", 1.0, end
    prof = f.harmonic[:, i:j].mean(axis=1)
    semi = np.array([prof[max(0, b * 3 - 1):b * 3 + 2].max() for b in range(len(prof) // 3)])
    prom = semi.max() - np.median(semi)
    return float(LOW_MIDI + int(np.argmax(semi))), "cqt", float(min(1.0, prom / 5.0)), end


def vocal_lyrics(track) -> list[tuple[int, str]]:
    """Lyric events, including the plain text events many older / Magma-built charts use for lyrics
    (bracketed text like ``[idle]`` is a marker, not a syllable)."""
    return [(t, m.text) for t, m in track.sorted_events()
            if m.type == "lyrics" or (m.type == "text" and m.text.strip() and not m.text.lstrip().startswith("["))]


def assign_pitches(est: list) -> list[int]:
    pitch = [None if e[0] is None else int(round(e[0])) for e in est]
    # octave-correct CQT-sourced pitches toward the local median of neighbours
    for i, e in enumerate(est):
        if e[1] != "cqt" or pitch[i] is None:
            continue
        nb = [pitch[j] for j in range(max(0, i - 5), min(len(pitch), i + 6)) if j != i and pitch[j] is not None]
        if nb:
            med = float(np.median(nb))
            for sh in (-12, 12):
                if abs(pitch[i] - med) >= 9 and abs(pitch[i] + sh - med) < abs(pitch[i] - med):
                    pitch[i] += sh
                    break
    for i, p in enumerate(pitch):
        if p is None:
            pitch[i] = next((pitch[j] for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(pitch)))
                             if pitch[j] is not None), 60)
        while pitch[i] > HIGH:
            pitch[i] -= 12
        while pitch[i] < LOW:
            pitch[i] += 12
    return pitch


@register
class VocalPitchCharter(PartCharter):
    key = "vocals"
    track_names = (instruments.VOCALS,)
    stem_keys = ("vocals",)
    description = "Pitches for already-timed lyrics, from the vocal stem (pyin + harmonic-sum fallback)"

    def is_charted(self, track) -> bool:
        return bool(track.notes(range(LOW, HIGH + 1)))  # phrase markers (105/106) alone don't count

    def generate(self, ctx: PartContext) -> PartResult:
        name = self.track_names[0]
        track = ctx.existing(name)
        lyr = vocal_lyrics(track)
        if not lyr:
            return PartResult([], summary=f"skipped: {name} has no timed lyrics yet")
        stem = ctx.stem(*self.stem_keys)
        if stem is None:
            return PartResult([], summary=f"skipped: no stem among {self.stem_keys}")

        min_conf = float(ctx.options.get("min_confidence", 0.6))
        feats = vocal_features(stem, ctx.cache_dir)
        feats.t = feats.t + ctx.audio_offset_s
        tm = ctx.tempo
        secs = [tm.t2s(k) for k, _ in lyr]
        est = [estimate(feats, s, secs[i + 1] if i + 1 < len(secs) else s + 1.0,
                        float(ctx.options.get("voiced_threshold", 0.5)))
               for i, s in enumerate(secs)]
        pitch = assign_pitches(est)

        track.remove_notes(range(LOW, HIGH + 1))
        min_len = ctx.chart.ppq // 4
        for i, (k, _) in enumerate(lyr):
            end = max(tm.s2t(est[i][3]), k + min_len)
            if i + 1 < len(lyr):
                end = min(end, lyr[i + 1][0] - 10)
            track.add_note(k, max(end, k + 30) - k, pitch[i])

        rows, review = [], []
        for i, ((_, txt), e) in enumerate(zip(lyr, est)):
            flagged = e[2] < min_conf
            rows.append({"n": i + 1, "time_s": round(secs[i], 2), "lyric": txt, "midi_pitch": pitch[i],
                         "source": e[1], "confidence": round(e[2], 2), "review": "yes" if flagged else ""})
            if flagged:
                review.append(ReviewItem(secs[i], self.key, f"'{txt}' pitch {pitch[i]} ({e[1]}, conf {e[2]:.2f})"))
        src = [e[1] for e in est]
        summary = (f"{len(lyr)} notes | pyin {src.count('pyin')}, cqt {src.count('cqt')}, "
                   f"none {src.count('none')} | {len(review)} flagged")
        return PartResult([track], review, rows, summary)


def _harmony(n: int, stems: tuple[str, ...]) -> type[VocalPitchCharter]:
    cls = type(f"Harmony{n}Charter", (VocalPitchCharter,), {
        "key": f"harm{n}", "track_names": (instruments.HARM[n],), "stem_keys": stems,
        "description": f"Pitches for already-timed {instruments.HARM[n]} lyrics",
    })
    return register(cls)


Harmony1Charter = _harmony(1, ("harm1", "vocals"))
Harmony2Charter = _harmony(2, ("harm2",))
Harmony3Charter = _harmony(3, ("harm3", "harm2"))
