"""Polyphonic note transcription (keys): Spotify Basic Pitch on a stem, or notes read from a MIDI file.

Both give the same thing: an array of notes ``(start_s, end_s, midi_pitch, amplitude 0-1)``.
Basic Pitch needs the ``keys`` extra (Python 3.11); results are cached like the other analyses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import _cached

NOTE_FIELDS = ("start", "end", "pitch", "amp")


def basic_pitch_notes(path: str | Path, cache_dir: Path | None = None, onset_threshold: float = 0.5,
                      frame_threshold: float = 0.3, min_note_ms: float = 60.0,
                      fmin: float | None = None, fmax: float | None = None) -> np.ndarray:
    """Notes from an audio stem via Basic Pitch -> ``(n, 4)`` array of start, end, pitch, amplitude."""
    def compute():
        try:
            import logging
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as e:
            raise RuntimeError("keys transcription needs the keys extra on Python 3.11: "
                               "pip install -e .[audio,keys]") from e
        _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=onset_threshold,
                               frame_threshold=frame_threshold, minimum_note_length=min_note_ms,
                               minimum_frequency=fmin, maximum_frequency=fmax, melodia_trick=True)
        notes = np.array([(s, e, p, a) for s, e, p, a, *_ in events], dtype=float).reshape(-1, 4)
        return {"notes": notes[np.argsort(notes[:, 0], kind="stable")] if len(notes) else notes}

    return _cached(cache_dir, "basicpitch", path, compute, onset=onset_threshold, frame=frame_threshold,
                   minlen=min_note_ms, fmin=fmin, fmax=fmax, v=1)["notes"]


def midi_notes(path: str | Path, tracks: list[str] | None = None) -> np.ndarray:
    """Notes from a MIDI file in seconds, using its own tempo map. Reads every track, including the
    first (single-track / type 0 files keep their notes there), or only ``tracks`` by name."""
    import mido

    from ..core.chart import Track
    from ..core.timing import TempoChange, TempoMap
    mf = mido.MidiFile(str(path))
    tempos, parsed = [], []
    for i, tr in enumerate(mf.tracks):
        tick, events, name = 0, [], f"TRACK {i}"
        for m in tr:
            tick += m.time
            if m.type == "set_tempo":
                tempos.append(TempoChange(tick, m.tempo))
            elif m.type == "track_name":
                name = m.name
            elif m.type in ("note_on", "note_off"):
                events.append((tick, m))
        parsed.append((name, Track(name, events)))
    tm = TempoMap(tempos, ppq=mf.ticks_per_beat)
    rows = []
    for name, tr in parsed:
        if tracks and name not in tracks:
            continue
        for n in tr.notes():
            if n.channel == 9:  # General MIDI drums
                continue
            rows.append((tm.t2s(n.tick), tm.t2s(n.end), n.pitch, n.velocity / 127))
    notes = np.array(rows, dtype=float).reshape(-1, 4)
    return notes[np.argsort(notes[:, 0], kind="stable")] if len(notes) else notes
