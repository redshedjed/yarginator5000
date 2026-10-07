"""Cached feature extraction from stems.

Each function returns plain numpy arrays and caches them under the song's ``.cache/`` folder,
keyed on the file's path, size, mtime and the analysis parameters, so iterating on a generator
doesn't re-run pyin every time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import require_librosa

SR = 22050
HOP = 256
BPO = 36                               # CQT bins per octave (3 per semitone)
HARMONIC_WEIGHTS = (1, 0.8, 0.64, 0.5, 0.4)
LOW_MIDI = 36                          # C2, bottom of the harmonic-sum profile


def _cache_file(cache_dir: Path | None, kind: str, path: Path, **params) -> Path | None:
    if cache_dir is None:
        return None
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{sorted(params.items())}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}-{path.stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"


def _cached(cache_dir, kind, path, compute, **params) -> dict[str, np.ndarray]:
    f = _cache_file(cache_dir, kind, Path(path), **params)
    if f and f.exists():
        with np.load(f) as z:
            return dict(z)
    data = compute()
    if f:
        np.savez_compressed(f, **data)
    return data


@dataclass
class VocalFeatures:
    t: np.ndarray          # frame times (s)
    midi: np.ndarray       # pyin f0 as fractional MIDI (a best guess even when unvoiced; check `voiced`)
    voiced: np.ndarray     # pyin voiced probability
    db: np.ndarray         # RMS level (dB)
    harmonic: np.ndarray   # harmonic-sum CQT salience, (3 bins/semitone from LOW_MIDI) x frames
    flatness: np.ndarray | None = None  # spectral flatness 0..1: ~0 tonal (singing), high = noise (cough, breath, growl)
    onset: np.ndarray | None = None     # onset strength (attacks), same frames
    db_fine: np.ndarray | None = None   # RMS level over a short (23 ms) window


def vocal_texture(path: str | Path, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Frame-aligned (HOP) spectral flatness and onset strength; cached separately from pyin."""
    def compute():
        librosa = require_librosa()
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=HOP)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        # short-window level (23 ms): resolves the brief closure of a re-articulated consonant
        # ("do-do") that the 93 ms window in `db` smooths away
        fine = 20 * np.log10(librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0] + 1e-6)
        return {"flatness": flat, "onset": onset, "db_fine": fine}

    return _cached(cache_dir, "voxtex", path, compute, sr=SR, hop=HOP, v=2)


def vocal_features(path: str | Path, cache_dir: Path | None = None) -> VocalFeatures:
    def compute():
        librosa = require_librosa()
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        f0, _, vp = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C5"), sr=sr,
                                 frame_length=2048, hop_length=HOP, fill_na=None)
        t = librosa.times_like(f0, sr=sr, hop_length=HOP)
        db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0] + 1e-6)
        C = np.log1p(30 * np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
                                              n_bins=BPO * 6, bins_per_octave=BPO)))
        nf = BPO * 3
        S = np.zeros((nf, C.shape[1]))
        for h, w in enumerate(HARMONIC_WEIGHTS, 1):
            off = int(round(BPO * np.log2(h)))
            S += w * C[off:off + nf]
        S -= np.median(S, axis=0, keepdims=True)
        n = min(len(t), len(db), S.shape[1])
        return {"t": t[:n], "midi": librosa.hz_to_midi(f0[:n]), "voiced": vp[:n], "db": db[:n], "harmonic": S[:, :n]}

    d = _cached(cache_dir, "vocal", path, compute, sr=SR, hop=HOP, bpo=BPO)
    tex = vocal_texture(path, cache_dir)
    n = len(d["t"])

    def fit(a):
        return np.pad(a[:n], (0, max(0, n - len(a))), mode="edge")
    return VocalFeatures(d["t"], d["midi"], d["voiced"], d["db"], d["harmonic"], fit(tex["flatness"]), fit(tex["onset"]),
                         fit(tex["db_fine"]))


@dataclass
class PitchFeatures:
    t: np.ndarray        # frame times (s)
    midi: np.ndarray     # pyin f0 as fractional MIDI (best guess; check `voiced`)
    voiced: np.ndarray   # pyin voiced probability
    db: np.ndarray       # RMS level (dB)
    onset: np.ndarray    # onset strength
    low_db: np.ndarray | None = None  # level below `low_hz` over short (46 ms) frames: a pluck re-excites
                                      # it even when the previous note is still ringing; a release kills it
    yin_midi: np.ndarray | None = None  # plain YIN pitch (no temporal smoothing): pyin's smoothing prefers
                                        # to keep the previous pitch and misses short notes in fast runs


def pitch_features(path: str | Path, cache_dir: Path | None = None, fmin: str = "C2", fmax: str = "C5",
                   frame_length: int = 2048, low_hz: float = 300.0) -> PitchFeatures:
    """Monophonic pitch track for any instrument stem (bass: fmin="B0", fmax="G4", frame_length=4096)."""
    def compute():
        librosa = require_librosa()
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        f0, _, vp = librosa.pyin(y, fmin=librosa.note_to_hz(fmin), fmax=librosa.note_to_hz(fmax), sr=sr,
                                 frame_length=frame_length, hop_length=HOP, fill_na=None)
        db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=frame_length // 2, hop_length=HOP)[0] + 1e-6)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        low = 10 * np.log10(S[freqs <= low_hz].sum(axis=0) + 1e-10)
        yin = librosa.yin(y, fmin=librosa.note_to_hz(fmin), fmax=librosa.note_to_hz(fmax), sr=sr,
                          frame_length=frame_length, hop_length=HOP)
        n = min(len(f0), len(db), len(onset), len(low), len(yin))
        t = librosa.times_like(f0, sr=sr, hop_length=HOP)[:n]
        return {"t": t, "midi": librosa.hz_to_midi(f0[:n]), "voiced": vp[:n], "db": db[:n], "onset": onset[:n],
                "low_db": low[:n], "yin_midi": librosa.hz_to_midi(yin[:n])}

    d = _cached(cache_dir, "pitch", path, compute, sr=SR, hop=HOP, fmin=fmin, fmax=fmax, frame=frame_length,
                low_hz=low_hz, v=3)
    return PitchFeatures(d["t"], d["midi"], d["voiced"], d["db"], d["onset"], d.get("low_db"), d.get("yin_midi"))


def rms_envelope(path: str | Path, cache_dir: Path | None = None, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """``(times, dB)`` loudness envelope, for energy-based decisions (lighting, sections)."""
    def compute():
        librosa = require_librosa()
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0] + 1e-6)
        return {"t": librosa.times_like(db, sr=sr, hop_length=hop), "db": db}

    d = _cached(cache_dir, "rms", path, compute, sr=SR, hop=hop)
    return d["t"], d["db"]


def onsets(path: str | Path, cache_dir: Path | None = None, backtrack: bool = False, hop: int = HOP,
           fmax: float | None = None, delta: float | None = None, wait_s: float | None = None) -> np.ndarray:
    """Onset times (s). ``backtrack`` moves each onset back to where its attack starts (best for
    note starts); without it you get the attack peak. ``fmax`` focuses on the low end (bass),
    ``delta`` lowers the peak threshold (librosa's default 0.07 suits full mixes and misses legato
    re-attacks), ``wait_s`` is the minimum gap between onsets."""
    def compute():
        librosa = require_librosa()
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        kw = {}
        if delta is not None:
            kw["delta"] = delta
        if wait_s is not None:
            kw["wait"] = max(1, int(wait_s * sr / hop))
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, **({"fmax": fmax} if fmax else {}))
        return {"t": librosa.onset.onset_detect(onset_envelope=env, y=y, sr=sr, hop_length=hop, units="time",
                                                backtrack=backtrack, **kw)}

    return _cached(cache_dir, "onsets", path, compute, sr=SR, hop=hop, backtrack=backtrack, fmax=fmax,
                   delta=delta, wait=wait_s)["t"]
