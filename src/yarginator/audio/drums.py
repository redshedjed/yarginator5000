"""Drum-stem analysis: onset candidates, what was hit at each one, and the sound of each hit.

Pipeline (developed and scored against charted songs with drum stems):

1. Candidates: peaks of spectral flux over the whole band and over low / mid / high bands, so a
   hi-hat under a kick still gets its own onset. These catch ~97% of charted hits.
2. NMF: the mel spectrogram is factored with bundled per-class templates (an attack and a decay
   spectrum for kick, snare, hat, ride, crash and three toms) plus a few free components that soak
   up bleed. The drum templates are allowed to drift toward this song's kit, which is what makes
   it hold up on kits it has never heard (separated stems, triggered metal kicks, live rooms).
3. Features per candidate: band rise / level / decay (9 octave-ish bands, levels relative to the
   song) plus each NMF family's activation and share at the onset.
4. A gradient-boosted classifier per family (kick, snare, cymbal, tom), trained from a bundled
   feature set (``data/drum_training.npz``) plus any you add with :func:`learn` (``yarginator
   drums-learn``) from your own finished charts. Kick and snare thresholds adapt per song (Otsu
   on the probability distribution), because a kit unlike the training songs shifts how sure the
   classifier is without changing which hits are real.

Which cymbal / which tom is decided later, per song, from the hit's sound (see parts/drums.py).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import require_librosa
from .analysis import _cached

log = logging.getLogger(__name__)

SR, HOP, NMEL = 44100, 441, 96            # 10 ms frames
FMIN, FMAX = 30.0, 18000.0
CLASSES = ("K", "S", "HH", "RD", "CR", "T1", "T2", "T3")
FAMILIES = {"kick": (0,), "snare": (1,), "cymbal": (2, 3, 4), "tom": (5, 6, 7)}
EDGES = (30, 100, 200, 400, 800, 1600, 3200, 6400, 12000, 18000)
N_FREE = 6                                # free NMF components for bleed / everything else
DATA_DIR = Path(__file__).parent / "data"
TEMPLATES = DATA_DIR / "drum_templates.npz"
TRAINING = DATA_DIR / "drum_training.npz"
FEATURE_VERSION = 1                       # bump when features change (old learned sets are skipped)


def user_data_dir() -> Path:
    """Where ``drums-learn`` keeps training sets learned from your charts."""
    return Path.home() / ".yarginator" / "drums"


# --- signal ------------------------------------------------------------------------------------------
def _load(paths) -> np.ndarray:
    librosa = require_librosa()
    ys = [librosa.load(str(p), sr=SR, mono=True)[0] for p in paths]
    n = max(len(y) for y in ys)
    return sum(np.pad(y, (0, n - len(y))) for y in ys)


def _mel(y) -> np.ndarray:
    librosa = require_librosa()
    return librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=NMEL,
                                          fmin=FMIN, fmax=FMAX, power=1.0)


def candidates(M: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Onset frames from flux peaks in the whole band and in low / mid / high bands (merged within 20 ms)."""
    librosa = require_librosa()
    L = np.log1p(100 * M)
    flux = np.maximum(0, np.diff(L, axis=1, prepend=L[:, :1]))
    n = L.shape[0]
    envs = [flux.mean(0), flux[: n * 3 // 16].mean(0), flux[n * 3 // 16: n * 5 // 8].mean(0), flux[n * 5 // 8:].mean(0)]
    peaks: set[int] = set()
    for e in envs:
        e = e / (np.percentile(e, 99) + 1e-9)
        peaks.update(int(x) for x in librosa.util.peak_pick(e, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                                             delta=delta, wait=3))
    merged: list[int] = []
    for p in sorted(peaks):
        if not merged or p - merged[-1] > 2:
            merged.append(p)
    return np.array(merged, dtype=int)


def nmf(V: np.ndarray, W0: np.ndarray, n_free: int = N_FREE, iters: int = 60, adapt: float = 0.3,
        seed: int = 0) -> np.ndarray:
    """KL-NMF with the drum templates ``W0`` (bands x components) drifting at most ``adapt`` of the
    way toward this song's spectra; returns activations H (components x frames)."""
    rng = np.random.default_rng(seed)
    k = W0.shape[1]
    W0 = W0 / (W0.sum(0, keepdims=True) + 1e-12)
    W = np.hstack([W0, rng.random((V.shape[0], n_free)) * W0.mean()])
    W /= W.sum(0, keepdims=True)
    H = rng.random((W.shape[1], V.shape[1])) * V.mean()
    eps = 1e-9
    for it in range(iters):
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.sum(0)[:, None] + eps)
        WH = W @ H + eps
        Wn = W * ((V / WH) @ H.T) / (H.sum(1)[None, :] + eps)
        Wn /= Wn.sum(0, keepdims=True) + eps
        a = adapt * it / iters
        W[:, :k] = (1 - a) * W0 + a * Wn[:, :k]
        W[:, k:] = Wn[:, k:]
    return H


def class_activations(H: np.ndarray) -> np.ndarray:
    """Per-class activation (attack + decay component), scaled by each class's 99th percentile."""
    A = np.stack([H[2 * c] + H[2 * c + 1] for c in range(len(CLASSES))])
    return A / (np.percentile(A, 99, axis=1, keepdims=True) + 1e-9)


def band_db(y) -> tuple[np.ndarray, np.ndarray]:
    """Power in the EDGES bands (dB, bands x frames) and the magnitude STFT below 500 Hz (for tom pitch)."""
    librosa = require_librosa()
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    P = S ** 2
    B = np.stack([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(EDGES, EDGES[1:])])
    low = f < 500
    return 10 * np.log10(B + 1e-10), S[low]


def features(D: np.ndarray, A: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-candidate features: band rise / level / three decay points / rise shape, then NMF family
    activation, rise and share."""
    T = D.shape[1]
    ref = np.percentile(D, 95, axis=1, keepdims=True)
    at = lambda X, o: X[:, np.clip(idx + o, 0, T - 1)]  # noqa: E731
    pre = np.mean([at(D, o) for o in (-4, -3, -2)], 0)
    peak = np.max([at(D, o) for o in (0, 1, 2)], 0)
    rise = peak - pre
    phys = np.vstack([rise, peak - ref, at(D, 8) - peak, at(D, 20) - peak, at(D, 40) - peak]).T
    phys = np.hstack([phys, (rise - rise.max(0)).T])
    a = np.max([at(A, o) for o in (0, 1, 2)], 0)
    a_pre = at(A, -3)
    fam = np.stack([a[0], a[1], a[2:5].sum(0), a[5:8].sum(0)])
    famr = np.stack([a[0] - a_pre[0], a[1] - a_pre[1], (a[2:5] - a_pre[2:5]).sum(0), (a[5:8] - a_pre[5:8]).sum(0)])
    return np.hstack([phys, np.vstack([fam, famr, fam / (fam.sum(0) + 1e-9)]).T]).astype(np.float32)


def tom_pitch(S_low: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Dominant frequency (Hz) of the body of each hit, 50-500 Hz (tom tuning)."""
    f = np.arange(S_low.shape[0]) * SR / 4096
    T = S_low.shape[1]
    band = f >= 50
    out = np.zeros(len(idx))
    for k, i in enumerate(idx):
        spec = S_low[:, np.clip(np.arange(i + 2, i + 9), 0, T - 1)].mean(1)
        j = int(np.argmax(np.where(band, spec, 0)))
        out[k] = f[j]
    return out


@dataclass
class DrumOnsets:
    t: np.ndarray          # candidate times (s)
    X: np.ndarray          # classifier features (n x F)
    pitch: np.ndarray      # body frequency (Hz): orders toms high -> low
    rise_db: np.ndarray    # loudest band rise (dB): accents vs ghost notes
    hf_decay: np.ndarray   # high-band level 200 ms / 400 ms after the hit (dB vs peak): hat < ride < crash
    hf_level: np.ndarray   # high-band level at the hit, vs the song's loud level (dB)
    act: np.ndarray        # per-class NMF activation at the hit (n x 8): which cymbal / which tom


def templates() -> np.ndarray:
    with np.load(TEMPLATES) as z:
        return z["W"]


def analyse(paths, cache_dir: Path | None = None, W0: np.ndarray | None = None) -> DrumOnsets:
    """Analyse a drum stem (several files, e.g. separate kick/snare/kit stems, are mixed)."""
    paths = [Path(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    W0 = templates() if W0 is None else W0

    def compute():
        y = _load(paths)
        M = _mel(y)
        idx = candidates(M)
        A = class_activations(nmf(M ** 0.75, W0))
        D, S_low = band_db(y)
        X = features(D, A, idx)
        T = D.shape[1]
        hf = D[7:].max(0)  # 6.4-18 kHz
        peak_hf = np.max([hf[np.clip(idx + o, 0, T - 1)] for o in (0, 1, 2)], 0)
        decay = np.stack([hf[np.clip(idx + 20, 0, T - 1)] - peak_hf, hf[np.clip(idx + 40, 0, T - 1)] - peak_hf], 1)
        act = np.max([A[:, np.clip(idx + o, 0, A.shape[1] - 1)] for o in (0, 1, 2)], 0).T
        return {"t": idx * HOP / SR, "X": X, "pitch": tom_pitch(S_low, idx), "rise_db": X[:, :9].max(1),
                "hf_decay": decay, "hf_level": peak_hf - np.percentile(hf, 95), "act": act}

    key = hashlib.sha1(W0.tobytes()).hexdigest()[:8]
    extra = "+".join(p.name for p in paths[1:])
    data = _cached(cache_dir, "drums", paths[0], compute, v=FEATURE_VERSION, a=2, w=key, extra=extra)
    return DrumOnsets(**{k: data[k] for k in DrumOnsets.__dataclass_fields__})


# --- learning from charts ------------------------------------------------------------------------------
def build_templates(songs) -> np.ndarray:
    """NMF templates from charted songs: per class, the median attack (first 20 ms) and decay
    (30-90 ms) spectra of hits charted on their own. ``songs``: ``[(stem paths, [(seconds, class)])]``,
    chart times already aligned to the stem."""
    att: dict[int, list] = {c: [] for c in range(len(CLASSES))}
    dec: dict[int, list] = {c: [] for c in range(len(CLASSES))}
    for paths, hits in songs:
        V = _mel(_load(paths)) ** 0.75
        idx = candidates(V ** (1 / 0.75))
        fake = DrumOnsets(idx * HOP / SR, *[np.zeros(len(idx))] * 6)
        Y = labels_for(fake, hits)
        for j in np.flatnonzero(Y.sum(1) == 1):
            c, i = int(np.argmax(Y[j])), idx[j]
            if i + 10 < V.shape[1]:
                a, b = V[:, i:i + 2].mean(1), V[:, i + 3:i + 9].mean(1)
                att[c].append(a / (a.sum() + 1e-9))
                dec[c].append(b / (b.sum() + 1e-9))
    W = []
    for c in range(len(CLASSES)):
        if not att[c]:
            raise ValueError(f"no isolated {CLASSES[c]} hits to build a template from")
        W += [np.median(att[c], 0), np.median(dec[c], 0)]
    return np.array(W).T.astype(np.float32)



def labels_for(onsets: DrumOnsets, hits: list[tuple[float, int]], tol: float = 0.035) -> np.ndarray:
    """Multi-hot class labels per candidate from charted ``(seconds, class index)`` hits."""
    Y = np.zeros((len(onsets.t), len(CLASSES)), dtype=np.int8)
    if len(onsets.t) == 0:
        return Y
    for t, c in hits:
        j = int(np.argmin(np.abs(onsets.t - t)))
        if abs(onsets.t[j] - t) <= tol:
            Y[j, c] = 1
    return Y


def align(onsets: DrumOnsets, hits: list[tuple[float, int]], search_s: float = 0.04, base: float = 0.0) -> float:
    """Offset (s, added to chart times) that best lines charted kicks/snares up with the onsets."""
    ks = np.array([t for t, c in hits if c in (0, 1)])
    ct = onsets.t
    if len(ks) == 0 or len(ct) < 2:
        return base
    best = (base, -1)
    for off in np.arange(base - search_s, base + search_s + 1e-9, 0.002):
        x = ks + off
        j = np.searchsorted(ct, x).clip(1, len(ct) - 1)
        score = int(np.sum(np.minimum(np.abs(ct[j] - x), np.abs(ct[j - 1] - x)) < 0.02))
        if score > best[1]:
            best = (float(off), score)
    return best[0]


def learn(onsets: DrumOnsets, hits: list[tuple[float, int]], out: Path, name: str) -> dict:
    """Save a training set (features + labels) from a finished chart; it joins the bundled set the
    next time the classifier is trained."""
    Y = labels_for(onsets, hits)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=onsets.X, Y=Y, song=np.array([name] * len(Y)), v=FEATURE_VERSION)
    return {"candidates": len(Y), "labelled": int((Y.sum(1) > 0).sum()), "hits": len(hits),
            "matched": int(Y.sum())}


# --- the classifier ------------------------------------------------------------------------------------
def training_sets(extra_dirs=()) -> list[Path]:
    sets = [TRAINING] if TRAINING.exists() else []
    for d in [user_data_dir(), *extra_dirs]:
        if Path(d).is_dir():
            sets += sorted(Path(d).glob("*.npz"))
    return sets


def _load_sets(sets, exclude: set[str] = frozenset()):
    Xs, Ys = [], []
    for p in sets:
        with np.load(p, allow_pickle=False) as z:
            if int(z["v"]) != FEATURE_VERSION:
                log.warning("drums: %s was learned with older features; re-run drums-learn on it", p.name)
                continue
            keep = ~np.isin(z["song"], list(exclude)) if exclude else slice(None)
            Xs.append(z["X"][keep]); Ys.append(z["Y"][keep])
    if not Xs:
        raise RuntimeError("drums: no training data found (the package's data/drum_training.npz is missing)")
    return np.concatenate(Xs), np.concatenate(Ys)


class DrumModel:
    """One gradient-boosted classifier per family (kick / snare / cymbal / tom)."""

    def __init__(self, models: dict):
        self.models = models

    @classmethod
    def train(cls, sets=None, exclude: set[str] = frozenset(), cache_dir: Path | None = None) -> "DrumModel":
        from sklearn import __version__ as skv
        from sklearn.ensemble import HistGradientBoostingClassifier
        sets = training_sets() if sets is None else sets
        key = hashlib.sha1("|".join(f"{p}:{p.stat().st_mtime_ns}" for p in sets).encode()
                           + f"{skv}|{sorted(exclude)}|{FEATURE_VERSION}".encode()).hexdigest()[:12]
        f = Path(cache_dir) / f"drum-model-{key}.pkl" if cache_dir else None
        if f and f.exists():
            try:
                with open(f, "rb") as fh:
                    return cls(pickle.load(fh))
            except Exception:  # a stale or foreign pickle: just retrain
                pass
        X, Y = _load_sets(sets, exclude)
        models = {}
        for fam, cl in FAMILIES.items():
            y = Y[:, list(cl)].max(1)
            models[fam] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                                         l2_regularization=1.0, random_state=0).fit(X, y)
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "wb") as fh:
                pickle.dump(models, fh)
        return cls(models)

    def proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if len(X) == 0:
            return {fam: np.zeros(0) for fam in FAMILIES}
        return {fam: m.predict_proba(X)[:, 1] for fam, m in self.models.items()}


def otsu(x: np.ndarray, bins: int = 64) -> float:
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    s = np.cumsum(h * c)
    m0 = s / np.maximum(w0, 1)
    m1 = (s[-1] - s) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def thresholds(P: dict[str, np.ndarray], settings: dict) -> dict[str, float]:
    """Probability threshold per family: a number, or "auto" (Otsu split of this song's
    probabilities, kept between 0.1 and 0.5: below 0.1 a song without a clear split starts doubling
    every kick with a phantom snare)."""
    out = {}
    for fam, p in P.items():
        s = settings.get(fam, "auto")
        if s == "auto" and len(p) >= 20:
            lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            out[fam] = float(np.clip(1 / (1 + np.exp(-otsu(lg))), 0.1, 0.5))
        else:
            out[fam] = 0.5 if s == "auto" else float(s)
    return out
