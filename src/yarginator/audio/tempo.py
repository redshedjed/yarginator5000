"""Beat detection for building tempo maps.

These are starting points: the downbeat in particular is a heuristic (the beat-in-bar with the most
low-end attack), so ``click`` results should be checked and the values locked into song.toml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import require_librosa

SR = 22050
HOP = 512


def detect_beats(path: str | Path) -> np.ndarray:
    librosa = require_librosa()
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    # trim=False: quiet intros are still beats, and dropping them shifts the offset measure
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="time", trim=False)
    return np.asarray(beats, dtype=float)


def fit_grid(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares constant grid through beat times -> ``(period_s, phase_s)``.

    Beats are numbered from consecutive gaps (a gap of ~2 periods = a skipped beat), so a dropout
    doesn't skew the fit the way numbering them 0, 1, 2... would."""
    if len(beats) < 8:
        raise ValueError("too few beats detected to fit a click tempo")
    gaps = np.diff(beats)
    p0 = float(np.median(gaps))
    idx = np.concatenate([[0], np.cumsum(np.maximum(1, np.round(gaps / p0)))])
    period, intercept = np.polyfit(idx, beats, 1)
    return float(period), float(intercept % period)


def comb_period(env_t: np.ndarray, env: np.ndarray, p0: float, span: float = 0.04) -> tuple[float, float]:
    """Constant grid that best explains the whole onset envelope -> ``(period_s, phase_s)``.

    For each candidate period near ``p0`` the envelope is folded onto one beat cycle
    (``|sum env * e^(2*pi*i*t/p)|``); the click period is where onsets pile up most coherently.
    Unlike fitting beat_track output, a local tracker glitch can't skew this."""
    w = env - env.mean()
    best = p0
    for rel, steps in ((span, 801), (span / 200, 201)):  # coarse ~0.01 % steps, then ~0.0001 %
        periods = best * (1 + np.linspace(-rel, rel, steps))
        score = np.abs(np.exp(2j * np.pi * env_t[None, :] / periods[:, None]) @ w)
        best = float(periods[int(np.argmax(score))])
    phase = float(np.angle(np.exp(2j * np.pi * env_t / best) @ w) / (2 * np.pi) * best % best)
    return best, phase


def fold_period(period: float, bpm_range: tuple[float, float]) -> float:
    """Double/halve the tempo into ``[lo, hi)`` -- beat trackers often lock onto half or double time."""
    lo, hi = bpm_range
    while 60 / period < lo:
        period /= 2
    while 60 / period >= hi:
        period *= 2
    return period


def first_downbeat(grid: np.ndarray, onset_t: np.ndarray, onset: np.ndarray, low_onset: np.ndarray,
                   beats_per_bar: int, start_fraction: float = 0.3) -> float:
    """Pick the first downbeat on a beat grid.

    1. Song start: the first grid beat whose onset strength reaches ``start_fraction`` of the
       song's typical on-beat strength.
    2. Bar phase: of the ``beats_per_bar`` possible phases, the one with the most low-end (kick)
       attack on average.
    """
    def at(env, times):
        idx = np.clip(np.searchsorted(onset_t, times), 0, len(env) - 1)
        win = np.stack([env[np.clip(idx + d, 0, len(env) - 1)] for d in (-1, 0, 1)])
        return win.max(axis=0)

    strength = at(onset, grid)
    typical = np.median(strength[strength > 0]) if np.any(strength > 0) else 0.0
    start = int(np.argmax(strength >= start_fraction * typical))
    low = at(low_onset, grid[start:])
    scores = np.array([low[p::beats_per_bar].mean() if len(low[p::beats_per_bar]) else 0
                       for p in range(beats_per_bar)])
    # Songs usually start on a downbeat, and kick-on-1-and-3 makes phases tie: only move off the
    # first beat when another phase is clearly stronger.
    phase = 0 if scores[0] >= 0.85 * scores.max() else int(np.argmax(scores))
    return float(grid[start + phase])


def refine_grid(period: float, phase: float, onset_times: np.ndarray, duration: float,
                fixed_period: bool = False) -> tuple[float, float]:
    """Re-fit the grid to precise onset times: match each grid beat to the nearest onset within a
    sixteenth note, then least-squares those matches (phase only if ``fixed_period``)."""
    grid = np.arange(phase, duration, period)
    if len(onset_times) < 2:
        return period, phase
    idx = np.clip(np.searchsorted(onset_times, grid), 1, len(onset_times) - 1)
    near = np.where(np.abs(onset_times[idx - 1] - grid) < np.abs(onset_times[idx] - grid),
                    onset_times[idx - 1], onset_times[idx])
    ok = np.abs(near - grid) < period / 4
    if ok.sum() < 8:
        return period, phase
    if fixed_period:
        return period, float((phase + np.median(near[ok] - grid[ok])) % period)
    p, b = np.polyfit(np.nonzero(ok)[0], near[ok], 1)
    return float(p), float(b % p)


def estimate_click(path: str | Path, beats_per_bar: int = 4, snap_bpm: float = 0.02,
                   bpm_range: tuple[float, float] = (80, 180)) -> tuple[float, float]:
    """-> ``(bpm, first_downbeat_s)``. The tempo is folded into ``bpm_range``; a BPM within
    ``snap_bpm`` of a whole number is snapped to it (click tracks usually are) and the phase re-fitted."""
    librosa = require_librosa()
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    low_onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmax=200, n_mels=32)
    onset_t = librosa.times_like(onset, sr=sr, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP, units="time", trim=False)
    # beat_track only gives the rough tempo (it's frame-accurate and can glitch); the comb over a
    # ~3 ms onset envelope gives the exact period, and matching to backtracked onsets the exact phase.
    period = fold_period(fit_grid(np.asarray(beats, dtype=float))[0], bpm_range)
    fine_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=64)
    period, phase = comb_period(librosa.times_like(fine_env, sr=sr, hop_length=64), fine_env, period)
    # The period comes from backtracked attack starts (consistent across instruments, so arrangement
    # changes don't tilt it). The phase comes from attack *peaks*: that's where a hit is heard, and a
    # grid on the starts sits 10-25 ms early, which makes every note quantized to it feel rushed.
    starts = np.asarray(librosa.onset.onset_detect(y=y, sr=sr, hop_length=64, units="time", backtrack=True))
    peaks = np.asarray(librosa.onset.onset_detect(y=y, sr=sr, hop_length=64, units="time", backtrack=False))
    period, phase = refine_grid(period, phase, starts, len(y) / sr)
    if snap_bpm and abs(60 / period - round(60 / period)) < snap_bpm:
        period = 60 / round(60 / period)
    period, phase = refine_grid(period, phase, peaks, len(y) / sr, fixed_period=True)
    grid = np.arange(phase, len(y) / sr, period)
    return 60.0 / period, first_downbeat(grid, onset_t, onset, low_onset, beats_per_bar)
