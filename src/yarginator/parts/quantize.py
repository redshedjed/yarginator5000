"""Snap times onto the chart's beat grid."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.timing import TempoMap


@dataclass
class Snap:
    tick: int
    error_ms: float   # distance from the original time to the chosen grid line
    grid: str         # "16th", "8th-triplet", ...


GRIDS = {"8th": 2, "16th": 4, "32nd": 8, "8th-triplet": 3, "16th-triplet": 6}


def snap(tm: TempoMap, seconds: float, straight: str = "16th", triplet: str | None = "8th-triplet",
         triplet_max_ms: float = 15.0, straight_min_ms: float = 35.0) -> Snap:
    """Nearest line on the straight grid, unless the note sits right on a triplet line
    (within ``triplet_max_ms``) and clearly off every straight line (``straight_min_ms``), so loose
    or laid-back straight playing doesn't turn into triplets."""
    tick = tm.s2t(seconds)
    best = _nearest(tm, seconds, tick, straight)
    if triplet:
        tri = _nearest(tm, seconds, tick, triplet)
        if tri.error_ms <= triplet_max_ms and best.error_ms >= straight_min_ms:
            best = tri
    return best


# how much a note "wants" each 16th position: on the beat > on the 8th > in between
METRIC_WEIGHT = {0: 1.0, 2: 0.8, 1: 0.55, 3: 0.55}


def feel_curve(tm: TempoMap, times: list[float], window_bars: float = 4.0, search_ms: tuple[float, float] = (-60, 100),
               step_ms: float = 2.0, sigma_ms: float = 20.0, min_notes: int = 6):
    """A part's lean against the grid, bar by bar (seconds, + = behind the beat) -> ``lean_at(t)``.

    Players drift ahead of / behind the click, and separately recorded parts can carry recording
    latency that changes from take to take. For each bar, the notes within ``window_bars`` are tried
    at every shift in ``search_ms``; the shift that lands the most notes near strong positions (beats,
    then 8ths, then 16ths) wins. The metrical preference is what tells "60 ms late on the beat" from
    "60 ms early on the next 16th". Per-bar values are median-smoothed; bars without enough notes use
    the whole-song value."""
    import numpy as np
    times = np.sort(np.asarray(times, dtype=float))
    if len(times) == 0:
        return lambda t: 0.0
    bar_ticks = tm.bar_ticks(tm.time_sig_at(0))
    offsets = np.arange(search_ms[0], search_ms[1] + step_ms / 2, step_ms) / 1000
    step16 = tm.ppq / 4

    def best(ts):
        scores = []
        for off in offsets:
            ticks = np.array([tm.s2t(t - off) for t in ts], dtype=float)
            pos = np.round(ticks / step16)
            resid_s = (ticks - pos * step16) * (60 / _bpm_at(tm, ts)) / tm.ppq
            w = np.array([METRIC_WEIGHT[int(p) % 4] for p in pos])
            scores.append(float(np.sum(w * np.exp(-(resid_s / (sigma_ms / 1000)) ** 2))))
        return float(offsets[int(np.argmax(scores))])

    whole = best(times)
    bars = np.array([tm.s2t(t) // bar_ticks for t in times])
    last = int(bars.max())
    raw = []
    for b in range(last + 1):
        sel = times[np.abs(bars - b) <= window_bars / 2]
        raw.append(best(sel) if len(sel) >= min_notes else whole)
    smooth = [float(np.median(raw[max(0, b - 1):b + 2])) for b in range(len(raw))]

    def lean_at(t: float) -> float:
        b = int(min(max(tm.s2t(t) // bar_ticks, 0), last))
        return smooth[b]
    lean_at.per_bar = smooth  # type: ignore[attr-defined]
    lean_at.whole = whole     # type: ignore[attr-defined]
    return lean_at


def _bpm_at(tm: TempoMap, ts) -> "float":
    import numpy as np
    mid = float(np.median(ts))
    tick = tm.s2t(mid)
    return next(t for t in reversed(tm.tempos) if t.tick <= tick).bpm


def feel_offset(tm: TempoMap, times: list[float], grid: str = "16th", max_ms: float = 60.0) -> float:
    """A part's consistent lead/lag against the grid, in seconds (+ = behind the beat).

    Players sit ahead of or behind the click, and separated stems smear attacks, so raw times
    straddle grid lines and snap to the wrong 16th. The circular mean of every note's offset from its
    nearest grid line measures that lean without being thrown off by which line each note snapped to."""
    if not times:
        return 0.0
    step = tm.ppq / GRIDS[grid]
    angles = []
    for t in times:
        tick = tm.s2t(t)
        angles.append(2 * math.pi * ((tick % step) / step))
    mean = math.atan2(sum(map(math.sin, angles)), sum(map(math.cos, angles)))
    off_ticks = mean / (2 * math.pi) * step
    # convert ticks to seconds at a typical point in the song
    mid = tm.s2t(sorted(times)[len(times) // 2])
    off = tm.t2s(mid + off_ticks) - tm.t2s(mid)
    return max(-max_ms, min(max_ms, off * 1000)) / 1000


def bar_grids(tm: TempoMap, times, lean_at, candidates=("16th", "8th-triplet"), margin: float = 0.7,
              min_notes: int = 3) -> dict[int, str]:
    """Pick one subdivision per bar from how *all* of its notes fit, instead of note by note (which
    mixes 16ths and triplets inside one figure when the playing is loose or laid back).

    A non-default grid (``candidates[1:]``) wins a bar only if its mean error is below ``margin`` x
    the default's. Each bar then takes the majority of itself and its neighbours, so sections stay
    consistent. Returns {bar index: grid}; bars without notes are absent."""
    import numpy as np
    bar_ticks = tm.bar_ticks(tm.time_sig_at(0))
    by_bar: dict[int, list[float]] = {}
    for t in times:
        by_bar.setdefault(int(tm.s2t(t - lean_at(t)) // bar_ticks), []).append(t - lean_at(t))
    raw = {}
    for b, ts in by_bar.items():
        if len(ts) < min_notes:
            raw[b] = candidates[0]
            continue
        err = {g: float(np.mean([_nearest(tm, t, tm.s2t(t), g).error_ms for t in ts])) for g in candidates}
        best = min(candidates[1:], key=lambda g: err[g]) if len(candidates) > 1 else candidates[0]
        raw[b] = best if len(candidates) > 1 and err[best] < margin * err[candidates[0]] else candidates[0]
    out = {}
    for b in raw:
        votes = [raw[x] for x in (b - 1, b, b + 1) if x in raw]
        out[b] = max(set(votes), key=lambda g: (votes.count(g), g == raw[b]))
    return out


def _nearest(tm: TempoMap, seconds: float, tick: int, grid: str) -> Snap:
    step = tm.ppq / GRIDS[grid]
    q = int(round(round(tick / step) * step))
    return Snap(q, abs(tm.t2s(q) - seconds) * 1000, grid)
