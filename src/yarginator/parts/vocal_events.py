"""Vocal events from a stem: find everything the voice does, then sort it into
sung notes, unpitched vocals (growls, screams, spoken words -> Rock Band "#" notes) and noise
(coughs, breaths, clicks -> dropped).

Detection is by energy, not by pitch, so unpitched vocals are found too. Events are split at strong
attacks (repeated syllables sung without a gap) and, where pitched, at sustained pitch jumps.
Every threshold is an option so it can be tuned per song in song.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio.analysis import VocalFeatures

SUNG, UNPITCHED, NOISE = "sung", "unpitched", "noise"

DEFAULTS = {
    "floor_db": 30.0,          # event = louder than (song's loud level - floor_db)
    "bridge_s": 0.06,          # gaps shorter than this don't end an event
    "min_event_s": 0.05,       # ignore anything shorter
    "split_attack": 2.5,       # split inside an event at attacks this many x the median attack strength...
    "split_dip_db": 3.0,       # ...where the level dipped at least this much and came back
    "split_semitones": 1.5,    # split pitched events where the pitch jumps and stays
    "min_note_s": 0.10,        # no split leaves a piece shorter than this
    # classification
    "sung_voiced": 0.5,        # fraction of pitched frames to count as sung
    "noise_max_s": 0.30,       # unpitched and shorter than this -> noise (coughs, clicks)
    "noise_quiet_db": -18.0,   # unpitched and quieter than this (vs loud level) -> noise (breaths)
    "unpitched": "keep",       # keep: unpitched vocals become "#" notes; drop: treat them as noise
    # consonants: a short unpitched burst right before a note is the start of that syllable
    "consonant_max_s": 0.30,
    "consonant_gap_s": 0.08,
}


@dataclass
class VocalEvent:
    start: float
    end: float
    kind: str = SUNG
    pitch: float | None = None   # median MIDI over pitched frames
    spread: float = 0.0          # semitone spread (10-90 %) over pitched frames
    voiced: float = 0.0          # fraction of pitched frames
    flatness: float = 0.0        # median spectral flatness
    level: float = 0.0           # peak dB relative to the song's loud level
    why: str = ""                # short reason for the classification (flatness is reported, not used)
    merged: int = 1              # how many detected pieces this event is made of (after fitting)
    split_strength: float = np.inf  # evidence for the boundary at `start` (inf = real silence before it)
    notes: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.end - self.start


def _runs(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and (mask[j] or mask[j + 1:j + 1 + bridge].any()):
            j += 1
        out.append((i, j))
        i = j
    return out


def detect(f: VocalFeatures, **opts) -> list[VocalEvent]:
    o = {**DEFAULTS, **opts}
    hop_s = float(np.median(np.diff(f.t))) if len(f.t) > 1 else 0.01
    loud = float(np.percentile(f.db, 95))
    active = f.db > loud - o["floor_db"]
    pitched = (f.voiced > 0.5) & np.isfinite(f.midi)
    onset = f.onset if f.onset is not None else np.zeros_like(f.db)
    attack_ref = float(np.median(onset[active])) if active.any() else 1.0
    min_piece = max(1, int(o["min_note_s"] / hop_s))

    events = []
    for i, j in _runs(active, max(1, int(o["bridge_s"] / hop_s))):
        if (j - i) * hop_s < o["min_event_s"]:
            continue
        # cut frame -> strength (dB-equivalent; how sure we are a new syllable starts there)
        cuts: dict[int, float] = {i: np.inf}  # a real silence before the event
        # re-attacks inside the event: a strong onset where the level dipped and comes back up
        # (repeated syllables sung without a gap), not just a note swelling
        w = max(2, int(0.05 / hop_s))
        lvl = f.db_fine if f.db_fine is not None else f.db
        last_cut = i
        for k in range(i + min_piece, j - min_piece):
            if (onset[k] >= o["split_attack"] * attack_ref and onset[k] == onset[k - 2:k + 3].max()
                    and k - last_cut >= min_piece):
                tk = k - w + int(np.argmin(lvl[k - w:k + 1]))  # the syllable starts at the bottom of the dip
                trough = lvl[tk]
                depth = min(lvl[max(i, k - 3 * w):k - w + 1].max() - trough, lvl[k:k + w].max() - trough)
                if depth >= o["split_dip_db"] and tk - last_cut >= min_piece:
                    cuts[tk] = max(0.0, float(depth))
                    last_cut = tk
        # sustained pitch jumps inside pitched stretches (2 dB-equivalent per semitone)
        bounds = sorted(cuts)
        for a, b in zip(bounds, bounds[1:] + [j]):
            idx = np.arange(a, b)[pitched[a:b]]
            if len(idx) < 2 * min_piece:
                continue
            sm = np.convolve(f.midi[idx], np.ones(5) / 5, mode="same")
            last = 0
            for k in range(min_piece, len(idx) - min_piece):
                jump = abs(np.median(sm[k:k + min_piece]) - np.median(sm[last:k]))
                if jump >= o["split_semitones"] and k - last >= min_piece:
                    cuts[int(idx[k])] = max(cuts.get(int(idx[k]), 0.0), 2.0 * float(jump))
                    last = k
        bounds = sorted(cuts)
        for a, b in zip(bounds, bounds[1:] + [j]):
            if (b - a) * hop_s >= o["min_event_s"]:
                e = _measure(f, a, b, pitched, loud)
                e.split_strength = cuts[a]
                events.append(e)
    for e in events:
        classify(e, **o)
    return attach_consonants(events, o["consonant_max_s"], o["consonant_gap_s"])


def attach_consonants(events: list[VocalEvent], max_s: float, gap_s: float) -> list[VocalEvent]:
    """Fold short unpitched bursts that run straight into a kept event ("f" + "a") into it, so the
    note starts at the syllable's start instead of the burst becoming a cough/noise or its own note."""
    out: list[VocalEvent] = []
    for k, e in enumerate(events):
        nxt = events[k + 1] if k + 1 < len(events) else None
        if (e.kind != SUNG and e.length <= max_s and nxt is not None and nxt.kind != NOISE
                and nxt.start - e.end <= gap_s):
            nxt.start = e.start
            nxt.notes.append("consonant")
            continue
        out.append(e)
    return out


def _measure(f: VocalFeatures, a: int, b: int, pitched: np.ndarray, loud: float) -> VocalEvent:
    end = f.t[min(b, len(f.t) - 1)]
    p = pitched[a:b]
    m = f.midi[a:b][p]
    flat = f.flatness[a:b] if f.flatness is not None else np.zeros(b - a)
    return VocalEvent(
        start=float(f.t[a]), end=float(end),
        pitch=float(np.median(m)) if len(m) else None,
        spread=float(np.percentile(m, 90) - np.percentile(m, 10)) if len(m) > 2 else 0.0,
        voiced=float(p.mean()) if len(p) else 0.0,
        flatness=float(np.median(flat)),
        level=float(f.db[a:b].max() - loud),
    )


def classify(e: VocalEvent, **opts) -> VocalEvent:
    o = {**DEFAULTS, **opts}
    if e.voiced >= o["sung_voiced"]:
        e.kind, e.why = SUNG, f"pitched {e.voiced:.0%}"
    elif e.length < o["noise_max_s"]:
        e.kind, e.why = NOISE, f"short unpitched burst ({e.length:.2f}s)"
    elif e.level < o["noise_quiet_db"]:
        e.kind, e.why = NOISE, f"quiet unpitched ({e.level:.0f} dB)"
    elif o["unpitched"] == "drop":
        e.kind, e.why = NOISE, "unpitched (unpitched = drop)"
    else:
        e.kind, e.why = UNPITCHED, f"unpitched {1 - e.voiced:.0%}, flatness {e.flatness:.2f}"
    return e


MERGE_COST_PER_DB = 0.02  # an internal split of N dB costs as much to undo as a N*0.02 s rest


def merge_cost(a: VocalEvent, b: VocalEvent) -> float:
    """Cost of treating ``a`` and ``b`` as one syllable: the rest between them, or for pieces that
    touch, how strong the split evidence was."""
    rest = b.start - a.end
    if rest > 0.02 or not np.isfinite(b.split_strength):
        return max(rest, 0.0) + (0.0 if np.isfinite(b.split_strength) else 0.02)
    return MERGE_COST_PER_DB * b.split_strength


def fit_count(events: list[VocalEvent], target: int) -> tuple[list[VocalEvent], list[str]]:
    """Reduce kept events to ``target`` by the cheapest step each time: undo the weakest split / merge
    across the shortest rest (one attempt detected as two pieces) or drop the least convincing event.
    Returns the events and a log of what changed."""
    ev = [e for e in events if e.kind != NOISE]
    log = []
    while len(ev) > target:
        gaps = [(merge_cost(ev[k], ev[k + 1]), k) for k in range(len(ev) - 1)]
        mcost, mk = min(gaps) if gaps else (np.inf, -1)
        # drop cost: short, unpitched, quiet events are cheap to lose
        drops = [(e.length * (0.3 + e.voiced) * (1 + max(-30.0, e.level) / 30), k) for k, e in enumerate(ev)]
        drop_cost, dk = min(drops)
        if mcost <= drop_cost:
            a, b = ev[mk], ev[mk + 1]
            rest = b.start - a.end
            why = f"rest {rest:.2f}s" if rest > 0.02 else f"weak split {b.split_strength:.1f} dB"
            log.append(f"merged {a.start:.2f}s+{b.start:.2f}s ({why})")
            pa, pb = a.pitch, b.pitch
            pitch = (pa if a.length >= b.length else pb) if pa is not None and pb is not None else (pa or pb)
            ev[mk] = VocalEvent(
                a.start, b.end, kind=SUNG if SUNG in (a.kind, b.kind) else a.kind, pitch=pitch,
                spread=max(a.spread, b.spread), voiced=max(a.voiced, b.voiced), flatness=min(a.flatness, b.flatness),
                level=max(a.level, b.level), why=a.why, merged=a.merged + b.merged,
                split_strength=a.split_strength, notes=a.notes + b.notes + ["merged"])
            del ev[mk + 1]
        else:
            e = ev.pop(dk)
            log.append(f"dropped {e.start:.2f}s ({e.length:.2f}s, {e.kind})")
    return ev, log
