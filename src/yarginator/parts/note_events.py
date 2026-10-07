"""Monophonic note transcription from an instrument stem (bass now; any single-note line later).

1. Candidate attacks come from a standard onset detector (trusted for timing) plus a deliberately
   sensitive low-end one (adds notes the standard one misses, e.g. legato re-attacks).
2. Standard attacks are kept unless they look like a release (the low end falls away after them);
   sensitive-detector candidates need real evidence of a new note:
   - the low end jumps (a pluck re-excites the string's fundamental even while the previous note
     still rings; a release or fret noise only makes it quieter), or
   - the pitch changes, or
   - it comes out of silence.
3. Each note is judged on its first ~250 ms:
   - pitched: pyin is confident
   - weak:    pyin isn't confident but its best guess is steady; kept as a note with that pitch and
              flagged (low notes, short notes and smeared stems often land here)
   - ghost:   short/quiet and unpitched: muted/dead notes
4. A held note whose pitch moves without a new attack (hammer-on, slide) is split.
5. Unpitched hits on another stem's attacks are bleed (kick drum in a separated bass stem); unpitched
   transients just before a pitched note are pickup noise (dropped by default).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio.analysis import PitchFeatures

PITCHED, GHOST, BLEED = "pitched", "ghost", "bleed"

DEFAULTS = {
    "floor_db": 30.0,        # attacks quieter than (loud level - floor_db) are ignored
    "min_gap_s": 0.05,       # two attacks closer than this are one
    "rise_db": 3.0,          # low-end jump that confirms a sensitive-detector candidate
    "release_drop_db": 4.0,  # a standard attack after which the low end falls this much is a release
    "pitch_change": 0.8,     # ...or a pitch change this big (semitones)
    "head_s": 0.25,          # how much of a note's start is used to judge it
    "voiced_min": 0.4,       # share of confident pyin frames in the head for "pitched"
    "weak_voicing": 0.15,    # frames above this pyin confidence feed the best-guess pitch
    "weak_share": 0.5,       # share of frames that must agree on a note name for a best guess
    "lowest_note": None,     # lowest note the instrument plays (MIDI); lower guesses are octave errors.
                             # None = estimate it from the stem (see estimate_lowest)
    "search_ceiling": 72,    # top of the pitch search range (MIDI); guesses at it are noise
    "ghost_max_s": 0.08,     # a note with only a best-guess pitch is a ghost if shorter than this...
    "ghost_quiet_db": -12.0, # ...unless it's at least this loud (vs the stem's loud level)
    "end_drop_db": 15.0,     # a note ends when its level falls this far below its peak
    "max_note_s": 4.0,
    "legato_semitones": 0.8, # held pitch change that starts a new note without a new attack
    "legato_min_s": 0.08,
    "attach_s": 0.12,        # an unpitched transient this close before a pitched note...
    "transient": "drop",     # ...is pickup noise to drop, or "attack": the note's real start
    "bleed_window_s": 0.03,  # unpitched attack this close to another stem's attack...
    "bleed_max_db": -10.0,   # ...and at least this quiet (vs the stem's loud level) = bleed
}


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: float | None
    voiced: float
    level: float               # peak dB vs the stem's loud level
    kind: str = PITCHED
    notes: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.end - self.start


def _window(f: PitchFeatures, a: float, b: float) -> slice:
    return slice(int(np.searchsorted(f.t, a)), max(int(np.searchsorted(f.t, a)) + 1, int(np.searchsorted(f.t, b))))


def low_rise(f: PitchFeatures, t: float) -> float:
    """Jump (dB) in low-end energy across ``t``: short frames, windows kept clear of the attack."""
    lv = f.low_db if f.low_db is not None else f.db
    pre = lv[_window(f, t - 0.07, t - 0.02)]
    post = lv[_window(f, t + 0.02, t + 0.07)]
    return float(post.max() - pre.max()) if len(pre) and len(post) else 0.0


def _pitch(f: PitchFeatures, a: float, b: float, min_conf: float = 0.3) -> float | None:
    s = _window(f, a, b)
    m = f.midi[s][f.voiced[s] > min_conf]
    return float(np.median(m)) if len(m) >= 3 else None


def estimate_lowest(f: PitchFeatures, standard: float = 28.0, margin: float = 2.0, min_frames: int = 10) -> float:
    """The lowest note the instrument plays, for any string count or tuning: at least a standard
    4-string's low E (``standard``), lower only when the stem *confidently* plays lower (5/6-string
    low B, drop D, drop A ...), with ``margin`` semitones of slack. Never higher than ``standard``:
    trackers are least confident on low notes, so a floor read from confident frames alone would
    sit too high and fold real low notes up an octave."""
    loud = float(np.percentile(f.db, 95))
    sure = f.midi[(f.voiced > 0.7) & (f.db > loud - 25) & np.isfinite(f.midi)]
    below = sure[sure < standard - 0.5]
    if len(below) < min_frames:
        return standard
    return float(min(standard, np.floor(np.percentile(below, 5)) - margin))


def crowds_edge(f: PitchFeatures, edge: float, side: str, within: float = 1.5, share: float = 0.01) -> bool:
    """Does the stem pile confident pitches against an edge of the search range? Then the
    instrument probably goes past it (drop-A 5-string at the bottom, 6-string high register at the
    top) and the search should be widened."""
    loud = float(np.percentile(f.db, 95))
    sure = f.midi[(f.voiced > 0.7) & (f.db > loud - 25) & np.isfinite(f.midi)]
    if len(sure) < 50:
        return False
    near = sure <= edge + within if side == "low" else sure >= edge - within
    return bool(near.mean() >= share)


def transcribe(f: PitchFeatures, attacks: np.ndarray, extra: np.ndarray | None = None, **opts) -> list[NoteEvent]:
    """``attacks``: a standard onset detector's output, trusted for timing; pitched ones are kept
    as they are (that catches same-pitch re-plucks), unpitched ones need evidence (release clicks
    don't have any). ``extra``: a sensitive detector's output; only candidates the standard detector
    missed are considered, and they always need evidence."""
    o = {**DEFAULTS, **opts}
    if o["lowest_note"] is None:
        o["lowest_note"] = estimate_lowest(f)
    loud = float(np.percentile(f.db, 95))

    def level(t0, t1):
        s = _window(f, t0, t1)
        return float(f.db[s].max())

    strong = np.sort(np.asarray(attacks, dtype=float))
    cands = [(float(a), True) for a in strong]
    for a in np.sort(np.asarray(extra if extra is not None else [], dtype=float)):
        if len(strong) == 0 or np.min(np.abs(strong - a)) > 0.04:
            cands.append((float(a), False))
    cands.sort()

    # 1-2. keep candidates with evidence of a new note
    accepted: list[float] = []
    for a, trusted in cands:
        if accepted and a - accepted[-1] < o["min_gap_s"]:
            continue
        if level(a, a + 0.06) < loud - o["floor_db"]:
            continue
        rise = low_rise(f, a)
        # A trusted attack is a note unless it looks like a release (the low end falls away after
        # it). Fast repeated plucks of one note never let the low end dip, so a positive jump can't
        # be required of them.
        if trusted and rise > -o["release_drop_db"]:
            accepted.append(a)
            continue
        silent_before = level(a - 0.12, a - 0.03) < loud - 25
        pb, pa = _pitch(f, a - 0.12, a - 0.01), _pitch(f, a + 0.02, a + 0.12)
        changed = pb is not None and pa is not None and abs(pa - pb) >= o["pitch_change"]
        if silent_before or changed or rise >= o["rise_db"]:
            accepted.append(a)

    # 3-4. judge each note on its head; split legato pitch moves
    raw: list[NoteEvent] = []
    for k, a in enumerate(accepted):
        nxt = accepted[k + 1] if k + 1 < len(accepted) else a + o["max_note_s"]
        s = _window(f, a + 0.015, min(nxt, a + o["max_note_s"]))
        lv = f.db[s]
        above = np.nonzero(lv > lv.max() - o["end_drop_db"])[0]
        end = min(float(f.t[min(s.start + above.max() + 1, len(f.t) - 1)]) if len(above) else a + 0.05, float(nxt))
        head = _window(f, a + 0.015, min(nxt, a + o["head_s"]))
        conf = float(np.mean(f.voiced[head] > 0.5))
        lvl = float(lv.max() - loud)
        if conf >= o["voiced_min"]:
            v = f.voiced[s.start:s.stop] > 0.5
            for n, (ps, pe, pitch) in enumerate(_legato_split(f, s.start, s.stop, v, o)):
                raw.append(NoteEvent(a if n == 0 else ps, end if pe >= end else pe, pitch, conf, lvl, PITCHED,
                                     ["legato"] if n else []))
            continue
        # pyin isn't sure. Candidates: plain YIN (no smoothing) and pyin's own best guess, each folded
        # into the instrument's range (YIN likes to guess an octave low). pyin's smoothing is biased
        # towards repeating the previous pitch, which is what hides the moving notes in a fast run, so
        # when the two disagree the one that differs from the previous note wins.
        lo, hi = o["lowest_note"], o["search_ceiling"]
        cands = [_fold(steady_pitch_class(f.yin_midi[head], o["weak_share"]), lo, hi) if f.yin_midi is not None else None,
                 _fold(steady_pitch_class(f.midi[head][f.voiced[head] > o["weak_voicing"]], o["weak_share"]), lo, hi)]
        cands = [c for c in cands if c is not None]
        prev = raw[-1].pitch if raw and raw[-1].pitch is not None else None
        moving = [c for c in cands if prev is None or round(c) % 12 != round(prev) % 12]
        pitch = (moving or cands or [None])[0]
        # a 16th at 123 BPM is 0.12 s: only very short or quiet unpitched hits are ghosts
        if pitch is not None and (end - a >= o["ghost_max_s"] or lvl >= o["ghost_quiet_db"]):
            raw.append(NoteEvent(a, end, pitch, conf, lvl, PITCHED, ["weak pitch"]))
        else:
            raw.append(NoteEvent(a, end, None, conf, lvl, GHOST))

    # 5. unpitched transients right before a pitched note
    out: list[NoteEvent] = []
    for k, e in enumerate(raw):
        nxt_e = raw[k + 1] if k + 1 < len(raw) else None
        if (e.kind == GHOST and nxt_e is not None and nxt_e.kind == PITCHED
                and nxt_e.start - e.start <= o["attach_s"]):
            if o["transient"] == "attack":
                nxt_e.start = e.start
                nxt_e.notes.append("attack")
            else:
                nxt_e.notes.append("pickup dropped")
            continue
        out.append(e)
    return out


def _fold(pitch: float | None, lowest: float, ceiling: float) -> float | None:
    """Pitches below the instrument's lowest note are octave errors: move them up. Guesses at the
    top of the search range are what YIN returns for noise: no pitch."""
    if pitch is None or pitch >= ceiling - 1.0:
        return None
    while pitch < lowest - 0.5:
        pitch += 12
    return pitch


def steady_pitch_class(guess: np.ndarray, min_share: float = 0.5) -> float | None:
    """A pitch from uncertain frame-by-frame guesses: the note name (pitch class) that at least
    ``min_share`` of the frames agree on, in its most common octave. Octave flips (slap/pop,
    percussive attacks) don't break the agreement, and a few frames of the previous note still
    ringing at the start of a short note don't either. None if no note name wins."""
    guess = np.asarray(guess, dtype=float)
    guess = guess[np.isfinite(guess)]
    if len(guess) < 3:
        return None
    semis = np.round(guess).astype(int)
    counts = np.bincount(semis % 12, minlength=12)
    pc = int(np.argmax(counts))
    if counts[pc] < min_share * len(guess):
        return None
    sel = guess[semis % 12 == pc]
    octaves = (np.round(sel).astype(int) - pc) // 12
    common = np.bincount(octaves - octaves.min()).argmax() + octaves.min()
    return float(np.median(sel[octaves == common]))


def _legato_split(f: PitchFeatures, i: int, j: int, v: np.ndarray, o) -> list[tuple[float, float, float]]:
    """Split a held, pitched note where the pitch moves and stays (hammer-on / slide)."""
    idx = np.arange(i, j)[v]
    if len(idx) == 0:
        return [(float(f.t[i]), float(f.t[min(j, len(f.t) - 1)]), float(np.median(f.midi[i:j])))]
    m = f.midi[idx]
    hop = float(np.median(np.diff(f.t))) if len(f.t) > 1 else 0.01
    min_frames = max(3, int(o["legato_min_s"] / hop))
    cuts = [0]
    for k in range(min_frames, len(idx) - min_frames):
        cur = np.median(m[cuts[-1]:k])
        if abs(np.median(m[k:k + min_frames]) - cur) >= o["legato_semitones"] and k - cuts[-1] >= min_frames:
            cuts.append(k)
    cuts.append(len(idx))
    return [(float(f.t[idx[a]]), float(f.t[min(idx[b - 1] + 1, len(f.t) - 1)]), float(np.median(m[a:b])))
            for a, b in zip(cuts, cuts[1:])]


def mark_bleed(events: list[NoteEvent], other_attacks: np.ndarray, window: float, max_level_db: float = -10.0) -> int:
    """Quiet unpitched hits that coincide with another stem's attacks are bleed. Loud ones stay:
    funk bass locks to the kick all the time, and a slap note on the kick is still a note.
    Returns how many were marked."""
    if len(other_attacks) == 0:
        return 0
    n = 0
    for e in events:
        if e.kind == GHOST and e.level <= max_level_db:
            k = np.searchsorted(other_attacks, e.start)
            near = [abs(other_attacks[x] - e.start) for x in (k - 1, k) if 0 <= x < len(other_attacks)]
            if near and min(near) <= window:
                e.kind = BLEED
                n += 1
    return n
