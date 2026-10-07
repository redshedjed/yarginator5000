"""Pro drums (PART DRUMS) from the drum stem, following the C3 Customs Book (chapter 11).

1. Hits: ``audio.drums`` finds onsets and classifies each as kick / snare / cymbal / tom (several
   at once is normal: kick + crash). Thresholds: ``"auto"`` per song for kick and snare, fixed for
   cymbals and toms; override any of them under ``[parts.drums.threshold]``.
2. Grid: the kit's lean against the tempo map (kick + snare) is measured and taken out, then each
   bar snaps to 16ths or, where the bar as a whole fits better, 8th-note triplets.
3. Kit: crashes are cymbal hits that act like accents (a bigger attack and more ring than the
   song's typical cymbal, on a strong beat, out of a fill); the rest keep time on the hi-hat (yellow); ``ride = "auto"`` puts sections
   that clearly sound different on the ride (blue). Toms are clustered by pitch and go high -> low to yellow / blue / green. Pro tom markers
   (110-112) cover tom gems; everything else on Y/B/G is a cymbal.
4. Expert playability: at most two hand gems at once (crash and snare win); a tom on the same
   colour as a cymbal moves to a free tom; kicks faster than ``kick_2x_ms`` become 2x kicks
   (note 95, shown only with double-kick enabled).
5. Reductions (book pp. 353-362):
   Hard: kicks thinned (no off-8th kick next to another), no kicks in fills, hands to 8ths at 140+ BPM,
   crashes on green. Medium: hands on 8ths (quarters at 140+), kicks on 8ths (quarters above
   110 BPM, one a bar at 170+) and only with a hand gem or on the beat, two gems at most, no kick
   under an off-beat crash, a breath after crashes at 140+. Easy: quarters (8ths below 100 BPM),
   a kick never shares a tick with anything, no kicks under crashes, one a bar at 170+.
6. Drum fills (120-124) over tom runs that land on a crash, ``[mix N drums0]`` events, drummer
   ``[idle]``/``[play]`` events and drum animation notes (24-51), each switchable.

Options (``[parts.drums]``)::

    stems = ["drums"]            # stem keys mixed for analysis (e.g. ["kick", "snare", "kit"])
    threshold = { kick = "auto", snare = "auto", cymbal = 0.3, tom = 0.3 }   # 0-1, higher = fewer
    grid = "16th"
    triplets = "8th-triplet"     # or "" to never use triplets
    timing_offset_ms = "auto"
    kick_2x_ms = 85              # Expert kicks closer than this are 2x-pedal kicks
    ride = "never"               # timekeeping on the hi-hat; "auto": blue where a section clearly changes cymbal
    toms = "pitch"               # tom colour from body frequency, high -> low; "sound": NMF tom templates
    crash_score = 1.0            # accent evidence a cymbal needs to be a crash (higher = fewer)
    fills = true
    animation = true
    mix = "drums0"               # [mix N drums0]; "" to leave mix events alone

Not generated (yet): disco flip, roll / swell lanes, overdrive, 2x-only Expert track (PART DRUMS_2X).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from ..core import instruments as ins
from ..core.chart import Chart, Track
from ..core.instruments import Difficulty
from ..feedback import ReviewItem
from .base import PartCharter, PartContext, PartResult, register
from .quantize import bar_grids, feel_curve, snap

KICK, RED, YEL, BLU, GRN = range(5)
LANE_NAMES = "KRYBG"
TOM_LANES = {5: YEL, 6: BLU, 7: GRN}            # audio.drums class T1/T2/T3 -> lane
CYM_KINDS = {"hat": YEL, "ride": BLU, "crash": GRN}
TOM_MARKER = {YEL: 110, BLU: 111, GRN: 112}
KICK_2X = ins.DRUM_KICK_2X
FILL = tuple(ins.DRUM_FILL)
ANIM_NOTES = range(24, 52)
ANIM_EVENTS = ("[idle]", "[idle_realtime]", "[idle_intense]", "[play]", "[mellow]", "[intense]", "[play_solo]")
# importance when a hand gem has to go (book: snare and crash matter most)
IMPORTANCE = {"crash": 4, "snare": 3, "tom": 2, "ride": 1, "hat": 1}

DEFAULTS = {
    "stems": ["drums"],
    "threshold": {},
    "grid": "16th",
    "triplets": "8th-triplet",
    "timing_offset_ms": "auto",
    "kick_2x_ms": 85,
    "ride": "never",
    "toms": "pitch",
    "crash_score": 1.0,
    "fills": True,
    "animation": True,
    "mix": "drums0",
}
DEFAULT_THRESHOLDS = {"kick": "auto", "snare": "auto", "cymbal": 0.3, "tom": 0.3}


@dataclass
class Gem:
    lane: int
    kind: str                  # kick / snare / hat / ride / crash / tom
    prob: float = 1.0
    double: bool = False       # 2x kick

    @property
    def is_tom(self) -> bool:
        return self.kind == "tom"


@dataclass
class Slot:
    """Everything charted at one tick."""
    tick: int
    gems: dict[int, Gem] = field(default_factory=dict)
    time_s: float = 0.0
    grid: str = ""
    error_ms: float = 0.0

    def hands(self) -> list[Gem]:
        return [g for lane, g in self.gems.items() if lane != KICK]

    def copy(self) -> "Slot":
        return Slot(self.tick, {k: Gem(g.lane, g.kind, g.prob, g.double) for k, g in self.gems.items()},
                    self.time_s, self.grid, self.error_ms)


# --- reading a chart (for drums-learn and scoring) ---------------------------------------------------
def expert_hits(chart: Chart) -> list[tuple[float, int]]:
    """Expert PART DRUMS as ``(seconds, audio.drums class index)``: K S HH RD CR T1 T2 T3."""
    tr = chart.tracks.get(ins.DRUMS)
    if tr is None:
        return []
    base = ins.DRUM_BASE[Difficulty.EXPERT]
    spans = {lane: [(n.tick, n.end) for n in tr.notes([TOM_MARKER[lane]])] for lane in TOM_MARKER}
    cym_class = {YEL: 2, BLU: 3, GRN: 4}
    tom_class = {YEL: 5, BLU: 6, GRN: 7}
    out = []
    for n in tr.notes([KICK_2X] + list(range(base, base + 5))):
        lane = KICK if n.pitch == KICK_2X else n.pitch - base
        if lane == KICK:
            c = 0
        elif lane == RED:
            c = 1
        else:
            tom = any(a <= n.tick < b for a, b in spans[lane])
            c = tom_class[lane] if tom else cym_class[lane]
        out.append((chart.tempo.t2s(n.tick), c))
    return sorted(out)


# --- helpers -----------------------------------------------------------------------------------------
def strength(tick: int, ppq: int, bar: int) -> int:
    """Metric weight: 4 bar line, 3 half bar, 2 beat, 1 8th, 0 anything finer."""
    if tick % bar == 0:
        return 4
    if tick % (bar // 2) == 0:
        return 3
    if tick % ppq == 0:
        return 2
    if tick % (ppq // 2) == 0:
        return 1
    return 0


def kmeans1d(x: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    c = np.quantile(x, np.linspace(0.15, 0.85, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), 1)
        c = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
    return np.argmin(np.abs(x[:, None] - c[None, :]), 1), c


def tom_lanes(on, is_tom: np.ndarray, method: str = "sound") -> np.ndarray:
    """Tom colour per onset. ``"sound"``: the strongest of the T1/T2/T3 NMF activations, each scaled
    by its typical level in this song. ``"pitch"``: the hit's body frequency, clustered into up to
    three toms (at least a minor third apart) and coloured high -> low."""
    n = len(on.t)
    lanes = np.full(n, YEL, dtype=int)
    if not is_tom.any():
        return lanes
    if method == "pitch":
        x = np.log2(np.maximum(on.pitch[is_tom], 40.0))
        best = np.zeros(len(x), int), np.array([x.mean()])
        for k in (3, 2):
            if len(x) < 3 * k:
                continue
            lab, c = kmeans1d(x, k)
            cs = np.sort(c)
            if np.all(np.diff(cs) >= 0.25) and min(np.sum(lab == j) for j in range(k)) >= max(2, len(x) // 20):
                best = lab, c
                break
        lab, c = best
        order = np.argsort(-c)                       # highest first
        colours = {1: [YEL if c[0] >= np.log2(150) else GRN], 2: [YEL, GRN], 3: [YEL, BLU, GRN]}[len(c)]
        rank = {int(j): colours[r] for r, j in enumerate(order)}
        lanes[is_tom] = [rank[int(j)] for j in lab]
        return lanes
    act = on.act[:, 5:8]
    a = act / (np.median(act[is_tom], 0, keepdims=True) + 1e-9)
    lanes[:] = [TOM_LANES[5 + int(j)] for j in np.argmax(a, 1)]
    return lanes


def _robust_z(x: np.ndarray) -> np.ndarray:
    return (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-9)


def crash_flags(slots: list[Slot], cym_idx: dict[int, int], on, ppq: int, bar: int, need: float) -> set[int]:
    """Ticks whose cymbal is a crash. Against charted songs the cues that hold up are a bigger
    attack than the song's typical cymbal and more ring 200 ms later (both as robust z-scores within
    the song), landing on a strong beat, and coming straight out of a tom fill."""
    ticks = sorted(cym_idx)
    if not ticks:
        return set()
    rise = _robust_z(np.array([on.rise_db[cym_idx[t]] for t in ticks]))
    ring = _robust_z(np.array([on.hf_decay[cym_idx[t], 0] for t in ticks]))
    tom_ticks = sorted(s.tick for s in slots if any(g.is_tom for g in s.hands()))
    out = set()
    for k, t in enumerate(ticks):
        score = rise[k] + ring[k] + {4: 1.0, 3: 0.5, 2: 0.0}.get(strength(t, ppq, bar), -1.0)
        j = bisect.bisect_left(tom_ticks, t)
        if j > 0 and t - tom_ticks[j - 1] <= ppq:
            score += 1.0          # a fill landing
        if score >= need:
            out.add(t)
    return out


def ride_sections(timekeeping: list[tuple[int, float]], bar: int, min_gap: float = 1.0) -> set[int]:
    """Bars to put on the ride: per-bar mean hat-vs-ride log ratio, split in two only when the
    groups are clearly apart (``min_gap``) and each spans at least two bars."""
    by_bar: dict[int, list[float]] = {}
    for tick, r in timekeeping:
        by_bar.setdefault(tick // bar, []).append(r)
    bars = sorted(b for b, v in by_bar.items() if len(v) >= 3)
    if len(bars) < 4:
        return set()
    x = np.array([np.mean(by_bar[b]) for b in bars])
    lab, c = kmeans1d(x, 2)
    if abs(c[0] - c[1]) < min_gap or min(np.sum(lab == 0), np.sum(lab == 1)) < 2:
        return set()
    ride = int(np.argmin(c))              # lower hat-vs-ride ratio = more ride-like
    # smooth: a bar follows its neighbours when both disagree with it
    lab = list(lab)
    for i in range(1, len(lab) - 1):
        if lab[i - 1] == lab[i + 1] != lab[i]:
            lab[i] = lab[i - 1]
    return {b for b, l in zip(bars, lab) if l == ride}


# --- expert -----------------------------------------------------------------------------------------
def playable(slot: Slot) -> list[str]:
    """Expert limits for one tick; returns notes about what changed."""
    notes = []
    hands = slot.hands()
    if len(hands) > 2:
        keep = sorted(hands, key=lambda g: (IMPORTANCE[g.kind], g.prob), reverse=True)[:2]
        for g in hands:
            if g not in keep:
                del slot.gems[g.lane]
                notes.append(f"dropped {g.kind} (3rd hand)")
    return notes


def mark_2x(slots: list[Slot], tm, min_ms: float) -> int:
    n, last = 0, None
    for s in slots:
        g = s.gems.get(KICK)
        if g is None:
            continue
        t = tm.t2s(s.tick)
        if last is not None and (t - last) * 1000 < min_ms:
            g.double = True
            n += 1
        else:
            last = t
    return n


def hands_only_stretches(slots: list[Slot], bar: int, min_bars: int = 2) -> list[tuple[int, int]]:
    """``(first, last)`` bar indexes of runs of ``min_bars``+ bars with cymbal/tom gems but no kick
    or snare (often another instrument bleeding into a separated drum stem)."""
    kit, hands = set(), set()
    for s in slots:
        b = s.tick // bar
        if KICK in s.gems or RED in s.gems:
            kit.add(b)
        elif s.gems:
            hands.add(b)
    runs, start, prev = [], None, None
    for b in sorted(hands - kit):
        if start is not None and b == prev + 1:
            prev = b
            continue
        if start is not None and prev - start + 1 >= min_bars:
            runs.append((start, prev))
        start = prev = b
    if start is not None and prev - start + 1 >= min_bars:
        runs.append((start, prev))
    return runs


def find_fills(slots: list[Slot], ppq: int, bar: int) -> list[tuple[int, int]]:
    """``(start, end)`` ticks of tom runs (3+ tom gems, no timekeeping cymbal) that land on a crash
    within a beat; fills at least two bars apart."""
    fills = []
    i = 0
    while i < len(slots):
        if not any(g.is_tom for g in slots[i].hands()):
            i += 1
            continue
        j, toms = i, 0
        while j < len(slots) and not any(g.kind in ("hat", "ride") for g in slots[j].hands()) \
                and (j == i or slots[j].tick - slots[j - 1].tick <= ppq):
            if any(g.kind == "crash" for g in slots[j].hands()):
                break
            toms += any(g.is_tom for g in slots[j].hands())
            j += 1
        if j < len(slots) and toms >= 3 and any(g.kind == "crash" for g in slots[j].hands()) \
                and slots[j].tick - slots[j - 1].tick <= ppq and slots[j].tick - slots[i].tick >= ppq // 2:
            start, end = slots[i].tick, slots[j].tick
            if not fills or start - fills[-1][1] >= 2 * bar:
                fills.append((start, end))
        i = max(j, i + 1)
    return fills


# --- reductions --------------------------------------------------------------------------------------
def _spacing(d: Difficulty, bpm: float, ppq: int) -> int:
    if d == Difficulty.EXPERT:
        return 0
    if d == Difficulty.HARD:
        return ppq // 4 if bpm < 140 else ppq // 2
    if d == Difficulty.MEDIUM:
        return ppq // 2 if bpm < 140 else ppq
    return ppq // 2 if bpm < 100 else ppq


def _thin(slots: list[Slot], spacing: int, ppq: int, bar: int) -> list[Slot]:
    """Hand gems at least ``spacing`` apart: strong beats first, then the more important sounds."""
    if spacing <= 0:
        return slots
    order = sorted((s for s in slots if s.hands()),
                   key=lambda s: (-strength(s.tick, ppq, bar), -max(IMPORTANCE[g.kind] for g in s.hands()), s.tick))
    kept: list[int] = []
    for s in order:
        k = bisect.bisect_left(kept, s.tick)
        if (k > 0 and s.tick - kept[k - 1] < spacing) or (k < len(kept) and kept[k] - s.tick < spacing):
            for lane in [l for l in s.gems if l != KICK]:
                del s.gems[lane]
            continue
        kept.insert(k, s.tick)
    return slots


def reduce(expert: list[Slot], d: Difficulty, ppq: int, bar: int, bpm: float,
           fills: list[tuple[int, int]]) -> list[Slot]:
    slots = [s.copy() for s in expert]
    for s in slots:                                   # single pedal below Expert
        if KICK in s.gems and s.gems[KICK].double:
            del s.gems[KICK]
    if d == Difficulty.EXPERT:
        return [s for s in slots if s.gems]

    in_fill = lambda t: any(a <= t < b for a, b in fills)  # noqa: E731
    x_kicks = sorted(s.tick for s in expert if KICK in s.gems)

    def kick_near(t, within):
        k = bisect.bisect_left(x_kicks, t - within + 1)
        return any(x != t for x in x_kicks[k:bisect.bisect_left(x_kicks, t + within)])
    # crashes all on green; a double crash becomes one
    for s in slots:
        crashes = [g for g in s.hands() if g.kind == "crash"]
        if crashes:
            for g in crashes:
                del s.gems[g.lane]
            if GRN not in s.gems:
                s.gems[GRN] = Gem(GRN, "crash", max(g.prob for g in crashes))
    _thin(slots, _spacing(d, bpm, ppq), ppq, bar)

    for s in slots:
        hands = s.hands()
        kick = s.gems.get(KICK)
        st = strength(s.tick, ppq, bar)
        if kick is not None:
            drop = False
            if d == Difficulty.HARD:
                drop = in_fill(s.tick) or (st == 0 and kick_near(s.tick, ppq // 2))  # off-8th, next to a kick
            elif d == Difficulty.MEDIUM:
                grid = ppq * 4 if bpm >= 170 else ppq if bpm > 110 else ppq // 2
                drop = (s.tick % grid != 0 or in_fill(s.tick) or (not hands and st < 2)
                        or any(g.kind == "crash" for g in hands) and st < 2)
                if bpm >= 170 and s.tick % bar != 0:
                    drop = True
            else:  # easy
                drop = bool(hands) or s.tick % ppq != 0 or in_fill(s.tick) or (bpm >= 170 and s.tick % bar != 0)
            if drop:
                del s.gems[KICK]
        hands = sorted(s.hands(), key=lambda g: (IMPORTANCE[g.kind], g.prob), reverse=True)
        limit = 2
        if d == Difficulty.MEDIUM and KICK in s.gems:
            limit = 1
        if d == Difficulty.EASY and not any(g.kind == "crash" for g in hands):
            limit = 1
        for g in hands[limit:]:
            del s.gems[g.lane]

    if d in (Difficulty.MEDIUM, Difficulty.EASY) and bpm >= 140:  # a breath after each crash
        crash_ticks = [s.tick for s in slots if any(g.kind == "crash" for g in s.hands())]
        for s in slots:
            k = bisect.bisect_left(crash_ticks, s.tick)
            if k > 0 and 0 < s.tick - crash_ticks[k - 1] < ppq // 2:
                for lane in [l for l in s.gems if l != KICK]:
                    del s.gems[lane]
    if d == Difficulty.MEDIUM:  # kicks only where the hands play, or on the beat
        for s in slots:
            if KICK in s.gems and not s.hands() and s.tick % ppq != 0:
                del s.gems[KICK]
    if d in (Difficulty.MEDIUM, Difficulty.EASY):
        # limited limb independence: a lone kick can't sit closer to a hand gem than the hands may to
        # each other (the snare matters more, so the kick goes)
        gap = _spacing(d, bpm, ppq)
        hand_ticks = sorted(s.tick for s in slots if s.hands())
        for s in slots:
            if KICK in s.gems and not s.hands():
                k = bisect.bisect_left(hand_ticks, s.tick - gap + 1)
                if any(abs(h - s.tick) < gap for h in hand_ticks[k:k + 2]):
                    del s.gems[KICK]
    return [s for s in slots if s.gems]


# --- writing -----------------------------------------------------------------------------------------
def write(track: Track, by_diff: dict[Difficulty, list[Slot]], fills, ppq: int, o: dict) -> Track:
    owned = set(FILL) | {KICK_2X} | set(TOM_MARKER.values())
    for d in Difficulty:
        owned |= set(range(ins.DRUM_BASE[d], ins.DRUM_BASE[d] + 5))
    if o["animation"]:
        owned |= set(ANIM_NOTES)
    track.remove_notes(owned)
    short = ppq // 8
    tom_ticks: dict[int, set[int]] = {lane: set() for lane in TOM_MARKER}
    for d, slots in by_diff.items():
        base = ins.DRUM_BASE[d]
        for s in slots:
            for lane, g in s.gems.items():
                if lane == KICK and g.double:
                    track.add_note(s.tick, short, KICK_2X)
                else:
                    track.add_note(s.tick, short, base + lane)
                if g.is_tom:
                    tom_ticks[lane].add(s.tick)
    # tom markers: one per tom gem (a marker spanning a cymbal of the same colour would turn it into a tom)
    for lane, ticks in tom_ticks.items():
        for t in sorted(ticks):
            track.add_note(t, short, TOM_MARKER[lane])
    for a, b in fills:
        for p in FILL:
            track.add_note(a, b - a + short, p)
    return track


def animate(track: Track, expert: list[Slot], ppq: int, bar: int) -> None:
    """Drum animation notes from Expert (simple sticking: right hand on hat / ride / crash, left on
    the snare under a cymbal, toms alternate), plus [idle] / [play] / [idle_realtime]."""
    short = ppq // 8
    right = True
    for s in expert:
        hands = s.hands()
        cym = any(g.kind in ("hat", "ride", "crash") for g in hands)
        for g in hands:
            if g.kind == "hat":
                track.add_note(s.tick, short, 31)
            elif g.kind == "ride":
                track.add_note(s.tick, short, 42)
            elif g.kind == "crash":
                track.add_note(s.tick, short, 38 if g.lane == GRN else 36)
            elif g.kind == "snare":
                track.add_note(s.tick, short, 26 if cym else 27)
            elif g.is_tom:
                base = {YEL: 46, BLU: 48, GRN: 50}[g.lane]
                track.add_note(s.tick, short, base + (1 if right else 0))
                right = not right
        if KICK in s.gems:
            track.add_note(s.tick, short, 24)
    track.remove(lambda t, m: m.type == "text" and m.text.strip() in ANIM_EVENTS)
    if not expert:
        return
    eighth = ppq // 2
    track.add_text(0, "[idle]")
    track.add_text(max(0, expert[0].tick - eighth), "[play]")
    for a, b in zip(expert, expert[1:]):
        if b.tick - a.tick >= 2 * bar:
            track.add_text(a.tick + eighth, "[idle]")
            track.add_text(b.tick - eighth, "[play]")
    track.add_text(expert[-1].tick + eighth, "[idle_realtime]")


def mix_events(track: Track, mix: str) -> None:
    if not mix:
        return
    track.remove(lambda t, m: m.type == "text" and m.text.strip().startswith("[mix "))
    for d in Difficulty:
        track.add_text(0, f"[mix {int(d)} {mix}]")


# --- the part ----------------------------------------------------------------------------------------
@dataclass
class Detected:
    on: object                     # audio.drums.DrumOnsets (times already on the chart's clock)
    P: dict                        # family -> probability per onset
    th: dict                       # family -> threshold used
    slots: dict                    # tick -> Slot (kick / snare / toms)
    cym_idx: dict                  # tick -> onset index of the cymbal hit there
    hat_ride: dict                 # tick -> hat-vs-ride log ratio (song-centred)
    review: list


@register
class ProDrums(PartCharter):
    key = "drums"
    track_names = (ins.DRUMS,)
    stem_keys = ("drums",)
    description = "Pro drums from the drum stem: kick/snare/cymbals/toms, tom markers, fills, 4 difficulties"

    def is_charted(self, track) -> bool:
        base = ins.DRUM_BASE[Difficulty.EXPERT]
        return bool(track.notes(range(base, base + 5)))

    def detect(self, ctx: PartContext, o: dict) -> "Detected | str":
        """Hits on the grid (kick / snare / toms placed; cymbals pending their type), or why not."""
        from ..audio import drums as dr
        keys = o["stems"] if isinstance(o["stems"], list) else [o["stems"]]
        paths = [ctx.stems[k] for k in keys if k in ctx.stems]
        if ctx.stem_override and ctx.stem_override in ctx.stems:
            paths = [ctx.stems[ctx.stem_override]]
        if not paths:
            return f"skipped: no stem among {keys}"
        tm, ppq = ctx.tempo, ctx.chart.ppq
        bar = tm.bar_ticks(tm.time_sig_at(0))

        on = dr.analyse(paths, ctx.cache_dir)
        on.t = on.t + ctx.audio_offset_s
        model = dr.DrumModel.train(cache_dir=ctx.cache_dir)
        P = model.proba(on.X)
        th = dr.thresholds(P, {**DEFAULT_THRESHOLDS, **(o["threshold"] or {})})
        hit = {fam: P[fam] >= th[fam] for fam in dr.FAMILIES}
        any_hit = np.any(np.stack(list(hit.values())), 0) if len(on.t) else np.zeros(0, bool)
        if not any_hit.any():
            return "no drum hits found"

        # --- grid
        ks = [t for t, k, s in zip(on.t, hit["kick"], hit["snare"]) if k or s]
        tom_opt = o["timing_offset_ms"]
        lean_at = feel_curve(tm, ks) if tom_opt == "auto" else (lambda t, v=float(tom_opt) / 1000: v)
        cands = (o["grid"],) + ((o["triplets"],) if o["triplets"] else ())
        grids = bar_grids(tm, list(on.t[any_hit]), lean_at, cands)
        review: list[ReviewItem] = []
        slots: dict[int, Slot] = {}
        cym_idx: dict[int, int] = {}
        tom_lane = tom_lanes(on, hit["tom"], o["toms"])
        ratio = np.log(on.act[:, 2] + 1e-3) - np.log(on.act[:, 3] + 1e-3)   # hat vs ride, per hit
        ratio = ratio - (np.median(ratio[hit["cymbal"]]) if hit["cymbal"].any() else 0)
        hat_ride: dict[int, float] = {}
        for i in np.flatnonzero(any_hit):
            t = float(on.t[i])
            lean = lean_at(t)
            grid = grids.get(int(tm.s2t(t - lean) // bar), o["grid"])
            sn = snap(tm, t - lean, grid, None)
            s = slots.setdefault(sn.tick, Slot(sn.tick, time_s=t, grid=sn.grid, error_ms=sn.error_ms))

            def put(lane, kind, p):
                g = s.gems.get(lane)
                if g is None or g.prob < p:
                    s.gems[lane] = Gem(lane, kind, float(p))
            if hit["kick"][i]:
                put(KICK, "kick", P["kick"][i])
            if hit["snare"][i]:
                put(RED, "snare", P["snare"][i])
            if hit["cymbal"][i]:
                if sn.tick not in cym_idx or P["cymbal"][i] > P["cymbal"][cym_idx[sn.tick]]:
                    cym_idx[sn.tick] = int(i)
                    hat_ride[sn.tick] = float(ratio[i])
            if hit["tom"][i]:
                put(int(tom_lane[i]), "tom", P["tom"][i])
            if sn.error_ms > 45:
                review.append(ReviewItem(t, self.key, f"hit {sn.error_ms:.0f} ms off the {sn.grid} grid"))
        return Detected(on, P, th, slots, cym_idx, hat_ride, review)

    def generate(self, ctx: PartContext) -> PartResult:
        o = {**DEFAULTS, **ctx.options}
        det = self.detect(ctx, o)
        if isinstance(det, str):
            return PartResult([], summary=det)
        on, P, th, slots, cym_idx, hat_ride, review = (det.on, det.P, det.th, det.slots, det.cym_idx,
                                                       det.hat_ride, det.review)
        tm, ppq = ctx.tempo, ctx.chart.ppq
        bar = tm.bar_ticks(tm.time_sig_at(0))
        ordered = [slots[t] for t in sorted(slots)]

        # --- cymbals: crashes, then hat / ride by section
        crashes = crash_flags(ordered, cym_idx, on, ppq, bar, float(o["crash_score"]))
        timekeeping = [(t, hat_ride[t]) for t in cym_idx if t not in crashes]
        ride_bars = ride_sections(timekeeping, bar) if o["ride"] == "auto" else set()
        for t, i in cym_idx.items():
            kind = "crash" if t in crashes else "ride" if t // bar in ride_bars else "hat"
            lane = CYM_KINDS[kind]
            s = slots[t]
            if lane in s.gems and s.gems[lane].is_tom:  # tom on the same colour: move it to a free tom
                tom = s.gems.pop(lane)
                free = [l for l in (YEL, BLU, GRN) if l not in s.gems and l != lane]
                if free:
                    s.gems[free[0]] = Gem(free[0], "tom", tom.prob)
            s.gems[lane] = Gem(lane, kind, float(P["cymbal"][i]))

        # --- expert limits
        rows = []
        dropped = 0
        for s in ordered:
            notes = playable(s)
            dropped += len(notes)
        n2x = mark_2x(ordered, tm, float(o["kick_2x_ms"]))
        ordered = [s for s in ordered if s.gems]
        for a, b in hands_only_stretches(ordered, bar):
            review.append(ReviewItem(tm.t2s(a * bar), self.key,
                                     f"bars {a + 1}-{b + 1}: cymbals/toms but no kick or snare; real (intro, "
                                     "percussion) or another instrument bleeding into the drum stem?"))
        fills = find_fills(ordered, ppq, bar) if o["fills"] else []
        bpm = 60_000_000 / tm.tempos[-1].us_per_beat
        by_diff = {d: reduce(ordered, d, ppq, bar, bpm, fills) for d in reversed(Difficulty)}

        track = write(ctx.existing(ins.DRUMS), by_diff, fills, ppq, o)
        if o["animation"]:
            animate(track, by_diff[Difficulty.EXPERT], ppq, bar)
        mix_events(track, o["mix"])

        on_ticks = {d: {s.tick: s for s in by_diff[d]} for d in Difficulty}
        for s in ordered:
            rows.append({"time_s": round(s.time_s, 3), "tick": s.tick, "grid": s.grid,
                         "grid_error_ms": round(s.error_ms, 1),
                         "expert": "".join(LANE_NAMES[l] + ("2" if g.double else "") for l, g in sorted(s.gems.items())),
                         "kinds": " ".join(g.kind for _, g in sorted(s.gems.items())),
                         "probs": " ".join(f"{g.prob:.2f}" for _, g in sorted(s.gems.items())),
                         **{d.name.lower(): "".join(LANE_NAMES[l] for l in sorted(on_ticks[d][s.tick].gems))
                            if s.tick in on_ticks[d] else "" for d in (Difficulty.HARD, Difficulty.MEDIUM, Difficulty.EASY)}})
        count = lambda d, f: sum(1 for s in by_diff[d] for g in s.gems.values() if f(g))  # noqa: E731
        kinds = {k: count(Difficulty.EXPERT, lambda g, k=k: g.kind == k) for k in ("kick", "snare", "hat", "ride", "crash", "tom")}
        counts = " ".join(f"{d.name[0]}{sum(len(s.gems) for s in by_diff[d])}" for d in reversed(Difficulty))
        thr = " ".join(f"{k}>{v:.2f}" for k, v in th.items())
        summary = (f"{counts} gems | X: " + " ".join(f"{v} {k}" for k, v in kinds.items())
                   + f" | {n2x} 2x kicks | {len(fills)} fills | {len(ride_bars)} ride bars | {dropped} 3rd-hand drops"
                   + f" | thresholds {thr} | {len(review)} flagged")
        return PartResult([track], review, rows, summary)
