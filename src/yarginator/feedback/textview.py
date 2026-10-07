"""Text view of a chart part, one row per bar: quick feedback without opening REAPER.

    bar  17 |G==. G==. G==. G==.|      16th-note columns, grouped by beat (row marked 3: triplet columns)
             G R Y B O = frets, '=' sustain, '.' nothing; several frets in one column = a chord
    drums:   K R Y B G = kick, snare, yellow, blue, green cymbals; y b g = toms; k = 2x kick
"""
from __future__ import annotations

from ..core.chart import Chart
from ..core.instruments import DRUM_BASE, DRUM_KICK_2X, DRUM_TOM_MARKER, FIVE_LANE_BASE, Difficulty

FIVE = "GRYBO"
DRUMS = "KRYBG"  # kick, red, yellow, blue, green


def view_keys(chart: Chart, track_name: str, bars: tuple[int, int]) -> str:
    """Pro keys, one row per bar: ``beat:notes`` (Rock Band names, C2 = MIDI 48; '~' = sustained)
    with range shifts shown as ``>>A2-C4``."""
    from ..parts.pro_keys import RANGES, rb_name
    tr = chart.tracks.get(track_name)
    if tr is None:
        return f"(no {track_name} track)"
    tm, ppq = chart.tempo, chart.ppq
    downbeats = tm.downbeats(max(chart.end_tick(), 1) + 8 * ppq)
    notes = tr.notes()
    lines = []
    for m in range(bars[0], min(bars[1], len(downbeats)) + 1):
        start = downbeats[m - 1]
        end = downbeats[m] if m < len(downbeats) else start + tm.bar_ticks(tm.time_sig_at(start))
        cells: dict[int, list[str]] = {}
        for n in notes:
            if start <= n.tick < end:
                if n.pitch in RANGES:
                    lo, hi = RANGES[n.pitch]
                    cells.setdefault(n.tick, []).insert(0, f">>{rb_name(lo)}-{rb_name(hi)}")
                elif n.pitch >= 48:
                    cells.setdefault(n.tick, []).append(rb_name(n.pitch) + ("~" if n.length > ppq // 4 else ""))
        parts = []
        for tick in sorted(cells):
            beat = (tick - start) / ppq + 1
            label = f"{beat:g}" if beat == int(beat) else f"{beat:.2f}".rstrip("0")
            parts.append(f"{label}:{'+'.join(cells[tick])}")
        lines.append(f"bar {m:3d} | " + ("  ".join(parts) if parts else "-"))
    return "\n".join(lines)


def view(chart: Chart, track_name: str, difficulty: Difficulty, bars: tuple[int, int],
         columns_per_beat: int | None = None) -> str:
    """``columns_per_beat`` None picks per bar: 4 (16ths) when every note fits, 3 (8th triplets,
    row marked ``3``) when they fit that instead, else 12."""
    if track_name.upper().startswith("PART REAL_KEYS"):
        from ..core.instruments import PRO_KEYS
        return view_keys(chart, PRO_KEYS[difficulty], bars)
    tr = chart.tracks.get(track_name)
    if tr is None:
        return f"(no {track_name} track)"
    drums = "DRUMS" in track_name
    letters, base = (DRUMS, DRUM_BASE[difficulty]) if drums else (FIVE, FIVE_LANE_BASE[difficulty])
    notes = tr.notes(range(base, base + 5))
    toms: dict[int, list[tuple[int, int]]] = {}
    if drums:  # toms lower-case (y/b/g), 2x kicks as 'k'
        toms = {lane: [(n.tick, n.end) for n in tr.notes([marker])]
                for lane, marker in ((2, DRUM_TOM_MARKER["yellow"]), (3, DRUM_TOM_MARKER["blue"]), (4, DRUM_TOM_MARKER["green"]))}
        if difficulty == Difficulty.EXPERT:
            notes = sorted(notes + [n.__class__(n.tick, n.end, base - 1) for n in tr.notes([DRUM_KICK_2X])],
                           key=lambda n: n.tick)
    tm, ppq = chart.tempo, chart.ppq
    downbeats = tm.downbeats(max(chart.end_tick(), 1) + 8 * ppq)
    lines = []
    for m in range(bars[0], bars[1] + 1):
        if m - 1 >= len(downbeats):
            break
        start = downbeats[m - 1]
        end = downbeats[m] if m < len(downbeats) else start + tm.bar_ticks(tm.time_sig_at(start))
        cols = columns_per_beat
        if cols is None:
            offs = [(n.tick - start) % ppq for n in notes if start <= n.tick < end]
            cols = next((c for c in (4, 3, 12) if all(o * c % ppq == 0 for o in offs)), 12)
        step = ppq / cols
        n_cols = int(round((end - start) / step))
        cells = [""] * n_cols
        sus = [False] * n_cols
        for n in notes:
            if start <= n.tick < end:
                col = int((n.tick - start) // step)
                lane = n.pitch - base
                ch = "k" if lane < 0 else letters[lane]
                if lane in toms and any(a <= n.tick < b for a, b in toms[lane]):
                    ch = ch.lower()
                cells[col] += ch
                if n.length > ppq // 4 and not drums:
                    for k in range(col + 1, min(n_cols, col + int(n.length // step))):
                        sus[k] = True
        out = []
        order = "Kk" + letters[1:] + letters[1:].lower() if drums else letters
        for k in range(n_cols):
            c = "".join(ch for ch in order if ch in cells[k])
            out.append(c if len(c) == 1 else (f"[{c}]" if c else ("=" if sus[k] else ".")))
        groups = ["".join(out[i:i + cols]) for i in range(0, n_cols, cols)]
        mark = {4: " ", 3: "3"}.get(cols, "*")
        lines.append(f"bar {m:3d}{mark}|{' '.join(groups)}|")
    return "\n".join(lines)
