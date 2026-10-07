"""Pro drums: the charting rules on hand-built slots, plus a smoke test on a synthetic kit."""
import struct

import numpy as np
import pytest

from yarginator.core import instruments as ins
from yarginator.core.chart import Chart
from yarginator.core.instruments import DRUM_BASE, Difficulty
from yarginator.core.timing import TempoMap
from yarginator.parts.drums import (BLU, GRN, KICK, RED, YEL, Gem, Slot, expert_hits, find_fills, mark_2x,
                                    playable, reduce, tom_lanes, write)

PPQ = 480
BAR = 4 * PPQ
OPTS = {"animation": True}


def slot(tick, *gems):
    s = Slot(tick)
    for lane, kind in gems:
        s.gems[lane] = Gem(lane, kind, 0.9)
    return s


def rock_bars(n_bars=2, start=BAR):
    """8th hats, snare on 2 & 4, kicks on 1, the 'and' of 2 and a 16th pickup into 3."""
    out = []
    for b in range(n_bars):
        t0 = start + b * BAR
        for e in range(8):
            t = t0 + e * PPQ // 2
            gems = [(YEL, "hat")]
            if e in (2, 6):
                gems.append((RED, "snare"))
            if e in (0, 3):
                gems.append((KICK, "kick"))
            out.append(slot(t, *gems))
        out.append(slot(t0 + 2 * PPQ - PPQ // 4, (KICK, "kick")))       # 16th before beat 3
    return sorted(out, key=lambda s: s.tick)


def gems_at(slots):
    return {s.tick: set(s.gems) for s in slots}


def test_playable_keeps_crash_and_snare():
    s = slot(0, (RED, "snare"), (YEL, "hat"), (GRN, "crash"), (KICK, "kick"))
    playable(s)
    assert set(s.gems) == {KICK, RED, GRN}


def test_reductions_follow_the_book():
    expert = rock_bars() + [slot(3 * BAR, (GRN, "crash"), (KICK, "kick"))]
    expert = sorted(expert, key=lambda s: s.tick)
    bpm = 120
    hard = gems_at(reduce(expert, Difficulty.HARD, PPQ, BAR, bpm, []))
    medium = reduce(expert, Difficulty.MEDIUM, PPQ, BAR, bpm, [])
    easy = reduce(expert, Difficulty.EASY, PPQ, BAR, bpm, [])
    # hard: the 16th pickup kick next to the 8th kick goes; the kicks on 8ths stay
    pickup = BAR + 2 * PPQ - PPQ // 4
    assert KICK not in hard.get(pickup, set())
    assert KICK in hard[BAR] and KICK in hard[BAR + 3 * PPQ // 2]
    # medium (120 BPM): kicks only on quarter notes, two gems at most, one hand with a kick
    for s in medium:
        assert len(s.gems) <= 2
        if KICK in s.gems:
            assert s.tick % PPQ == 0 and len(s.hands()) <= 1
    # easy: a kick never shares a tick; hands on quarters at 120 BPM
    for s in easy:
        if KICK in s.gems:
            assert len(s.gems) == 1
        assert s.tick % PPQ == 0
    # the crash survives everywhere, on green
    for d in Difficulty:
        r = gems_at(reduce(expert, d, PPQ, BAR, bpm, []))
        assert GRN in r[3 * BAR]


def test_kicks_in_fills_and_2x_kicks_go_below_expert():
    expert = [slot(BAR + k * PPQ // 4, (YEL, "tom"), (KICK, "kick")) for k in range(8)]
    expert.append(slot(BAR + 2 * PPQ, (GRN, "crash"), (KICK, "kick")))
    expert.sort(key=lambda s: s.tick)
    fills = find_fills(expert, PPQ, BAR)
    assert fills == [(BAR, BAR + 2 * PPQ)]
    hard = reduce(expert, Difficulty.HARD, PPQ, BAR, 120, fills)
    assert not any(KICK in s.gems for s in hard if s.tick < BAR + 2 * PPQ)
    tm = TempoMap.constant(200)
    fast = [slot(BAR + k * PPQ // 4, (KICK, "kick")) for k in range(8)]   # 16ths at 200 BPM = 75 ms
    assert mark_2x(fast, tm, 85) == 4                                       # every other one
    assert all(KICK not in s.gems or not s.gems[KICK].double
               for s in reduce(fast, Difficulty.EXPERT, PPQ, BAR, 200, []))  # expert keeps them (as 2x)
    assert sum(1 for s in reduce(fast, Difficulty.HARD, PPQ, BAR, 200, []) if KICK in s.gems) <= 4


def test_write_marks_only_toms_and_reads_back():
    tm = TempoMap.constant(120)
    expert = [slot(BAR, (YEL, "hat"), (KICK, "kick")),
              slot(BAR + PPQ, (YEL, "tom")),
              slot(BAR + 2 * PPQ, (BLU, "tom"), (GRN, "crash"))]
    by_diff = {d: [s.copy() for s in expert] for d in Difficulty}
    tr = write(Chart(tm).track(ins.DRUMS), by_diff, [], PPQ, OPTS)
    chart = Chart(tm)
    chart.tracks[ins.DRUMS] = tr
    hits = sorted((tm.s2t(t), c) for t, c in expert_hits(chart))
    assert hits == [(BAR, 0), (BAR, 2), (BAR + PPQ, 5), (BAR + 2 * PPQ, 4), (BAR + 2 * PPQ, 6)]  # tom marker only on toms
    assert len(tr.notes([DRUM_BASE[Difficulty.EASY] + YEL])) == 2


def test_tom_colours_by_pitch():
    class On:
        t = np.zeros(9)
        pitch = np.array([220, 230, 210, 140, 150, 145, 90, 95, 85], float)
    lanes = tom_lanes(On, np.ones(9, bool), "pitch")
    assert list(lanes) == [YEL] * 3 + [BLU] * 3 + [GRN] * 3


def test_chart_load_clips_bad_data_bytes(tmp_path):
    # a type-0 file with a note-on whose velocity byte is 200 (> 127)
    trk = bytes([0x00, 0xFF, 0x03, 0x04]) + b"TEST" + bytes([0x00, 0x90, 60, 200, 0x60, 0x80, 60, 0, 0x00, 0xFF, 0x2F, 0x00])
    data = b"MThd" + struct.pack(">IHHH", 6, 1, 2, 480)
    data += b"MTrk" + struct.pack(">I", 4) + bytes([0x00, 0xFF, 0x2F, 0x00])
    data += b"MTrk" + struct.pack(">I", len(trk)) + trk
    p = tmp_path / "bad.mid"
    p.write_bytes(data)
    chart = Chart.load(p)
    assert chart.tracks["TEST"].notes()[0].velocity == 127


def test_text_view_shows_toms_lower_case():
    from yarginator.feedback.textview import view
    tm = TempoMap.constant(120)
    expert = [slot(0, (YEL, "hat"), (KICK, "kick")), slot(PPQ, (YEL, "tom"))]
    chart = Chart(tm)
    chart.tracks[ins.DRUMS] = write(Chart(tm).track(ins.DRUMS), {d: [s.copy() for s in expert] for d in Difficulty},
                                    [], PPQ, OPTS)
    assert view(chart, ins.DRUMS, Difficulty.EXPERT, (1, 1)).startswith("bar   1 |[KY]... y...")


# --- end to end on a synthetic kit (needs the audio extra) ---------------------------------------------
def test_synthetic_groove_end_to_end(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("sklearn")
    sf = pytest.importorskip("soundfile")
    from drumkit import groove
    from yarginator.parts import get_charter
    from yarginator.parts.base import PartContext
    y, sr, truth = groove(bpm=120, start=2.0, bars=6)
    sf.write(tmp_path / "drums.wav", y, sr)
    tm = TempoMap.click_tracked(120, 2.0)
    chart = Chart(tm)
    res = get_charter("drums").generate(PartContext(chart, stems={"drums": tmp_path / "drums.wav"},
                                                    cache_dir=tmp_path / "cache"))
    tr = res.tracks[0]
    x = DRUM_BASE[Difficulty.EXPERT]
    kicks = {n.tick for n in tr.notes([x + KICK])}
    want = {tm.s2t(t) for t, k in truth if k == "kick"}
    assert len(kicks & want) >= 0.8 * len(want) and len(kicks - want) <= 2
    last = tm.s2t(max(t for t, k in truth if k == "crash"))
    assert last in {n.tick for n in tr.notes([x + GRN])}                       # crash on green ...
    assert last in kicks                                                         # ... with its kick
    tom_ticks = {n.tick for p in (110, 111, 112) for n in tr.notes([p])}
    gem_ticks = {n.tick for n in tr.notes(range(60, 101))}
    assert tom_ticks <= gem_ticks
    assert tr.notes([120]), "the tom fill into the crash should be a drum fill"
    assert {txt for _, txt in tr.texts()} >= {"[mix 0 drums0]", "[mix 3 drums0]", "[idle]", "[play]"}
    for d in Difficulty:  # easy never pairs a kick
        if d == Difficulty.EASY:
            b = DRUM_BASE[d]
            k = {n.tick for n in tr.notes([b])}
            others = {n.tick for n in tr.notes(range(b + 1, b + 5))}
            assert not k & others
