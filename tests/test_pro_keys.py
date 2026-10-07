"""Pro keys charting rules (C3 Customs Book), driven by MIDI input so no transcription is needed."""
import mido
import numpy as np
import pytest

from yarginator.core.chart import Chart
from yarginator.core.instruments import PRO_KEYS, Difficulty
from yarginator.core.timing import TempoMap
from yarginator.parts import get_charter
from yarginator.parts import pro_keys as pk
from yarginator.parts.base import PartContext

PPQ = 480


def test_right_hand_drops_left_hand_notes():
    # chord C5 E5 G5 with a C3 bass note under it, then a lone C3 between chords
    notes = np.array([[0.0, 0.4, 72, .8], [0.0, 0.4, 76, .8], [0.0, 0.4, 79, .8], [0.0, 0.4, 48, .8],
                      [0.5, 0.9, 48, .8],
                      [1.0, 1.4, 74, .8], [1.0, 1.4, 77, .8]])
    kept, dropped = pk.right_hand(notes, cluster_s=0.04, span=12, register_s=4.0)
    assert sorted(kept[:, 2].astype(int)) == [72, 74, 76, 77, 79] and dropped == 2


def test_octave_fit_and_wrapping():
    assert pk.best_octave_shift([84, 86, 88, 91]) == -24 or pk.best_octave_shift([84, 86, 88, 91]) == -12
    assert all(48 <= p - 12 <= 72 for p in [74, 76, 79])
    assert pk.wrap_into(66, 48, 64) == 54 and pk.wrap_into(45, 53, 69) == 57


def test_chord_limits_keep_the_top():
    assert pk.limit_chord([72, 67, 64, 60, 55], 4, 12) == [72, 67, 64, 60]
    assert pk.limit_chord([72, 67, 64, 60], 3, 11) == [72, 67, 64]
    assert pk.limit_chord([72, 67, 64], 2, 9) == [72, 67]
    assert pk.limit_chord([72, 67, 64], 1, 0) == [72]
    assert pk.limit_chord([72, 58], 4, 12) == [72]          # more than an octave under the top: dropped


def test_range_plan_prefers_primary_ranges_and_few_shifts():
    low = [[50, 53, 55]] * 4          # fits C2-E3
    high = [[67, 69, 71, 72]] * 4     # fits A2-C4 only
    plan = pk.plan_ranges(low + high, shift_cost=6, nonprimary_cost=1, outside_cost=1)
    assert plan[:4] == [0] * 4 and plan[4:] == [9] * 4
    # a single high bar isn't worth two shifts: it stays and gets wrapped
    plan = pk.plan_ranges(low + [[66]] + low, shift_cost=6, nonprimary_cost=1, outside_cost=1)
    assert len(set(plan)) == 1


def test_shift_goes_a_bar_early_but_not_before_notes_it_cant_show():
    chords = [pk.Chord(0, [50], [240]), pk.Chord(4 * PPQ, [52], [4 * PPQ + 240]),
              pk.Chord(7 * PPQ, [50], [7 * PPQ + 240]),    # bar 2, beat 4: still needs C2
              pk.Chord(8 * PPQ, [70], [8 * PPQ + 240])]    # bar 3: needs A2-C4
    shifts = pk.place_shifts(chords, [0, 0, 9], lambda t: t // (4 * PPQ), 4 * PPQ, 1.0, PPQ)
    assert shifts[0] == (0, 0)
    tick, r = shifts[1]
    assert r == 9 and 7 * PPQ < tick <= 8 * PPQ               # after the last C2 note, before the A2-C4 note


def test_thin_prefers_strong_beats():
    chords = [pk.Chord(t, [60], [t + 60]) for t in range(0, 4 * PPQ, PPQ // 4)]   # a bar of 16ths
    assert [c.tick for c in pk.thin(chords, PPQ, PPQ)] == [0, PPQ, 2 * PPQ, 3 * PPQ]
    assert [c.tick for c in pk.thin(chords, 2 * PPQ, PPQ)] == [0, 2 * PPQ]


def write_keys_midi(path, notes, bpm=120):
    """notes: (beat, length_beats, pitch) in a one-track MIDI at a fixed tempo."""
    mf = mido.MidiFile(ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    ev = []
    for b, ln, p in notes:
        ev += [(int(b * PPQ), mido.Message("note_on", note=p, velocity=100)),
               (int((b + ln) * PPQ), mido.Message("note_off", note=p, velocity=0))]
    last = 0
    for t, m in sorted(ev, key=lambda e: (e[0], e[1].type == "note_on")):
        tr.append(m.copy(time=t - last))
        last = t
    mf.tracks.append(tr)
    mf.save(path)


def test_pro_keys_from_midi_end_to_end(tmp_path):
    # right hand: C major / F major chords in the C5 octave, a 16th run, a high run that needs another range;
    # left hand: low C2 octave bass notes
    rh = []
    for bar in range(4):
        rh += [(bar * 4 + b, 1.0, p) for b in (0, 2) for p in ((72, 76, 79) if bar % 2 == 0 else (77, 81, 84))]
    rh += [(16 + k * 0.25, 0.25, 72 + (k % 5)) for k in range(16)]            # bar 5: 16th run
    rh += [(24 + k, 1.0, 88 + (k % 3)) for k in range(8)]                    # bars 7-8: high
    lh = [(b, 2.0, 36) for b in range(0, 32, 2)]
    write_keys_midi(tmp_path / "keys.mid", rh + lh)

    chart = Chart(TempoMap.constant(120))
    ctx = PartContext(chart, {}, options={"midi": "keys.mid", "timing_offset_ms": 0}, root=tmp_path)
    res = get_charter("prokeys").generate(ctx)
    tracks = {t.name: t for t in res.tracks}
    assert set(PRO_KEYS.values()) <= set(tracks) and "PART KEYS_ANIM_RH" in tracks

    def notes(d):
        return [n for n in tracks[PRO_KEYS[d]].notes() if n.pitch >= 48]

    def shifts(d):
        return [(n.tick, n.pitch) for n in tracks[PRO_KEYS[d]].notes() if n.pitch <= 9]

    x, h, m, e = (notes(d) for d in (Difficulty.EXPERT, Difficulty.HARD, Difficulty.MEDIUM, Difficulty.EASY))
    assert all(48 <= n.pitch <= 72 for n in x + h + m + e)
    assert 36 not in {n.pitch for n in x} and not any(n.pitch < 48 for n in x)  # left hand gone
    # every difficulty starts with a range shift; medium/easy have only that one
    for d in Difficulty:
        assert shifts(d) and shifts(d)[0][0] == 0
    assert len(shifts(Difficulty.MEDIUM)) == 1 and len(shifts(Difficulty.EASY)) == 1
    # every note is inside the range in force at its tick
    for d in Difficulty:
        sh = shifts(d)
        for n in notes(d):
            r = [p for t, p in sh if t <= n.tick][-1]
            lo, hi = pk.RANGES[r]
            assert lo <= n.pitch <= hi, (d, n)

    def chord_sizes(ns):
        by = {}
        for n in ns:
            by.setdefault(n.tick, []).append(n.pitch)
        return by
    assert max(len(v) for v in chord_sizes(x).values()) <= 4
    assert max(len(v) for v in chord_sizes(h).values()) <= 3
    assert max(len(v) for v in chord_sizes(m).values()) <= 2
    assert max(len(v) for v in chord_sizes(e).values()) == 1
    ticks = sorted(chord_sizes(m))
    assert all(b - a >= PPQ for a, b in zip(ticks, ticks[1:]))              # medium: a quarter apart
    ticks = sorted(chord_sizes(e))
    assert all(b - a >= 2 * PPQ for a, b in zip(ticks, ticks[1:]))          # easy: a half note apart
    assert len(x) > len(h) > len(m) > len(e)
    # animation
    texts = [t for _, t in tracks[PRO_KEYS[Difficulty.EXPERT]].texts()]
    assert texts[0] == "[idle]" and "[play]" in texts and texts[-1] == "[idle_realtime]"


def test_missing_midi_is_reported(tmp_path):
    ctx = PartContext(Chart(TempoMap.constant(120)), {}, options={"midi": "nope.mid"}, root=tmp_path)
    res = get_charter("prokeys").generate(ctx)
    assert res.tracks == [] and "not found" in res.summary
