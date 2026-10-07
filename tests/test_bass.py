"""5-lane charting rules and bass transcription."""
import numpy as np
import pytest

from yarginator.core.instruments import FIVE_LANE_BASE, Difficulty
from yarginator.core.timing import TempoMap
from yarginator.parts import five_lane as fl
from yarginator.parts.quantize import snap

PPQ = 480


def test_same_pitch_keeps_its_fret_and_contour_is_kept():
    # bar 1: E E E E ; bar 2-3 (same 8-beat window): A C D C A
    ticks = [0, 480, 960, 1440] + [3840 + k * 480 for k in range(5)]
    pitches = [28, 28, 28, 28, 33, 36, 38, 36, 33]
    lanes = fl.assign_lanes(ticks, pitches, PPQ, window_beats=8)
    assert lanes[:4] == [0, 0, 0, 0]
    assert lanes[4:] == [0, 1, 2, 1, 0]


def test_window_boundary_does_not_split_a_repeated_pitch():
    # end of one window on F, next window starts with four Es: all four Es share a fret
    ticks = [3600] + [3840 + k * 480 for k in range(4)]
    lanes = fl.assign_lanes(ticks, [29, 28, 28, 28, 28], PPQ, window_beats=8)
    assert len(set(lanes[1:])) == 1


def test_more_than_five_pitches_compress_in_order():
    ticks = [k * 240 for k in range(8)]
    lanes = fl.assign_lanes(ticks, [40 + k for k in range(8)], PPQ, window_beats=8)
    assert lanes == sorted(lanes) and lanes[0] == 0 and lanes[-1] == 4


def test_ghosts_sit_on_previous_fret_and_never_sustain():
    notes = [fl.LaneNote(0, 960, 2, 40.0), fl.LaneNote(960, 1440, 2, None, ghost=True), fl.LaneNote(1920, 1960, 1, 38.0)]
    out = fl.add_sustains(notes, PPQ, min_beats=0.75, gap_beats=0.25)
    assert out[0].sustain > 0 and out[1].sustain == 0 and out[2].sustain == 0
    assert fl._fill([40.0, None, 38.0]) == [40.0, 40.0, 38.0]


def test_reductions():
    # a bar of 16ths, pitched, alternating frets
    notes = [fl.LaneNote(k * 120, k * 120 + 60, k % 5, 40.0 + k % 5) for k in range(16)]
    notes[3].ghost = True
    exp = fl.reduce(notes, Difficulty.EXPERT, PPQ)
    hard = fl.reduce(notes, Difficulty.HARD, PPQ)
    med = fl.reduce(notes, Difficulty.MEDIUM, PPQ)
    easy = fl.reduce(notes, Difficulty.EASY, PPQ)
    assert len(exp) == 16 and len(hard) == 8 and len(med) == 8 and len(easy) == 4
    assert all(n.tick % 240 == 0 for n in med) and all(n.tick % 480 == 0 for n in easy)
    assert max(n.lane for n in med) <= 3 and max(n.lane for n in easy) <= 2
    assert all(not n.ghost for n in hard)


def test_write_uses_difficulty_bases():
    from yarginator.core.chart import Track
    tr = fl.write(Track("PART BASS"), {d: [fl.LaneNote(0, 60, 1, 40.0)] for d in Difficulty}, PPQ)
    assert sorted(n.pitch for n in tr.notes()) == sorted(FIVE_LANE_BASE[d] + 1 for d in Difficulty)


def test_feel_offset_recovers_a_laid_back_player():
    from yarginator.parts.quantize import feel_offset
    tm = TempoMap.constant(123)
    s16 = 60 / 123 / 4
    rng = np.random.default_rng(0)
    # 16th positions, played ~45 ms behind with +-15 ms of human jitter: raw snapping would send many
    # of them to the *next* 16th (only 122 ms apart)
    times = [k * s16 + 0.045 + rng.normal(0, 0.015) for k in range(8, 200, 3)]
    lean = feel_offset(tm, times)
    assert lean == pytest.approx(0.045, abs=0.006)
    raw_wrong = sum(snap(tm, t).tick != tm.s2t(round((t - 0.045) / s16) * s16) for t in times)
    fixed_wrong = sum(snap(tm, t - lean).tick != tm.s2t(round((t - 0.045) / s16) * s16) for t in times)
    assert fixed_wrong < raw_wrong and fixed_wrong <= 1


def test_feel_curve_follows_a_lag_that_changes_between_sections():
    """Bars 1-16 played 15 ms late, bars 17-32 75 ms late (a different take with more latency).
    75 ms is past half a 16th (61 ms): only the metrical preference (8ths/beats over 'e'/'a')
    keeps those notes from snapping a 16th early."""
    from yarginator.parts.quantize import feel_curve
    tm = TempoMap.constant(123)
    s8 = 60 / 123 / 2
    rng = np.random.default_rng(2)
    intended, played = [], []
    for k in range(0, 4 * 2 * 32, 1):           # 8ths through 32 bars, skip some for groove
        if k % 3 == 2:
            continue
        t = k * s8
        lag = 0.015 if t < 16 * 4 * 2 * s8 else 0.075
        intended.append(t)
        played.append(t + lag + rng.normal(0, 0.008))
    lean_at = feel_curve(tm, played)
    assert lean_at(5.0) == pytest.approx(0.015, abs=0.01)
    assert lean_at(50.0) == pytest.approx(0.075, abs=0.01)
    wrong = sum(snap(tm, p - lean_at(p)).tick != tm.s2t(t) for p, t in zip(played, intended))
    assert wrong <= 3  # only around the section change


def test_slap_octave_flips_still_give_a_pitch():
    from yarginator.parts.note_events import steady_pitch_class
    guesses = np.array([26.1, 38.0, 38.1, 26.0, 38.2, 37.9, 38.1])   # D1/D2 flipping (slap/pop)
    assert round(steady_pitch_class(guesses)) == 38                   # D, in the more common octave
    assert steady_pitch_class(np.array([26.0, 31.0, 35.5, 29.0])) is None  # no agreement
    # a short fifth with the previous root still ringing in its first frames: the fifth wins
    assert round(steady_pitch_class(np.array([34.0, 34.1, 41.0, 41.1, 40.9, 41.0, 41.2]))) == 41


def test_note_names():
    from yarginator.parts.bass import midi_name, to_midi
    assert [to_midi(x) for x in ("E1", "D1", "B0", "A0", "Bb0", "C5", 31, "auto")] == [28, 26, 23, 21, 22, 72, 31, None]
    assert midi_name(23) == "B0" and midi_name(26.2) == "D1"


def test_lowest_note_is_read_from_the_stem():
    """A 5-string part that confidently plays low B: B0 is real, nothing folds it up an octave."""
    from yarginator.audio.analysis import PitchFeatures
    from yarginator.parts.note_events import _fold, estimate_lowest
    n = 400
    midi = np.where(np.arange(n) % 4 == 0, 23.0, 35.0)       # B0 and B1, confidently
    f = PitchFeatures(np.arange(n) * 0.01, midi, np.full(n, 0.95), np.full(n, -6.0), np.zeros(n))
    lowest = estimate_lowest(f)
    assert 20 <= lowest <= 23
    assert _fold(23.0, lowest, 72) == 23.0                    # low B stays low
    assert _fold(16.0, lowest, 72) == 28.0                    # an E0 "guess" is an octave error
    assert _fold(71.5, lowest, 72) is None                    # top-of-range guess = noise
    # a part that only confidently plays high notes still gets a standard 4-string floor (E1)
    f_hi = PitchFeatures(np.arange(n) * 0.01, np.full(n, 40.0), np.full(n, 0.95), np.full(n, -6.0), np.zeros(n))
    assert estimate_lowest(f_hi) == 28.0


def test_search_range_widens_only_when_notes_crowd_an_edge():
    from yarginator.audio.analysis import PitchFeatures
    from yarginator.parts.note_events import crowds_edge
    n = 400
    t, conf, db, z = np.arange(n) * 0.01, np.full(n, 0.95), np.full(n, -6.0), np.zeros(n)
    normal = PitchFeatures(t, np.where(np.arange(n) % 3, 33.0, 40.0), conf, db, z)
    drop_a = PitchFeatures(t, np.where(np.arange(n) % 3, 23.2, 33.0), conf, db, z)   # piles up at B0
    assert not crowds_edge(normal, 23, "low") and not crowds_edge(normal, 67, "high")
    assert crowds_edge(drop_a, 23, "low")


def test_only_quiet_unpitched_hits_on_the_kick_are_bleed():
    from yarginator.parts.note_events import GHOST, NoteEvent, mark_bleed
    loud = NoteEvent(1.00, 1.1, None, 0.0, level=-3.0, kind=GHOST)    # slap note on the kick
    quiet = NoteEvent(2.00, 2.1, None, 0.0, level=-20.0, kind=GHOST)  # kick thump leaking in
    assert mark_bleed([loud, quiet], np.array([1.01, 2.01]), 0.03, -10.0) == 1
    assert loud.kind == GHOST and quiet.kind == "bleed"


def test_snap_prefers_straight_unless_triplet_clearly_closer():
    tm = TempoMap.constant(120)  # 0.5 s per beat
    assert snap(tm, 0.130).grid == "16th"            # 5 ms from a 16th
    s = snap(tm, 1.0 + 0.5 / 3)                      # exactly an 8th-triplet
    assert s.grid == "8th-triplet" and s.error_ms < 1
    assert snap(tm, 1.0 + 0.5 / 3, triplet=None).grid == "16th"


# --- transcription on a synthetic funk-ish bass line ---------------------------------------------
librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")
SR = 22050


def bass_line(path):
    """120 BPM: E1 (beat 1, held), A1 (beat 3), a ghost thump on the 'e' of 4, G1 on the next 1."""
    y = np.zeros(int(SR * 4.0))

    def note(t0, midi, dur):
        t = np.arange(int(SR * dur)) / SR
        f = 440 * 2 ** ((midi - 69) / 12)
        sig = sum(0.6 ** h * np.sin(2 * np.pi * f * (h + 1) * t) for h in range(5))
        env = np.minimum(1, t / 0.005) * np.exp(-t * 1.5)
        i = int(t0 * SR)
        y[i:i + len(t)] += 0.4 * sig * env
    note(0.5, 28, 0.9)
    note(1.5, 33, 0.45)
    rng = np.random.default_rng(1)
    i = int(2.125 * SR)
    thump = rng.normal(size=int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 60)
    y[i:i + len(thump)] += 0.3 * np.convolve(thump, np.ones(30) / 30, mode="same")
    note(2.5, 31, 0.9)
    sf.write(path, y, SR)


def test_bass_chart_from_stem(tmp_path):
    from yarginator.core.chart import Chart
    from yarginator.parts import get_charter
    from yarginator.parts.base import PartContext
    bass_line(tmp_path / "bass.wav")
    chart = Chart(TempoMap.constant(120))
    res = get_charter("bass").generate(PartContext(chart, {"bass": tmp_path / "bass.wav"}, cache_dir=tmp_path / "c"))
    rows = res.rows
    pitched = [(round(float(r["time_s"]) * 2) / 2, round(float(r["midi_pitch"]))) for r in rows if r["kind"] == "pitched"]
    assert pitched == [(0.5, 28), (1.5, 33), (2.5, 31)]
    assert [round(float(r["time_s"]), 2) for r in rows if r["kind"] == "ghost"] == [2.13]  # the thump; no release clicks
    exp = res.tracks[0].notes(range(96, 101))
    assert [n.tick for n in exp if n.tick % 480 == 0] == [480, 1440, 2400]
    assert exp[0].length > 240  # the held E1 sustains
