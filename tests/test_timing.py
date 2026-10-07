import pytest

from yarginator.core.timing import TempoMap


def test_constant_roundtrip():
    tm = TempoMap.constant(120)
    assert tm.t2s(480) == pytest.approx(0.5)
    assert tm.s2t(0.5) == 480


@pytest.mark.parametrize("downbeat", [0.3, 1.0, 2.0, 2.2, 7.77])
def test_click_tracked_lands_on_downbeat(downbeat):
    tm = TempoMap.click_tracked(bpm=120, first_downbeat_s=downbeat)
    bar = 4 * tm.ppq
    downbeat_ticks = [t for t in tm.downbeats(bar * 20) if t >= bar]
    # some bar line after the offset measure sits exactly on the audio's first downbeat
    assert min(abs(tm.t2s(t) - downbeat) for t in downbeat_ticks) < 1e-4
    # and the tempo after the offset measure is locked
    assert all(t.bpm == pytest.approx(120, abs=1e-3) for t in tm.tempos[1:])
    # with at least half a bar of lead-in, the offset measure's tempo stays within 1.5x/2x of the song's
    if downbeat >= 1.0:
        assert 80 - 0.1 < tm.tempos[0].bpm <= 240 + 0.1


def test_from_beats_hits_every_beat():
    beats = [0.5, 1.0, 1.52, 2.01, 2.55, 3.04]
    tm = TempoMap.from_beats(beats)
    for i, b in enumerate(beats):
        assert tm.t2s(4 * tm.ppq + i * tm.ppq) == pytest.approx(b, abs=1e-5)


def test_beats_flags_downbeats():
    tm = TempoMap.constant(100, numerator=3)
    beats = list(tm.beats(6 * tm.ppq))
    assert [d for _, d in beats] == [True, False, False, True, False, False]
