import pytest

from yarginator.core.chart import Chart, Track
from yarginator.core.timing import TempoMap


def make_chart():
    chart = Chart(TempoMap.click_tracked(130, 1.7))
    voc = chart.track("PART VOCALS")
    voc.add_note(1920, 240, 105).add_lyric(1920, "Hel-").add_lyric(2160, "lo")
    voc.add_note(1920, 200, 60).add_note(2160, 200, 62)
    chart.track("EVENTS").add_text(1920, "[section verse]")
    return chart


def test_roundtrip(tmp_path):
    chart = make_chart()
    chart.save(tmp_path / "a.mid")
    back = Chart.load(tmp_path / "a.mid")
    assert list(back.tracks) == ["PART VOCALS", "EVENTS"]
    assert back.tracks["PART VOCALS"].lyrics() == [(1920, "Hel-"), (2160, "lo")]
    assert [n.pitch for n in back.tracks["PART VOCALS"].notes()] == [60, 105, 62]
    assert [t.us_per_beat for t in back.tempo.tempos] == [t.us_per_beat for t in chart.tempo.tempos]
    # saving again is byte-identical
    back.save(tmp_path / "b.mid")
    assert (tmp_path / "a.mid").read_bytes() == (tmp_path / "b.mid").read_bytes()


def test_back_to_back_notes_dont_overlap():
    tr = Track("x").add_note(0, 100, 60).add_note(100, 100, 60)
    assert [(n.tick, n.end) for n in tr.notes()] == [(0, 100), (100, 200)]


def test_retime_preserves_seconds():
    chart = make_chart()
    before = [(chart.tempo.t2s(t), m.type) for t, m in chart.tracks["PART VOCALS"].sorted_events()]
    chart.retime(TempoMap.constant(97))
    after = [(chart.tempo.t2s(t), m.type) for t, m in chart.tracks["PART VOCALS"].sorted_events()]
    assert [a[1] for a in after] == [b[1] for b in before]
    for (sa, _), (sb, _) in zip(after, before):
        assert sa == pytest.approx(sb, abs=0.002)


def test_sliding_the_grid_moves_grid_parts_and_keeps_vocals_in_place():
    from yarginator.core.chart import grid_shift_s
    chart = make_chart()
    chart.track("PART BASS").add_note(3840, 120, 96)
    old = chart.tempo
    new = TempoMap.click_tracked(130, 1.712)           # same tempo, downbeat 12 ms later
    shift = grid_shift_s(old, new)
    assert shift == pytest.approx(0.012, abs=0.001)
    voc_s = [old.t2s(t) for t, _ in chart.tracks["PART VOCALS"].lyrics()]
    chart.retime(new, only={"PART VOCALS"})
    assert chart.tracks["PART BASS"].notes()[0].tick == 3840                     # still on the grid
    assert [chart.tempo.t2s(t) for t, _ in chart.tracks["PART VOCALS"].lyrics()] == pytest.approx(voc_s, abs=0.002)
    assert grid_shift_s(old, TempoMap.click_tracked(128, 1.7)) is None          # a different tempo isn't a slide


def test_remove_notes_keeps_lyrics_and_phrases():
    tr = make_chart().tracks["PART VOCALS"]
    tr.remove_notes(range(36, 85))
    assert [n.pitch for n in tr.notes()] == [105]
    assert len(tr.lyrics()) == 2
