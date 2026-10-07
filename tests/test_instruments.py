"""Per-song instrument lists: unused instruments are left out of the chart and REAPER project,
listed-but-uncharted ones get empty tracks, BEAT/EVENTS/VENUE are always there."""
import pytest

from yarginator import pipeline
from yarginator.core import instruments as ins
from yarginator.core.chart import Chart
from yarginator.core.timing import TempoMap
from yarginator.project import SongProject, suggest_instruments
from yarginator.reaper import project as rproject

from test_rpp import OLD_SONG_TEMPLATE


def song(tmp_path, instruments):
    proj = SongProject.init(tmp_path / "s")
    cfg = proj.root / "song.toml"
    text = cfg.read_text().replace("# bpm = 120.0", "bpm = 120").replace("# first_downbeat_s = 1.234", "first_downbeat_s = 1")
    import re
    text = re.sub(r"^instruments = .*$", f"instruments = {instruments!r}".replace("'", '"'), text, flags=re.M)
    cfg.write_text(text)
    return SongProject(proj.root)


def test_tracks_for():
    assert ins.tracks_for(None) is None
    assert ins.tracks_for(["harmonies"]) == {"HARM1", "HARM2", "HARM3", "BEAT", "EVENTS", "VENUE"}
    with pytest.raises(ValueError, match="kazoo"):
        ins.tracks_for(["kazoo"])


def test_suggest_from_stems(tmp_path):
    stems = {k: tmp_path / k for k in ("drums", "bass", "keys", "vocals", "song")}
    assert suggest_instruments(stems) == ["drums", "bass", "keys", "vocals"]
    assert "keys5" not in suggest_instruments({"song": tmp_path / "x"})


def test_chart_prunes_empty_unlisted_tracks_but_keeps_work(tmp_path, caplog):
    proj = song(tmp_path, ["bass", "vocals"])
    chart = Chart(TempoMap.constant(120))
    chart.track("PART GUITAR")                              # empty -> pruned
    chart.track("PART DRUMS").add_note(960, 120, 96)        # has notes -> kept, warned
    proj.save_chart(chart)
    caplog.set_level("INFO")
    pipeline.step_chart(proj)
    names = set(proj.load_chart().tracks)
    assert "PART GUITAR" not in names and "PART DRUMS" in names
    assert {"BEAT", "EVENTS"} <= names
    assert "PART DRUMS isn't in [chart] instruments" in caplog.text


def test_reaper_keeps_only_listed_instruments(tmp_path):
    tpl = tmp_path / "t.RPP"
    tpl.write_text(OLD_SONG_TEMPLATE)  # has Guitar audio, PART GUITAR, PART DRUMS
    chart = Chart(TempoMap.constant(120))
    chart.track("BEAT").add_note(0, 60, 12)
    wanted = pipeline.wanted_tracks(["bass", "harmonies"])
    root = rproject.build(chart, {}, [], tpl, wanted=wanted)
    names = [t.value("NAME")[0] for t in root.nodes("TRACK")]
    assert "PART GUITAR" not in names and "PART DRUMS" not in names and "Guitar" not in names
    for t in ("PART BASS", "HARM1", "HARM2", "HARM3", "BEAT", "EVENTS", "VENUE"):
        assert t in names
    bass = next(t for t in root.nodes("TRACK") if t.value("NAME")[0] == "PART BASS")
    assert bass.find("ITEM").find("SOURCE").args == ["MIDI"]  # empty, ready to chart
    # chart tracks follow the instrument order, BEAT/EVENTS/VENUE last
    chart_names = [n for n in names if n in set(wanted)]
    assert chart_names == ["PART BASS", "HARM1", "HARM2", "HARM3", "BEAT", "EVENTS", "VENUE"]
