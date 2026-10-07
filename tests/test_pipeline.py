"""End-to-end on a song folder with no audio: tempo from config, structural parts, lights, ini, REAPER."""
import re

from yarginator import pipeline
from yarginator.core import instruments
from yarginator.lighting import cues
from yarginator.project import SongProject


def make_song(tmp_path):
    proj = SongProject.init(tmp_path / "song", name="Test Song", artist="Band")
    cfg = proj.root / "song.toml"
    text = cfg.read_text()
    text = text.replace("# bpm = 120.0", "bpm = 128.0").replace("# first_downbeat_s = 1.234", "first_downbeat_s = 1.5")
    cfg.write_text(text)
    return SongProject(proj.root)


def test_full_build_without_audio(tmp_path):
    proj = make_song(tmp_path)
    pipeline.step_tempo(proj)
    chart = proj.load_chart()
    ev = chart.track(instruments.EVENTS)
    bar = 4 * chart.ppq
    for i, name in enumerate(["intro", "verse 1", "chorus 1", "bridge"]):
        ev.add_text(bar * (1 + 4 * i), f"[section {name}]")
    ev.add_text(bar * 18, "[music_end]")
    ev.add_text(bar * 18 + 480, "[end]")
    proj.save_chart(chart)

    pipeline.build(proj)

    chart = proj.load_chart()
    assert chart.tempo.tempos[1].bpm == 128.0
    beat = chart.tracks[instruments.BEAT].notes()
    assert beat[0].pitch == instruments.BEAT_DOWNBEAT and len(beat) == 18 * 4 + 1  # runs to [end]
    texts = [t for _, t in chart.tracks[instruments.EVENTS].texts()]
    assert "[music_start]" in texts and texts.count("[end]") == 1

    venue = chart.tracks[instruments.VENUE]
    lighting = [cues.parse_cue(t) for _, t in venue.texts() if cues.parse_cue(t) is not None]
    assert lighting == ["intro", "verse", "flare_fast", "chorus", "manual_cool", "blackout_slow"]
    assert venue.notes(cues.KEYFRAME_NOTES)  # bridge -> manual_cool gets keyframes

    assert "name = Test Song" in proj.song_ini_path.read_text()
    rpp = (proj.root / "reaper" / "song.RPP").read_text()
    # no stems -> init lists every instrument, so uncharted parts get empty tracks to chart by hand
    assert re.search(r'NAME "?PART VOCALS"?', rpp) and re.search(r'NAME "?PART REAL_KEYS_X"?', rpp)
    assert "PART VOCALS" not in proj.load_chart().tracks  # ...but nothing empty is written to notes.mid
    assert "NAME VENUE" in rpp and "<TEMPOENVEX" in rpp


def test_lights_regenerate_keeps_other_venue_events(tmp_path):
    proj = make_song(tmp_path)
    pipeline.step_tempo(proj)
    chart = proj.load_chart()
    chart.track(instruments.VENUE).add_text(960, "[coop_all_far]").add_text(960, "[lighting (stomp)]")
    proj.save_chart(chart)
    pipeline.step_lights(proj, force=True)
    texts = [t for _, t in proj.load_chart().tracks[instruments.VENUE].texts()]
    assert "[coop_all_far]" in texts and "[lighting (stomp)]" not in texts


def test_phrase_markers_alone_dont_count_as_charted(tmp_path, caplog):
    proj = make_song(tmp_path)
    chart = proj.load_chart()
    chart.track(instruments.VOCALS).add_lyric(1920, "la").add_note(1900, 400, instruments.VOCAL_PHRASE)
    proj.save_chart(chart)
    caplog.set_level("INFO")
    pipeline.step_chart(proj)
    assert "vocals: already charted" not in caplog.text


def test_one_backup_per_run_and_hand_edits_always_backed_up(tmp_path):
    proj = make_song(tmp_path)
    pipeline.step_tempo(proj)
    history = proj.root / ".history"
    n0 = len(list(history.glob("*"))) if history.exists() else 0
    pipeline.build(proj, ["chart", "lights"])  # two more writes by this process: no new backups
    assert len(list(history.glob("*"))) == n0
    chart = proj.load_chart()
    chart.track("PART GUITAR").add_note(1920, 100, 96)
    chart.save(proj.chart_path)  # an edit from outside (REAPER), same second
    pipeline.step_lights(proj, force=True)
    assert len(list(history.glob("*"))) == n0 + 1


def test_charted_parts_are_skipped_unless_asked(tmp_path, caplog):
    proj = make_song(tmp_path)
    chart = proj.load_chart()
    chart.track(instruments.VOCALS).add_lyric(1920, "la").add_note(1920, 100, 64)
    proj.save_chart(chart)
    caplog.set_level("INFO")
    pipeline.step_chart(proj)
    assert "vocals: already charted" in caplog.text
