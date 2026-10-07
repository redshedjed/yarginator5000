"""Video -> song folder workflow: names, layout, stems, finalize (needs ffmpeg via imageio-ffmpeg)."""
import numpy as np
import pytest

from yarginator.media import parse_title


def test_parse_title_from_downloads():
    assert parse_title("I Prevail - VIOLENT NATURE (Official Music Video) - IPrevailBand (1080p, h264).mp4") \
        == ("I Prevail", "VIOLENT NATURE")
    assert parse_title("Electric Callboy - THE WAY YOU ARE (OFFICIAL VIDEO).mp4") == ("Electric Callboy", "THE WAY YOU ARE")
    assert parse_title("Silence (1080p_25fps_H264-128kbit_AAC).mp4") == ("", "Silence")
    assert parse_title("Ladrones - Así Nomás (Video Oficial) - Ladrones (1080p, h264) (1).mp4") == ("Ladrones", "Así Nomás")


@pytest.fixture
def ffmpeg():
    pytest.importorskip("imageio_ffmpeg")
    from yarginator import media
    return media


def make_video(media, path, seconds=4):
    media.run(["-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
               "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)])


def test_new_stems_finalize(tmp_path, ffmpeg):
    sf = pytest.importorskip("soundfile")
    from yarginator import workflow
    from yarginator.project import WORK_DIR, SongProject
    song = tmp_path / "In Progress" / "Band - Tune"
    song.mkdir(parents=True)
    make_video(ffmpeg, song / "Tune (1080p).mp4")

    proj = workflow.new_from_video(song / "Tune (1080p).mp4", None, detect_tempo=False)
    assert proj.root == song and proj.work == song / WORK_DIR
    assert (proj.config.song.artist, proj.config.song.name) == ("Band", "Tune")
    top = {p.name for p in song.iterdir()}
    assert top == {"song.ogg", "video.mp4", "song.ini", WORK_DIR}            # download moved out of the way
    assert (song / WORK_DIR / "source" / "Tune (1080p).mp4").exists()
    probe = ffmpeg.probe(song / "video.mp4")
    assert probe.video_codec == "h264" and probe.audio_codec is None          # picture only

    # stems from a separator: vocals + instrumental + a silent guitar
    sep = tmp_path / "separated"
    sep.mkdir()
    t = np.arange(4 * 22050) / 22050
    sf.write(sep / "tune_vocals.wav", 0.3 * np.sin(2 * np.pi * 330 * t), 22050)
    sf.write(sep / "tune_instrum.wav", 0.3 * np.sin(2 * np.pi * 110 * t), 22050)
    sf.write(sep / "tune_guitar.wav", 1e-5 * np.sin(2 * np.pi * 220 * t), 22050)
    rep = workflow.register_stems(proj, sep)
    assert set(rep.stems) == {"song", "vocals", "instrumental", "guitar"}
    assert rep.empty == ["guitar"] and rep.instruments == ["vocals"]
    proj = SongProject(song)                                                   # re-read song.toml
    assert proj.config.instruments == ["vocals"] and "guitar" in proj.config.stems

    notes, final = workflow.finalize(proj)
    assert final == song
    assert {"song.ogg", "vocals.ogg", "song.ini"} <= {p.name for p in song.iterdir()}
    assert any("instrumental" in n for n in notes)
    # history lives in the work folder (song.ini rewritten by finalize; song.toml by stems)
    hist = {p.name.split(".")[0] for p in (song / WORK_DIR / ".history").iterdir()}
    assert "song" in hist
    assert not (song / ".history").exists()

    done = tmp_path / "Complete"
    _, moved = workflow.finalize(SongProject(song), done)
    assert moved == done / "Band - Tune" and (moved / WORK_DIR / "song.toml").exists() and not song.exists()


def test_new_into_a_projects_folder(tmp_path, ffmpeg):
    from yarginator import workflow
    dl = tmp_path / "Downloads"
    dl.mkdir()
    make_video(ffmpeg, dl / "Some Band - Big Song (Official Video) - SomeBandVEVO (1080p, h264).mp4")
    proj = workflow.new_from_video(next(dl.iterdir()), tmp_path / "In Progress", detect_tempo=False)
    assert proj.root == tmp_path / "In Progress" / "Some Band - Big Song"
    assert len(list(dl.iterdir())) == 1                                         # the download stays put
    with pytest.raises(FileExistsError):
        workflow.new_from_video(next(dl.iterdir()), tmp_path / "In Progress", detect_tempo=False)
