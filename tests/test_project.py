import pytest

from yarginator.project import SongProject, stem_key


@pytest.mark.parametrize("name,key", [
    # stem-separator output: the meaningful word is last, and the leading "song" must not win
    ("song.ogg_bs6stem_mt_0_vocals (1)", "vocals"),
    ("song.ogg_bs6stem_mt_0_instrum (1)", "instrumental"),
    ("song.ogg_bs6stem_mt_0_piano (1)", "keys"),
    ("song_original", "song"),
    ("Drums Kit", "drums"),
    ("lead vocal", "vocals"),
    ("HARM2", "harm2"),
    ("backing vox", "vocals"),
    ("Mystery Stem", "mystery_stem"),
])
def test_stem_key(name, key):
    assert stem_key(name) == key


def test_init_with_stems_in_place(tmp_path):
    src = tmp_path / "Some Band - Some Song"
    src.mkdir()
    for f in ("song.ogg_x_drums (1).mp3", "song_original.ogg", "video.mp4", "desktop.ini"):
        (src / f).write_bytes(b"")
    proj = SongProject.init(tmp_path / "work", stems_dir=str(src))
    assert set(proj.config.stems) == {"drums", "song"}
    assert all(proj.root.joinpath(p).exists() for p in proj.config.stems.values())
