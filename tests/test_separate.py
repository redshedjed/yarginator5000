"""Built-in separation: pass chaining, output naming, skip-if-done, registration (model calls faked)."""
import sys
import types

import numpy as np
import pytest

from yarginator import separate as sep_mod

OUTPUTS = {  # model -> labels it writes, as audio-separator names them: "<input>_(<Label>)_<model>.flac"
    "voc.ckpt": ["Vocals", "Instrumental"],
    "six.yaml": ["Drums", "Bass", "Guitar", "Piano", "Other", "Vocals"],
    "drumsep.ckpt": ["kick", "snare", "toms", "hh", "ride", "crash"],
}


@pytest.fixture
def fake_separator(monkeypatch):
    sf = pytest.importorskip("soundfile")
    calls = []

    class FakeSeparator:
        def __init__(self, output_dir, **kw):
            self.out = output_dir

        def load_model(self, model_filename):
            self.model = model_filename

        def separate(self, path):
            calls.append((self.model, path))
            names = []
            for label in OUTPUTS[self.model]:
                n = f"x_({label})_{self.model.split('.')[0]}.flac"
                level = 1e-6 if label == "Guitar" else 0.3          # a silent guitar stem
                t = np.arange(22050 * 2) / 22050
                sf.write(f"{self.out}/{n}", level * np.sin(2 * np.pi * 220 * t), 22050)
                names.append(n)
            return names

    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    monkeypatch.setattr(sep_mod, "ensure_ffmpeg_on_path", lambda: None)
    return calls


MODELS = {"vocals_model": "voc.ckpt", "instruments_model": "six.yaml", "drums_model": "drumsep.ckpt"}


def test_passes_chain_and_name_stems(tmp_path, fake_separator):
    mix = tmp_path / "mix.flac"
    mix.write_bytes(b"")
    out = sep_mod.separate(mix, tmp_path / "stems", drum_split=True, models=MODELS, device="cpu")
    assert set(out) == {"vocals", "instrumental", "drums", "bass", "guitar", "keys", "other",
                        "kick", "snare", "toms", "hihat", "ride", "crash"}
    assert out["keys"].name == "keys.flac" and out["hihat"].parent.name == "drum_split"
    # pass 2 runs on the instrumental, pass 3 on the drums; the 6-stem model's residual vocals don't
    # overwrite the real vocal stem
    assert [m for m, _ in fake_separator] == ["voc.ckpt", "six.yaml", "drumsep.ckpt"]
    assert fake_separator[1][1].endswith("instrumental.flac") and fake_separator[2][1].endswith("drums.flac")
    assert not (tmp_path / "stems" / "_tmp").exists()
    # done passes are skipped next time
    sep_mod.separate(mix, tmp_path / "stems", drum_split=True, models=MODELS, device="cpu")
    assert len(fake_separator) == 3


def test_separate_song_registers_stems(tmp_path, fake_separator):
    pytest.importorskip("librosa")
    from yarginator.project import SongProject
    from yarginator.workflow import separate_song
    song = tmp_path / "Band - Tune"
    proj = SongProject.init(song, workspace=True, stems={})
    (proj.work / "source").mkdir()
    (proj.work / "source" / "mix.flac").write_bytes(b"")
    text = proj.config_path.read_text(encoding="utf-8")
    text = text.replace("[separation]\n", "[separation]\n" + "".join(f'{k} = "{v}"\n' for k, v in MODELS.items()))
    proj.config_path.write_text(text, encoding="utf-8")
    proj.reload()
    rep = separate_song(proj, device="cpu", engine="local")
    assert rep.empty == ["guitar"]
    assert rep.instruments == ["drums", "bass", "keys", "vocals"]
    proj = SongProject(song)
    assert proj.config.stems["vocals"] == "stems/vocals.flac" and proj.config.stems["song"] == "source/mix.flac"


def test_device_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(sep_mod, "gpu_env", lambda: tmp_path / "none")
    assert sep_mod.pick_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="setup-gpu"):
        sep_mod.pick_device("gpu")
    env = tmp_path / "env"
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    py.parent.mkdir(parents=True)
    py.write_bytes(b"")
    monkeypatch.setattr(sep_mod, "gpu_env", lambda: env)
    assert sep_mod.pick_device("auto") == "gpu" and sep_mod.pick_device("cpu") == "cpu"
