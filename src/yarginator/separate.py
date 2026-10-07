"""Stem separation, built in.

Three passes, each a model from the UVR / audio-separator catalogue:

1. vocals: the full mix -> vocals + instrumental (BS-RoFormer by viperx, the strongest open vocal
   model; the instrumental is what finalize uses for song.ogg).
2. instruments: the instrumental -> drums, bass, guitar, piano (as "keys"), other (Demucs v4 6-stem).
   Running it on the instrumental instead of the mix keeps vocal bleed out of the instrument stems.
3. drum split (optional): drums -> kick, snare, toms, hihat, ride, crash (MDX23C DrumSep).

Where it runs (``[separation] device``):
- ``gpu``: DirectML (any DirectX 12 GPU, AMD included) in its own environment at
  ~/.yarginator/separator-gpu, made by ``yarginator setup-gpu``. It's separate because the
  DirectML build needs numpy 2, which would break TensorFlow (keys) in the main environment.
  Passes run there through ``separate_worker.py``.
- ``cpu``: in this environment (``pip install -e .[separate]``).
- ``auto`` (default): the GPU environment if it exists, else the CPU.

Swap any model in ``[separation]`` (song.toml or ~/.yarginator.toml); `audio-separator --list_models`
lists them. Models download once to ~/.yarginator/models (shared by both). Each pass is skipped when
its outputs are already there (``force`` redoes them). Outputs are FLAC, the same length as the mix,
so they stay in sync with the video.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULTS = {
    "vocals_model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "instruments_model": "htdemucs_6s.yaml",
    "drums_model": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
}
# model output label (lower case) -> stem key
LABELS = {"vocals": "vocals", "instrumental": "instrumental", "drums": "drums", "bass": "bass",
          "guitar": "guitar", "piano": "keys", "other": "other",
          "kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "ride": "ride", "crash": "crash"}
PASSES = {
    "vocals": ("vocals_model", ("vocals", "instrumental")),
    "instruments": ("instruments_model", ("drums", "bass", "guitar", "keys", "other")),
    "drums": ("drums_model", ("kick", "snare", "toms", "hihat", "ride", "crash")),
}
DRUM_SPLIT_DIR = "drum_split"
GPU_PACKAGES = ["audio-separator[dml]", "setuptools<81"]  # setuptools: librosa/resampy still import pkg_resources


def model_dir() -> Path:
    return Path.home() / ".yarginator" / "models"


def gpu_env() -> Path:
    return Path.home() / ".yarginator" / "separator-gpu"


def gpu_python() -> Path | None:
    py = gpu_env() / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return py if py.exists() else None


def pick_device(device: str = "auto") -> str:
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"[separation] device must be auto, cpu or gpu, not {device!r}")
    if device == "gpu" and gpu_python() is None:
        raise RuntimeError("no GPU environment yet: run `yarginator setup-gpu`")
    if device == "auto":
        return "gpu" if gpu_python() is not None else "cpu"
    return device


def setup_gpu() -> str:
    """Create (or update) the DirectML environment; returns the GPU it found."""
    env = gpu_env()
    if gpu_python() is None:
        log.info("setup-gpu: creating %s", env)
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    py = str(gpu_python())
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    log.info("setup-gpu: installing %s (PyTorch + DirectML, ~1 GB)", ", ".join(GPU_PACKAGES))
    subprocess.run([py, "-m", "pip", "install", *GPU_PACKAGES], check=True)
    p = subprocess.run([py, "-c", "import torch_directml as d; print(d.device_name(d.default_device()))"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"DirectML isn't working in {env}: {p.stderr.strip().splitlines()[-1:]}")
    return p.stdout.strip()


def ensure_ffmpeg_on_path() -> None:
    """audio-separator (and pydub under it) run ``ffmpeg`` from the PATH. Without one there, expose
    the bundled imageio-ffmpeg binary as ~/.yarginator/bin/ffmpeg(.exe) and put that on the PATH."""
    if shutil.which("ffmpeg"):
        return
    from .media import ffmpeg_exe
    src = Path(ffmpeg_exe())
    bin_dir = Path.home() / ".yarginator" / "bin"
    dst = bin_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if not dst.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)             # same drive: no extra space
        except OSError:
            shutil.copy2(src, dst)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _separate_cpu(src: Path, out_dir: Path, model: str) -> list[Path]:
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise RuntimeError("CPU separation needs the separate extra (pip install -e .[separate]), "
                           "or set up the GPU one with `yarginator setup-gpu`") from e
    sep = Separator(log_level=logging.WARNING, model_file_dir=str(model_dir()), output_dir=str(out_dir),
                    output_format="FLAC", normalization_threshold=1.0)  # 1.0: don't rescale stems apart
    sep.load_model(model_filename=model)
    return [out_dir / Path(f).name for f in sep.separate(str(src))]


def _separate_gpu(src: Path, out_dir: Path, model: str) -> list[Path]:
    from .separate_worker import MARKER
    args = {"src": str(src), "out_dir": str(out_dir), "model": model, "model_dir": str(model_dir()),
            "directml": True}
    worker = Path(__file__).with_name("separate_worker.py")
    # progress bars stream straight to the terminal (stderr); the result comes back on stdout
    p = subprocess.run([str(gpu_python()), str(worker), json.dumps(args)], stdout=subprocess.PIPE,
                       text=True, env=os.environ.copy())
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith(MARKER)), None)
    if p.returncode != 0 or line is None:
        raise RuntimeError(f"GPU separation failed ({model}); see the output above, "
                           "or set [separation] device = \"cpu\"")
    return [out_dir / Path(f).name for f in json.loads(line[len(MARKER):])]


def _run_pass(src: Path, out_dir: Path, model: str, want: tuple[str, ...], force: bool,
              device: str) -> dict[str, Path]:
    targets = {k: out_dir / f"{k}.flac" for k in want}
    if not force and all(p.exists() for p in targets.values()):
        log.info("separate: %s already done (%s)", model, ", ".join(want))
        return targets
    tmp = out_dir / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    ensure_ffmpeg_on_path()
    log.info("separate: %s on %s (%s)", model, src.name, device.upper())
    t0 = time.monotonic()
    out: dict[str, Path] = {}
    try:
        made = (_separate_gpu if device == "gpu" else _separate_cpu)(src, tmp, model)
        for f in made:
            m = re.search(r"_\(([^)]+)\)", f.stem)
            key = LABELS.get(m[1].lower()) if m else None
            if key in targets:
                shutil.move(str(f), str(targets[key]))
                out[key] = targets[key]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("separate: %s took %.0f s", model, time.monotonic() - t0)
    missing = [k for k in want if k not in out]
    if missing:
        log.warning("separate: %s gave no %s", model, ", ".join(missing))
    return out


# --- MVSEP (mvsep.com) -------------------------------------------------------------------------------
# One job for the six stems (BS Roformer SW); optionally the best vocals/instrumental model; DrumSep
# (MelBand Roformer, 6 stems) on our drum stem for the split. All jobs go through mvsep.Client and
# its guard rails.
MVSEP_STEMS = (63, {})
MVSEP_VOCALS = (40, {"add_opt1": 81})                       # BS Roformer ver 2025.07
MVSEP_DRUMS = (37, {"add_opt1": 7, "add_opt2": 1})          # MelBand Roformer 6 stems, input is drums only
# (substring in MVSEP's file name, stem key): drum parts before "drum", "instrum" before "vocal"
MVSEP_NAMES = [("kick", "kick"), ("snare", "snare"), ("tom", "toms"), ("hi-hat", "hihat"), ("hihat", "hihat"),
               ("_hh", "hihat"), ("ride", "ride"), ("crash", "crash"), ("cymbal", "cymbals"), ("residual", None),
               ("instrum", "instrumental"), ("vocal", "vocals"), ("bass", "bass"), ("drum", "drums"),
               ("guitar", "guitar"), ("piano", "keys"), ("other", "other")]


def mvsep_key(filename: str) -> str | None:
    n = Path(filename).stem.lower()
    return next((key for word, key in MVSEP_NAMES if word in n), None)


def _mvsep_pass(client, src: Path, stems_dir: Path, job: tuple[int, dict], want: set[str], force: bool,
                rename: dict[str, str] | None = None) -> dict[str, Path]:
    from .mvsep import Job
    import json
    targets = {k: stems_dir / f"{k}.flac" for k in want}
    # what this job produced last time: skip when those files are still here, even if MVSEP didn't
    # return every stem we'd have liked (otherwise each run would submit the job again)
    marker = stems_dir / f".mvsep-{job[0]}.json"
    if not force and marker.exists():
        done = [k for k in json.loads(marker.read_text("utf-8")) if k in targets]
        if all(targets[k].exists() for k in done):
            log.info("separate: mvsep sep_type %s already done (%s)", job[0], ", ".join(done))
            return {k: targets[k] for k in done}
    raw = stems_dir / "_mvsep"
    files = client.separate(src, Job(job[0], dict(job[1])), raw)
    if client.dry_run:
        return {}
    out: dict[str, Path] = {}
    for f in files:
        key = mvsep_key(f.name)
        key = (rename or {}).get(key, key)
        log.info("separate: mvsep %s -> %s", f.name, key or "(not used)")
        if key in targets:
            f.replace(targets[key])
            out[key] = targets[key]
    shutil.rmtree(raw, ignore_errors=True)
    marker.write_text(json.dumps(sorted(out)), encoding="utf-8")
    missing = sorted(set(want) - set(out))
    if missing:
        log.info("separate: mvsep sep_type %s gave no %s", job[0], ", ".join(missing))
    return out


def separate_mvsep(mix: Path, stems_dir: Path, drum_split: bool = False, force: bool = False,
                   best_vocals: bool = False, settings: dict | None = None, dry_run: bool = False) -> dict[str, Path]:
    from .mvsep import Client, Limits, find_token
    cfg = settings or {}
    client = Client(find_token(cfg), Limits.from_config(cfg), dry_run=dry_run)
    out = _mvsep_pass(client, mix, stems_dir, MVSEP_STEMS,
                      {"vocals", "bass", "drums", "guitar", "keys", "other", "instrumental"} - ({"vocals", "instrumental"} if best_vocals else set()),
                      force)
    if best_vocals:
        # model 40 calls its instrumental "other"
        out.update(_mvsep_pass(client, mix, stems_dir, MVSEP_VOCALS, {"vocals", "instrumental"}, force,
                               rename={"other": "instrumental"}))
    if drum_split and "drums" in out:
        out.update(_mvsep_pass(client, out["drums"], stems_dir / DRUM_SPLIT_DIR, MVSEP_DRUMS,
                               {"kick", "snare", "toms", "hihat", "ride", "crash"}, force))
    return out


def pick_engine(engine: str) -> str:
    """``local`` (default) or ``mvsep``. MVSEP is opt-in only: it uploads audio to a third party and
    spends credits, so having a token set up is never enough on its own to use it."""
    if engine not in ("mvsep", "local"):
        raise ValueError(f"[separation] engine must be local or mvsep, not {engine!r}")
    return engine


def separate(mix: Path, stems_dir: Path, drum_split: bool = False, models: dict | None = None,
             force: bool = False, device: str = "auto") -> dict[str, Path]:
    """Run the local passes on ``mix``; returns {stem key: file} for everything produced."""
    m = {**DEFAULTS, **(models or {})}
    dev = pick_device(device)
    out = _run_pass(mix, stems_dir, m["vocals_model"], PASSES["vocals"][1], force, dev)
    if "instrumental" in out:
        out.update(_run_pass(out["instrumental"], stems_dir, m["instruments_model"], PASSES["instruments"][1],
                             force, dev))
    if drum_split and "drums" in out:
        out.update(_run_pass(out["drums"], stems_dir / DRUM_SPLIT_DIR, m["drums_model"], PASSES["drums"][1],
                             force, dev))
    return out
