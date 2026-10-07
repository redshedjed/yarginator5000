"""The video-first workflow: one music video in, a YARG song folder out, then iterate and finalize.

    yarginator new "C:\\...\\Downloads\\Band - Song (Official Video).mp4" --dest "C:\\...\\In Progress"
        -> <dest>\\Band - Song\\  video.mp4, song.ogg (full mix), notes.mid (tempo map), song.ini
                                _yarginator\\ (song.toml, source\\mix.flac, ...)
    (separate _yarginator\\source\\mix.flac with your stem separator)
    yarginator stems  "<song>" --from "C:\\...\\separator output"   # copies stems in, picks instruments
    yarginator build  "<song>"                                    # chart, lights, REAPER project; iterate
    yarginator finalize "<song>" [--move-to "C:\\...\\Complete"]
        -> song.ogg = instrumental, vocals.ogg = vocals (from the stems), song.ini, checks

Until ``finalize``, song.ogg is the full mix so the song plays in YARG as-is. YARG plays every
audio file in the song folder together, so a full-mix song.ogg next to vocals.ogg would double the
vocals. That's why vocals.ogg only appears once there's an instrumental to replace the mix with.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import media
from .config import _read_toml, user_config_path
from .project import AUDIO_EXTS, WORK_DIR, SongProject, detect_stems, suggest_instruments

log = logging.getLogger(__name__)

# stems that aren't part of the instrumental mix (finalize builds one when there's no instrumental stem)
VOCAL_STEMS = ("song", "vocals", "lead_vocals", "backing_vocals", "harm1", "harm2", "harm3", "crowd")

YARG_FILES = {"song.ini", "song.ogg", "vocals.ogg", "video.mp4", "video.webm", "notes.mid", "album.png",
              "album.jpg"}


def user_paths() -> dict:
    """``[paths]`` in ~/.yarginator.toml: ``projects`` (where `new` puts songs), ``complete``."""
    return dict(_read_toml(user_config_path()).get("paths", {}))


# --- new ---------------------------------------------------------------------------------------------
def new_from_video(video: Path, dest: Path | None, name: str = "", artist: str = "", charter: str = "",
                   detect_tempo: bool = True) -> SongProject:
    """``dest`` None: the video's own folder becomes the song folder (its name read as "Artist -
    Title"), and the downloaded file moves into _yarginator/source once it's split."""
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"{video} not found")
    probe = media.probe(video)
    if probe.audio_codec is None:
        raise ValueError(f"{video.name} has no audio track")
    in_place = dest is None
    guess_artist, guess_title = media.parse_title(video.name)
    if in_place and " - " in video.parent.name:
        guess_artist, guess_title = media.parse_title(video.parent.name + ".x")
    artist, name = artist or guess_artist, name or guess_title
    folder = video.parent if in_place else \
        Path(dest).resolve() / media.safe_folder_name(f"{artist} - {name}" if artist else name)
    if (folder / WORK_DIR).exists() or (folder / "notes.mid").exists():
        raise FileExistsError(f"{folder} already has a song in it")
    work = folder / WORK_DIR
    (work / "stems").mkdir(parents=True, exist_ok=True)

    log.info("new: %s -> %s", video.name, folder)
    mix = media.extract_lossless(video, work / "source" / "mix.flac")
    media.to_ogg(mix, folder / "song.ogg")
    if probe.video_codec:
        out = media.extract_video(video, folder, probe)
        log.info("new: picture (%s) -> %s, audio (%s) -> song.ogg + %s", probe.video_codec, out.name,
                 probe.audio_codec, mix.relative_to(folder))
    else:
        log.warning("new: %s has no picture; no video file made", video.name)

    tempo = None
    if detect_tempo:
        try:
            from .audio.tempo import estimate_click
            bpm, downbeat = estimate_click(mix, 4)
            tempo = (round(bpm, 4), round(downbeat, 4))
            log.info("new: detected %.2f BPM, first downbeat at %.3f s (written to song.toml; check it)", *tempo)
        except Exception as e:  # no audio extra, or nothing beat-like: leave tempo to the user
            log.warning("new: tempo not detected (%s); set [tempo] bpm and first_downbeat_s", e)
    proj = SongProject.init(folder, name=name, artist=artist, charter=charter, workspace=True,
                            stems={"song": mix}, tempo=tempo)
    if in_place:  # keep the song folder to what YARG reads
        kept = work / "source" / video.name
        shutil.move(str(video), str(kept))
        log.info("new: moved the download to %s", kept.relative_to(folder))
    else:
        (work / "source" / "video.txt").write_text(f"{video}\n", encoding="utf-8")  # where it came from

    from . import pipeline
    if tempo:
        pipeline.step_tempo(proj)
    pipeline.step_ini(proj)
    return proj


# --- stems -------------------------------------------------------------------------------------------
def stem_activity(paths: dict[str, Path]) -> dict[str, float]:
    """How loud each stem plays (dB, 95th percentile of 1 s windows), to spot near-empty stems."""
    import numpy as np
    out = {}
    try:
        import librosa
    except ImportError:
        return {}
    for key, p in paths.items():
        y, sr = librosa.load(str(p), sr=11025, mono=True)
        n = len(y) // sr
        if n == 0:
            out[key] = -120.0
            continue
        rms = np.sqrt(np.mean(y[: n * sr].reshape(n, sr) ** 2, axis=1))
        out[key] = float(np.percentile(20 * np.log10(rms + 1e-9), 95))
    return out


@dataclass
class StemReport:
    stems: dict[str, Path]
    instruments: list[str]
    levels: dict[str, float] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)


def register_stems(proj: SongProject, from_dir: Path | None = None, empty_db: float = 30.0) -> StemReport:
    """Copy separator output into <work>/stems (if ``from_dir``), map files to stem keys by name,
    and pick instruments from the stems that actually play (``empty_db`` below the loudest = empty)."""
    stems_dir = proj.dir("stems")
    copied = []
    if from_dir is not None:
        for p in sorted(Path(from_dir).iterdir()):
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                shutil.copy2(p, stems_dir / p.name)
                copied.append(p.name)
    found = detect_stems(stems_dir)
    found.pop("song", None)  # a separator's copy of the full mix: we already have source/mix.flac
    rep = apply_stems(proj, found, empty_db)
    rep.copied = copied
    return rep


def apply_stems(proj: SongProject, found: dict[str, Path], empty_db: float = 30.0) -> StemReport:
    """Write ``found`` (plus the full mix as "song") into song.toml and pick the instruments from
    the stems that actually play (``empty_db`` below the loudest instrument stem = empty)."""
    mix = proj.work / "source" / "mix.flac"
    song = {"song": mix} if mix.exists() else {k: v for k, v in proj.stems().items() if k == "song"}
    stems = {**song, **found}
    # lead_vocals / backing_vocals stay in: an audible backing stem suggests harmonies
    not_parts = {"instrumental", "other", "crowd", "kick", "snare", "toms", "hihat", "ride", "crash"}
    instrument_stems = {k: v for k, v in found.items() if k not in not_parts}
    levels = stem_activity(instrument_stems)
    loudest = max(levels.values(), default=0.0)
    empty = [k for k, db in levels.items() if db < loudest - empty_db]
    instruments = suggest_instruments({k: v for k, v in instrument_stems.items() if k not in empty})
    proj.set_stems(stems, instruments)
    return StemReport(stems, instruments, levels, empty)


def separate_song(proj: SongProject, drum_split: bool = False, force: bool = False,
                  device: str | None = None, engine: str | None = None, best_vocals: bool | None = None,
                  dry_run: bool = False, vocal_split: bool | None = None) -> StemReport | None:
    """Separate source/mix.flac into <work>/stems (MVSEP or the local models), then register the
    stems. ``vocal_split`` adds lead/backing vocal stems (default ``[separation] vocal_split``).
    ``dry_run`` (MVSEP only) reports the jobs and changes nothing."""
    from .separate import pick_engine, separate, separate_mvsep
    mix = proj.work / "source" / "mix.flac"
    if not mix.exists():
        mix = proj.stems().get("song")
        if mix is None:
            raise FileNotFoundError("nothing to separate: no source/mix.flac and no [stems] song")
    cfg = proj.config.raw.get("separation", {})
    mv = proj.config.raw.get("mvsep", {})
    eng = pick_engine(engine or cfg.get("engine", "local"))
    vsplit = bool(cfg.get("vocal_split", False) if vocal_split is None else vocal_split)
    if eng == "mvsep":
        log.info("separate: engine MVSEP (uploads %s to mvsep.com)", mix.name)
        found = separate_mvsep(mix, proj.dir("stems"), drum_split=drum_split, force=force,
                               best_vocals=bool(cfg.get("best_vocals", False) if best_vocals is None else best_vocals),
                               settings=mv, dry_run=dry_run, vocal_split=vsplit)
        if dry_run:
            return None
    else:
        if dry_run:
            raise ValueError("--dry-run is for the MVSEP engine (the local models send nothing anywhere)")
        models = {k: v for k, v in cfg.items() if k.endswith("_model")}
        found = separate(mix, proj.dir("stems"), drum_split=drum_split, models=models, force=force,
                         device=device or cfg.get("device", "auto"), vocal_split=vsplit)
    return apply_stems(proj, found)


# --- finalize ----------------------------------------------------------------------------------------
def finalize(proj: SongProject, move_to: Path | None = None) -> tuple[list[str], Path]:
    """song.ogg / vocals.ogg from the stems, song.ini, then checks. Returns (notes, final folder)."""
    from . import pipeline
    from .core.chart import Chart
    from .core.instruments import tracks_for
    from .feedback import read_review
    notes: list[str] = []
    root = proj.root
    stems = proj.stems()

    instrumental, vocals = stems.get("instrumental"), stems.get("vocals")
    if vocals is not None and instrumental is None:
        others = [p for k, p in stems.items() if k not in VOCAL_STEMS]
        if others:
            instrumental = media.mix_to_flac(others, proj.dir("source") / "instrumental.flac")
            notes.append(f"no instrumental stem: mixed {len(others)} stems for song.ogg")
    if vocals is not None and instrumental is not None:
        media.to_ogg(instrumental, root / "song.ogg")
        media.to_ogg(vocals, root / "vocals.ogg")
        notes.append("song.ogg = instrumental, vocals.ogg = vocals")
    else:
        mix = stems.get("song")
        if mix is not None and not (root / "song.ogg").exists():
            media.to_ogg(mix, root / "song.ogg")
        (root / "vocals.ogg").unlink(missing_ok=True)
        notes.append("WARNING: no vocal stem: song.ogg stays the full mix, no vocals.ogg")

    pipeline.step_ini(proj)

    chart_path = proj.chart_path
    if not chart_path.exists():
        notes.append("WARNING: no notes.mid yet")
    else:
        chart = Chart.load(chart_path)
        wanted = tracks_for(proj.config.instruments) or set(chart.tracks)
        empty = sorted(t for t in wanted if t not in chart.tracks or not chart.tracks[t].has_notes())
        empty = [t for t in empty if t not in ("EVENTS", "VENUE")]
        if empty:
            notes.append("WARNING: no notes yet on " + ", ".join(empty))
    if not any((root / v).exists() for v in ("video.mp4", "video.webm")):
        notes.append("no video file")
    flagged = [r for r in read_review(proj.dir("reports")) if r.severity == "warn"]
    if flagged:
        notes.append(f"{len(flagged)} spots still flagged in {WORK_DIR}/reports/review.csv")
    extra = sorted(p.name for p in root.iterdir() if p.is_file() and p.name.lower() not in YARG_FILES)
    if extra:
        notes.append("WARNING: extra files next to the chart (YARG may load audio ones as stems): " + ", ".join(extra))

    final = root
    if move_to is not None:
        final = Path(move_to).resolve() / root.name
        if final.exists():
            raise FileExistsError(f"{final} already exists")
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(final))
        notes.append(f"moved to {final} (regenerate the REAPER project there with `reaper --force`: "
                     "it points at the old location)")
    return notes, final
