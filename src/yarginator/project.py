"""A song folder on disk and the files in it.

Song folders made from a video (``yarginator new``) keep only what YARG reads at the top, and
everything else in a work folder YARG ignores (no chart or song.ini in it, and no audio named like
a stem next to the chart):

    Artist - Title/
      song.ini  song.ogg  vocals.ogg  video.mp4  notes.mid
      _yarginator/
        song.toml      config (see templates/song.toml); paths in it are relative to this folder
        source/        mix.flac: the video's soundtrack, lossless (feed this to your stem separator)
        stems/         separated stems (``yarginator stems`` registers them)
        reaper/        generated REAPER project
        reports/       per-part CSVs and review.csv
        .history/      timestamped copies of notes.mid / song.ini / the REAPER project before overwrites
        .cache/        audio analysis cache

Older folders (``yarginator init``) have song.toml and the work folders at the top level; both work.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from importlib import resources
from pathlib import Path
from string import Template

from .config import SONG_FILE, SongConfig, load_config
from .core.chart import Chart

log = logging.getLogger(__name__)

WORK_DIR = "_yarginator"
AUDIO_EXTS = {".wav", ".ogg", ".opus", ".mp3", ".flac"}

# Word prefix -> stem key, used by `init` to pre-fill [stems]. Filenames are split into words and
# scanned from the end, because separators name files like "song.ogg_bs6stem_mt_0_vocals (1).mp3":
# the meaningful word is last, and an early "song" must not win.
STEM_KEYWORDS = [
    ("drum", "drums"), ("kick", "drums"), ("bass", "bass"), ("rhythm", "rhythm"),
    ("guitar", "guitar"), ("gtr", "guitar"), ("key", "keys"), ("piano", "keys"), ("synth", "keys"),
    ("harm1", "harm1"), ("harm2", "harm2"), ("harm3", "harm3"), ("backing", "harm2"), ("bv", "harm2"),
    ("vocal", "vocals"), ("vox", "vocals"), ("voice", "vocals"),
    ("instrum", "instrumental"), ("karaoke", "instrumental"), ("accomp", "instrumental"),
    ("other", "other"), ("crowd", "crowd"),
    ("original", "song"), ("mix", "song"), ("full", "song"), ("song", "song"),
]


class SongProject:
    def __init__(self, root: str | Path):
        root = Path(root).resolve()
        if root.name == WORK_DIR and (root / SONG_FILE).exists():
            root = root.parent                     # pointed at the work folder itself
        self.root = root                           # the song folder YARG reads
        if (root / WORK_DIR / SONG_FILE).exists():
            self.work = root / WORK_DIR            # everything that isn't for YARG
        elif (root / SONG_FILE).exists():
            self.work = root
        else:
            raise FileNotFoundError(f"no song.toml in {root} or {root / WORK_DIR} "
                                    "(start one with `yarginator new <video>` or `yarginator init`)")
        self.config_path = self.work / SONG_FILE
        self.config: SongConfig = load_config(self.config_path)
        self._written: dict[Path, bytes] = {}

    @property
    def name(self) -> str:
        return self.config.song.name or self.root.name

    # --- paths ------------------------------------------------------------------------------
    @property
    def chart_path(self) -> Path:
        return self.root / self.config.chart_file

    @property
    def song_ini_path(self) -> Path:
        return self.root / "song.ini"

    def dir(self, name: str) -> Path:
        """A working subfolder (reports, reaper, .cache, .history, ...), created on demand."""
        d = self.work / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stems(self) -> dict[str, Path]:
        out = {}
        for key, rel in self.config.stems.items():
            p = (self.work / rel).resolve()
            if p.exists():
                out[key] = p
            else:
                log.warning("stem %r: %s does not exist", key, p)
        return out

    # --- chart ------------------------------------------------------------------------------
    def load_chart(self) -> Chart:
        return Chart.load(self.chart_path) if self.chart_path.exists() else Chart()

    def save_chart(self, chart: Chart) -> None:
        self.backup(self.chart_path)
        chart.save(self.chart_path)
        self._written[self.chart_path] = self.chart_path.read_bytes()
        log.info("wrote %s", self.chart_path)

    def reload(self) -> None:
        self.config = load_config(self.config_path)

    def write_text(self, path: Path, text: str) -> None:
        """Write a generated text file (song.ini, song.toml), backing up a changed previous copy first."""
        self.backup(path)
        path.write_text(text, encoding="utf-8")
        self._written[path] = path.read_bytes()

    def backup(self, path: Path) -> Path | None:
        """Copy ``path`` into .history/ unless it's exactly what this process last wrote there
        (so a multi-step `build` keeps one pre-run copy, not one per step)."""
        if not path.exists() or self._written.get(path) == path.read_bytes():
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self.dir(".history") / f"{path.stem}.{stamp}{path.suffix}"
        n = 1
        while dest.exists():
            dest = dest.with_name(f"{path.stem}.{stamp}-{n}{path.suffix}")
            n += 1
        shutil.copy2(path, dest)
        return dest

    # --- creation ---------------------------------------------------------------------------
    @classmethod
    def init(cls, root: str | Path, name: str = "", artist: str = "", charter: str = "",
             template: str | None = None, chart: str | None = None,
             stems_dir: str | None = None, workspace: bool = False, stems: dict[str, Path] | None = None,
             tempo: tuple[float, float] | None = None, instruments: list[str] | None = None) -> "SongProject":
        """``stems_dir``: use audio from another folder in place (absolute paths) instead of <work>/stems.
        ``workspace``: keep song.toml and working files in <root>/_yarginator (the video layout).
        ``stems`` / ``tempo`` (bpm, first downbeat) / ``instruments``: known up front instead of detected."""
        root = Path(root).resolve()
        work = root / WORK_DIR if workspace else root
        (work / "stems").mkdir(parents=True, exist_ok=True)
        cfg = work / SONG_FILE
        if cfg.exists():
            raise FileExistsError(f"{cfg} already exists")
        if chart:
            shutil.copy2(chart, root / "notes.mid")
        if stems is None:
            stems = detect_stems(Path(stems_dir).resolve() if stems_dir else work / "stems")
        tempo_lines = ("# bpm = 120.0                 # leave unset to auto-detect, then copy the detected value here to lock it\n"
                       "# first_downbeat_s = 1.234\n")
        if tempo:
            bpm = f"bpm = {tempo[0]}"
            tempo_lines = (f"{bpm:<30}# detected: check it against the audio in REAPER\n"
                           f"first_downbeat_s = {tempo[1]}\n")
        blank_tpl = "# template = 'C:/path/to/template.RPP'   # unset = built-in template; \"none\" = blank project\n"
        text = Template(resources.files("yarginator.templates").joinpath("song.toml").read_text("utf-8"))
        cfg.write_text(text.substitute(
            name=name or root.name, artist=artist, charter=charter,
            instruments=json.dumps(instruments if instruments is not None else suggest_instruments(stems)),
            tempo_mode="chart" if chart else "click",  # an imported chart already has a tempo map
            stems=stem_lines(stems, work) or '# drums = "stems/drums.wav"\n',
            tempo_lines=tempo_lines,
            template=(f"template = {json.dumps(Path(template).resolve().as_posix(), ensure_ascii=False)}\n"
                      if template else blank_tpl),
        ), encoding="utf-8")
        return cls(root)

    def set_stems(self, stems: dict[str, Path], instruments: list[str] | None = None) -> None:
        """Rewrite the [stems] table (and optionally [chart] instruments) in song.toml, keeping the rest."""
        text = self.config_path.read_text("utf-8-sig")
        body = stem_lines(stems, self.work)
        m = re.search(r"^\[stems\][^\n]*\n(.*?)(?=^\[)", text, flags=re.M | re.S)
        text = (text[:m.start(1)] + body + "\n" + text[m.end(1):]) if m else text + f"\n[stems]\n{body}"
        if instruments is not None:
            text = re.sub(r"^instruments\s*=.*$", f"instruments = {json.dumps(instruments)}", text, count=1,
                          flags=re.M)
        self.write_text(self.config_path, text)
        self.reload()


def stem_lines(stems: dict[str, Path], base: Path) -> str:
    def rel(p: Path) -> str:
        return (p.relative_to(base) if p.is_relative_to(base) else p).as_posix()
    return "".join(f"{k} = {json.dumps(rel(Path(p)), ensure_ascii=False)}\n" for k, p in stems.items())


STEM_INSTRUMENTS = {"drums": "drums", "guitar": "guitar", "rhythm": "rhythm", "bass": "bass", "keys": "keys",
                    "vocals": "vocals", "harm1": "harmonies", "harm2": "harmonies", "harm3": "harmonies"}


def suggest_instruments(stems: dict[str, Path]) -> list[str]:
    """Instruments implied by the stems found (all of them if there are no instrument stems)."""
    from .core.instruments import INSTRUMENTS
    found = {STEM_INSTRUMENTS[k] for k in stems if k in STEM_INSTRUMENTS}
    return [i for i in INSTRUMENTS if i in found] or [i for i in INSTRUMENTS if i != "keys5"]


def stem_key(filename_stem: str) -> str:
    words = re.findall(r"[a-z]+\d*", filename_stem.lower())
    for w in reversed(words):
        for prefix, key in STEM_KEYWORDS:
            if w.startswith(prefix):
                return key
    return "_".join(words) or filename_stem


def detect_stems(folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in sorted(folder.iterdir()) if folder.exists() else []:
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        key = stem_key(p.stem)
        if key in found:
            log.warning("stems %s and %s both look like %r; keeping the first", found[key].name, p.name, key)
            continue
        found[key] = p
    return found
