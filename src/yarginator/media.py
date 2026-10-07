"""Video in: split a downloaded music video into the files a YARG song folder needs.

    video  -> video.mp4 (or .webm): picture only, stream-copied (no re-encode, no quality loss)
           -> song.ogg: the soundtrack as Ogg Vorbis (YARG's song stem)
           -> <work>/source/mix.flac: the same soundtrack, lossless, for stem separation and analysis

Both audio files are decoded with the container's timeline, padded so that time 0 is the start
of the file. The audio therefore lines up with the video exactly as it played in the download.
The audio is never trimmed or moved after that; the tempo map bends to it.

ffmpeg comes from the ``imageio-ffmpeg`` package (``pip install -e .[video]``) or the PATH.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}
# codecs YARG's video player handles per container
MP4_VIDEO = {"h264", "hevc", "mpeg4"}
WEBM_VIDEO = {"vp8", "vp9", "av1"}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found: pip install -e .[video] (bundles it) or put ffmpeg on the PATH")


def run(args: list[str]) -> str:
    """Run ffmpeg; returns stderr (where ffmpeg reports), raises with its last lines on failure."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg failed: {tail}")
    return p.stderr


@dataclass
class Probe:
    video_codec: str | None
    audio_codec: str | None
    duration_s: float | None


def probe(path: Path) -> Probe:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    err = p.stderr
    v = re.search(r"Stream #\S+.*?: Video: (\w+)", err)
    a = re.search(r"Stream #\S+.*?: Audio: (\w+)", err)
    d = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(d[1]) * 3600 + int(d[2]) * 60 + float(d[3]) if d else None
    return Probe(v[1] if v else None, a[1] if a else None, dur)


# --- names ---------------------------------------------------------------------------------------
_JUNK = re.compile(r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|visuali[sz]er|4k|hd\b|hq\b|remaster|"
                   r"\d{3,4}p|explicit|clean)[^\)\]]*[\)\]]", re.I)


def parse_title(filename: str) -> tuple[str, str]:
    """``(artist, title)`` from a downloaded video's file name, e.g. JDownloader's
    ``"I Prevail - VIOLENT NATURE (Official Music Video) - IPrevailBand (1080p, h264)"``
    -> ``("I Prevail", "VIOLENT NATURE")``. No " - " in the name: ``("", whole name)``."""
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)          # "... (1)" duplicate-download suffix
    stem = _JUNK.sub("", stem)
    stem = re.sub(r"\s*[_|•]\s*official.*$", "", stem, flags=re.I).strip()
    parts = [p.strip() for p in stem.split(" - ") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]                    # a third part is the uploader / channel
    return "", stem


def safe_folder_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name).strip().rstrip(".")
    return re.sub(r"\s+", " ", name) or "song"


# --- extraction ----------------------------------------------------------------------------------
def extract_video(src: Path, dest_dir: Path, probe_: Probe) -> Path:
    """The picture only, stream-copied, as video.mp4 (H.264/HEVC) or video.webm (VP8/VP9/AV1)."""
    codec = (probe_.video_codec or "").lower()
    if codec in WEBM_VIDEO:
        out = dest_dir / "video.webm"
    else:
        out = dest_dir / "video.mp4"
        if codec not in MP4_VIDEO:
            log.warning("video codec %s may not play in YARG; re-encode it to H.264 if it doesn't", codec or "?")
    args = ["-y", "-i", str(src), "-map", "0:v:0", "-c:v", "copy", "-an", "-sn", "-dn"]
    if out.suffix == ".mp4":
        args += ["-movflags", "+faststart"]
    run(args + [str(out)])
    return out


def _audio_args(src: Path) -> list[str]:
    # pad to the container's time 0, so the audio sits where it played against the picture
    return ["-y", "-i", str(src), "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", "aresample=async=1:first_pts=0"]


def extract_lossless(src: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(_audio_args(src) + ["-c:a", "flac", str(out)])
    return out


def mix_to_flac(srcs: list[Path], out: Path) -> Path:
    """Sum several stems (no level normalisation, so they add back up to the mix)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y"]
    for s in srcs:
        args += ["-i", str(s)]
    args += ["-filter_complex", f"amix=inputs={len(srcs)}:normalize=0:duration=longest", "-c:a", "flac", str(out)]
    run(args)
    return out


def to_ogg(src: Path, out: Path, quality: int = 8) -> Path:
    """Ogg Vorbis (YARG's song.ogg / vocals.ogg); quality 8 is ~256 kbps."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run(_audio_args(src) + ["-c:a", "libvorbis", "-q:a", str(quality), str(out)])
    return out
