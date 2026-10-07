"""Command line entry point: ``yarginator <command> ...``.

Each command is a small function registered with ``@command``; add new ones the same way.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from . import __version__, lighting, parts, pipeline
from .project import SongProject

COMMANDS: dict[str, tuple[Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], int | None], str]] = {}


def command(name: str, help: str):
    def deco(fn):
        COMMANDS[name] = (getattr(fn, "_args", lambda p: None), fn, help)
        return fn
    return deco


def args(setup: Callable[[argparse.ArgumentParser], None]):
    def deco(fn):
        fn._args = setup
        return fn
    return deco


def _song(p):
    p.add_argument("song", nargs="?", default=".", help="song folder (default: current directory)")


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# --- commands --------------------------------------------------------------------------------
@command("init", "create a song folder with song.toml (drop stems into <song>/stems first to auto-fill [stems])")
@args(lambda p: (_song(p), p.add_argument("--name", default=""), p.add_argument("--artist", default=""),
                 p.add_argument("--charter", default=""), p.add_argument("--template", help="template .RPP"),
                 p.add_argument("--chart", help="existing notes.mid to start from"),
                 p.add_argument("--stems", help="use audio from this folder in place instead of <song>/stems")))
def cmd_init(a):
    proj = SongProject.init(a.song, a.name, a.artist, a.charter, a.template, a.chart, a.stems)
    print(f"created {proj.root / 'song.toml'}")
    stems = proj.config.stems
    print("stems: " + (", ".join(f"{k}={v}" for k, v in stems.items()) if stems else "none found; add them under [stems]"))


@command("new", "start a song from a music video: video.mp4 + song.ogg + tempo + song.ini in <dest>/<Artist - Title>")
@args(lambda p: (p.add_argument("video", help="the downloaded video file"),
                 p.add_argument("--dest", help="folder to create the song folder in (default: [paths] projects "
                                               "in ~/.yarginator.toml; unset = the video's own folder)"),
                 p.add_argument("--name", default="", help="song title (default: from the file name)"),
                 p.add_argument("--artist", default="", help="(default: from the file name)"),
                 p.add_argument("--charter", default=""),
                 p.add_argument("--no-tempo", action="store_true", help="don't detect the tempo"),
                 p.add_argument("--separate", action="store_true", help="also separate stems with the built-in models"),
                 p.add_argument("--drum-split", action="store_true", help="with --separate: split the drums too")))
def cmd_new(a):
    from . import workflow
    dest = a.dest or workflow.user_paths().get("projects")
    proj = workflow.new_from_video(Path(a.video), Path(dest) if dest else None, a.name, a.artist, a.charter,
                                   not a.no_tempo)
    t = proj.config.tempo
    print(f"created {proj.root}")
    print(f"  {', '.join(sorted(p.name for p in proj.root.iterdir() if p.is_file()))}")
    print(f"  tempo: " + (f"{t.bpm} BPM, first downbeat {t.first_downbeat_s} s" if t.bpm else "not set"))
    if a.separate:
        _print_stems(workflow.separate_song(proj, a.drum_split))
        print(f'next: yarginator build "{proj.root}"')
    else:
        print(f'next: yarginator separate "{proj.root}"   (or run your own separator on '
              f'{proj.work / "source" / "mix.flac"}, then: yarginator stems "{proj.root}" --from <its output>)\n'
              f'      yarginator build "{proj.root}"')


def _print_stems(rep):
    for k, p in rep.stems.items():
        lvl = rep.levels.get(k)
        note = " (near silent: not charted)" if k in rep.empty else ""
        print(f"  {k:<13} {p.name}" + (f"  {lvl:6.1f} dB" if lvl is not None else "") + note)
    print("instruments: " + ", ".join(rep.instruments) + "   (edit [chart] instruments in song.toml to change)")


@command("separate", "split the song's audio into stems with the built-in models (vocals, instrumental, drums, "
                     "bass, guitar, keys, other) and register them")
@args(lambda p: (_song(p), p.add_argument("--drum-split", action="store_true",
                                          help="also split drums into kick/snare/toms/hihat/ride/crash"),
                 p.add_argument("--force", action="store_true", help="redo passes that already have output"),
                 p.add_argument("--device", choices=["auto", "cpu", "gpu"],
                                help="default: [separation] device, else auto (GPU if `setup-gpu` was run)")))
def cmd_separate(a):
    from . import workflow
    rep = workflow.separate_song(SongProject(a.song), a.drum_split, a.force, a.device)
    _print_stems(rep)


@command("setup-gpu", "install GPU stem separation (DirectML: AMD / NVIDIA / Intel) in its own environment")
def cmd_setup_gpu(a):
    from .separate import gpu_env, setup_gpu
    name = setup_gpu()
    print(f"GPU separation ready on {name} ({gpu_env()}); `separate` now uses it by default")


@command("stems", "register separated stems (copied from --from into the song's _yarginator/stems) and pick instruments")
@args(lambda p: (_song(p), p.add_argument("--from", dest="from_dir", help="folder with the separator's output")))
def cmd_stems(a):
    from . import workflow
    rep = workflow.register_stems(SongProject(a.song), Path(a.from_dir) if a.from_dir else None)
    if rep.copied:
        print(f"copied {len(rep.copied)} file(s)")
    _print_stems(rep)


@command("finalize", "song.ogg/vocals.ogg from the stems, song.ini, and a last check (optionally move the folder)")
@args(lambda p: (_song(p), p.add_argument("--move-to", help="move the song folder here when done (e.g. your Complete folder)")))
def cmd_finalize(a):
    from . import workflow
    notes, final = workflow.finalize(SongProject(a.song), Path(a.move_to) if a.move_to else None)
    for n in notes:
        print(("! " if n.startswith("WARNING") else "- ") + n.removeprefix("WARNING: "))
    print(f"ready: {final}")


@command("tempo", "build the tempo map (song.toml [tempo]) and the BEAT track")
@args(_song)
def cmd_tempo(a):
    pipeline.step_tempo(SongProject(a.song))


@command("chart", "run part generators (skips parts that already have notes unless named or --force)")
@args(lambda p: (_song(p), p.add_argument("--parts", type=_csv, help="comma-separated, e.g. vocals,harm1"),
                 p.add_argument("--force", action="store_true")))
def cmd_chart(a):
    pipeline.step_chart(SongProject(a.song), only=a.parts, force=a.force)


@command("lights", "(re)generate the VENUE light show; keeps camera/other VENUE events")
@args(lambda p: (_song(p), p.add_argument("--preset", help="preset name or .toml path (overrides song.toml)")))
def cmd_lights(a):
    pipeline.step_lights(SongProject(a.song), force=True, preset=a.preset)


@command("ini", "write song.ini from song.toml")
@args(_song)
def cmd_ini(a):
    pipeline.step_ini(SongProject(a.song))


@command("reaper", "generate the REAPER project (template + stems + tempo + chart + review markers)")
@args(lambda p: (_song(p), p.add_argument("--force", action="store_true", help="overwrite an existing project")))
def cmd_reaper(a):
    pipeline.step_reaper(SongProject(a.song), force=a.force)


@command("build", "run every step in order: " + " -> ".join(pipeline.STEPS))
@args(lambda p: (_song(p), p.add_argument("--steps", type=_csv, help="subset of steps"),
                 p.add_argument("--force", action="store_true", help="redo charted parts / lights / REAPER project")))
def cmd_build(a):
    pipeline.build(SongProject(a.song), a.steps, force=a.force)


@command("info", "summarise a notes.mid (or a song folder's chart)")
@args(lambda p: p.add_argument("path", nargs="?", default="."))
def cmd_info(a):
    from .core.chart import Chart
    path = Path(a.path)
    if path.is_dir():
        path = SongProject(path).chart_path
    chart = Chart.load(path)
    tm = chart.tempo
    bpms = [t.bpm for t in tm.tempos]
    print(f"{path}  ppq={chart.ppq}  length={tm.t2s(chart.end_tick()):.1f}s")
    print(f"tempo: {len(bpms)} change(s), {min(bpms):.2f}-{max(bpms):.2f} BPM; first {bpms[0]:.2f}"
          + (f", then {bpms[1]:.2f}" if len(bpms) > 1 else ""))
    for name, tr in chart.tracks.items():
        notes, kinds = tr.notes(), Counter(m.type for _, m in tr.events)
        extra = ", ".join(f"{k} {v}" for k, v in kinds.items() if k in ("lyrics", "text"))
        print(f"  {name:<20} {len(notes):>5} notes" + (f"  ({extra})" if extra else ""))


@command("view", "text view of a part, one row per bar (e.g. view songs/x bass --bars 17-24 --diff hard)")
@args(lambda p: (_song(p), p.add_argument("part", help="instrument (bass, guitar, rhythm, drums, keys, keys5) or track name"),
                 p.add_argument("--bars", default="1-16", help="range, e.g. 17-24"),
                 p.add_argument("--diff", default="expert", choices=["expert", "hard", "medium", "easy", "all"])))
def cmd_view(a):
    from .core.chart import Chart
    from .core.instruments import INSTRUMENTS, Difficulty
    from .feedback.textview import view
    path = Path(a.song)
    chart = Chart.load(SongProject(path).chart_path if path.is_dir() else path)
    track = INSTRUMENTS[a.part][0] if a.part in INSTRUMENTS else a.part
    lo, _, hi = a.bars.partition("-")
    diffs = list(reversed(Difficulty)) if a.diff == "all" else [Difficulty[a.diff.upper()]]
    for d in diffs:
        print(f"--- {track} {d.name.lower()}")
        print(view(chart, track, d, (int(lo), int(hi or lo))))


@command("generators", "list registered part and lighting generators")
def cmd_generators(a):
    print("parts:")
    for key, cls in sorted(parts.REGISTRY.items()):
        print(f"  {key:<10} {'' if cls.implemented else '(not yet) '}{cls.description}")
    print("lighting:")
    for key, cls in sorted(lighting.LIGHTING.items()):
        print(f"  {key:<10} {cls.description}")
    from .lighting.base import builtin_presets
    print("lighting presets: " + ", ".join(builtin_presets()))


@command("drums-learn", "teach the drum charter from a finished chart + its drum stem (song folder, or notes.mid with --stems)")
@args(lambda p: (p.add_argument("chart", help="song folder, or a notes.mid"),
                 p.add_argument("--stems", nargs="+", help="drum stem file(s) (several are mixed); default: the song's drums stem"),
                 p.add_argument("--offset", type=float, default=None,
                                help="where the stem starts on the chart's timeline, in seconds (REAPER: item position "
                                     "minus its start offset). A song folder uses its [audio] offset_s."),
                 p.add_argument("--name", help="name for the training set (default: the folder name)")))
def cmd_drums_learn(a):
    from .audio import drums as dr
    from .core.chart import Chart
    from .parts.drums import expert_hits
    path = Path(a.chart)
    if path.is_dir() and any((d / "song.toml").exists() for d in (path, path / "_yarginator")):
        proj = SongProject(path)
        chart = Chart.load(proj.chart_path)
        stems = proj.stems()
        keys = proj.config.parts.get("drums").options.get("stems", ["drums"]) if "drums" in proj.config.parts else ["drums"]
        paths = [Path(s) for s in a.stems] if a.stems else [stems[k] for k in keys if k in stems]
        offset = a.offset if a.offset is not None else -proj.config.audio_offset_s
        cache = proj.dir(".cache")
    else:
        chart = Chart.load(path / "notes.mid" if path.is_dir() else path)
        paths = [Path(s) for s in a.stems or []]
        offset = a.offset or 0.0
        cache = None
    if not paths:
        raise FileNotFoundError("no drum stem: pass --stems")
    hits = expert_hits(chart)
    if not hits:
        raise ValueError("the chart has no Expert PART DRUMS notes to learn from")
    on = dr.analyse(paths, cache)
    shifted = [(t - offset, c) for t, c in hits]
    fine = dr.align(on, shifted)
    name = a.name or (path if path.is_dir() else path.parent).name
    out = dr.user_data_dir() / f"{name}.npz"
    stats = dr.learn(on, [(t + fine, c) for t, c in shifted], out, name)
    share = stats["matched"] / max(stats["hits"], 1)
    print(f"{name}: {stats['hits']} charted hits, {stats['matched']} matched to onsets ({share:.0%}), "
          f"alignment {fine * 1000:+.0f} ms on top of the offset -> {out}")
    if share < 0.7:
        logging.warning("fewer than 70%% of hits matched: check --offset (where the stem starts on the chart's timeline)")


@command("vocals-pitch", "standalone: pitch an existing chart's timed lyrics from a vocal stem")
@args(lambda p: (p.add_argument("audio"), p.add_argument("chart"),
                 p.add_argument("-o", "--out", default="notes_with_vocals.mid"),
                 p.add_argument("--part", default="vocals", help="vocals | harm1 | harm2 | harm3"),
                 p.add_argument("--offset", type=float, default=0.0, help="seconds to shift the stem later(+)/earlier(-)"),
                 p.add_argument("--cache", help="cache folder")))
def cmd_vocals_pitch(a):
    from .core.chart import Chart
    from .feedback import write_part_report
    from .parts.base import PartContext
    chart = Chart.load(a.chart)
    charter = parts.get_charter(a.part)
    ctx = PartContext(chart, stems={charter.stem_keys[0]: Path(a.audio)}, audio_offset_s=a.offset,
                      cache_dir=Path(a.cache) if a.cache else None)
    result = charter.generate(ctx)
    for tr in result.tracks:
        chart.set_track(tr)
    chart.save(a.out)
    out = Path(a.out)
    rep = write_part_report(out.parent, out.stem + ".report", result.rows)
    print(f"{result.summary} | wrote {out}" + (f" and {rep}" if rep else ""))
    return 0 if result.tracks else 1


# --- main ------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yarginator", description="Charting toolkit for YARG.")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (setup, fn, help_) in COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=help_)
        setup(p)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        return a.fn(a) or 0
    except (FileNotFoundError, FileExistsError, RuntimeError, KeyError, ValueError) as e:
        if a.verbose:
            raise
        logging.error("%s", e.args[0] if isinstance(e, KeyError) and e.args else e)  # KeyError's str() adds quotes
        return 1


if __name__ == "__main__":
    sys.exit(main())
