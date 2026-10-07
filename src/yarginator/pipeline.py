"""The steps that turn a song folder into a chart, in order. Each step loads what it needs from disk
and writes its result back, so any step can be re-run on its own after hand edits.

    tempo  -> tempo map into notes.mid (re-timing existing events so they keep their position in seconds)
    chart  -> run part generators
    lights -> VENUE lighting
    ini    -> song.ini
    reaper -> REAPER project for fine-tuning
"""
from __future__ import annotations

import logging
from typing import Callable

from . import lighting, parts
from .config import PartConfig
from .core import instruments as _ins
from .core.chart import Chart, grid_shift_s
from .core.timing import TempoMap
from .feedback import read_review, write_part_report, write_review
from .parts.base import PartContext
from .project import SongProject

log = logging.getLogger(__name__)


# tracks timed freely to the audio (not quantized): they keep their seconds when the grid slides
VOCAL_TRACKS = (_ins.VOCALS, *_ins.HARM.values())


def part_context(project: SongProject, chart: Chart, pc: PartConfig | None = None) -> PartContext:
    pc = pc or PartConfig()
    return PartContext(chart=chart, stems=project.stems(), options=pc.options, stem_override=pc.stem,
                       cache_dir=project.dir(".cache"), audio_offset_s=project.config.audio_offset_s,
                       root=project.work)


# --- steps -----------------------------------------------------------------------------------
def step_tempo(project: SongProject, **_) -> None:
    cfg = project.config.tempo
    if cfg.mode == "chart":
        log.info("tempo: mode=chart, keeping the tempo map in %s", project.chart_path.name)
        return
    stems = project.stems()
    stem = next((stems[k] for k in (cfg.stem, "song", "drums") if k in stems), None)
    if cfg.mode == "click":
        bpm, downbeat = cfg.bpm, cfg.first_downbeat_s
        if bpm is None or downbeat is None:
            if stem is None:
                raise RuntimeError(f"tempo: set bpm and first_downbeat_s in song.toml, or provide stem {cfg.stem!r} to detect them")
            from .audio.tempo import estimate_click
            est_bpm, est_downbeat = estimate_click(stem, cfg.beats_per_bar, bpm_range=tuple(cfg.bpm_range))
            bpm = bpm if bpm is not None else round(est_bpm, 4)
            downbeat = downbeat if downbeat is not None else round(est_downbeat, 4)
            log.warning("tempo: detected bpm=%s first_downbeat_s=%s -- check them, then copy into [tempo] to lock",
                        bpm, downbeat)
        new = TempoMap.click_tracked(bpm, downbeat, cfg.beats_per_bar)
    elif cfg.mode == "beats":
        if stem is None:
            raise RuntimeError(f"tempo: mode=beats needs stem {cfg.stem!r}")
        from .audio.tempo import detect_beats
        new = TempoMap.from_beats(detect_beats(stem), cfg.beats_per_bar)
    else:
        raise ValueError(f"unknown tempo mode {cfg.mode!r} (click | beats | chart)")

    chart = project.load_chart()
    shift = grid_shift_s(chart.tempo, new) if chart.has_events() else None
    if shift is not None and abs(shift) < 1e-6:
        chart.tempo = new
    elif shift is not None:
        # the grid only slid: parts quantized to it move with it; free-timed vocals stay with the audio
        log.info("tempo: grid moved %+.0f ms; grid parts move with it, vocals stay put", shift * 1000)
        chart.retime(new, only=set(VOCAL_TRACKS) & set(chart.tracks))
    elif chart.has_events():
        log.warning("tempo: re-timing existing chart events onto the new tempo map")
        chart.retime(new)
    else:
        chart.tempo = new
    # The BEAT track is derived from tempo; rebuild it straight away.
    for tr in parts.get_charter("beat").generate(part_context(project, chart)).tracks:
        chart.set_track(tr)
    project.save_chart(chart)


def step_chart(project: SongProject, only: list[str] | None = None, force: bool = False, **_) -> None:
    from .core.instruments import tracks_for
    chart = project.load_chart()
    cfg_parts = project.config.parts
    wanted = tracks_for(project.config.instruments)
    keys = only or [k for k, pc in cfg_parts.items() if pc.enabled]
    reports = project.dir("reports")
    review, ran = [], set()
    for key in keys:
        pc = cfg_parts.get(key, PartConfig())
        charter = parts.get_charter(pc.generator or key)
        if wanted is not None and not set(charter.track_names) & wanted:
            if only:
                log.warning("chart: %s: not in [chart] instruments; add it there first", key)
            continue
        if not charter.implemented:
            log.info("chart: %s: no generator yet; the REAPER project gets an empty track to chart by hand", key)
            continue
        existing = [chart.tracks.get(n) for n in charter.track_names]
        if not (force or only or charter.always_run) and any(t is not None and charter.is_charted(t) for t in existing):
            log.info("chart: %s: already charted, skipping (name it with --parts or use --force to redo)", key)
            continue
        result = charter.generate(part_context(project, chart, pc))
        for tr in result.tracks:
            chart.set_track(tr)
        for r in result.review:  # file under the part (not the generator), so re-runs replace them
            r.part = key
        write_part_report(reports, key, result.rows)
        review += result.review
        ran.add(key)
        log.info("chart: %s: %s", key, result.summary or f"{len(result.tracks)} track(s)")
    if wanted is not None:
        prune_tracks(chart, wanted)
    project.save_chart(chart)
    write_review(reports, review, replace_parts=ran)


def prune_tracks(chart: Chart, wanted: set[str]) -> None:
    """Drop tracks for instruments the song doesn't have. Tracks with content are kept (with a
    warning) so dropping an instrument from the list never destroys charting work."""
    for name in [n for n in chart.tracks if n not in wanted]:
        tr = chart.tracks[name]
        content = tr.has_notes() or any(m.type in ("lyrics", "text") for _, m in tr.events)
        if content:
            log.warning("chart: %s isn't in [chart] instruments but has charted content; keeping it "
                        "(delete it in REAPER if you really want it gone)", name)
        else:
            del chart.tracks[name]


def step_lights(project: SongProject, force: bool = False, preset: str | None = None, **_) -> None:
    from .core import instruments
    from .lighting.cues import is_lighting_event
    chart = project.load_chart()
    venue = chart.tracks.get(instruments.VENUE)
    if venue and not force and any(is_lighting_event(m) for _, m in venue.events):
        log.info("lights: VENUE already has lighting, skipping (use --force / the lights command to redo)")
        return
    cfg = project.config.lighting
    ctx = lighting.LightingContext(part=part_context(project, chart),
                                   preset=lighting.load_preset(preset or cfg.preset, project.work),
                                   options=cfg.options, energy_stem=cfg.stem)
    chart.set_track(lighting.get_lighting(cfg.generator).generate(ctx))
    project.save_chart(chart)


def step_ini(project: SongProject, **_) -> None:
    meta = project.config.song
    keys = {"name": meta.name or project.name, "artist": meta.artist, "album": meta.album, "genre": meta.genre,
            "year": meta.year, "charter": meta.charter, "delay": 0}
    keys.update(project.config.song_ini)
    text = "[song]\n" + "".join(f"{k} = {v}\n" for k, v in keys.items())
    project.write_text(project.song_ini_path, text)
    log.info("wrote %s", project.song_ini_path)


def wanted_tracks(instruments: list[str] | None) -> list[str] | None:
    """Ordered chart track names for a song's instruments, then BEAT/EVENTS/VENUE."""
    from .core.instruments import ALWAYS_TRACKS, INSTRUMENTS, tracks_for
    if instruments is None:
        return None
    tracks_for(instruments)  # validates the names
    return [t for i in instruments for t in INSTRUMENTS[i]] + list(ALWAYS_TRACKS)


def step_reaper(project: SongProject, force: bool = False, **_) -> None:
    from .reaper import project as rproject
    from .reaper.rpp import dumps
    out = project.dir("reaper") / f"{project.root.name}.RPP"
    if out.exists() and not force:
        log.warning("reaper: %s exists (it may have your edits); use --force to regenerate (old copy goes to .history)", out)
        return
    project.backup(out)
    from .reaper import DEFAULT_TEMPLATE, bundled, resolve
    rc, meta = project.config.reaper, project.config.song
    tpl = resolve(rc.template, bundled(DEFAULT_TEMPLATE), project.work)
    if tpl is not None and not tpl.is_file():
        raise FileNotFoundError(f"reaper template {tpl} not found")
    cmaps = resolve(rc.color_maps, bundled("color_maps"), project.work)
    if cmaps is not None and not cmaps.is_dir():
        log.warning("reaper: color_maps folder %s not found; notes won't be coloured", cmaps)
        cmaps = None
    keymaps = resolve(rc.note_names, bundled("note_names"), project.work)
    if keymaps is not None and not keymaps.is_dir():
        log.warning("reaper: note_names folder %s not found; tracks keep the template's note names", keymaps)
        keymaps = None
    title = " - ".join(x for x in (meta.artist, project.name) if x)
    review = read_review(project.dir("reports")) if rc.review_markers else []  # [] also clears old markers
    root = rproject.build(project.load_chart(), project.stems(), review,
                          tpl, rc.tracks, cmaps, title, wanted_tracks(project.config.instruments), keymaps,
                          rc.preview_fx, rc.preview_fx_on)
    text = dumps(root)
    if tpl is not None and b"\r\n" in tpl.read_bytes()[:4096]:
        text = text.replace("\n", "\r\n")  # keep the template's line endings
    out.write_bytes(text.encode("utf-8", errors="surrogateescape"))
    log.info("wrote %s", out)


STEPS: dict[str, Callable[..., None]] = {
    "tempo": step_tempo,
    "chart": step_chart,
    "lights": step_lights,
    "ini": step_ini,
    "reaper": step_reaper,
}


def build(project: SongProject, steps: list[str] | None = None, **kw) -> None:
    for name in steps or list(STEPS):
        log.info("== %s ==", name)
        STEPS[name](project, **kw)
