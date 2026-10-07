"""Default light show: one cue per section (from the preset), keyframes for keyframed cues,
accents and fog at section starts, blackout at the end. Falls back to stem energy when the chart
has no sections."""
from __future__ import annotations

import logging
import re

import numpy as np

from ..core import instruments
from ..core.chart import Track
from . import cues
from .base import LightingContext, LightingGenerator, register_lighting

log = logging.getLogger(__name__)


def normalise(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def match_section(name: str, table: dict[str, str]) -> tuple[str, str]:
    """-> (matched keyword, cue)"""
    n = normalise(name)
    for keyword, cue in table.items():
        if keyword != "default" and keyword in n:
            return keyword, cue
    return "default", table.get("default", "loop_warm")


def energy_sections(ctx: LightingContext, end_tick: int) -> list[tuple[int, str, str]]:
    """``(tick, keyword, cue)`` from stem loudness, merging neighbouring chunks of the same grade."""
    # Drums first: modern masters are compressed flat (the mix barely moves +-2 dB between sections),
    # while the drum stem drops out clearly in intros, breaks and breakdowns.
    stem = ctx.part.stem(*([ctx.energy_stem] if ctx.energy_stem else []), "drums", "song", "bass", "guitar")
    if stem is None:
        return []
    from ..audio.analysis import rms_envelope
    t, db = rms_envelope(stem, ctx.part.cache_dir)
    t = t + ctx.part.audio_offset_s
    cfg = ctx.preset.get("energy", {})
    tm = ctx.chart.tempo
    # count chunks from the first real downbeat, not from the offset measure
    bars = [b for b in tm.downbeats(end_tick) if b >= tm.first_downbeat_tick()] or tm.downbeats(end_tick)
    step = int(cfg.get("segment_bars", 4))
    starts = bars[::step]
    levels = []
    for a, b in zip(starts, starts[1:] + [end_tick]):
        i, j = np.searchsorted(t, tm.t2s(a)), np.searchsorted(t, tm.t2s(b))
        levels.append(float(np.mean(db[i:j])) if j > i else -120.0)
    med = float(np.median([lv for lv in levels if lv > -100] or [0.0]))
    out = []
    for tick, lv in zip(starts, levels):
        grade = ("high" if lv >= med + cfg.get("high_db", 3.0)
                 else "low" if lv <= med + cfg.get("low_db", -6.0) else "mid")
        if not out or out[-1][1] != grade:
            out.append((tick, grade, cfg.get(grade, "loop_warm")))
    if out:  # the first look also covers the offset measure
        out[0] = (0, *out[0][1:])
    return out


@register_lighting
class DefaultShow(LightingGenerator):
    key = "default"
    description = "Section-driven show from a TOML preset; energy-driven when there are no sections"

    def generate(self, ctx: LightingContext) -> Track:
        chart, preset = ctx.chart, ctx.preset
        tm, ppq = chart.tempo, chart.ppq
        venue = ctx.part.existing(instruments.VENUE).remove(lambda t, m: cues.is_lighting_event(m))
        end = ctx.part.song_end_tick()
        events = chart.tracks.get(instruments.EVENTS)
        music_end = next((t for t, txt in (events.texts() if events else []) if txt.strip() == "[music_end]"), end)

        table = preset.get("sections", {"default": "loop_warm"})
        plan = [(tick, *match_section(name, table)) for tick, name in ctx.sections()]
        if not plan:
            plan = energy_sections(ctx, music_end)
            log.info("no [section] events; %s", f"using stem energy ({len(plan)} segments)" if plan
                     else "no stem for energy either, using one default cue")
        if not plan:
            plan = [(0, "default", table.get("default", "loop_warm"))]

        accents = preset.get("accents", {})
        fog_on = set(preset.get("fog", {}).get("on_sections", []))
        every = int(preset.get("keyframes", {}).get("every_beats", 2)) * ppq
        fogged = False
        for n, (tick, keyword, cue) in enumerate(plan):
            stop = plan[n + 1][0] if n + 1 < len(plan) else music_end
            main_at = tick
            if keyword in accents:
                acc = accents[keyword]
                venue.add_text(tick, cues.cue_text(acc["cue"]))
                main_at = min(tick + int(acc.get("beats", 1)) * ppq, stop)
            venue.add_text(main_at, cues.cue_text(cue))
            if cue in cues.KEYFRAMED:
                for k in range(main_at + every, stop, every):
                    venue.add_note(k, ppq // 8, cues.KEYFRAME_NEXT)
            if fogged and keyword not in fog_on:
                venue.add_text(tick, cues.FOG_OFF)
                fogged = False
            elif keyword in fog_on and not fogged:
                venue.add_text(tick, cues.FOG_ON)
                fogged = True
        if fogged:
            venue.add_text(music_end, cues.FOG_OFF)
        end_cue = preset.get("end", {}).get("cue", "blackout_slow")
        if end_cue:
            venue.add_text(music_end, cues.cue_text(end_cue))
        return venue
