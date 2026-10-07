"""Tracks derived from the tempo map and song length rather than from audio: BEAT and EVENTS."""
from __future__ import annotations

from ..core import instruments
from ..core.chart import Track
from .base import PartCharter, PartContext, PartResult, register


@register
class BeatTrack(PartCharter):
    key = "beat"
    track_names = (instruments.BEAT,)
    description = "BEAT track (downbeat/beat notes) from the tempo map; rebuilt every run"
    always_run = True

    def generate(self, ctx: PartContext) -> PartResult:
        tr = Track(instruments.BEAT)
        length = ctx.chart.ppq // 4
        for tick, down in ctx.tempo.beats(ctx.song_end_tick()):
            tr.add_note(tick, length, instruments.BEAT_DOWNBEAT if down else instruments.BEAT_UPBEAT)
        return PartResult([tr])


@register
class EventsTrack(PartCharter):
    key = "events"
    track_names = (instruments.EVENTS,)
    description = "Adds [music_start]/[music_end]/[end] to EVENTS if missing; never touches sections"
    always_run = True

    def generate(self, ctx: PartContext) -> PartResult:
        tr = ctx.existing(instruments.EVENTS)
        have = {txt.strip("[]") for _, txt in tr.texts()}
        end = ctx.song_end_tick()
        if "music_start" not in have:
            tr.add_text(min(ctx.tempo.first_downbeat_tick(), end), "[music_start]")
        if "music_end" not in have:
            tr.add_text(end, "[music_end]")
        if "end" not in have:
            tr.add_text(end + ctx.chart.ppq, "[end]")
        return PartResult([tr])
