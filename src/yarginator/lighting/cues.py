"""VENUE-track lighting vocabulary (Rock Band 3 style), which YARG reads and drives its lighting /
DMX output from.

Verify additions against YARG's docs before relying on them; keep every format fact in this file.
"""
from __future__ import annotations

import re

from ..core.chart import Msg

# [lighting (<cue>)] text events
CUES = {
    # automatic, mood-based
    "intro", "verse", "chorus", "loop_warm", "loop_cool", "harmony", "frenzy", "silhouettes",
    "silhouettes_spot", "searchlights", "sweep", "bre",
    # keyframed: step to the next look on each [next] keyframe
    "manual_warm", "manual_cool", "dischord", "stomp",
    # one-shots / accents
    "strobe_slow", "strobe_fast", "blackout_slow", "blackout_fast", "blackout_spot", "flare_slow", "flare_fast",
    "",  # [lighting ()] = default
}
KEYFRAMED = {"manual_warm", "manual_cool", "dischord", "stomp"}

# Lighting keyframe notes on the VENUE track
KEYFRAME_NEXT = 48
KEYFRAME_PREV = 49
KEYFRAME_FIRST = 50
KEYFRAME_NOTES = (KEYFRAME_NEXT, KEYFRAME_PREV, KEYFRAME_FIRST)

FOG_ON = "[FogOn]"
FOG_OFF = "[FogOff]"

_CUE_RE = re.compile(r"^\[lighting \((.*)\)\]$")


def cue_text(cue: str) -> str:
    if cue not in CUES:
        raise ValueError(f"unknown lighting cue {cue!r}")
    return f"[lighting ({cue})]"


def parse_cue(text: str) -> str | None:
    m = _CUE_RE.match(text.strip())
    return m.group(1) if m else None


def is_lighting_event(msg: Msg) -> bool:
    """Events the lighting generator owns (and may delete when regenerating). Camera cuts,
    post-processing and anything else on VENUE are left alone."""
    if msg.type in ("note_on", "note_off"):
        return msg.note in KEYFRAME_NOTES
    if msg.type == "text":
        return parse_cue(msg.text) is not None or msg.text.strip() in (FOG_ON, FOG_OFF)
    return False
