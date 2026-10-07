"""Placeholders for the instrument generators that aren't written yet (drums: see drums.py).

Each stub says which tracks it owns and which stems it reads, so the pipeline, config and CLI
already treat them as parts. Write the real thing by filling in ``generate`` (or registering a new
class under a different key and pointing ``[parts.<x>] generator`` at it).

Building blocks you'll likely want: ``audio.analysis.onsets`` (note placement), quantising onset
seconds onto ``ctx.tempo`` ticks, and ``core.instruments`` for note numbers per difficulty.
"""
from __future__ import annotations

from ..core import instruments as ins
from .base import PartCharter, PartContext, PartResult, register


class _NotYet(PartCharter):
    implemented = False

    def generate(self, ctx: PartContext) -> PartResult:
        raise NotImplementedError(f"the {self.key!r} generator isn't written yet")


@register
class Guitar(_NotYet):
    key = "guitar"
    track_names = (ins.GUITAR,)
    stem_keys = ("guitar",)
    description = "5-fret lead guitar"


@register
class Rhythm(_NotYet):
    key = "rhythm"
    track_names = (ins.RHYTHM,)
    stem_keys = ("rhythm", "guitar")
    description = "5-fret rhythm guitar"

