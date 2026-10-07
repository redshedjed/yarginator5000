"""Preview FX on chart tracks: a note filter in front of the template's synth, so a part can be heard.

The filter (REAPER's stock JSFX "MIDI Note Filter") passes only the playable note range, so the
synth doesn't sound range shifts, phrase markers, percussion, overdrive and the like. The synth is
the one on the template's chart tracks (ReaSynth); tracks without one get a copy. The chain ships
bypassed (toggle it in REAPER) unless ``enabled``. Running it again updates the filter in place.

5-lane guitar / rhythm / bass get no FX at all (``NO_FX``): their gem notes aren't pitches.
"""
from __future__ import annotations

import re
import uuid

from .rpp import Line, Node

FILTER_JS = "MIDI/midi_note_filter"

# track -> playable note range passed to the synth
PREVIEW_RANGES = [
    (re.compile(r"^PART REAL_KEYS_[XHME]$", re.I), (48, 72)),   # pro keys C2-C4
    (re.compile(r"^(PART VOCALS|HARM[123])$", re.I), (36, 84)),  # vocal pitches
    (re.compile(r"^PART KEYS$", re.I), (96, 100)),               # 5-lane keys: Expert gems (rhythm only)
]
NO_FX = re.compile(r"^PART (GUITAR|RHYTHM|BASS)$", re.I)


def range_for(track_name: str) -> tuple[int, int] | None:
    return next((rng for pattern, rng in PREVIEW_RANGES if pattern.match(track_name)), None)


def _filter_node(lo: int, hi: int) -> Node:
    node = Node(f'JS {FILTER_JS} ""')
    # sliders: lowest key, highest key, pass other events (no); the rest unused
    node.append(Line(" ".join([str(lo), str(hi), "0"] + ["-"] * 61)))
    return node


def apply(track: Node, track_name: str, synth_chain=None, enabled: bool = False) -> bool:
    """Filter + synth for a previewable chart track (bypassed unless ``enabled``); strips the FX
    from 5-lane guitar/rhythm/bass. ``synth_chain()`` makes a fresh FX chain for tracks without one.
    False if nothing applies."""
    if NO_FX.match(track_name):
        track.remove_nodes("FXCHAIN")
        return False
    rng = range_for(track_name)
    if rng is None:
        return False
    chain = track.find("FXCHAIN")
    if chain is None:
        if synth_chain is None:
            return False
        chain = synth_chain()
        track.append(chain)
    kids = chain.children
    existing = next((c for c in kids if isinstance(c, Node) and c.name == "JS" and FILTER_JS in c.header), None)
    if existing is not None:
        chain.children[kids.index(existing)] = _filter_node(*rng)
    else:
        first_fx = next((i for i, c in enumerate(kids) if isinstance(c, Line) and c.key == "BYPASS"), len(kids))
        block = [Line("BYPASS 0 0 0"), _filter_node(*rng), Line("FLOATPOS 0 0 0 0"),
                 Line(f"FXID {{{str(uuid.uuid4()).upper()}}}"), Line("WAK 0 0")]
        chain.children[first_fx:first_fx] = block
    flag = "0" if enabled else "1"
    for i, c in enumerate(chain.children):
        if isinstance(c, Line) and c.key == "BYPASS":
            chain.children[i] = Line(f"BYPASS {flag} 0 0")
    return True
