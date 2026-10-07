"""MIDI note-name maps (REAPER's "key maps", e.g. RBN2 Drums.txt) for chart tracks.

REAPER saves a map as ``<note><TAB><name>`` lines; inside a project it lives on the track as

    <MIDINOTENAMES
      -1 96 "EXPERT Kick" 0 96
    >

Each chart track gets the first map found for it in the configured folder, replacing whatever
the template had, so the folder is the one place note names come from.
"""
from __future__ import annotations

import re
from pathlib import Path

from .rpp import Line, Node, format_line

# chart track -> map files to try, in order (your own variants first)
MAP_FILES = [
    (re.compile(r"^PART DRUMS", re.I), ["RBN2 Drums.txt"]),
    (re.compile(r"^PART (GUITAR|RHYTHM)$", re.I), ["RBN2 Guitar.txt"]),
    (re.compile(r"^PART BASS$", re.I), ["RBN2 Bass.txt"]),
    (re.compile(r"^PART KEYS$", re.I), ["RBN2 Keys.txt"]),
    (re.compile(r"^PART REAL_KEYS_X$", re.I), ["RBN2 Pro Keys Expert.txt"]),
    (re.compile(r"^PART REAL_KEYS_H$", re.I), ["RBN2 Pro Keys Hard.txt"]),
    (re.compile(r"^PART REAL_KEYS_[ME]$", re.I), ["RBN2 Pro Keys Medium-Easy.txt"]),
    (re.compile(r"^PART VOCALS$", re.I), ["RBN2 Vox_RSJ.txt", "RBN2 Vox.txt"]),
    (re.compile(r"^HARM1$", re.I), ["RBN2 Harmony 1.txt", "RBN2 Harmony.txt"]),
    (re.compile(r"^HARM[23]$", re.I), ["RBN2 Harmony.txt"]),
    (re.compile(r"^BEAT$", re.I), ["RBN2 Beat.txt"]),
    (re.compile(r"^VENUE$", re.I), ["RBN2 Venue.txt"]),
]


def load_map(path: Path) -> dict[int, str]:
    names = {}
    for raw in path.read_text("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        note, _, name = line.partition("\t") if "\t" in line else line.partition(" ")
        if note.strip().isdigit():
            names[int(note)] = name.strip()
    return names


def map_for(track_name: str, folder: Path) -> Path | None:
    for pattern, files in MAP_FILES:
        if pattern.match(track_name):
            return next((folder / f for f in files if (folder / f).exists()), None)
    return None


def midinotenames(names: dict[int, str]) -> Node:
    node = Node("MIDINOTENAMES")
    for note in sorted(names, reverse=True):
        node.append(Line(f"-1 {note} {format_line(names[note])} 0 {note}"))
    return node


def apply(track: Node, track_name: str, folder: Path) -> bool:
    """Replace the track's note names with the map for ``track_name``. False if there's no map."""
    path = map_for(track_name, folder)
    if path is None:
        return False
    track.remove_nodes("MIDINOTENAMES")
    idx = next((i for i, c in enumerate(track.children) if isinstance(c, Node)), len(track.children))
    track.children.insert(idx, midinotenames(load_map(path)))
    return True
