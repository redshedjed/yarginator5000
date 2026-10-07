"""Build a REAPER project for fine-tuning: your template + stems + the chart's tempo map + one MIDI
item per chart track + a marker at every flagged review spot.

Template handling (templates are often a previous song's project, so they carry old content):
- Tracks are matched by name (case-insensitive, or via ``[reaper] tracks``). A matched track keeps
  everything (FX, note names, colours, item fades, MIDI editor settings); only its item's audio file /
  MIDI events are replaced.
- Template chart tracks the new chart doesn't use are emptied, and template audio tracks that get no
  stem lose their items, so nothing from the old song survives.
- If the "song" stem matches no track, it goes to the first unmatched audio track (templates usually
  open with a full-mix track named after the song), which is renamed.
- New chart tracks copy MIDI editor settings from the template's first MIDI item.
- Stems always start at 0: the tempo map bends to the audio, never the other way round.
"""
from __future__ import annotations

import base64
import re
import uuid
from pathlib import Path

from .. import audio
from ..core.chart import Chart, Track
from ..feedback import ReviewItem
from . import note_names as key_maps
from . import preview_fx
from .rpp import Line, Node, format_line, parse

MINIMAL_TEMPLATE = """<REAPER_PROJECT 0.1 "7.0" 0
  RIPPLE 0
  SAMPLERATE 44100 0 0
  TEMPO 120 4 4
>
"""

SOURCE_TYPES = {".wav": "WAVE", ".mp3": "MP3", ".ogg": "VORBIS", ".opus": "OPUS", ".flac": "FLAC"}
REVIEW_PREFIX = "review: "
CHART_TRACK_RE = re.compile(r"^(PART .+|HARM\d|EVENTS|VENUE|BEAT)$", re.I)
# Inside a MIDI source, these are rewritten rather than copied from the template item
_SOURCE_DROP = {"E", "e", "HASDATA", "POOLEDEVTS", "GUID"}
_ITEM_DROP = {"IGUID", "GUID", "POSITION", "LENGTH", "NAME", "SOFFS"}

# Default MIDI note colour map per chart track (files from the Rock Band REAPER setup)
DEFAULT_COLOR_MAPS = [
    (re.compile(r"^PART DRUMS", re.I), "rockband_drums.png"),
    (re.compile(r"^PART (GUITAR|RHYTHM|BASS|KEYS)$", re.I), "rockband_guitarbass.png"),
    (re.compile(r".*"), "rockband_vox-other.png"),
]


def _guid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def _name(tr: Node) -> str:
    return (tr.value("NAME") or [""])[0]


def _insert_before_tracks(root: Node, child: Node | Line) -> None:
    idx = next((i for i, c in enumerate(root.children) if isinstance(c, Node) and c.name == "TRACK"),
               len(root.children))
    root.children.insert(idx, child)


def _is_midi_track(tr: Node) -> bool:
    if CHART_TRACK_RE.match(_name(tr)):
        return True
    return any((it.find("SOURCE") is not None and it.find("SOURCE").args[:1] == ["MIDI"]) for it in tr.nodes("ITEM"))


# --- project-level ------------------------------------------------------------------------------
def set_tempo(root: Node, chart: Chart) -> None:
    tm = chart.tempo
    first_ts = tm.time_sigs[0]
    root.set("TEMPO", f"{tm.tempos[0].bpm:.10g}", first_ts.numerator, first_ts.denominator)
    old = root.find("TEMPOENVEX")
    root.remove_nodes("TEMPOENVEX")
    env = Node("TEMPOENVEX")
    kept = [c for c in (old.children if old else []) if isinstance(c, Line) and c.key not in ("PT", "ACT")]
    env.append(Line("ACT 1 -1"))
    for c in kept or [Line("VIS 1 0 1"), Line("DEFSHAPE 1 -1 -1")]:
        env.append(c)
    sig_ticks = {s.tick: s for s in tm.time_sigs}
    for tick in sorted({t.tick for t in tm.tempos} | set(sig_ticks)):
        bpm = 60_000_000 / next(t for t in reversed(tm.tempos) if t.tick <= tick).us_per_beat
        tokens = [f"{tm.t2s(tick):.12f}", f"{bpm:.10f}", 1]
        if tick in sig_ticks:  # time signature encoded as numerator + (denominator << 16)
            tokens.append(sig_ticks[tick].numerator + (sig_ticks[tick].denominator << 16))
        env.append(Line.of("PT", *tokens))
    _insert_before_tracks(root, env)


def set_markers(root: Node, review: list[ReviewItem]) -> None:
    def ours(c):
        return isinstance(c, Line) and c.key == "MARKER" and len(c.tokens) > 3 and c.tokens[3].startswith(REVIEW_PREFIX)
    root.remove(ours)
    used = [int(c.tokens[1]) for c in root.children if isinstance(c, Line) and c.key == "MARKER"]
    start = max(used, default=0) + 1
    for n, item in enumerate(review):
        _insert_before_tracks(root, Line.of("MARKER", start + n, f"{item.seconds:.6f}",
                                            f"{REVIEW_PREFIX}{item.part}: {item.message}", 0))


# --- items --------------------------------------------------------------------------------------
def _item_shell(proto: Node | None, name: str, length: float) -> Node:
    """A fresh ITEM with the prototype's settings (fades, volume, ...) but new identity/position."""
    item = Node("ITEM")
    for ln in (("POSITION", 0), ("SNAPOFFS", 0), ("LENGTH", f"{length:.6f}")):
        item.append(Line.of(*ln))
    if proto is not None:
        for c in proto.children:
            if isinstance(c, Line) and c.key not in _ITEM_DROP and c.key != "SNAPOFFS":
                item.append(c)
    else:
        item.append(Line("LOOP 0"))
    item.append(Line.of("IGUID", _guid()))
    item.append(Line.of("NAME", name))
    item.append(Line.of("SOFFS", 0, 0) if proto is not None else Line.of("SOFFS", 0))
    return item


def audio_item(path: Path, length: float, proto: Node | None = None) -> Node:
    item = _item_shell(proto, path.name, length)
    src = Node.of("SOURCE", SOURCE_TYPES.get(path.suffix.lower(), "WAVE"))
    src.append(Line.of("FILE", str(path)))
    item.append(src)
    return item


def _meta_node(delta: int, msg) -> Node:
    data = bytes(msg.bytes())
    if hasattr(msg, "text") or hasattr(msg, "name"):  # REAPER writes text metas with a readable header
        text = getattr(msg, "text", None) or getattr(msg, "name", "")
        node = Node(f"X {delta} 0 0 0 {data[1]} {format_line(text)}")
    else:
        node = Node(f"X {delta} 0")
    node.append(Line(base64.b64encode(data).decode()))
    return node


def midi_events(track: Track, ppq: int) -> list[Node | Line]:
    import mido
    out: list[Node | Line] = [_meta_node(0, mido.MetaMessage("track_name", name=track.name))]
    last = 0
    for tick, m in track.sorted_events():
        delta, last = tick - last, tick
        if m.is_meta or m.type == "sysex":
            out.append(_meta_node(delta, m))
        else:
            b = (list(m.bytes()) + [0, 0])[:3]
            out.append(Line(f"E {delta} {b[0]:02x} {b[1]:02x} {b[2]:02x}"))
    out.append(Line("E 0 b0 7b 00"))  # all notes off
    return out


def midi_item(track: Track, ppq: int, length: float, proto: Node | None = None) -> Node:
    item = _item_shell(proto, track.name, length)
    src = Node("SOURCE MIDI")
    src.append(Line(f"HASDATA 1 {ppq} QN"))
    proto_src = proto.find("SOURCE") if proto is not None else None
    settings = [c for c in (proto_src.children if proto_src is not None else [])
                if isinstance(c, Line) and c.key not in _SOURCE_DROP]
    # template order: HASDATA, CCINTERP, <events>, remaining settings
    head = [c for c in settings[:1] if c.key == "CCINTERP"]
    src.children += head + midi_events(track, ppq) + settings[len(head):] + [Line.of("GUID", _guid())]
    item.append(src)
    return item


# --- tracks -------------------------------------------------------------------------------------
def find_track(root: Node, *names: str) -> Node | None:
    wanted = [n.lower() for n in names if n]
    return next((tr for tr in root.nodes("TRACK") if _name(tr).lower() in wanted), None)


def new_track(root: Node, name: str, after: Node | None = None) -> Node:
    """Append a track, or insert it right after ``after`` (e.g. keep audio tracks together)."""
    tr = Node(f"TRACK {_guid()}")
    tr.append(Line.of("NAME", name))
    if after is not None and after in root.children:
        root.children.insert(root.children.index(after) + 1, tr)
    else:
        root.append(tr)
    return tr


def new_chart_track(root: Node, name: str, proto_track: Node | None, before: Node | None) -> Node:
    """A chart track with the template's preview synth (FX chain) but not its instrument-specific
    note names, placed before ``before`` (or at the end)."""
    tr = Node(f"TRACK {_guid()}")
    tr.append(Line.of("NAME", name))
    fx = proto_track.find("FXCHAIN") if proto_track is not None else None
    if fx is not None and name.upper() not in ("BEAT", "EVENTS", "VENUE"):  # no beeping on marker tracks
        tr.append(copy_chain(fx))
    if before is not None:
        root.children.insert(root.children.index(before), tr)
    else:
        root.append(tr)
    return tr


def copy_chain(fx: Node) -> Node:
    """Deep copy of an FX chain with fresh FX ids."""
    chain = parse(fx.dumps())
    chain.children = [Line.of("FXID", _guid()) if isinstance(c, Line) and c.key == "FXID" else c
                      for c in chain.children]
    return chain


def _next_in_order(root: Node, name: str, order: dict[str, int]) -> Node | None:
    """The existing track that should come right after ``name`` in the song's instrument order."""
    later = [(order[_name(t)], t) for t in root.nodes("TRACK") if order.get(_name(t), -1) > order[name]]
    return min(later, key=lambda x: x[0])[1] if later else None


def set_items(track: Node, *items: Node) -> None:
    track.remove_nodes("ITEM")
    for it in items:
        track.append(it)


def color_map_for(track_name: str, folder: Path | None) -> Path | None:
    if folder is None:
        return None
    for pattern, filename in DEFAULT_COLOR_MAPS:
        if pattern.match(track_name):
            p = folder / filename
            return p if p.exists() else None
    return None


def song_length(chart: Chart, stems: dict[str, Path]) -> float:
    lengths = [d for d in (audio.duration(p) for p in stems.values()) if d]
    return max(lengths) if lengths else chart.tempo.t2s(chart.end_tick()) + 5.0


def build(chart: Chart, stems: dict[str, Path], review: list[ReviewItem], template: Path | None = None,
          track_map: dict[str, str] | None = None, color_maps: Path | None = None,
          song_title: str | None = None, wanted: list[str] | None = None,
          note_names: Path | None = None, preview: bool = True, preview_on: bool = False) -> Node:
    """``wanted``: the chart tracks this song should have (from its instruments), in order. Each gets
    a track even when it has no data yet (an empty item, ready to chart by hand); template chart
    tracks outside it are removed. None keeps every template track."""
    track_map = track_map or {}
    root = parse(template.read_text("utf-8", errors="surrogateescape") if template else MINIMAL_TEMPLATE)
    set_tempo(root, chart)
    set_markers(root, review)
    length = song_length(chart, stems)

    template_tracks = list(root.nodes("TRACK"))
    midi_tracks = [t for t in template_tracks if _is_midi_track(t)]
    proto_track = next((t for t in midi_tracks if _name(t).upper() == "PART DRUMS"), midi_tracks[0] if midi_tracks else None)
    midi_proto = proto_track.find("ITEM") if proto_track is not None else None
    used: set[int] = set()

    # audio
    for key, path in stems.items():
        tr = find_track(root, track_map.get(key, ""), key, path.stem)
        if tr is None and key == "song":
            tr = next((t for t in template_tracks if id(t) not in used and not _is_midi_track(t)
                       and find_track_by_stem_name(t, stems, track_map) is None), None)
        if tr is not None and key == "song" and song_title:
            tr.set("NAME", song_title)  # the full-mix track is named after the song
        if tr is None:
            audio_tracks = [t for t in root.nodes("TRACK") if not _is_midi_track(t)]
            tr = new_track(root, track_map.get(key, key), audio_tracks[-1] if audio_tracks else None)
        used.add(id(tr))
        set_items(tr, audio_item(path, audio.duration(path) or length, tr.find("ITEM")))

    # chart (plus empty tracks for wanted parts that have no data yet)
    tracks = dict(chart.tracks)
    for name in wanted or []:
        tracks.setdefault(name, Track(name))
    order = {n: k for k, n in enumerate(wanted or [])}
    for name, track in tracks.items():
        tr = find_track(root, track_map.get(name, ""), name)
        proto = tr.find("ITEM") if tr is not None else None
        if tr is None:
            tr = new_chart_track(root, track_map.get(name, name), proto_track,
                                 _next_in_order(root, name, order) if name in order else None)
        used.add(id(tr))
        set_items(tr, midi_item(track, chart.ppq, length, proto or midi_proto))

    # leftovers from the template's previous song
    for tr in template_tracks:
        if id(tr) in used:
            continue
        if _is_midi_track(tr):
            if wanted is None:
                set_items(tr, midi_item(Track(_name(tr)), chart.ppq, length, tr.find("ITEM") or midi_proto))
            else:
                root.children.remove(tr)  # an instrument this song doesn't have
        elif wanted is not None and tr.find("ITEM") is not None:
            root.children.remove(tr)      # old song's audio with no stem to replace it
        else:
            tr.remove_nodes("ITEM")

    # note colour maps, note-name ("key") maps, preview FX
    synth = next((t.find("FXCHAIN") for t in midi_tracks if t.find("FXCHAIN") is not None), None)
    synth_chain = (lambda: copy_chain(synth)) if synth is not None else None
    for tr in root.nodes("TRACK"):
        if _is_midi_track(tr):
            cmap = color_map_for(_name(tr), color_maps)
            if cmap is not None:
                tr.set("MIDICOLORMAPFN", str(cmap))
            if note_names is not None:
                key_maps.apply(tr, _name(tr), note_names)
            if preview:
                preview_fx.apply(tr, _name(tr), synth_chain, preview_on)
    return root


def find_track_by_stem_name(tr: Node, stems: dict[str, Path], track_map: dict[str, str]) -> str | None:
    """The stem key whose name matches this track, if any (used to avoid stealing another stem's track)."""
    n = _name(tr).lower()
    for key, path in stems.items():
        if n in {key.lower(), path.stem.lower(), track_map.get(key, "").lower()}:
            return key
    return None
