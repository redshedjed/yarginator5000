# yarginator5000

A Python CLI (`yarginator`) for charting songs for YARG. One song folder per song, configured by
`song.toml`. See README.md for the user workflow.

## Layout

- `src/yarginator/core/`: format-level code with no audio. `timing.py` (TempoMap and its
  click/beats constructors), `chart.py` (lossless notes.mid model: tracks of `(tick, mido msg)`),
  `instruments.py` (track names and note numbers; every format constant lives here).
- `src/yarginator/parts/`: part generators. `PartCharter` + `@register` in `base.py`; `PartContext`
  gives stems, options, cache dir, and song length. Generators **return** tracks and never mutate the chart.
  `vocal_events.py` is the shared vocal layer: energy-based event detection; sung/unpitched/noise
  classification; consonant attachment; and `fit_count`, which matches a lyric count by undoing the
  weakest splits. It needs to hold up on growled/screamed vocals (death metal is coming) as well as clean singing.
- Instrument parts share three layers: `note_events.py` (monophonic transcription: pitched / ghost /
  bleed, transients, legato splits, release-click rejection), `quantize.py` (grid snapping and
  `feel_offset`, which measures a part's lean against the beat), and `five_lane.py` (frets ranked
  within fixed bar-aligned windows, sustains, reductions, writing). `bass.py` wires them together;
  guitar/rhythm/keys5 should reuse them. Check the result with `yarginator view`.
- Built-in REAPER data (template, colour maps, RBN2 note-name maps) lives in
  `src/yarginator/templates/reaper/` and is the default whenever `[reaper] template / color_maps /
  note_names` are unset (`reaper.resolve`). `songs/` is git-ignored, so anything the user drops there
  that should be a default has to be copied into the package.
- Song folders: `yarginator new` (`workflow.py`, `media.py`) makes the video layout. YARG's files sit at the top,
  everything else in `_yarginator/` (`SongProject.work`, which holds song.toml and the working dirs). Old
  `init` folders have work == root. Paths in song.toml are relative to `work`; the chart and song.ini
  live in `root`. Don't put anything named notes.mid / song.ini, or audio named like a stem, in the song
  root besides YARG's own files.
- User defaults live in `~/.yarginator.toml` (deep-merged under song.toml; tests isolate it in `conftest.py`).
- `src/yarginator/lighting/`: VENUE light shows. `cues.py` (vocabulary), `base.py` (registry and
  preset loading), `default.py`, and `presets/*.toml`.
- `src/yarginator/reaper/`: `rpp.py` (lossless RPP tree parser/writer) and `project.py` (template +
  stems + tempo envelope + inline MIDI items + review markers).
- `src/yarginator/audio/`: librosa-based analysis, imported lazily. Results are cached as `.npz` in `<song>/.cache`.
  `audio/drums.py` is drum analysis: onset candidates, semi-adaptive NMF with bundled per-class templates,
  physical + NMF features, and a per-family gradient-boosted classifier trained from
  `audio/data/drum_training.npz` plus the user's `~/.yarginator/drums/*.npz` (`drums-learn`). Bump
  `FEATURE_VERSION` if features change (that invalidates every training set, bundled ones included).
  The bundled templates/training set came from the user's charted songs (Bruno, abcdefu, Surface
  Pressure: their charts on separated stems; Black Dahlia: Harmonix on multitracks). To rebuild them,
  align each chart to its stem with the REAPER item position. Mind that kits shift classifier
  confidence, hence the per-song Otsu thresholds for kick/snare.
- `src/yarginator/feedback/`: `reports/<part>.csv` and the merged `reports/review.csv`.
- `pipeline.py`: ordered `STEPS` (tempo → chart → lights → ini → reaper). `cli.py`: `@command` registry.

## Principles (from the user's workflow)

- Never move the audio. The user often syncs a music video. The tempo map bends to the audio
  (offset measure for click-tracked songs, per-beat tempo for live/older ones).
- Don't destroy hand edits. Back up before writes (`SongProject.save_chart` / `backup`), skip
  charted parts by default, remove only the events a generator owns, and don't overwrite the
  REAPER project without `--force`.
- Expect requirements to change. Add a registry entry or a config key rather than special-casing.
  Unknown song.toml keys flow through `PartConfig.options` / `LightingConfig.options` / `SongConfig.raw`.

## Format facts

Confirmed against a REAPER 7.27-saved project (`songs/Template/Reaper Template.rpp`): inline MIDI is
`HASDATA 1 <ppq> QN`, then `E <delta> <status> <d1> <d2>` lines; text metas are written as
`<X <delta> 0 0 0 <type> "<text>"` plus a base64 line of the raw FF event; sysex as `<X <delta> 0`;
lower-case `e`/`x` mark selected events; per-track MIDI note colours come from `MIDICOLORMAPFN`.

Still to verify:
- REAPER: `PT` line fields in TEMPOENVEX (time, bpm, shape, timesig). The template had no tempo points.
- Whether REAPER needs a particular `CFGEDIT` colour mode for `MIDICOLORMAPFN` to show.
- VENUE: the list of lighting cues and keyframe notes 48/49/50 in `lighting/cues.py`, and how YARG maps
  them to DMX output.
- Pro keys range-shift notes and the HARM2/3 phrase-marker conventions in `core/instruments.py`.

## Dev

- Python 3.11 venv (Basic Pitch / TensorFlow for keys doesn't support newer Pythons; the old
  3.14 venv is kept as `.venv-py314`). `pip install -e ".[audio,keys,dev]"`, then `python -m pytest`.
- Pro drums (`parts/drums.py`) follows the Customs Book chapter 11 (pp. 325-376 of CustomsBookv1.pdf):
  tom markers 110-112 apply to every difficulty, so they're written per tom gem (a span would turn a
  same-colour cymbal into a tom); 2x kicks are note 95; reductions in `reduce()`. Detection is scored
  against reference charts (leave-one-song-out). Kick/snare are solid, but cymbal type and tom colour are weak.
- Pro keys (`parts/pro_keys.py`) follows the C3 Customs Book (songs/Template/CustomsBookv1.pdf,
  pp. 241-251): RBN naming has C2 = MIDI 48; range-shift notes 0/2/4/5/7/9; per-difficulty
  limits are in `LIMITS`. Transcription is `audio/transcribe.py` (Basic Pitch or MIDI input).
- Keep the core importable without librosa; `tests/test_vocals.py` skips without it.
