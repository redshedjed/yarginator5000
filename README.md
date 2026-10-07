# yarginator5000

Chart songs for YARG from scratch or on top of an existing chart. Give it stems; get back a
`notes.mid`, a default light show, `song.ini`, and a REAPER project built from your template for
fine-tuning.

## Install

```powershell
py -3.11 -m venv .venv                             # 3.11: Basic Pitch (keys) doesn't support newer Pythons yet
.\.venv\Scripts\pip install -e ".[audio,keys,video,separate,dev]" # keys pulls in TensorFlow (~1 GB); audio alone is light
.\.venv\Scripts\yarginator --help
```

## Workflow: from a music video

```powershell
# 1. Point it at the downloaded video. With --dest it makes "<dest>\Artist - Title" (from the file
#    name, e.g. JDownloader's "Band - Song (Official Video) - Channel (1080p, h264).mp4"); without
#    it, the video's own folder becomes the song folder (name that folder "Artist - Title").
yarginator new "C:\Users\me\Downloads\Band - Song (Official Video).mp4" --dest "C:\...\Rock Band\In Progress"

# 2. Separate the stems with the built-in models (or `new ... --separate` to do it in one go)...
yarginator separate "C:\...\In Progress\Band - Song" --drum-split
#    ...or with your own separator on <song>\_yarginator\source\mix.flac, then register its output
yarginator stems "C:\...\In Progress\Band - Song" --from "C:\...\separator output"
#    Either way, near-silent stems are left out of [chart] instruments.

# 3. Chart and iterate (see below), then finalize, optionally moving it to your finished folder
yarginator build "C:\...\In Progress\Band - Song"
yarginator finalize "C:\...\In Progress\Band - Song" --move-to "C:\...\Rock Band\Complete"
```

The song folder holds only what YARG reads. Everything else is in `_yarginator\`, which YARG ignores:

```
Band - Song\
  song.ini  song.ogg  vocals.ogg  video.mp4  notes.mid
  _yarginator\   song.toml, source\ (mix.flac + the download), stems\, reaper\, reports\,
                 .history\ (copies of notes.mid / song.ini / song.toml before each overwrite), .cache\
```

- `video.mp4` is the picture only, copied without re-encoding (VP9/AV1 downloads become `video.webm`).
  The audio is padded to the file's start, so it plays against the picture exactly as in the download.
- `song.ogg` is the full mix until `finalize`. Then it becomes the instrumental stem and `vocals.ogg`
  the vocal stem. If the separator gave no instrumental, the other stems are mixed for it. YARG plays
  every audio file in the folder, so a full mix next to `vocals.ogg` would double the vocals.
- `new` detects the tempo and writes it into song.toml; check it against the audio in REAPER.
- `separate` runs three passes on the CPU (expect several minutes a song): BS-RoFormer for vocals /
  instrumental, Demucs 6-stem on the instrumental for drums / bass / guitar / keys / other, and with
  `--drum-split` MDX23C DrumSep for kick / snare / toms / hihat / ride / crash (in `stems\drum_split`).
  `--vocal-split` (or `[separation] vocal_split = true`) runs a Mel-Roformer karaoke model on the vocal
  stem for `lead_vocals` and `backing_vocals` (in `stemsocal_split`). HARM1 then reads the lead stem
  and HARM2 / HARM3 the backing stem, and an audible backing stem adds harmonies to the instruments.
  PART VOCALS keeps the full vocal stem, which is also what `vocals.ogg` is made from.
  Models download once to `~/.yarginator/models`. Swap one with `[separation] vocals_model = "..."`
  (also `instruments_model`, `drums_model`, `vocal_split_model`; `audio-separator --list_models` lists them).
- `[paths] projects = "C:/.../In Progress"` in `~/.yarginator.toml` makes `--dest` optional.
- After `--move-to`, run `reaper --force` there, because the REAPER project points at the old location.

### Stem separation: MVSEP or local

`yarginator separate` runs on your machine by default (`local`: CPU, or GPU after `yarginator setup-gpu`;
on AMD cards DirectML is slower than the CPU for the vocal model). `--engine mvsep` (or
`[separation] engine = "mvsep"`) uses [mvsep.com](https://mvsep.com)'s GPUs instead: one BS Roformer SW
job for vocals / bass / drums / guitar / piano / other, `--best-vocals` adds a BS-RoFormer vocals /
instrumental job, `--drum-split` a DrumSep job on the drum stem, `--vocal-split` an MVSep Karaoke job
(BS Roformer by the MVSep team) on the vocal stem. All four together are exactly `max_jobs_per_run`.
MVSEP is opt-in only, because it
uploads the song audio to a third party and uses your account's credits. The API token comes from the
`MVSEP_API_TOKEN` environment variable.

Guard rails (all in `mvsep.py`, tunable under `[mvsep]` in `~/.yarginator.toml`):

| | |
|---|---|
| one job at a time | a lock file stops a second run from submitting in parallel |
| no duplicate work | every job is keyed by file content + algorithm + options in `~/.yarginator/mvsep/ledger.json`; finished work is reused, an interrupted job is resumed, never resubmitted |
| caps | `max_jobs_per_run = 4`, `max_jobs_per_day = 20` (rolling 24 h) |
| pacing | `min_interval_s = 3` between requests; polls every 15 s backing off to 60 s; `job_timeout_s = 3600` |
| retries | only status checks and downloads (3 tries with backoff); job creation is never retried; "too many requests" stops the run |
| uploads | checked first: real audio, at least 5 s and 64 KB, at most `max_upload_mb = 150` |
| downloads | HTTPS from mvsep.com hosts only |
| privacy | the token is never logged or shown; jobs are deleted on the server after download; Ctrl+C cancels a queued job |
| `--dry-run` | shows the jobs, sends nothing |

## Workflow: song folders by hand

```powershell
# 1. Make a song folder; put stems in <song>/stems first and they're picked up by name
yarginator init songs\my-song --artist "Band" --template C:\path\to\template.RPP
#    ...or use stems where they already are (e.g. a stem separator's output folder), no copying
yarginator init songs\my-song --stems "C:\...\In Progress\Band - Song"
#    ...or start from an existing chart (tempo mode is then "chart", so its tempo map is kept)
yarginator init songs\my-song --chart C:\charts\my-song\notes.mid

# 2. Edit songs\my-song\song.toml: stems, tempo (bpm + first_downbeat_s for click-tracked songs)

# 3. Run everything, or one step at a time
yarginator build songs\my-song
yarginator tempo  songs\my-song
yarginator chart  songs\my-song --parts vocals,harm1
yarginator lights songs\my-song --preset presets\my_rig.toml
yarginator reaper songs\my-song --force

yarginator info songs\my-song       # what's in the chart
yarginator view songs\my-song bass --bars 17-24 --diff all   # text view of a part, one row per bar
yarginator generators               # what can chart what
```

### Tempo, without moving the audio

The audio always starts at 0 (so a music video stays in sync), and the tempo map is fitted to it:

| `[tempo] mode` | Use for | What it does |
|---|---|---|
| `click` | modern click-tracked songs | Offset measure ending on `first_downbeat_s`, then the locked `bpm`. Leave them unset to auto-detect, then copy the detected values in to lock them. |
| `beats` | older / live songs | One tempo change per detected beat. |
| `chart` | after tuning in REAPER | Keep the tempo map in `notes.mid`. |

If the chart already has events when the tempo changes, they're re-timed so they stay at the same
point in the audio.

### Choosing instruments

`[chart] instruments` in song.toml lists what the song has: `drums guitar rhythm bass keys` (pro
keys) `keys5` (5-lane) `vocals` (PART VOCALS) `harmonies` (HARM1-3). `init` suggests a list from the
stems it finds. Only listed instruments get chart and REAPER tracks; BEAT, EVENTS and VENUE always
do. To add one back later, add it to the list and run `chart` + `reaper --force`. Its track comes
from your template if the template has one, otherwise it's new, empty and ready to chart. Tracks that
already have charted notes are never deleted when you remove an instrument; you get a warning instead.

For harmonies only, list `harmonies` without `vocals` and chart them from scratch with the
`harm1-scratch` / `harm2-scratch` / `harm3-scratch` generators:

```toml
[parts.harm1]
generator = "harm1-scratch"
lyrics = "..."
```

### Iterating

- Every write to `notes.mid` saves the previous copy in `.history/` first.
- `chart` skips parts that already have notes (so your hand edits survive) unless you name them
  with `--parts` or pass `--force`.
- `lights` only replaces lighting events on VENUE. Camera cuts and other VENUE events are left alone.
- `reaper` won't overwrite an existing project without `--force`.
- Generators write `reports/<part>.csv` and add spots they're unsure about to `reports/review.csv`.
  The REAPER project gets a `review: ...` marker at each one.

## Parts

| Part | Status |
|---|---|
| `vocals`, `harm1-3` | Pitches for lyrics already timed in the chart (pyin, with a harmonic-sum fallback for processed vocals) |
| `vocals-scratch` | Vocals with no timed lyrics: notes, pitches, lyrics and phrases from the stem plus lyric text (`"Do*4 Re*3 ..."`). Sung → pitched notes, growls/screams/spoken → `#` notes, coughs/breaths → dropped; consonants fold into their syllable. With lyric counts, `fit` matches notes to syllables by undoing the weakest splits. Use it with `[parts.vocals] generator = "vocals-scratch"` |
| `beat`, `events` | BEAT track from the tempo map; `[music_start]`/`[music_end]`/`[end]` |
| `bass` | 5-lane bass from the bass stem, all four difficulties. Transcribes notes and sorts unpitched hits into ghost notes, kick-drum bleed (checked against the drum stem) and release noise. Measures and corrects the player's feel (how far behind/ahead of the beat) before snapping to 16ths (triplets only where they clearly fit). Frets follow the riff's shape within 2-bar windows; sustains for held notes. Works with 4/5/6-string basses and dropped tunings: the range is read from the stem (`lowest_note` / `highest_note` = `"auto"`), or set it, e.g. `lowest_note = "B0"` |
| `prokeys` | Pro keys (all four difficulties) from the keys stem via Spotify Basic Pitch, or from a MIDI you supply (`[parts.prokeys] midi = "keys.mid"`). Right hand only (follows the top voice); one subdivision per bar (16ths or triplets); two-octave fit with range shifts planned per bar (primary ranges first, few shifts, each placed about a bar early, one at the start); Expert/Hard/Medium/Easy limits from the C3 Customs Book; `[idle]`/`[play]` animation and PART KEYS_ANIM_RH. Needs the `keys` extra (Python 3.11). Trill/glissando/overdrive markers aren't generated |
| `drums` | Pro drums (all four difficulties) from the drum stem. Each onset is classified as kick / snare / cymbal / tom by NMF templates adapted to the song's kit plus a classifier trained on charted songs; one subdivision per bar; crashes from accent evidence (attack, ring, strong beat, fill landing), timekeeping on the hi-hat (`ride = "auto"` for ride sections), toms coloured high → low by pitch with pro tom markers. Expert: two hands at most, fast kicks as 2x kicks. Hard / Medium / Easy reductions from the C3 Customs Book; drum fills over tom runs into a crash; `[mix N drums0]`, `[idle]`/`[play]` and drum animation notes. Flags stretches with cymbals but no kick/snare (bleed in separated stems). See "Drums" below |
| `guitar`, `rhythm` | Not written yet. Registered, so config and CLI already know about them |

### Drums

Scored against charted songs with drum stems (onset level, F1): kick ~0.6-0.8, snare ~0.85-0.95,
cymbal-vs-not ~0.4-0.65, tom-vs-not ~0.2-0.5. Which cymbal (hat / ride / crash) and which tom is the
weak part: expect to fix crash placement and tom colours by hand. `view` shows toms in lower case
(`y b g`) and 2x kicks as `k`; `reports/drums.csv` has every hit with its probability.

The classifier learns from finished charts. Teach it yours (it's retrained on the next run):

```powershell
yarginator drums-learn songs\my-song                      # a yarginator song folder
yarginator drums-learn "C:\...\Band - Song
otes.mid" --stems "C:\...\drums.wav" --offset 0.249
```

`--offset` is where the stem starts on the chart's timeline (in REAPER: the item's position minus
its start offset). Training sets go to `~/.yarginator/drums/`; delete a file there to forget it.

## REAPER data and personal defaults

yarginator ships its own REAPER data in
[src/yarginator/templates/reaper/](src/yarginator/templates/reaper/), and uses it unless told otherwise:
- `Reaper Template.rpp`: the project template
- `color_maps/rockband_*.png`: MIDI note colours (drums, guitar/bass, vocals/other)
- `note_names/RBN2 *.txt`: note-name ("key") maps for every chart track: drums, guitar/rhythm,
  bass, 5-lane keys, pro keys (Expert / Hard / Medium-Easy), vocals (`RBN2 Vox_RSJ.txt` before the
  stock `RBN2 Vox.txt`), harmonies, BEAT and VENUE

To change a default for everyone, edit those files. To override per user or per song, set a path in
`~/.yarginator.toml` (or the file named by `YARGINATOR_CONFIG`) or in a song's `song.toml`
(song.toml wins); `"none"` switches that piece off:

```toml
[song]
charter = "RedShedJed"
[reaper]
template = "C:/.../My Template.rpp"   # unset = built-in
color_maps = "none"                   # no note colours
note_names = "C:/.../my key maps"     # folder of RBN2 *.txt
review_markers = true                 # also show reports/review.csv spots as markers (default off)
preview_fx = false                    # default on: keys/vocal/harmony tracks get a note filter + synth
preview_fx_on = true                  # default off: that FX is added bypassed
```

With `preview_fx`, pro keys, 5-lane keys, vocals and HARM1-3 get REAPER's stock MIDI Note Filter in
front of the template's ReaSynth. The filter passes only playable notes (pro keys 48-72, vocals 36-84,
5-lane keys Expert gems 96-100), so range shifts, phrase markers and the like stay silent. The chain is
added bypassed; un-bypass it in REAPER (or set `preview_fx_on`) to hear a part. 5-lane guitar,
rhythm and bass get no FX at all.

## Lighting

The `default` generator picks one cue per `[section]` using a TOML preset (keyword → cue, accents,
keyframe rate, fog, end blackout). If the chart has no sections, it uses the stem's loudness
instead. To tune the show for your DMX rig, copy
[src/yarginator/lighting/presets/default.toml](src/yarginator/lighting/presets/default.toml) into the
song (or a shared folder) and point `[lighting] preset` at it. For a different approach entirely,
register a new generator.

## Extending

- **New or replacement part generator:** subclass `PartCharter` in `src/yarginator/parts/`, decorate it
  with `@register`, and import its module in `parts/__init__.py`. Then select it per song with
  `[parts.drums] generator = "my-drums"`. Any extra keys in that table reach it as `ctx.options`.
- **New lighting generator:** subclass `LightingGenerator`, decorate it with `@register_lighting`, and set `[lighting] generator`.
- **New pipeline step:** add a function to `STEPS` in `pipeline.py`.
- **New command:** add an `@command` function in `cli.py`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
