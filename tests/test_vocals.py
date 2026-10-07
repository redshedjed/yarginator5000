"""Vocal pitching against a synthetic stem with known pitches (needs the audio extra)."""
import numpy as np
import pytest

pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from yarginator.core.chart import Chart  # noqa: E402
from yarginator.core.timing import TempoMap  # noqa: E402
from yarginator.parts import get_charter  # noqa: E402
from yarginator.parts.base import PartContext  # noqa: E402

SR = 22050
MELODY = [(1.0, 57), (1.5, 60), (2.0, 64), (2.5, 62)]  # (seconds, midi)


def synth(path):
    y = np.zeros(int(SR * 3.5))
    for s, p in MELODY:
        f = 440 * 2 ** ((p - 69) / 12)
        t = np.arange(int(SR * 0.4)) / SR
        tone = sum(0.5 ** h * np.sin(2 * np.pi * f * (h + 1) * t) for h in range(4)) * np.hanning(len(t))
        i = int(s * SR)
        y[i:i + len(t)] += 0.3 * tone
    sf.write(path, y, SR)


def test_pitches_follow_the_stem(tmp_path):
    synth(tmp_path / "vocals.wav")
    chart = Chart(TempoMap.constant(120))
    voc = chart.track("PART VOCALS")
    for i, (s, _) in enumerate(MELODY):
        voc.add_lyric(chart.tempo.s2t(s), f"la{i}")
    voc.add_note(chart.tempo.s2t(0.9), chart.tempo.s2t(1.8), 105)

    ctx = PartContext(chart, stems={"vocals": tmp_path / "vocals.wav"}, cache_dir=tmp_path / "cache")
    result = get_charter("vocals").generate(ctx)
    notes = [n for n in result.tracks[0].notes() if n.pitch <= 84]
    assert [n.pitch for n in notes] == [p for _, p in MELODY]
    assert 105 in {n.pitch for n in result.tracks[0].notes()}  # phrase marker kept
    assert len(result.rows) == 4


def test_scratch_vocals_place_notes_and_lyrics(tmp_path):
    synth(tmp_path / "vocals.wav")
    chart = Chart(TempoMap.constant(120))
    opts = {"lyrics": "do re me", "lyric_mode": "cycle", "note_start": "attack"}
    ctx = PartContext(chart, stems={"vocals": tmp_path / "vocals.wav"}, cache_dir=tmp_path / "cache", options=opts)
    result = get_charter("vocals-scratch").generate(ctx)
    tr = result.tracks[0]
    notes = [n for n in tr.notes() if n.pitch <= 84]
    assert [n.pitch for n in notes] == [p for _, p in MELODY]
    attack = [chart.tempo.t2s(n.tick) for n in notes]
    assert [round(t, 1) for t in attack] == [s for s, _ in MELODY]
    assert [txt for _, txt in tr.lyrics()] == ["do", "re", "me", "do"]
    assert tr.notes([105])  # phrase marker(s)
    # default: start where the pitch sounds (these tones fade in), later than the attack but inside the note
    ctx.options = {**opts, "note_start": "vowel"}
    vowel = [chart.tempo.t2s(n.tick) for n in get_charter("vocals-scratch").generate(ctx).tracks[0].notes()
             if n.pitch <= 84]
    assert all(0 < v - a <= 0.2 for v, a in zip(vowel, attack))
    # start_delay_ms nudges every note on top
    ctx.options = {**opts, "start_delay_ms": 30}
    delayed = [chart.tempo.t2s(n.tick) for n in get_charter("vocals-scratch").generate(ctx).tracks[0].notes()
               if n.pitch <= 84]
    assert all(abs(d - a - 0.03) < 0.003 for d, a in zip(delayed, attack))


def noisy_stem(path):
    """1.0 s: sung A3 · 2.0 s: cough · 3.0 s: "s" straight into a sung C4 · 4.5 s: 0.9 s growl."""
    rng = np.random.default_rng(3)
    y = np.zeros(int(SR * 6.0))

    def put(t, sig):
        i = int(t * SR)
        y[i:i + len(sig)] += sig

    def tone(midi, dur):
        t = np.arange(int(SR * dur)) / SR
        f = 440 * 2 ** ((midi - 69) / 12)
        return 0.3 * sum(0.5 ** h * np.sin(2 * np.pi * f * (h + 1) * t) for h in range(4)) * np.minimum(1, t / 0.02)

    put(1.0, tone(57, 0.5))
    cough = rng.normal(size=int(SR * 0.15)) * np.exp(-np.arange(int(SR * 0.15)) / SR * 25) * 0.6
    put(2.0, cough)
    hiss = np.diff(rng.normal(size=int(SR * 0.12) + 1)) * 0.15  # "s": high, noisy
    put(3.0, hiss)
    put(3.12, tone(60, 0.5))
    gt = np.arange(int(SR * 0.9)) / SR
    growl = rng.normal(size=len(gt)) * (0.6 + 0.4 * np.sin(2 * np.pi * 45 * gt))  # rough, unpitched
    growl = np.convolve(growl, np.ones(8) / 8, mode="same") * 0.5
    put(4.5, growl)
    sf.write(path, y, SR)


def test_vocal_events_sort_sung_consonant_cough_growl(tmp_path):
    from yarginator.audio.analysis import vocal_features
    from yarginator.parts import vocal_events as ve
    noisy_stem(tmp_path / "v.wav")
    events = ve.detect(vocal_features(tmp_path / "v.wav", tmp_path / "cache"))
    kept = [(round(e.start, 1), e.kind) for e in events if e.kind != ve.NOISE]
    noise = [round(e.start, 1) for e in events if e.kind == ve.NOISE]
    assert kept == [(1.0, "sung"), (3.0, "sung"), (4.5, "unpitched")]  # "s" folded into the C4 note
    assert 2.0 in noise                                                  # the cough
    # in a chart: growl -> "#" note, cough nowhere
    chart = Chart(TempoMap.constant(120))
    ctx = PartContext(chart, stems={"vocals": tmp_path / "v.wav"}, cache_dir=tmp_path / "cache",
                      options={"lyrics": "la la RAH"})
    tr = get_charter("vocals-scratch").generate(ctx).tracks[0]
    assert [txt for _, txt in tr.lyrics()] == ["la", "la", "RAH#"]
    ctx.options["unpitched"] = "drop"
    tr = get_charter("vocals-scratch").generate(ctx).tracks[0]
    assert [txt for _, txt in tr.lyrics()] == ["la", "la"]


def test_repeated_syllables_without_gaps_are_split(tmp_path):
    """Same pitch three times, legato (level dips between syllables but never stops)."""
    from yarginator.audio.analysis import vocal_features
    from yarginator.parts import vocal_events as ve
    t = np.arange(int(SR * 1.2)) / SR
    env = np.ones_like(t)
    for s in (0.4, 0.8):  # 60 ms dips to -14 dB
        m = (t > s - 0.03) & (t < s + 0.03)
        env[m] = 0.2
    env = np.convolve(env, np.ones(200) / 200, mode="same")
    y = np.zeros(int(SR * 2.5))
    y[SR // 2:SR // 2 + len(t)] = 0.3 * env * sum(0.5 ** h * np.sin(2 * np.pi * 220 * (h + 1) * t) for h in range(4))
    sf.write(tmp_path / "v.wav", y, SR)
    kept = [e for e in ve.detect(vocal_features(tmp_path / "v.wav", tmp_path / "c")) if e.kind != ve.NOISE]
    assert [round(e.start, 1) for e in kept] == [0.5, 0.9, 1.3]


def test_fit_merges_split_attempts_and_drops_blips():
    from yarginator.parts import vocal_events as ve
    ev = [ve.VocalEvent(1.0, 1.4, voiced=0.9, level=-2), ve.VocalEvent(1.4, 1.9, voiced=0.9, level=-2),  # one attempt, split
          ve.VocalEvent(3.0, 3.1, voiced=0.6, level=-15),                                           # blip
          ve.VocalEvent(5.0, 6.0, voiced=0.9, level=-1)]
    out, log = ve.fit_count(ev, 2)
    assert [(e.start, e.end) for e in out] == [(1.0, 1.9), (5.0, 6.0)]
    assert len(log) == 2


def test_syllable_repeats():
    from yarginator.parts.vocals_scratch import syllables
    assert syllables("Do*2 Re Hel- lo*2") == ["Do", "Do", "Re", "Hel-", "lo", "lo"]


def test_text_event_lyrics_and_harmonic_fallback(tmp_path):
    """Older charts store lyrics as text events; and with pyin disabled the CQT path must still work."""
    synth(tmp_path / "vocals.wav")
    chart = Chart(TempoMap.constant(120))
    voc = chart.track("PART VOCALS").add_text(0, "[idle]")
    for i, (s, _) in enumerate(MELODY):
        voc.add_text(chart.tempo.s2t(s), f"la{i}")
    ctx = PartContext(chart, stems={"vocals": tmp_path / "vocals.wav"}, options={"voiced_threshold": 1.01},
                      cache_dir=tmp_path / "cache")
    result = get_charter("vocals").generate(ctx)
    assert "cqt 4" in result.summary
    assert [n.pitch for n in result.tracks[0].notes()] == [p for _, p in MELODY]
