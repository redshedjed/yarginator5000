"""Click-tempo detection on a synthetic drum stem with a quiet intro (needs the audio extra).

Regression for: beat_track trimming quiet intro beats (downbeat detected 15 s late), frame-level
timing error (~30 ms), and picking beat 3 as the downbeat when the kick is on 1 and 3.
"""
import numpy as np
import pytest

pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from yarginator.audio.tempo import estimate_click  # noqa: E402

SR = 22050


def drum_stem(path, bpm, downbeat, bars=20, quiet_bars=6):
    beat = 60 / bpm
    y = np.zeros(int((downbeat + bars * 4 * beat + 1) * SR))
    t = np.arange(int(0.2 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    rng = np.random.default_rng(0)
    snare = rng.normal(size=len(t)) * np.exp(-t * 25) * 0.6
    for b in range(bars * 4):
        loud = 0.3 if b // 4 < quiet_bars else 1.0
        i = int((downbeat + b * beat) * SR)
        if b % 4 in (0, 2):
            y[i:i + len(kick)] += loud * kick
        elif b // 4 >= quiet_bars:
            y[i:i + len(snare)] += loud * snare
    sf.write(path, y / np.abs(y).max(), SR)


@pytest.mark.parametrize("bpm,downbeat", [(128, 2.37), (97, 0.81), (150, 4.1), (123.45, 1.6)])
def test_estimate_click(tmp_path, bpm, downbeat):
    drum_stem(tmp_path / "drums.wav", bpm, downbeat)
    est_bpm, est_downbeat = estimate_click(tmp_path / "drums.wav")
    assert est_bpm == pytest.approx(bpm, abs=0.01)
    assert est_downbeat == pytest.approx(downbeat, abs=0.008)
