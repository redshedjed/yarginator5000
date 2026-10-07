"""A tiny synthetic drum kit for tests: kick, snare, closed hat, crash and three toms."""
import numpy as np

SR = 44100


def _env(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def kick(rng):
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 50 + 110 * np.exp(-t / 0.03)                       # pitch drop 160 -> 50 Hz
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.12)
    click = rng.normal(size=n) * _env(n, 0.004) * 0.3
    return 0.9 * body + click


def snare(rng):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * _env(n, 0.05)
    noise = np.convolve(rng.normal(size=n), [1, -0.6], mode="same") * _env(n, 0.09)
    return 0.5 * tone + 0.45 * noise


def hat(rng):
    n = int(0.12 * SR)
    noise = np.diff(rng.normal(size=n + 1)) * _env(n, 0.03)   # bright, short
    return 0.25 * noise


def crash(rng):
    n = int(1.6 * SR)
    noise = np.diff(rng.normal(size=n + 1)) * _env(n, 0.6) + 0.3 * rng.normal(size=n) * _env(n, 0.4)
    return 0.45 * noise


def tom(rng, hz):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = hz * (1 + 0.25 * np.exp(-t / 0.05))
    return 0.8 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.18) + 0.1 * rng.normal(size=n) * _env(n, 0.01)


def groove(bpm=120.0, start=2.0, bars=6, seed=0):
    """Rock beat (kick 1 & 3, snare 2 & 4, 8th hats) for ``bars - 1`` bars, a 16th tom fill
    high -> low over the last two beats before the final bar, a crash + kick on its downbeat.
    Returns (audio, sr, [(seconds, kind)])."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    y = np.zeros(int((start + bars * 4 * beat + 2) * SR))
    truth = []

    def put(t, sig, kind):
        i = int(round(t * SR))
        y[i:i + len(sig)] += sig[: len(y) - i]
        truth.append((round(t, 4), kind))
    for b in range(bars - 1):
        t0 = start + b * 4 * beat
        for e in range(8):
            t = t0 + e * beat / 2
            fill = b == bars - 2 and e >= 4
            if not fill:
                put(t, hat(rng), "hat")
            if e in (0, 4) and not fill:
                put(t, kick(rng), "kick")
            if e in (2, 6) and not fill:
                put(t, snare(rng), "snare")
        if b == bars - 2:  # tom fill over beats 3-4
            for k, hz in enumerate([220] * 3 + [150] * 3 + [95] * 2):
                put(t0 + 2 * beat + k * beat / 4, tom(rng, hz), "tom")
    t_end = start + (bars - 1) * 4 * beat
    put(t_end, crash(rng), "crash")
    put(t_end, kick(rng), "kick")
    y /= np.abs(y).max() * 1.1
    return y.astype(np.float32), SR, sorted(truth)
