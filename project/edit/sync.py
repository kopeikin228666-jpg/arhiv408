"""Musical grid for each version, shared by render.py and mix2.py.  Version comes from env VOY_VER (B or C).
B: Mara only.        Pulse 0.534 s (a quarter of Mara's 2.136 s hit) before 26 s, 0.5356 s (1/16 of the 8.57 s boom) after.
C: «аллея кошмаров».  Beat 0.5769 s (104 bpm), downbeat = drop at 10.455; after the creak the track re-enters on a downbeat at 30.0.
"""
import os, numpy as np

VER = os.environ.get('VOY_VER', 'B')
E = '/home/claude/voynich/edit'

T1_P, T1_PH = 0.5769, 0.54             # track1 beat period / phase (song seconds), fitted on the kick/snare onsets of the whole song
T1_FILM0 = 5.3                         # song 0 at film 5.3 (until the smear)
T1_BACK = T1_PH + 44 * T1_P            # song time re-entering at film 30.0 (a downbeat)

ACT2 = 30.0
SMEAR = 26.0

# every musical time below is the real ATTACK of the hit (measured: the waveform rises 12-18 ms after the beat grid)
if VER == 'C':
    PULSE = T1_P
    LAG = 0.012

    def _pts():
        """(time, is_accent).  The trailer track accents every 3rd beat (song beat k = 1 mod 3), not every 4th."""
        a = [(T1_FILM0 + T1_PH + k * T1_P + LAG, k % 3 == 1) for k in range(0, 60)]
        a = [p for p in a if p[0] < SMEAR]
        b = [(ACT2 + k * T1_P + LAG, k % 3 == 2) for k in range(0, 200)]
        return a + b
else:
    PULSE = 0.5356
    LAG = 0.016
    MARA_HIT0 = 8.5914 - 6.66          # first Mara hit on film (1.93)
    MARA_STEP = 2.1409

    def _pts():
        """(time, is_hit).  Before 26 s: Mara's hit every 4 quarter-pulses.  Act II: the boom every 16 pulses (8.57 s);
        the wall 90..116: a hit every 8 pulses (4.285 s)."""
        a = [(MARA_HIT0 + k * MARA_STEP / 4 + LAG, k % 4 == 0) for k in range(0, 60)]
        a = [p for p in a if p[0] < SMEAR]
        b = []
        for k in range(0, 200):
            t = ACT2 + k * 0.53564 + LAG
            hit = (k % 16 == 0 and k <= 112) or (112 <= k <= 160 and k % 8 == 0)
            b.append((t, hit))
        return a + b


_P = _pts()


def grid(t0, t1):
    return [t for t, h in _P if t0 <= t <= t1]


def strong(t0, t1):
    return [t for t, h in _P if h and t0 <= t <= t1]


def is_strong(t):
    return any(h and abs(t - x) < 0.002 for x, h in _P)


def nearest(t, pts):
    return min(pts, key=lambda x: abs(x - t)) if pts else t


_FL = None


def beat_events():
    """[(time, strength 0..1, is_strong)] for every grid pulse; strength = how hard the music really hits there,
    normalised per version (median -> 0, 90th percentile -> 1)"""
    global _FL
    if _FL is None:
        tt, flux, nf = np.load(f'{E}/flux_{VER}.npy')
        _FL = (tt, nf)
    tt, nf = _FL
    raw = []
    for g, h in _P:
        i0, i1 = np.searchsorted(tt, g - 0.04), np.searchsorted(tt, g + 0.04)
        raw.append(float(nf[i0:max(i1, i0 + 1)].max()) if i1 > i0 else 0.0)
    raw = np.array(raw)
    lo, hi = np.percentile(raw, 50), np.percentile(raw, 90)
    ev = []
    for (g, h), v in zip(_P, raw):
        s = float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))
        if h:
            s = max(s, 0.7)
        ev.append((g, s, h))
    return ev


# the book intro, on the music of each version
if VER == 'B':
    ASM1 = 1.93                         # the pixels land on Mara's first hit
    OPEN0 = 4.06                        # the cover opens on the second
    FLIPS = [5.40, 6.20, 7.27, 8.34, 8.87, 9.41, 9.67, 9.81, 9.94]
else:
    ASM1 = 2.45                         # track1 has not started yet: the book builds in silence
    OPEN0 = 4.15
    _b = T1_FILM0 + T1_PH
    FLIPS = [_b + k * T1_P for k in (0, 2, 3, 4, 5, 5.5, 6, 6.5, 6.75, 7)]
DROP = 10.48 if VER == 'B' else T1_FILM0 + T1_PH + 8 * T1_P + LAG   # the drop / wall hit (B: Mara wall, C: track1 drop = beat 8)
