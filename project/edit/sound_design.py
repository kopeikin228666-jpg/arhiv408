"""sound_design.py - the sound designer's layer for version B (Mara).   Owner: sound agent.

    design(ver, shots, words, sync, SR, N) -> (N, 2) float32   (added to mix2's fx bus)   | None for ver != 'B'
    duck(ver, shots, words, sync, SR, N)   -> (N,)   float32   (gain on mix2's ducked music bed)

Everything is synthesised here or re-processed from material that already exists (Mara itself, the narrator /
whisper recordings, the sound tracks of the two generated video clips).  Every event sits on the music grid
(sync.grid / sync.strong) or on a picture event (shot cut, subliminal frame, key word).

Layers (film seconds, version B):
  * VACUUMS: 0.3-0.6 s of TOTAL silence before the act II / wall booms that have no voice around them (music bed -> 0
    via duck(), mix2's fx bus cancelled sample-exactly, this layer gated), each preceded by a reversed swell made
    from the boom itself (reverse reverb) that cuts dead into the vacuum.
  * SUB DROPS (+ phone-audible 2nd/3rd harmonics) on every Mara boom of act II and the wall, and on the 10.48 drop.
  * HEARTBEAT from 47.16 (s19): every 2 pulses (56 bpm), every pulse from the wall (112 bpm), alone in the held-breath
    pause, every half pulse in the collapse (s36), dead on the cut to the museum (s37); one last beat on «книгой».
  * BREATH: a gasp on the cut into the held-breath pause and a shaking exhale out of it, a surfacing gasp out of the
    underwater shot, close wide breathing around the whisperer (s35).
  * CHOIR: a dark pad granulated / time-stretched (x4) and pitched an octave down from Mara's own wall.
  * WATER: drips on the grid in a cave (s19), bubbles (s20), the v21 clip's own water track in sync (v21),
    underwater pressure + big bubbles (s22), a wet slap of parchment on s24.
  * METAL / GLASS: a nail on glass after «стекло» (s25), a bowed-metal scrape on the s34 cut, a rising metallic
    screech through the collapse (s36).
  * WHISPERS circling the head (ITD/ILD/behind-the-head low-pass): after «с другой стороны» and after «внутри».
  * PRE-VERBS (the voice's own reversed reverb) sucking into «И кто-то шепчет…» and «внутри»; low swells into
    «нельзя видеть» and «книгой».
  * TINNITUS: a 7 kHz ring after the collapse into the clean museum ending.
  * subliminal frames / scares get a crushed low 'whump'.
Safety: the layer's 300-3400 Hz band is carved under every word, then a joint solve (music bed + this layer) keeps
every narration word >= +8.6 dB over music+fx in 300-3400 Hz; a peak guard keeps (mix without layer + layer) under
0.72 x the mix reference peak so the final tanh limiter is never driven harder than it already is.
"""
import os, sys, json, subprocess
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, lfilter
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

E = '/home/claude/voynich/edit'
AUD = '/home/claude/voynich/audio'
CLIPS = '/home/claude/voynich/clips'
REPORT_PATH = '/home/claude/voynich/team/sound/sd_report.json'
TARGET_DB = 8.6            # per word, voice vs music+fx in 300-3400 Hz (hard rule is 8.0)
CEIL_FRAC = 0.70           # |mix without layer + layer| <= CEIL_FRAC * reference peak
_STATE = {}
REPORT = {}


# =================================================================== small tools
def _S(t, SR):
    return int(round(t * SR))


def _place(buf, x, t, SR, g=1.0):
    i = _S(t, SR)
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if i < 0:
        x = x[-i:]
        i = 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i] * g


def _sos(kind, f, SR, order=2):
    return butter(order, f, kind, fs=SR, output='sos')


def _lp(x, f, SR, order=2):
    return sosfilt(_sos('lowpass', f, SR, order), x, axis=0)


def _hp(x, f, SR, order=2):
    return sosfilt(_sos('highpass', f, SR, order), x, axis=0)


def _bp(x, lo, hi, SR, order=2):
    return sosfilt(_sos('bandpass', (lo, hi), SR, order), x, axis=0)


def _fade(x, a, b, SR):
    x = np.array(x, np.float64)
    na, nb = min(len(x), _S(a, SR)), min(len(x), _S(b, SR))
    w = np.ones(len(x))
    if na:
        w[:na] = np.linspace(0, 1, na)
    if nb:
        w[len(x) - nb:] *= np.linspace(1, 0, nb)
    return x * (w[:, None] if x.ndim == 2 else w)


def _crush(x, bits, hold):
    h = max(1, int(hold))
    y = np.repeat(x[::h], h, axis=0)[:len(x)] if h > 1 else np.array(x)
    st = 2.0 ** bits
    return np.round(y * st) / st


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _st(x, w=1.0):
    """mono -> stereo"""
    return np.stack([x, x * w], 1)


def _pan(x, p):
    """equal-power pan, p in [-1, 1] (scalar or per-sample)"""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2)


def _ir(SR, dur=2.5, decay=2.4, lp=3000., seed=3, pre=.015):
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-t * decay)[:, None]
    ir = _lp(ir, lp, SR, 2)
    ir[:_S(pre, SR)] = 0
    ir = _fade(ir, 0, .1, SR)
    return ir / np.sqrt((ir ** 2).sum(0)) * .6


def _reverb(x, SR, **kw):
    x = _st(x) if x.ndim == 1 else x
    ir = _ir(SR, **kw)
    return np.stack([fftconvolve(x[:, c], ir[:, c]) for c in (0, 1)], 1)


def _varispeed(x, speed):
    """resample by a constant or per-output-sample speed (speed < 1: slower and lower)"""
    if np.ndim(speed) == 0:
        pos = np.arange(0, len(x) - 1, speed)
    else:
        pos = np.cumsum(speed)
        pos = pos[pos < len(x) - 1]
    i = pos.astype(int)
    fr = pos - i
    if x.ndim == 2:
        fr = fr[:, None]
    return x[i] * (1 - fr) + x[i + 1] * fr


def _load(path, SR, filt=None, start=None, dur=None):
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    if dur is not None:
        cmd += ['-t', str(dur)]
    cmd += ['-i', path]
    if filt:
        cmd += ['-af', filt]
    cmd += ['-vn', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    try:
        return np.frombuffer(subprocess.check_output(cmd), np.float32).reshape(-1, 2).astype(np.float64)
    except Exception:
        return None


def _ctx():
    """mix2's namespace (bed_d, vo, fx, bed, mara ...) when called from mix2.py; None otherwise"""
    f = sys._getframe(1)
    while f is not None:
        g = f.f_globals
        if 'bed_d' in g and 'vo' in g and 'fx' in g:
            return g
        f = f.f_back
    return None


# =================================================================== synthesis
def sub_drop(SR, dur=1.7, f0=62., f1=26., seed=0):
    """sub drop with a saturated upper body (2nd/3rd harmonics) so it is felt on a phone speaker too"""
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 2.6)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, t / .004) * np.exp(-t * 2.0)
    x = np.sin(ph) * env
    body = np.tanh(np.sin(ph) * 3.0) * env * np.exp(-t * 3) * .35          # harmonics 2f..5f
    body = _lp(body, 260, SR, 2)
    thud = _lp(r.standard_normal(n) * np.exp(-t * 60), 300, SR, 2) * .5
    y = x + body + thud
    y = _crush(y, 7, 3)
    return _fade(_norm(y), 0, .05, SR)


def rev_swell(SR, src, dur, seed=0, lp=3500.):
    """reversed reverb of src (mono) whose crest lands exactly on the last sample (cut dead there)"""
    src = src if src.ndim == 1 else src.mean(1)
    wet = _reverb(src[::-1], SR, dur=dur + .4, decay=2.2, lp=lp, seed=seed, pre=.0)
    wet = wet[:_S(dur, SR)][::-1]
    u = np.linspace(0, 1, len(wet))
    wet = wet * (u ** 1.6)[:, None]
    wet = _hp(wet, 35, SR)
    wet = _fade(wet, .05, .006, SR)
    return wet / (np.abs(wet).max() + 1e-12)


def heartbeat(SR, rate_s, seed=0, muffle=False):
    """one lub-dub (lub at sample 0); rate_s = time to the next beat, sets the lub-dub gap"""
    r = np.random.default_rng(seed)
    gap = float(np.clip(rate_s * .32, .11, .24))
    n = _S(gap + .38, SR)
    y = np.zeros(n)

    def thump(f0, f1, dec, amp):
        m = _S(.30, SR)
        t = np.arange(m) / SR
        f = f1 + (f0 - f1) * np.exp(-t * 30)
        ph = 2 * np.pi * np.cumsum(f) / SR
        e = np.minimum(1, t / .006) * np.exp(-t * dec)
        x = np.sin(ph) * e + np.tanh(np.sin(ph) * 2.5) * e * .45
        nz = _bp(r.standard_normal(m), 70, 420, SR, 2) * np.exp(-t * 45) * .6
        return (x + nz) * amp

    a = thump(78, 46, 16, 1.0)
    b = thump(92, 58, 22, .62)
    y[:len(a)] += a
    k = _S(gap, SR)
    y[k:k + len(b)] += b[:n - k]
    y = _lp(y, 420 if not muffle else 220, SR, 2)
    y = _crush(y / (np.abs(y).max() + 1e-12), 7, 2)
    return _fade(y, 0, .04, SR)


def breath(SR, dur, kind='in', seed=0, shake=0.0, width=1.0):
    """close breath: two-formant noise + air, decorrelated L/R (wide); kind in / out / gasp"""
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    u = t / dur
    out = []
    for ch in range(2):
        nz = r.standard_normal(n)
        f1 = _bp(nz, 650, 1250, SR, 2) * 1.0
        f2 = _bp(nz, 1500, 2600, SR, 2) * .55
        air = _hp(nz, 3500, SR, 2) * .35
        out.append(f1 + f2 + air)
    x = np.stack(out, 1)
    x[:, 1] = x[:, 1] * width + x[:, 0] * (1 - width)
    if kind == 'in':
        env = np.sin(np.pi / 2 * np.minimum(1, u / .8)) ** 1.5 * np.where(u > .9, (1 - u) / .1, 1)
    elif kind == 'gasp':
        env = np.minimum(1, u / .06) * (1 - u) ** .6
    else:
        env = np.minimum(1, u / .12) * (1 - u) ** 1.4
    if shake:
        env = env * (1 + shake * np.sign(np.sin(2 * np.pi * r.uniform(7, 10) * t)) * .5)
    x = x * env[:, None]
    return _fade(x / (np.abs(x).max() + 1e-12), .005, .02, SR)


def granular_choir(SR, src, out_dur, stretch=4.0, pitch=.5, grain=.30, dens=8, seed=0):
    """time-stretch x stretch and pitch src (mono) by granular overlap-add; returns mono out_dur s"""
    r = np.random.default_rng(seed)
    n = _S(out_dur, SR)
    y = np.zeros(n + _S(grain / pitch, SR) + 10)
    g_in = _S(grain, SR)
    g_out = int(g_in / pitch)
    win = np.hanning(g_out)
    hop = max(1, _S(grain / dens, SR))
    for o in range(0, n, hop):
        p = o / stretch + r.uniform(-.06, .06) * SR
        p = int(np.clip(p, 0, len(src) - g_in - 2))
        seg = src[p:p + g_in]
        seg = np.interp(np.arange(g_out) * pitch, np.arange(g_in), seg)
        y[o:o + g_out] += seg * win
    return y[:n]


def metal(SR, dur, f0, f1, seed=0, ratios=(1, 1.41, 2.13, 2.97, 3.61, 4.93), jitter=24., bits=6, hold=3):
    """bowed / scraped metal: an inharmonic cluster gliding f0 -> f1, friction AM, crushed"""
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    y = np.zeros(n)
    for k, q in enumerate(ratios):
        am = np.abs(_lp(r.standard_normal(n), jitter * r.uniform(.6, 1.4), SR, 2))
        am = am / (am.max() + 1e-9)
        ph = 2 * np.pi * np.cumsum(f * q * (1 + r.uniform(-.004, .004))) / SR
        y += np.sin(ph + r.uniform(0, 6)) * am * (.8 ** k)
    y = _crush(_norm(y), bits, hold)
    return _fade(y, .04, .08, SR)


def scrape_glass(SR, dur, seed=0):
    """fingernail on wet glass: stick-slip squeal with a wandering pitch"""
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    f = 1450 + 420 * np.sin(2 * np.pi * .7 * t + r.uniform(0, 6)) + 160 * np.sin(2 * np.pi * 3.1 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    slip = (np.sin(2 * np.pi * np.cumsum(28 + 18 * np.sin(2 * np.pi * .9 * t)) / SR) > .2).astype(float)
    slip = uniform_filter1d(slip, _S(.003, SR))
    y = (np.sin(ph) + .45 * np.sin(2 * ph + 1) + .25 * np.sin(3.02 * ph)) * (.35 + .65 * slip)
    y += _bp(r.standard_normal(n), 2500, 6000, SR) * .25 * slip
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** .7
    return _fade(_norm(y * env), .03, .1, SR)


def drip(SR, f0, seed=0):
    r = np.random.default_rng(seed)
    n = _S(.09, SR)
    t = np.arange(n) / SR
    f = f0 * (1 + .9 * (1 - np.exp(-t * 60)))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / .0012) * np.exp(-t * 55)
    x += _bp(r.standard_normal(n), 2000, 5000, SR) * np.exp(-t * 600) * .25
    return _fade(x, 0, .01, SR)


def rustle(SR, dur, seed=0, density=500):
    """dry crawling scratch: sparse crackle through a band + soft paper noise"""
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    imp = (r.random(n) < density / SR) * r.uniform(-1, 1, n)
    cr = _bp(imp, 1800, 6500, SR, 2) * 3
    pap = _bp(r.standard_normal(n), 500, 3000, SR, 2) * .25
    env = np.abs(_lp(r.standard_normal(n), 4, SR, 1))
    env = env / (env.max() + 1e-9)
    return _fade(_norm((cr + pap) * (.3 + .7 * env)), .05, .1, SR)


def circle(x, SR, turns=1.0, start=0.0, behind_lp=1800.):
    """move a mono source around the head: ITD (<= .6 ms), ILD, a low-pass when it is behind"""
    n = len(x)
    th = start + 2 * np.pi * turns * np.arange(n) / n
    s, c = np.sin(th), np.cos(th)                 # s: right(+)/left(-), c: front(+)/behind(-)
    itd = .00062 * SR * s
    idx = np.arange(n, dtype=float)
    L = np.interp(idx - np.maximum(itd, 0), idx, x)
    R = np.interp(idx - np.maximum(-itd, 0), idx, x)
    L *= (1 - .55 * np.maximum(s, 0))
    R *= (1 - .55 * np.maximum(-s, 0))
    dark = np.stack([_lp(L, behind_lp, SR), _lp(R, behind_lp, SR)], 1)
    w = ((1 - c) / 2)[:, None]
    return np.stack([L, R], 1) * (1 - w) + dark * w


def tinnitus(SR, dur, f=7040., seed=0):
    r = np.random.default_rng(seed)
    n = _S(dur, SR)
    t = np.arange(n) / SR
    wob = 1 + .0015 * np.sin(2 * np.pi * .35 * t)
    y = np.sin(2 * np.pi * np.cumsum(f * wob) / SR) + .22 * np.sin(2 * np.pi * (f * 1.0071) * t + r.uniform(0, 6))
    env = np.minimum(1, t / .025) * np.exp(-t / 2.4)
    return _fade(y * env, 0, .3, SR)


def whump(SR, seed=0):
    """subliminal frame: a crushed low whump + a short sucked breath of air"""
    r = np.random.default_rng(seed)
    n = _S(.5, SR)
    t = np.arange(n) / SR
    f = 34 + 70 * np.exp(-t * 22)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / .003) * np.exp(-t * 7)
    x = np.tanh(x * 2.2)
    air = _bp(r.standard_normal(n), 900, 3000, SR) * np.exp(-t * 30) * .25
    y = _crush(_lp(x, 300, SR) + air, 5, 6)
    return _fade(_norm(y), 0, .05, SR)


# =================================================================== the score
def _shot(shots, name):
    for s in shots:
        if getattr(s, 'name', None) == name:
            return float(s.t0), float(s.t1)
    return None


def _word(words, prefix, after=0.0):
    for w in words:
        if w['w'].upper().startswith(prefix) and w['t0'] >= after:
            return w
    return None


def _spans(words):
    return [(w['t0'], w['t1']) for w in words]


def _free(words, a, b, pad=(.05, .15)):
    """no word sounds in [a, b]"""
    return not any(w['t0'] - pad[0] < b and a < w['t1'] + pad[1] for w in words)


def _build(shots, words, sync, SR, N, C):
    """returns (elements layer (N,2), vacuum windows, dead-stop time, gate (N,))"""
    L = np.zeros((N, 2), np.float32)
    rep = REPORT.setdefault('events', [])
    sh = {s.name: (float(s.t0), float(s.t1)) for s in shots if hasattr(s, 'name')}
    S37 = sh.get('s37', (118.933, 127.5))[0]
    S36 = sh.get('s36', (115.733, S37))[0]
    P32 = sh.get('s32', (99.1, 100.733))
    bed_raw = C.get('bed') if C else None

    def add(x, t, g, tag):
        _place(L, x, t, SR, g)
        rep.append((round(float(t), 3), tag, round(float(g), 3)))

    strong = [h for h in sync.strong(29.5, S37) if h < S37 - .1]
    grid = sync.grid(0, 131)
    P = sync.PULSE

    # ---------------- vacuums before booms (no voice around, act II + wall)
    vac = []
    for i, h in enumerate(strong):
        if h < 40 or abs(h - P32[0]) < .7:
            continue
        cut = min([s[0] for s in sh.values() if abs(s[0] - h) < .03] + [h])
        end = cut - .010
        d = [.48, .42, .55, .36, .5, .6, .45, .56][i % 8]
        if not _free(words, end - d - .05, end + .02):
            continue
        vac.append((end - d, end, h))
    REPORT['vacuums'] = [(round(a, 3), round(b, 3)) for a, b, h in vac]

    def boom_src(h, dur=.9):
        if bed_raw is not None:
            return np.asarray(bed_raw[_S(h, SR):_S(h + dur, SR)], np.float64).mean(1)
        r = np.random.default_rng(int(h * 10))
        return _lp(r.standard_normal(_S(dur, SR)), 1200, SR) * np.exp(-np.arange(_S(dur, SR)) / SR * 4)

    # ---------------- reversed swells: into each vacuum (cut dead into the silence) / into the other booms
    vac_h = {h for a, b, h in vac}
    for h in strong:
        if h in vac_h:
            a, b, _ = next(v for v in vac if v[2] == h)
            dur = 2.4 if h > 90 else 1.9
            add(rev_swell(SR, boom_src(h), dur, seed=int(h)), a - dur, .42, f'rev swell -> vacuum {a:.2f}')
        elif h > 31 and abs(h - P32[0]) > .7:
            add(rev_swell(SR, boom_src(h), 1.4, seed=int(h)), h - 1.4 - .01, .22, f'rev swell -> boom {h:.2f}')
    # into the drop of act I
    add(rev_swell(SR, boom_src(sync.DROP, .8), 1.5, seed=7), sync.DROP - 1.5 - .01, .30, 'rev swell -> drop')

    # ---------------- sub drops on the booms
    for h in [sync.DROP] + [x for x in strong if x > 31]:
        g = .50 if h > 90 else .42
        add(sub_drop(SR, 1.8 if h > 90 else 1.5, seed=int(h * 10)), h - .002, g, f'sub drop {h:.2f}')

    # ---------------- heartbeat
    t_h0 = 47.156
    beats = []
    for gt in grid:
        if gt < t_h0 - .01 or gt >= S37 - .05:
            continue
        k = int(round((gt - 30.016) / P))
        if gt < 90.0:
            if k % 2:
                continue
            iv, lvl = 2 * P, np.interp(gt, [47.1, 52, 64.2, 64.3, 68.0, 68.1, 90], [.10, .22, .30, .48, .48, .32, .40])
        elif gt < S36 - .01:
            iv, lvl = P, np.interp(gt, [90, 99.0, 99.1, 100.7, 100.8, 111.4, 115.7], [.40, .46, .70, .70, .48, .55, .62])
        else:
            iv, lvl = P, .66
        beats.append((gt, iv, lvl))
        if gt >= S36 - .01:                                   # the collapse: a half-pulse flutter
            beats.append((gt + P / 2, P / 2, .62))
    for gt, iv, lvl in beats:
        if gt >= S37 - .03:
            continue
        uw = (sh.get('s22', (64.3, 68.03))[0] <= gt < sh.get('s22', (64.3, 68.03))[1])
        add(heartbeat(SR, iv, seed=int(gt * 100), muffle=uw), gt - .004, lvl, 'heart')
    REPORT['heartbeats'] = len(beats)

    # ---------------- held-breath pause (s32): a gasp on the cut, a shaking exhale out of it
    add(breath(SR, .34, 'gasp', seed=1), P32[0] + .004, .30, 'gasp into pause')
    add(breath(SR, .5, 'out', seed=2, shake=.8), P32[1] - .52, .16, 'shaking exhale out of pause')

    # ---------------- choir pad from Mara (stretched x4, an octave down)
    mara = C.get('mara') if C else None
    if mara is None:
        mara = _load(f'{AUD}/track2.mp3', SR)
    if mara is not None:
        src_t0 = 154.0                                            # Mara's wall (song s) = the film's 90 s
        src = np.asarray(mara[_S(src_t0, SR):_S(src_t0 + 26, SR)], np.float64).mean(1)
        t0, t1 = 30.0, S37
        ch = np.stack([granular_choir(SR, src, t1 - t0, seed=11), granular_choir(SR, src, t1 - t0, seed=12)], 1)
        ch = _lp(_hp(ch, 45, SR, 2), 1400, SR, 4)
        ch = ch / (np.sqrt((ch ** 2).mean()) + 1e-12)
        tt = t0 + np.arange(len(ch)) / SR
        lvl = np.interp(tt, [30, 38.6, 47.2, 64.3, 68.0, 90.0, 99.1, 100.7, 112, S36, S37 - .3, S37],
                        [0, .02, .035, .045, .03, .05, .0, .05, .06, .08, .16, .16])
        ch *= lvl[:, None]
        add(ch, t0, 1.0, 'choir pad')

    # ---------------- water
    s19, s20, v21, s22, s24 = (sh.get(k) for k in ('s19', 's20', 'v21', 's22', 's24'))
    if s19:
        r = np.random.default_rng(19)
        dd = np.zeros((N, 2))
        for gt in sync.grid(s19[0] + .3, (s20 or s19)[1] - .2):
            for sub in (0, .5):
                if sub and r.random() < .55:
                    continue
                f0 = r.uniform(700, 1500)
                dr = drip(SR, f0, seed=int(gt * 100 + sub * 10))
                _place(dd, _pan(dr, r.uniform(-.7, .7)), gt + sub * P, SR, r.uniform(.5, 1.0))
        a0, a1 = _S(s19[0], SR), _S((s20 or s19)[1] + 2.6, SR)
        seg = dd[a0:a1]
        wet = _reverb(seg, SR, dur=2.6, decay=1.6, lp=2600, seed=19)[:len(seg)]
        add(seg * .5 + wet * 1.1, s19[0], .14, 'drips (cave) s19-s20')
    pools = _load(f'{CLIPS}/v21_pools.mp4', SR)
    if pools is not None and v21:
        x = _varispeed(pools, .93)                               # the clip plays at 0.93 in v21: sound in sync
        x = _lp(_hp(x, 120, SR), 5000, SR)
        x = x[:_S(v21[1] - v21[0] + .3, SR)]
        add(_fade(x / (np.sqrt((x ** 2).mean()) + 1e-12), .2, .3, SR), v21[0], .035, 'v21 clip water (in sync)')
    if pools is not None and s20:
        x = _varispeed(pools, .7)                                # slower, deeper bubbles under the pools
        x = _lp(_hp(x, 90, SR), 2500, SR)[:_S(s20[1] - s20[0], SR)]
        add(_fade(x / (np.sqrt((x ** 2).mean()) + 1e-12), .5, .3, SR), s20[0], .022, 's20 bubbles')
    if s22:
        r = np.random.default_rng(22)
        d = s22[1] - s22[0]
        n = _S(d + .4, SR)
        t = np.arange(n) / SR
        rum = _lp(np.cumsum(r.standard_normal((n, 2)), 0) * .02, 220, SR, 2)
        rum = _hp(rum, 25, SR)
        rum = rum / (np.abs(rum).max() + 1e-9)
        pr = np.interp(t, [0, .4, d - .3, d], [0, 1, 1.25, 0])
        throb = np.sin(2 * np.pi * 38 * t) * (.5 + .5 * np.cos(2 * np.pi * t / (2 * P))) * .5
        uw = rum * pr[:, None] + _st(throb * pr)
        if pools is not None:
            bub = _lp(_varispeed(pools, .5), 900, SR)[:n]
            bub = bub / (np.sqrt((bub ** 2).mean()) + 1e-12) * .12
            uw[:len(bub)] += bub * pr[:len(bub), None]
        add(_fade(uw, .05, .25, SR), s22[0], .26, 'underwater pressure s22')
    if s24:
        r = np.random.default_rng(24)
        n = _S(.6, SR)
        t = np.arange(n) / SR
        sl = _bp(r.standard_normal(n), 300, 2400, SR) * np.exp(-t * 18) + np.sin(2 * np.pi * 70 * t) * np.exp(-t * 20)
        add(_fade(_norm(sl), .002, .1, SR), s24[0] - .003, .28, 'wet parchment slap s24')
        add(breath(SR, .38, 'gasp', seed=24, width=.8), s24[0] + .02, .24, 'surfacing gasp s24')

    # ---------------- letters crawl (s18) / glass (s25) / metal (s34) / collapse (s36)
    s18 = sh.get('s18')
    if s18:
        w0 = _word(words, 'ДВИГАЮТСЯ', s18[0] - 1)
        t0 = w0['t0'] if w0 else s18[0] + .5
        x = rustle(SR, s18[1] - .6 - t0, seed=18, density=700)
        add(_pan(x, np.sin(np.linspace(0, 5, len(x))) * .8), t0, .10, 'letters crawl s18')
    s25 = sh.get('s25')
    st_w = _word(words, 'СТЕКЛО', 70)
    if s25 and st_w:
        hs = [h for h in strong if s25[0] - .1 <= h <= s25[1]]
        if hs:
            r = np.random.default_rng(25)
            n = _S(1.2, SR)
            t = np.arange(n) / SR
            ring = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) * w for f, d, w in ((1870, 5, 1), (2990, 7, .6), (4410, 9, .4)))
            slap = np.sin(2 * np.pi * 85 * t) * np.exp(-t * 25) * 1.6 + _lp(r.standard_normal(n), 800, SR) * np.exp(-t * 40)
            add(_fade(_norm(slap + ring * .25), .001, .2, SR), hs[0] - .003, .34, 'hand slaps the glass')
        g0 = sync.nearest(st_w['t1'] + .25, sync.grid(st_w['t1'], s25[1]))
        dur = min(2.4, s25[1] - g0 - .2)
        if dur > .5:
            add(_pan(scrape_glass(SR, dur, seed=25), np.linspace(.6, -.5, _S(dur, SR))), g0, .09, 'nail on glass s25')
    s34 = sh.get('s34')
    if s34:
        add(_pan(metal(SR, 1.3, 310, 240, seed=34), -.3), s34[0] - .005, .12, 'bowed metal s34')
    if S36:
        d = S37 - S36
        m = metal(SR, d + .02, 260, 1250, seed=36, jitter=40, bits=5, hold=4)
        m = m * np.linspace(.25, 1, len(m)) ** 1.5
        add(_st(m), S36, .16, 'metal screech rising s36')
        add(rev_swell(SR, boom_src(strong[-1] if strong else S36), 2.2, seed=36), S37 - 2.2 - .002, .40,
            'implosion swell -> s37 (dead stop)')

    # ---------------- whispers around the head + pre-verbs (the voice's own reversed reverb)
    wsrc = _load(f'{AUD}/vo_whisper.mp3', SR)
    nsrc = _load(f'{AUD}/vo_narrator.mp3', SR, f'aresample={SR},asetrate={SR}*0.86,aresample={SR}')
    sto = _word(words, 'СТОРОНЫ', 85)
    vac_after = [a for a, b, h in vac]
    if sto and nsrc is not None:
        t0 = sto['t1'] + .1
        t1 = min([a for a in vac_after if a > t0] + [t0 + 2.4]) - .05
        x = nsrc[_S(50.4, SR):_S(50.4 + (t1 - t0) * 1.3 + .5, SR)].mean(1)[::-1]
        x = _bp(x, 400, 5000, SR)[:_S(t1 - t0, SR)]
        x = _fade(x / (np.abs(x).max() + 1e-9), .25, .2, SR)
        add(circle(x, SR, 1.2, start=np.pi / 2), t0, .20, 'whisper circles the head (after «стороны»)')
    vn = _word(words, 'ВНУТРИ', 100)
    if vn and wsrc is not None:
        t0 = vn['t1'] + .15
        t1 = min([a for a in vac_after if a > t0] + [S36 - .4]) - .05
        if t1 - t0 > .6:
            acc = np.zeros((_S(t1 - t0, SR), 2))
            for j, (sp, tr, st0) in enumerate(((.82, 1.5, 0.), (.7, -1.1, 2.), (.95, 2.2, 4.))):
                x = _varispeed(wsrc.mean(1)[::-1] if j != 1 else wsrc.mean(1), sp)
                x = np.tile(x, 3)[:len(acc)]
                x = _bp(x, 350, 5000, SR)
                acc += circle(x / (np.abs(x).max() + 1e-9), SR, tr, start=st0) * (.6 + .2 * j)
            acc = _fade(acc, .3, .25, SR) * np.linspace(.4, 1, len(acc))[:, None]
            add(acc, t0, .13, 'three whispers circle (after «внутри»)')
    vo_ctx = C.get('vo') if C else None

    def preverb(t_on, max_len, tag, g):
        if vo_ctx is None:
            return
        on = _S(t_on, SR)
        x = np.asarray(vo_ctx[on:on + _S(.45, SR)], np.float64).mean(1)
        sw = rev_swell(SR, x, max_len, seed=int(t_on * 10), lp=5000)
        add(sw, t_on - max_len - .015, g, tag)

    sh_l = _word(words, 'И', 88.9)
    sama = _word(words, 'САМА', 85)
    if sh_l and sama and sh_l['t0'] - sama['t1'] > .3:
        preverb(sh_l['t0'], min(.9, sh_l['t0'] - sama['t1'] - .1), 'pre-verb -> «И кто-то шепчет»', .16)
    uzh = _word(words, 'УЖЕ', 105)
    if vn and uzh and vn['t0'] - uzh['t1'] > .3:
        preverb(vn['t0'], min(.9, vn['t0'] - uzh['t1'] - .12), 'pre-verb -> «внутри»', .22)
        add(sub_drop(SR, 1.4, 55, 24, seed=77), vn['t0'] - .002, .30, 'sub on «внутри»')
    nel = _word(words, 'НЕЛЬЗЯ', 55)
    if nel:
        sw = rev_swell(SR, sub_drop(SR, .9, 60, 30, seed=8), 1.6, seed=9, lp=500)
        add(sw, nel['t0'] - 1.6, .26, 'low swell -> «нельзя видеть»')
        hit = sync.nearest(nel['t0'] + .25, sync.grid(nel['t0'] - .2, nel['t1'] + .3))
        add(whump(SR, seed=61), hit - .003, .26, 'whump on «нельзя видеть» (scribe dissolves)')

    # ---------------- subliminal frames (render: s8 bt(2), s17 bt(4), s32 bt(1)) + vfx scares
    def bt(n):
        return round(n * P * 15) / 15
    subs = []
    for nm, n in (('s8', 2), ('s17', 4), ('s32', 1)):
        if nm in sh:
            subs.append(sh[nm][0] + bt(n))
    try:
        import vfx as _vfx
        subs += [float(x) for x in getattr(_vfx, 'SCARES', [])]
    except Exception:
        pass
    for i, ts in enumerate(sorted(set(round(x, 3) for x in subs))):
        add(whump(SR, seed=100 + i), ts - .003, .30, f'subliminal whump {ts:.2f}')

    # gate: everything except the after-collapse elements dies on the S37 cut
    gate = np.ones(N, np.float32)
    i37 = _S(S37, SR)
    gate[i37:] = 0
    gate[i37 - _S(.004, SR):i37] = np.linspace(1, 0, _S(.004, SR))
    return L, vac, S37, gate


def _after37(shots, words, sync, SR, N, S37):
    """the elements that live after the dead stop (tinnitus, last heartbeat, swell into «книгой»)"""
    L = np.zeros((N, 2), np.float32)
    rep = REPORT.setdefault('events', [])
    _place(L, _st(tinnitus(SR, 7.0), .97), S37, SR, .045)
    rep.append((round(S37, 3), 'tinnitus 7.04 kHz (after the dead stop)', .045))
    kn = _word(words, 'КНИГОЙ', 120)
    if kn:                                                       # one last beat: it is alive
        _place(L, heartbeat(SR, 1.1, seed=5, muffle=True), kn['t0'] - .004, SR, .34)
        rep.append((round(kn['t0'], 3), 'last heartbeat on «книгой»', .34))
        sw = rev_swell(SR, sub_drop(SR, .8, 50, 30, seed=3), 1.4, seed=4, lp=400)
        _place(L, sw, kn['t0'] - 1.4, SR, .18)
        rep.append((round(kn['t0'] - 1.4, 3), 'low swell -> «книгой»', .18))
    return L


def _vac_mask(vac, N, SR, att=.012, rel=.006):
    m = np.zeros(N, np.float32)
    for a, b, h in vac:
        i0, i1 = _S(a, SR), _S(b, SR)
        m[i0:i1] = 1
        na, nr = _S(att, SR), _S(rel, SR)
        m[i0:i0 + na] = np.linspace(0, 1, na)
        m[i1 - nr:i1] = np.linspace(1, 0, nr)
    return m


def _bed_shape(N, SR, S37, vac):
    """music automation: vacuums, the dead stop on the museum cut and a faint dark return"""
    g = 1.0 - _vac_mask(vac, N, SR)
    i37 = _S(S37, SR)
    t = np.arange(N - i37) / SR
    g[i37:] *= np.interp(t, [0, 1.6, 4.5, 99], [0, 0, .45, .45]).astype(np.float32)
    k = _S(.004, SR)
    g[i37 - k:i37] *= np.linspace(1, 0, k)
    return g.astype(np.float32)


_BP = None


def _band(x, SR):
    global _BP
    if _BP is None:
        _BP = butter(4, [300, 3400], 'band', fs=SR, output='sos')
    return sosfilt(_BP, x, axis=0).astype(np.float32)


def _word_rms(x, words, SR, pad=.08):
    out = []
    for w in words:
        i0, i1 = _S(w['t0'], SR), _S(w['t1'] + pad, SR)
        out.append(float(np.sqrt((np.asarray(x[i0:i1], np.float64) ** 2).mean()) + 1e-12))
    return np.array(out)


def design(ver, shots, words, sync, SR, N):
    if ver != 'B' or os.environ.get('SD_OFF'):
        return None
    REPORT.clear()
    C = _ctx() or {}
    L, vac, S37, gate = _build(shots, words, sync, SR, N, C)
    L = L * gate[:, None]
    L += _after37(shots, words, sync, SR, N, S37)
    mvac = _vac_mask(vac, N, SR)
    L *= (1 - mvac)[:, None]

    vo = C.get('vo')
    bed_d = C.get('bed_d')
    fx = C.get('fx')
    if vo is None:
        _STATE.update(gb=_bed_shape(N, SR, S37, vac))
        return L.astype(np.float32)

    # ---- 1) carve this layer's speech band under every word (16 dB under the voice), zero-phase split
    sos = butter(4, [300, 3400], 'band', fs=SR, output='sos')
    mid = sosfiltfilt(sos, L, axis=0).astype(np.float32)
    rest = L - mid
    vb = _band(vo.mean(1), SR)
    rv = _word_rms(vb, words, SR) * 1.5
    gm = np.ones(N, np.float32)
    for k in range(2):
        mb = _band((mid * gm[:, None]).mean(1), SR)
        rm = _word_rms(mb, words, SR)
        for w, a, b in zip(words, rv, rm):
            lim = a / 10 ** (16 / 20)
            if b > lim:
                i0, i1 = _S(w['t0'] - .1, SR), _S(w['t1'] + .2, SR)
                gm[i0:i1] = np.minimum(gm[i0:i1], lim / b)
        gm = uniform_filter1d(minimum_filter1d(gm, _S(.12, SR)), _S(.06, SR))
    L = rest + mid * gm[:, None]

    # ---- 2) the music bed shape and mix2's fx cancelled in the vacuums / after the dead stop
    gb = _bed_shape(N, SR, S37, vac)
    i37 = _S(S37, SR)
    dead = np.zeros(N, np.float32)
    dead[i37:i37 + _S(1.6, SR)] = 1
    dead[i37 + _S(1.6, SR):i37 + _S(2.6, SR)] = np.linspace(1, 0, _S(1.0, SR))
    canc = np.maximum(mvac, dead)
    fx_c = (-fx * canc[:, None]).astype(np.float32) if fx is not None else 0

    # ---- 3) peak guard: (mix without this layer) + layer stays under CEIL_FRAC x reference peak
    hook_l = None
    try:
        import hook as _hook
        if hasattr(_hook, 'design'):
            hook_l = _hook.design('B', shots, words, sync, SR, N)
    except Exception:
        hook_l = None
    without = bed_d * .9 * gb[:, None] + vo * 1.5 + (fx if fx is not None else 0) + fx_c
    if hook_l is not None:
        without = without + hook_l[:N]
    ref = float(np.abs(bed_d * .9 + vo * 1.5 + (fx if fx is not None else 0) + (hook_l[:N] if hook_l is not None else 0)).max())
    ceil = CEIL_FRAC * ref
    head = np.maximum(ceil - np.abs(without), 0)
    la = np.abs(L) + 1e-9
    gp = np.minimum(1, (head / la).min(1)).astype(np.float32)
    w = _S(.08, SR)
    gp = uniform_filter1d(minimum_filter1d(gp, 2 * w + 1), w)
    L = L * gp[:, None]
    REPORT['peak_guard_min_gain'] = float(gp.min())
    REPORT['peak_guard_time_below_0.9_s'] = float((gp < .9).sum() / SR)
    REPORT['ref_peak'] = ref
    L = L + fx_c

    # ---- 4) joint solve: every word >= TARGET_DB over music + fx + this layer (+ hook) in 300-3400 Hz,
    #         measured on the mono sum (as mix2 does) AND on each ear separately
    rest_all = (fx if fx is not None else 0) + L + (hook_l[:N] if hook_l is not None else 0)
    chans = []
    for c in ('m', 0, 1):
        sel = (lambda x: x.mean(1)) if c == 'm' else (lambda x, c=c: x[:, c])
        chans.append((_word_rms(_band(sel(vo), SR), words, SR) * 1.5, _band(sel(bed_d) * .9, SR), _band(sel(rest_all), SR)))
    gw = np.ones(N, np.float32)
    tgt = 10 ** (TARGET_DB / 10)
    for _pass in range(5):
        g = gb * gw
        bad = 0
        for k, w in enumerate(words):
            i0, i1 = _S(w['t0'], SR), _S(w['t1'] + .08, SR)
            f = 1.0
            for rvc, bbc, rbc in chans:
                pb = float(((bbc[i0:i1] * g[i0:i1]) ** 2).mean())
                pr = float((rbc[i0:i1] ** 2).mean())
                pt = rvc[k] ** 2 / tgt
                if pb + pr > pt:
                    f = min(f, np.sqrt(max(pt * .9 - pr, 0) / (pb + 1e-20)))
            if f < 1:
                bad += 1
                j0, j1 = _S(w['t0'] - .12, SR), _S(w['t1'] + .2, SR)
                gw[j0:j1] = np.minimum(gw[j0:j1], max(f, .05))
        gw = -maximum_filter1d(-gw, _S(.25, SR))
        gw = uniform_filter1d(gw, _S(.2, SR))
        if not bad:
            break
    gfin = gb * gw
    _STATE['gb'] = gfin.astype(np.float32)

    # ---- report
    ob = _band((bed_d * .9 * gfin[:, None] + rest_all).mean(1), SR)
    ro = _word_rms(ob, words, SR)
    db = 20 * np.log10(rv / ro)
    per_ch = []
    for c in (0, 1):
        vbc = _band(vo[:, c], SR)
        obc = _band(bed_d[:, c] * .9 * gfin + rest_all[:, c], SR)
        per_ch.append(20 * np.log10(_word_rms(vbc, words, SR) * 1.5 / _word_rms(obc, words, SR)))
    worst = np.minimum(db, np.minimum(*per_ch))
    order = np.argsort(worst)
    REPORT['words_worst'] = [(round(float(worst[i]), 2), words[i]['w'], round(words[i]['t0'], 2), round(float(db[i]), 2))
                             for i in order[:10]]
    REPORT['words_min_db'] = float(worst.min())
    REPORT['words_median_db'] = float(np.median(db))
    REPORT['layer_peak'] = float(np.abs(L).max())
    REPORT['bed_gain_min_on_words'] = float(gw.min())
    try:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        json.dump(REPORT, open(REPORT_PATH, 'w'), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass
    print(f'sound_design: words min {worst.min():.2f} dB (mean-ch {db.min():.2f}), median {np.median(db):.1f}; '
          f'vacuums {REPORT["vacuums"]}; peak guard min gain {gp.min():.2f}')
    return L.astype(np.float32)


def duck(ver, shots, words, sync, SR, N):
    if ver != 'B' or os.environ.get('SD_OFF'):
        return np.ones(N, np.float32)
    g = _STATE.get('gb')
    if g is None or len(g) != N:
        return np.ones(N, np.float32)
    return g
