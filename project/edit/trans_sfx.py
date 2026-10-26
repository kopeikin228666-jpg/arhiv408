"""trans_sfx.py - synthesised transition sound effects for the Voynich reel.

    sfx(kind, dur, center, seed=0) -> np.float32 array (n, 2) at SR = 48000

dur    : length (s) of the visual transition
center : offset (s) inside the returned array where the visual switch happens (the beat).
         Build-ups rise INTO center, the accent lands on center, tails ring out after it
         (never more than 1.2 s past dur).  Place the array at (switch_time - center).
seed   : any int; the result is deterministic per (kind, dur, center, seed).

Same palette as mix2.py: crushed / sample-held / bit-reduced 'pixel' hits, stick-slip wood,
dark rooms.  Everything is band-limited (a dull reconstruction low-pass per kind, mostly
below 3 kHz), DC-free (20 Hz high-pass), every edge faded, peak normalised to ~0.25-0.35,
with one clear accent and quiet textures around it.  No pitched melodies: the only tones
are falling kick pitches and one static inharmonic shimmer cluster.

    python3 trans_sfx.py   ->  qa/trans_sfx_demo.wav + qa/trans_sfx_demo.png + a level report
"""
from functools import lru_cache
import numpy as np
from scipy.signal import butter, sosfilt, lfilter, fftconvolve
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 48000
ALL_KINDS = ['tear', 'flash', 'slice', 'zoom', 'glyph', 'iris', 'sortwipe',
             'swirl', 'grow', 'scatter', 'melt', 'blocks', 'shutter']
MAX_TAIL = 1.2
LEAD = 288                  # samples (6 ms): kick onsets sit this far before center so their peak lands on it


def S(sec):
    return int(round(sec * SR))


def _lead(c):
    """kick onset lead; shortened when the accent sits at the very start of the array"""
    return min(LEAD, c)


# ------------------------------------------------------------------ filters / helpers
@lru_cache(maxsize=256)
def _sos(btype, f, order):
    return butter(order, f, btype, fs=SR, output='sos')


def _lp(x, f, order=2):
    return sosfilt(_sos('lowpass', float(f), order), x, axis=0)


def _hp(x, f, order=2):
    return sosfilt(_sos('highpass', float(f), order), x, axis=0)


def _bp(x, lo, hi, order=2):
    return sosfilt(_sos('bandpass', (float(lo), float(hi)), order), x, axis=0)


def _vlp(x, fc, blk=512, poles=2):
    """low-pass whose cutoff follows the per-sample curve fc (one-pole stages, block-wise)"""
    x2 = x[:, None] if x.ndim == 1 else x
    fc = np.broadcast_to(np.asarray(fc, float), (len(x2),))
    y = np.asarray(x2, float)
    for _ in range(poles):
        out = np.empty_like(y)
        zi = np.zeros((1, y.shape[1]))
        for k in range(0, len(y), blk):
            a = np.exp(-2 * np.pi * float(fc[k:k + blk].mean()) / SR)
            out[k:k + blk], zi = lfilter([1 - a], [1, -a], y[k:k + blk], axis=0, zi=zi)
        y = out
    return y if x.ndim == 2 else y[:, 0]


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _fade(x, a=0.0, b=0.0):
    """linear fade-in a seconds, fade-out b seconds (mono or stereo, returns a copy)"""
    x = np.array(x, float)
    na, nb = min(len(x), S(a)), min(len(x), S(b))
    w = np.ones(len(x))
    if na:
        w[:na] = np.linspace(0, 1, na)
    if nb:
        w[len(x) - nb:] *= np.linspace(1, 0, nb)
    return x * (w[:, None] if x.ndim == 2 else w)


def _add(out, x, i0, g=1.0, pan=0.0):
    """mix x (mono -> balance-panned, or stereo) into out starting at sample i0"""
    x = np.asarray(x, float)
    if x.ndim == 1:
        pan = np.asarray(pan, float)
        x = np.stack([x * np.minimum(1, 1 - pan), x * np.minimum(1, 1 + pan)], 1)
    i0 = int(i0)
    if i0 < 0:
        x, i0 = x[-i0:], 0
    j = min(len(out), i0 + len(x))
    if j > i0:
        out[i0:j] += x[:j - i0] * g


def _crush(x, bits, hold, dither=0.0, rng=None):
    """sample-and-hold (hold: samples, scalar or per-sample curve) + bit reduction (bits: scalar or curve)"""
    x = np.asarray(x, float)
    n = len(x)
    if np.ndim(hold) == 0:
        h = max(1, int(round(hold)))
        y = np.repeat(x[::h], h, axis=0)[:n] if h > 1 else x.copy()
    else:
        acc = np.floor(np.cumsum(1.0 / np.maximum(1.0, hold)))
        ch = np.ones(n, bool)
        ch[1:] = acc[1:] != acc[:-1]
        y = x[np.maximum.accumulate(np.where(ch, np.arange(n), 0))]
    step = 2.0 ** np.asarray(bits, float)
    if step.ndim == 1 and y.ndim == 2:
        step = step[:, None]
    if dither and rng is not None:
        y = y + (rng.random(y.shape) - 0.5) * dither / step
    return np.round(y * step) / step


def _interp(x, pos):
    pos = np.clip(pos, 0, len(x) - 2)
    i = pos.astype(int)
    fr = pos - i
    if x.ndim == 2:
        fr = fr[:, None]
    return x[i] * (1 - fr) + x[i + 1] * fr


def _rmod(rng, n, fc, bw, stereo=False):
    """band noise around a (moving) centre fc: low-passed noise ring-modulated by a gliding carrier.
    Half-bandwidth = bw, so it is a breathy band, never a tone."""
    ch = 2 if stereo else 1
    b = _lp(rng.standard_normal((n, ch), dtype=np.float32), bw, 4)
    b /= b.std() + 1e-12
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    ph = 2 * np.pi * np.cumsum(fc) / SR + rng.uniform(0, 2 * np.pi)
    y = b * np.cos(ph)[:, None]
    return y if stereo else y[:, 0]


def _kick(rng, f0=110., f1=38., dur=.35, decay=9., drive=2.2, bits=5, hold=6, sweep=18.,
          noise=.3, nd=260., punch=.3):
    """the mix2 'pixel' kick: pitch-dropping sine, saturated, sample-held, bit-reduced.
    A post-saturation transient shaper makes the first half-cycle the true peak."""
    n = S(dur)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * sweep)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * decay)
    x += rng.standard_normal(n) * np.exp(-t * nd) * noise
    x = np.tanh(x * drive) / np.tanh(drive)
    x *= (1 - punch) + punch * np.exp(-t * 70)
    x = _crush(x, bits, hold)
    x *= np.minimum(1.0, t / 0.0012)
    return _fade(x, 0, min(.012, dur / 4))


def _wood(rng, f_res, damp, dur=.03, noise=.25):
    """one muted knock on old wood: a few damped resonances + a tiny noise scrape"""
    m = S(dur)
    t = np.arange(m) / SR
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) * w for f, d, w in zip(f_res, damp, (1, .55, .3)))
    x += rng.standard_normal(m) * np.exp(-t * 900) * noise
    x *= np.minimum(1.0, t / 0.0004)
    return _fade(_norm(x), 0, .004)


@lru_cache(maxsize=16)
def _ir(dur=1.0, seed=9, decay=3.5, lp=2600., predelay=.012):
    """dark room (mix2 reverb_ir, low-passed, with a little pre-delay so the accent stays dry)"""
    r = np.random.default_rng(seed)
    n = S(dur)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-t * decay)[:, None]
    ir = _lp(ir, lp, 2)
    ir[:S(predelay)] = 0
    ir[S(predelay):S(predelay + .004)] *= np.linspace(0, 1, S(.004))[:, None]
    ir = _fade(ir, 0, .05)
    ir = (ir / np.sqrt((ir ** 2).sum(0)) * 0.6).astype(np.float32)
    ir.flags.writeable = False
    return ir


RV_DEC = 4      # the rooms are dark (< 2.6 kHz), so the reverb runs at 12 kHz: 16x cheaper


@lru_cache(maxsize=16)
def _ir_dec(rv):
    d = np.ascontiguousarray(_ir(*rv)[::RV_DEC]) * RV_DEC
    d.flags.writeable = False
    return d


def _reverb(x, rv):
    m = len(x)
    xd = _lp(x, 2800, 4)[::RV_DEC].astype(np.float32)
    y = fftconvolve(xd, _ir_dec(rv), axes=0)
    pos = np.arange(m) / RV_DEC
    k = np.arange(len(y))
    return np.stack([np.interp(pos, k, y[:, 0]), np.interp(pos, k, y[:, 1])], 1)


# ------------------------------------------------------------------ the kinds
# each generator writes the dry sound into o and a reverb send into s; c = accent sample


def _tear(o, s, c, dur, rng):
    """hard 3-frame glitch cut: very short crushed tick + low thud (+ two ghost ticks, frames 2 and 3)"""
    thud = _kick(rng, 92, 40, .26, decay=15, drive=1.8, bits=5, hold=7, sweep=34, noise=.25, nd=300)
    m = S(.018)
    t = np.arange(m) / SR
    tk = _norm(_bp(rng.standard_normal(m), 500, 2200))
    tk = _crush(tk * np.exp(-t * 170), 2, 10) * np.minimum(1, t / .0006)
    tk = _fade(tk, 0, .004)
    _add(o, thud, c - _lead(c))
    _add(o, tk, c - S(.0004), .40)
    _add(o, tk, c + S(1 / 24), .14, pan=-.45)
    _add(o, tk, c + S(2 / 24), .08, pan=.5)
    _add(s, thud, c, .22)
    _add(s, tk, c, .20)


def _flash(o, s, c, dur, rng):
    """white flash frame: short sub thump + bitcrushed air burst"""
    P = min(c, S(.09))
    if P > S(.01):                                   # a breath of air leaning into the flash
        u = np.linspace(0, 1, P)
        air = _bp(rng.standard_normal((P, 2)), 600, 2400)
        air = air / air.std() * (u ** 3)[:, None] * .05
        _add(o, _fade(air, .003, .002), c - P)
    sub = _kick(rng, 72, 34, .55, decay=6.5, drive=1.6, bits=6, hold=4, sweep=14, noise=.12)
    _add(o, sub, c - _lead(c))
    m = S(.32)
    t = np.arange(m) / SR
    b = _bp(rng.standard_normal((m, 2)), 450, 2600)
    b = b / b.std() * (np.minimum(1, t / .0015) * np.exp(-t * 24))[:, None] * .35
    b = _crush(b, 4, 6)
    b = _vlp(b, 2600 * 0.25 ** (t / t[-1]))           # the burst dims as it dies
    _add(o, _fade(b, 0, .02), c - S(.001), .55)
    _add(s, b, c, .5)
    _add(s, sub, c, .15)


def _slice(o, s, c, dur, rng):
    """VHS bands sliding: a tape scrub / shuttle that slides into a transport 'clack'"""
    P = min(c, S(1.2))
    W = S(.2)
    L = int(P * 1.5) + S(.7)
    tt = np.arange(L) / SR
    prog = _lp(rng.standard_normal(L), 900) * 1.0          # the 'programme' on the tape: dark mumble + hum
    prog += _rmod(rng, L, 420, 90) * .6 + _rmod(rng, L, 1150, 140) * .35
    am = _lp(rng.standard_normal(L), 7)
    prog *= np.clip(am / am.std() * .6 + .5, 0, 1.3)
    prog += .25 * np.sin(2 * np.pi * 50 * tt) + .1 * np.sin(2 * np.pi * 100 * tt + 1)
    prog = _norm(prog)
    p_end = S(.2)
    if P > S(.02):
        u = np.linspace(0, 1, P)
        tp = np.arange(P) / SR
        sp = (.25 + 2.3 * u ** 2
              + .55 * (1 - u) ** 1.5 * np.sin(2 * np.pi * (2.5 * tp + 4 * tp ** 2 / max(.1, P / SR))))
        pos = S(.2) + np.cumsum(sp)
        p_end = pos[-1]
        sc = np.stack([_interp(prog, pos), _interp(prog, pos + 37)], 1)       # two heads, slightly apart
        sc = _crush(_vlp(sc, 700 + 1900 * u ** 1.5), 7, 3)
        sc = sc * (u ** 1.6)[:, None] * .42
        _add(o, _fade(sc, .005, .003), c - P)
    k = _kick(rng, 125, 46, .24, decay=14, drive=2.0, bits=5, hold=6, sweep=30, noise=.3, nd=350)
    clk = _wood(rng, (620, 1050, 1580), (260, 380, 520), .03)
    _add(o, k, c - _lead(c))
    _add(o, _lp(clk, 1800), c - S(.0005), .35)
    v = np.linspace(0, 1, W)                            # ... and the tape winds down behind it
    pos2 = p_end + np.cumsum(1.4 * (1 - v) ** 1.6 + .03)
    wd = _vlp(_interp(prog, pos2), 1600 * (250 / 1600) ** v) * (1 - v) ** 1.5 * .2
    _add(o, _fade(wd, .004, .01), c + S(.003), pan=.25)
    _add(s, k, c, .2)
    _add(s, wd, c, .3)


def _zoom(o, s, c, dur, rng):
    """zoom-through punch: reversed-reverb suck rising into a dull low hit"""
    hit = _kick(rng, 88, 31, .7, decay=5.5, drive=2.0, bits=5, hold=6, sweep=13, noise=.3, nd=180)
    m = S(.12)
    th = _norm(_lp(rng.standard_normal(m), 380)) * np.exp(-np.arange(m) / SR * 45) * .5
    hit[:m] += th * np.minimum(1, np.arange(m) / S(.0015))
    hit = _lp(hit, 1400)
    P = min(c, S(.9))
    if P > S(.02):
        ir = _ir(1.2, 23, 3.0, 3200., .0)
        z = fftconvolve(hit[::-1, None], ir, axes=0)[::-1]
        K = len(ir) - 1
        suck = z[K - P:K]                                    # ends exactly where the hit begins
        u = np.linspace(0, 1, P)
        suck = _vlp(suck, 300 * 10 ** (u ** 1.2)) * (u ** 1.4)[:, None]
        suck = _norm(suck) * .42 * np.abs(hit).max()
        _add(o, _fade(suck, .004, .002), c - P)
    _add(o, hit, c - _lead(c))
    _add(s, hit, c, .18)


def _glyph(o, s, c, dur, rng):
    """occult glyph reveal: low ritual drum, then a quiet, dark, inharmonic metallic shimmer"""
    m = S(.9)
    t = np.arange(m) / SR
    f = 54 + 20 * np.exp(-t * 22)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    for r_, w, d in zip((1.59, 2.14, 2.30, 2.65), (.45, .3, .22, .15), (9, 13, 16, 20)):
        x += np.sin(2 * np.pi * 54 * r_ * t) * np.exp(-t * d) * w          # membrane modes
    x += _norm(_lp(rng.standard_normal(m), 650)) * np.exp(-t * 55) * .6     # the hand/skin slap
    x = np.tanh(1.7 * x) / np.tanh(1.7) * (.75 + .25 * np.exp(-t * 60))
    hit = _fade(_crush(x, 6, 5) * np.minimum(1, t / .0015), 0, .05)
    # shimmer: one static inharmonic cluster, L/R detuned so it beats slowly; swells in after the hit
    m2 = S(1.0)
    t2 = (np.arange(m2) / SR).astype(np.float32)
    base = 311 * rng.uniform(.97, 1.03)
    sh = np.zeros((m2, 2))
    for k, r_ in enumerate((1, 1.483, 1.932, 2.546, 3.31, 4.12)):
        env = (1 - np.exp(-t2 * rng.uniform(14, 30))) * np.exp(-t2 * (2.2 + .9 * k)) / (1 + .6 * k)
        for ch, dt in ((0, -.7), (1, .7)):
            sh[:, ch] += np.sin(np.float32(2 * np.pi * (base * r_ + dt + rng.uniform(-.3, .3))) * t2
                                + np.float32(rng.uniform(0, 2 * np.pi))) * env
    sh *= (.6 + .4 * np.sin(2 * np.pi * 41 * t2))[:, None]          # ring-mod: darker, more 'metal'
    sh = _lp(_crush(_norm(sh), 5, 3), 2400)
    sh = _fade(_norm(sh), .01, .1)
    P = min(c, S(.8))
    if P > S(.02):                                    # a low drawn breath + the shimmer, reversed, rising in
        u = np.linspace(0, 1, P)
        sw = _rmod(rng, P, 150 + 110 * u, 70, True) * .7
        sw += _lp(rng.standard_normal((P, 2)), 300) * 6.0
        sw = _norm(sw) * (u ** 2.2)[:, None] * .17
        Q = min(P, S(.4))
        rs = sh[:Q][::-1] * (np.linspace(0, 1, Q) ** 2)[:, None] * .07
        _add(o, _fade(sw, .005, .004), c - P)
        _add(o, _fade(rs, .005, .004), c - Q)
    _add(o, hit, c - _lead(c))
    _add(o, sh, c + S(.012), .13)
    _add(s, hit, c, .25)
    _add(s, sh, c + S(.012), .07)


def _iris(o, s, c, dur, rng):
    """blocky circle opening: a breathy swell into a soft pixel pop"""
    P = min(c, S(.8))
    if P > S(.02):
        u = np.linspace(0, 1, P)
        sc = .82 + .18 * u                              # the breath opens as the iris opens
        br = (_rmod(rng, P, 620 * sc, 110, True) + _rmod(rng, P, 1080 * sc, 150, True) * .6
              + _rmod(rng, P, 1900 * sc, 220, True) * .25)
        br += _lp(rng.standard_normal((P, 2)), 1200) * 2.0
        br = _crush(_norm(br), 8, 3) * (u ** 2.4)[:, None] * .3
        _add(o, _fade(br, .004, .006), c - P)
    m = S(.07)
    t = np.arange(m) / SR
    pop = np.sin(2 * np.pi * np.cumsum(95 + 230 * np.exp(-t * 140)) / SR) * np.exp(-t * 55)
    pop = _crush(np.tanh(1.5 * pop) / np.tanh(1.5), 4, 8) * np.minimum(1, t / .0012)
    pop = _fade(pop, 0, .006)
    th = _kick(rng, 70, 42, .22, decay=16, drive=1.4, bits=6, hold=5, sweep=25, noise=.08)
    _add(o, pop, c - S(.004), .7)
    _add(o, th, c - _lead(c), .65)
    m = S(.35)
    t = np.arange(m) / SR
    ex = _rmod(rng, m, 700, 140, True) * (np.minimum(1, t / .01) * np.exp(-t * 9))[:, None]
    ex = _lp(ex, 1500) * .1
    _add(o, _fade(ex, 0, .03), c + S(.01))
    _add(s, pop, c, .2)
    _add(s, ex, c + S(.01), .4)


def _grain_bank(rng, m):
    return {bw: _lp(rng.standard_normal(m), bw, 4) / np.sqrt(bw / 12000) for bw in (60, 150, 400)}


def _band_grain(rng, bank, fc, L):
    bw = 60 if fc < 300 else 150 if fc < 900 else 400
    buf = bank[bw]
    off = rng.integers(0, len(buf) - L)
    tl = np.arange(L) / SR
    return buf[off:off + L] * np.cos(2 * np.pi * fc * tl + rng.uniform(0, 6.28)) * np.sin(np.linspace(0, np.pi, L)) ** 2


def _sortwipe(o, s, c, dur, rng):
    """pixel-sorted streak wipe: a granular, sample-and-hold descending sweep landing on a thud"""
    P = min(c, S(.9))
    bank = _grain_bank(rng, S(.6))
    if P > S(.03):
        st = np.zeros((P, 2))
        tau = 0.0
        while tau < P - S(.012):
            u = tau / P
            fc = 2400 * (180 / 2400) ** (u ** .9)
            fc = 2 ** (np.round(np.log2(fc) * 3) / 3) * rng.uniform(.97, 1.03)     # sorted in 1/3-octave steps
            L = min(S(rng.uniform(.010, .024)), P - int(tau))
            g = _band_grain(rng, bank, fc, L) * (.12 + .32 * u ** 1.5) * rng.uniform(.6, 1)
            _add(st, g, int(tau), 1.0, pan=np.clip(-.7 + 1.4 * u + rng.normal(0, .2), -1, 1))
            tau += S(.024 + (.006 - .024) * u ** 1.2) * rng.uniform(.7, 1.3)
        u = np.linspace(0, 1, P)
        st = _crush(st, 5, 2 + 9 * u)                       # more 'pixel' the lower it sorts
        _add(o, _fade(st, .004, .003), c - P)
    k = _kick(rng, 100, 40, .3, decay=12, drive=2.2, bits=5, hold=7, sweep=26, noise=.25)
    _add(o, k, c - _lead(c))
    tau = c + S(.006)
    for i in range(7):                                  # the last streaks settle
        g = _crush(_band_grain(rng, bank, rng.uniform(150, 320), S(rng.uniform(.012, .022))), 5, 9)
        _add(o, g, tau, .22 * .7 ** i, pan=rng.uniform(-.8, .8))
        tau += S(rng.uniform(.02, .035))
    _add(s, k, c, .18)


def _swirl(o, s, c, dur, rng):
    """whirlpool: swirling, circling filtered noise, its pitch slowly dipping; accent = the collapse"""
    P = min(c, S(1.2))
    R = S(.3)
    if P > S(.03):
        T = P + R
        tt = np.arange(T) / SR
        tc = P / SR
        u = np.arange(T) / P
        after = u > 1
        rate = np.where(after, 12.5 * np.exp(-(tt - tc) * 8), 2.5 + 10 * np.minimum(u, 1) ** 1.6)
        phi = 2 * np.pi * np.cumsum(rate) / SR
        dip = np.where(after, 1100 * 2 ** -2.1 * 2 ** (-(tt - tc) * 4),
                       1100 * 2 ** (-1.1 * u - np.clip((u - .75) / .25, 0, 1) ** 2))
        A = _rmod(rng, T, dip * (1 + .35 * np.sin(phi)), 160)
        B = _rmod(rng, T, dip * 1.45 * (1 + .35 * np.sin(phi + np.pi)), 200) * .7
        pa = .8 * np.sin(phi)
        sw = np.stack([A * np.minimum(1, 1 - pa) + B * np.minimum(1, 1 + pa),
                       A * np.minimum(1, 1 + pa) + B * np.minimum(1, 1 - pa)], 1)
        rum = _lp(rng.standard_normal((T, 2)), 170) * 9.0
        env = np.where(after, .3 * np.exp(-(tt - tc) * 14), np.minimum(u, 1) ** 1.8 * (.85 + .15 * np.sin(phi)))
        renv = np.where(after, .3 * np.exp(-(tt - tc) * 10), np.minimum(u, 1) ** 2.5 * .6)
        sw = sw * env[:, None] + rum * renv[:, None]
        sw = _crush(sw, 7, 2)
        sw[P - S(.003):P] *= np.linspace(1, .3, S(.003))[:, None]   # the vortex is sucked in on the beat
        sw = sw / (np.abs(sw[:P]).max() + 1e-12) * .33
        _add(o, _fade(sw, .006, .03), c - P)
        _add(s, sw[P:], c, .3)
    k = _kick(rng, 92, 28, .6, decay=6.5, drive=2.3, bits=4, hold=8, sweep=16, noise=.25, punch=.6)
    m = S(.12)
    t = np.arange(m) / SR
    gulp = _rmod(rng, m, 240 * (70 / 240) ** (t / t[-1]), 60) * np.exp(-t * 30) * np.minimum(1, t / .002)
    _add(o, k, c - _lead(c))
    _add(o, _fade(gulp, 0, .01), c, .4)
    _add(s, k, c, .2)


def _grow(o, s, c, dur, rng):
    """pixels swell into huge blocks and back: the texture crushes harder and harder into the accent, then un-crushes"""
    tc = c / SR
    E = max(dur, tc + .15)
    T = S(E + .15)
    tt = np.arange(T) / SR
    up = np.clip(tt / max(tc, 1e-3), 0, 1)
    dn = np.clip(1 - (tt - tc) / max(E - tc, 1e-3), 0, 1)
    g = np.where(tt <= tc, up, dn)
    g = g * g * (3 - 2 * g)
    br = np.cumsum(rng.standard_normal((T, 2)), axis=0)
    tex = _norm(_hp(br, 40)) + _rmod(rng, T, 330, 120, True) * .25 + _rmod(rng, T, 780, 160, True) * .12
    tex = _norm(tex)
    cr = _crush(tex, 11 - 8 * g, 1 + 47 * g ** 2.2)          # 48x sample-rate reduction and 3 bits at the peak
    cr = cr * (.07 + .26 * g ** 1.5)[:, None]
    _add(o, _fade(cr, .02, .15), 0)
    k = _kick(rng, 108, 38, .4, decay=9, drive=2.6, bits=3, hold=24, sweep=18, noise=.3)
    _add(o, k, c - _lead(c))
    _add(s, k, c, .2)


def _resbank(imp, modes):
    """impulses through damped-sine resonators [(f, damp, weight)]: every impulse rings like a small wooden chip"""
    y = 0
    for f, d, w in modes:
        r, wq = np.exp(-d / SR), 2 * np.pi * f / SR
        y = y + lfilter([0, w * np.sin(wq)], [1, -2 * r * np.cos(wq), r * r], imp, axis=0)
    return y


def _scatter(o, s, c, dur, rng):
    """blocks flying apart: a crackling granular burst that spreads out in stereo"""
    n = len(o)
    B = 5                                                    # five kinds of chip + one kind of heavier chunk
    chips = np.zeros((B + 1, n, 2))

    def hit_at(i, a, pan, bank):
        if 0 <= i < n:
            chips[bank, i] += a * np.array([min(1, 1 - pan), min(1, 1 + pan)])
    P = min(c, S(.15))
    for _ in range(int(P / SR * 60)):                         # tension: faint crackle rising into the break
        u = rng.random() ** .5
        hit_at(c - P + int(u * (P - S(.02))), (.05 + .1 * u) * .6, rng.uniform(-.3, .3), rng.integers(B))
    dts = np.sort(np.minimum(rng.exponential(.13, 90), .75))
    for i, dt in enumerate(dts):                              # the burst: dense at the break, fanning out wide
        chunk = rng.random() < .2
        pan = (1 if i % 2 else -1) * min(1, dt / .3) * rng.uniform(.5, 1)
        a = np.exp(-dt * 4.5) * rng.uniform(.25, 1) * (.5 if chunk else .3)
        hit_at(c + S(.004 + dt), a, pan, B if chunk else rng.integers(B))
    first = max(0, c - P)
    st = np.zeros((n - first, 2))
    for b_ in range(B):
        f = rng.uniform(380, 2100)
        st += _resbank(chips[b_, first:], ((f, rng.uniform(140, 300), 1), (f * rng.uniform(1.4, 2.2), 420, .45)))
    f = rng.uniform(140, 300)
    st += _resbank(chips[B, first:], ((f, 80, 1), (f * rng.uniform(1.5, 2.3), 130, .5), (f * 3.1, 200, .25)))
    st = _crush(st * .6, 6, 4)
    hit = _kick(rng, 112, 40, .28, decay=13, drive=2.2, bits=5, hold=6, sweep=28, noise=.35)
    m = S(.12)
    t = np.arange(m) / SR
    brk = _norm(_bp(rng.standard_normal(m), 300, 1800)) * np.exp(-t * 45)
    brk = _fade(_crush(brk, 3, 10) * np.minimum(1, t / .001), 0, .01)
    _add(o, hit, c - _lead(c))
    _add(o, brk, c - S(.0008), .35)
    _add(o, st, first)
    _add(s, hit, c, .15)
    _add(s, st, first, .2)


def _melt(o, s, c, dur, rng):
    """datamosh drip: a slow, viscous downward smear; accent = the heavy drop"""
    P = min(c, S(1.0))
    T = P + S(1.05)
    tt = np.arange(T) / SR
    u = np.arange(T) / T
    octs = 1.7 * u ** .8
    tex = sum(_rmod(rng, T, F * 2 ** -octs, bw) * w
              for F, bw, w in ((1350, 170, .25), (720, 110, .5), (320, 70, .8)))
    am = _lp(rng.standard_normal(T), 4)
    tex *= np.clip(.75 + .35 * am / am.std(), .2, 1.3)                # viscous, gloopy swell and sag
    tex = _crush(_norm(tex), 6, 2 + 10 * u)
    tex = _vlp(tex, 2300 * (220 / 2300) ** u)
    tex = np.stack([tex, np.roll(tex, S(.011)) * .9], 1)             # width from a short smear, not more noise
    tp = tt - P / SR
    env = np.where(tp < 0, np.clip(1 + tp / max(P / SR, 1e-3), 0, 1) ** 1.3, np.exp(-np.maximum(tp, 0) * 2.6))
    sm = _norm(tex) * env[:, None] * .38
    _add(o, _fade(sm, .01, .1), c - P)
    drop = _lp(_kick(rng, 82, 29, .7, decay=5.5, drive=1.9, bits=5, hold=8, sweep=12, noise=.15, punch=.5), 900)
    _add(o, drop, c - _lead(c))
    m = S(.6)
    t = np.arange(m) / SR
    drag = _rmod(rng, m, 260 * 2 ** (-t * 3.5), 60, True) * (np.minimum(1, t / .015) * np.exp(-t * 5))[:, None]
    _add(o, _fade(drag, 0, .05), c, .25)
    _add(s, drop, c, .25)
    _add(s, sm, c - P, .35)


def _blocks(o, s, c, dur, rng):
    """dithered block wipe: a soft crushed stutter on 32nd notes at 103 bpm, rising into a heavier block"""
    step = S(60 / 103 / 8)
    m = S(.048)
    t = np.arange(m) / SR
    blk = _kick(rng, 96, 52, .048, decay=30, drive=1.8, bits=8, hold=1, sweep=40, noise=0)
    blk = blk * .8 + _rmod(rng, m, 650, 260) * np.exp(-t * 45) * .12
    blk = _fade(_crush(blk, 4, 10, dither=.9, rng=rng) * np.minimum(1, t / .0015), 0, .012)
    blk = _lp(blk, 2200)
    k = 1
    while c - k * step >= S(.004) and k <= 12:
        _add(o, blk, c - k * step - LEAD, .52 * .74 ** (k - 1), pan=.22 * (-1) ** k)
        k += 1
    big = _kick(rng, 104, 40, .3, decay=11, drive=2.2, bits=4, hold=10, sweep=26, noise=.2)
    big[:m] += blk * .5
    _add(o, _lp(big, 2400), c - _lead(c))
    _add(o, blk, c + step - LEAD, .3, pan=-.3)
    _add(o, blk, c + 2 * step - LEAD, .15, pan=.3)
    _add(s, big, c, .15)


def _shutter(o, s, c, dur, rng):
    """vertical blinds: a quick cascade of muted wooden clicks fanning out in stereo, closing on a clack"""
    def click(g_lo=1.0):
        return _lp(_crush(_wood(rng, (rng.uniform(420, 560), rng.uniform(820, 1000), rng.uniform(1350, 1650)),
                                np.array((260, 380, 520)) * rng.uniform(.85, 1.2), .028), 6, 3), 1800)
    P = min(c, S(.32))
    times, gap, tau = [], .017, c - S(.017)
    while tau >= c - P + S(.003):
        times.append(tau)
        gap = min(.036, gap + .0035)
        tau -= S(gap)
    times = times[::-1]
    K = len(times)
    for j, tau in enumerate(times):
        pr = (j + 1) / K
        _add(o, click(), tau, .18 + .22 * pr, pan=(-1) ** j * (.15 + .7 * pr))
        _add(s, click(), tau, .1)
    clack = _lp(_wood(rng, (360, 690, 1180), (170, 260, 380), .05), 1600)
    body = _kick(rng, 150, 62, .14, decay=24, drive=1.4, bits=6, hold=4, sweep=40, noise=.1)
    _add(o, body, c - _lead(c), .85)
    _add(o, clack, c - S(.0005), .55)
    for dt, g, p in ((.031, .26, -.85), (.067, .16, .85), (.109, .09, -.7)):
        _add(o, click(), c + S(dt), g, pan=p)
    _add(s, body, c, .15)
    _add(s, clack, c, .25)


# kind -> (generator, tail after the accent (s), peak, final low-pass (Hz), reverb (dur, seed, decay, lp))
_KINDS = {
    'tear':     (_tear,     .55, .30, 3200, (.6, 11, 7.0, 2200)),
    'flash':    (_flash,    .9,  .30, 3000, (1.0, 12, 4.5, 2600)),
    'slice':    (_slice,    .6,  .30, 2800, (.7, 13, 6.0, 2200)),
    'zoom':     (_zoom,     1.0, .32, 2600, (1.0, 14, 4.0, 2000)),
    'glyph':    (_glyph,    1.2, .32, 2800, (1.2, 15, 3.0, 2200)),
    'iris':     (_iris,     .7,  .28, 2800, (.8, 16, 5.0, 2400)),
    'sortwipe': (_sortwipe, .6,  .30, 3000, (.7, 17, 6.0, 2200)),
    'swirl':    (_swirl,    1.0, .32, 2800, (1.0, 18, 4.0, 2000)),
    'grow':     (_grow,     .6,  .30, 3500, (.7, 19, 6.0, 2200)),
    'scatter':  (_scatter,  .9,  .30, 3000, (.8, 20, 5.0, 2400)),
    'melt':     (_melt,     1.2, .32, 2400, (1.2, 21, 3.0, 1800)),
    'blocks':   (_blocks,   .55, .28, 2400, (.6, 22, 7.0, 2000)),
    'shutter':  (_shutter,  .6,  .28, 2600, (.6, 23, 7.0, 2200)),
}


def _accent_guard(o, c, lo=.004, hi=.008, ratio=.85):
    """keep the accent the loudest moment: anything outside [c-lo, c+hi] that would reach the accent
    peak (a later kick lobe, a resonant build-up on an unlucky seed) is gently held to ratio x accent peak"""
    a = np.abs(o).max(1)
    i0, i1 = max(0, c - S(lo)), min(len(o), c + S(hi))
    if i1 <= i0:
        return o
    lim = ratio * a[i0:i1].max()
    env = maximum_filter1d(a, S(.006))
    g = np.minimum(1.0, lim / (env + 1e-12))
    g[i0:i1] = 1.0
    if g.min() >= 1.0:
        return o
    g = uniform_filter1d(minimum_filter1d(g, S(.008)), S(.008))      # smooth, but never lets a peak through
    g[i0:i1] = np.maximum(g[i0:i1], 1.0)
    g = np.minimum(g, uniform_filter1d(g, S(.002)))                  # soften the edges of the protected window
    return o * g[:, None]


for _k in _KINDS.values():
    _ir_dec(_k[4])


def _generic(o, s, c, dur, rng):
    """fallback for any kind this module does not know yet: a short crushed air lean-in + a crushed 'pixel' hit"""
    P = min(c, S(.25))
    if P > S(.01):
        u = np.linspace(0, 1, P)
        air = _bp(rng.standard_normal((P, 2)), 500, 2200)
        air = _crush(air / (air.std() + 1e-12) * (u ** 3)[:, None] * .06, 5, 4)
        _add(o, _fade(air, .003, .002), c - P)
    hit = _kick(rng, 108, 38, .34, decay=9, drive=2.2, bits=4, hold=8, sweep=22, noise=.25)
    _add(o, hit, c - _lead(c))
    _add(s, hit, c, .2)


# aliases: names a transitions director is likely to invent -> the closest existing sound
_ALIAS = {'pixel': 'blocks', 'pixels': 'blocks', 'dissolve': 'blocks', 'dither': 'blocks', 'mosaic': 'blocks',
          'spread': 'scatter', 'burst': 'scatter', 'explode': 'scatter', 'shatter': 'scatter', 'sand': 'scatter',
          'spiral': 'swirl', 'vortex': 'swirl', 'twist': 'swirl', 'whirl': 'swirl', 'drain': 'swirl',
          'drip': 'melt', 'bleed': 'melt', 'ink': 'melt', 'smear': 'melt', 'liquid': 'melt', 'ripple': 'melt',
          'wipe': 'sortwipe', 'sort': 'sortwipe', 'datamosh': 'sortwipe', 'scan': 'slice', 'vhs': 'slice',
          'glitch': 'tear', 'cut': 'tear', 'strobe': 'flash', 'white': 'flash', 'blink': 'flash',
          'eye': 'iris', 'lens': 'iris', 'circle': 'iris', 'blinds': 'shutter', 'venetian': 'shutter',
          'page': 'glyph', 'pageturn': 'glyph', 'letters': 'glyph', 'script': 'glyph', 'rune': 'glyph',
          'push': 'zoom', 'punch': 'zoom', 'dive': 'zoom', 'zoomin': 'zoom', 'scale': 'grow', 'pixelate': 'grow'}


def _resolve(kind):
    """unknown kinds never crash: exact match, alias, alias by substring, else the generic crushed hit"""
    k = str(kind).lower()
    if k in _KINDS:
        return k
    if k in _ALIAS:
        return _ALIAS[k]
    for a, b in _ALIAS.items():
        if a in k:
            return b
    for b in _KINDS:
        if b in k:
            return b
    return None


def sfx(kind, dur, center, seed=0):
    """synthesise one transition sound: (n, 2) float32 at SR, accent on sample round(center*SR).
    Unknown kinds fall back to the closest known sound or a generic crushed hit (never raises)."""
    try:
        return _sfx(kind, dur, center, seed)
    except Exception:                                   # absolutely never break the mix over one transition sound
        try:
            return _sfx('__generic__', dur, center, seed)
        except Exception:
            return np.zeros((S(max(float(dur), .05)), 2), np.float32)


def _sfx(kind, dur, center, seed=0):
    if kind == '__generic__':
        gen, tail, peak, lp, rv = (_generic, .6, .28, 2600, (.7, 24, 6.0, 2200))
        kid = len(ALL_KINDS)
    else:
        rk = _resolve(kind)
        if rk is None:
            return _sfx('__generic__', dur, center, seed)
        gen, tail, peak, lp, rv = _KINDS[rk]
        kid = ALL_KINDS.index(rk) if rk in ALL_KINDS else len(ALL_KINDS) + 1
    kind = rk if kind != '__generic__' else kind
    dur = max(float(dur), .02)
    center = min(max(float(center), 0.0), dur)
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, kid, 7177])
    n = S(min(dur + MAX_TAIL, max(dur + .05, center + tail)))
    c = S(center)
    o = np.zeros((n, 2))
    snd = np.zeros((n, 2))
    gen(o, snd, c, dur, rng)
    nz = np.flatnonzero(np.any(snd != 0, axis=1))
    if len(nz):
        i0 = nz[0]
        o[i0:] += _reverb(snd[i0:], rv)
    o = _lp(_hp(o, 20), lp)
    o = _fade(o, .003 if c >= LEAD + S(.003) else .0005, .04)   # the kick ramps itself in from 0
    o = _accent_guard(o, c)
    o *= peak / (np.abs(o).max() + 1e-12)
    return o.astype(np.float32)


# ------------------------------------------------------------------ demo / QA
def _demo(wav='/home/claude/voynich/qa/trans_sfx_demo.wav', png='/home/claude/voynich/qa/trans_sfx_demo.png',
          dur=.4, center=.2, gap=1.2, lead=.5):
    import time
    from scipy.io import wavfile
    slot = dur + gap
    total = S(lead + slot * len(ALL_KINDS) + .5)
    mix = np.zeros((total, 2), np.float32)
    rows, accents = [], []
    for i, kind in enumerate(ALL_KINDS):
        t0 = time.perf_counter()
        x = sfx(kind, dur, center, seed=i)
        ms = (time.perf_counter() - t0) * 1e3
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 2
        assert np.array_equal(x, sfx(kind, dur, center, seed=i)), 'not deterministic'
        a = np.abs(x).max(1)
        k = S(.004)
        env = np.convolve(a.astype(float) ** 2, np.ones(k) / k, 'same') ** .5     # 4 ms RMS envelope
        off_env = (np.argmax(env) - S(center)) / SR * 1e3
        off_pk = (np.argmax(a) - S(center)) / SR * 1e3
        pk = a.max()
        win = x[max(0, S(center - .2)):S(center + .4)]
        rms_core = np.sqrt((win.astype(float) ** 2).mean())
        rms_all = np.sqrt((x.astype(float) ** 2).mean())
        dc = np.abs(x.astype(float).mean(0)).max()
        X = np.abs(np.fft.rfft(x.mean(1)))
        fr = np.fft.rfftfreq(len(x), 1 / SR)
        hf = (X[fr > 3000] ** 2).sum() / (X ** 2).sum()
        edge = max(np.abs(x[0]).max(), np.abs(x[-1]).max())
        pre = np.abs(x[:max(1, S(center) - S(.012))]).max() / pk if S(center) > S(.012) else 0
        rows.append((kind, len(x) / SR, pk, 20 * np.log10(pk), 20 * np.log10(rms_core), 20 * np.log10(rms_all),
                     off_pk, off_env, dc, hf * 100, pre, edge, ms))
        i0 = S(lead + i * slot)
        mix[i0:i0 + len(x)] += x
        accents.append(lead + i * slot + center)
    hdr = (f'{"kind":9s} {"len s":>5s} {"peak":>6s} {"pk dB":>6s} {"RMScore":>7s} {"RMSall":>7s} '
           f'{"off pk":>7s} {"off env":>7s} {"DC":>8s} {">3k %":>6s} {"pre/pk":>6s} {"edge":>8s} {"ms":>5s}')
    print(hdr)
    for r in rows:
        print(f'{r[0]:9s} {r[1]:5.2f} {r[2]:6.3f} {r[3]:6.1f} {r[4]:7.1f} {r[5]:7.1f} {r[6]:+6.1f}ms {r[7]:+6.1f}ms '
              f'{r[8]:8.1e} {r[9]:6.2f} {r[10]:6.2f} {r[11]:8.1e} {r[12]:5.1f}')
    print('demo peak', float(np.abs(mix).max()), 'clipped samples', int((np.abs(mix) >= 1).sum()))
    wavfile.write(wav, SR, np.round(np.clip(mix, -1, 1) * 32767).astype(np.int16))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = mix.mean(1)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(26, 9), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    tt = np.arange(len(m)) / SR
    ax0.plot(tt[::16], np.abs(mix).max(1)[::16], lw=.6, color='k')
    ax0.set_ylim(0, .4)
    ax0.set_ylabel('|x|')
    ax1.specgram(m, NFFT=2048, Fs=SR, noverlap=1792, cmap='magma', vmin=-115, vmax=-35)
    ax1.set_ylim(0, 6000)
    ax1.set_ylabel('Hz')
    ax1.axhline(3000, color='c', lw=.5, ls=':')
    for i, (kind, a) in enumerate(zip(ALL_KINDS, accents)):
        for ax in (ax0, ax1):
            ax.axvline(a, color='c', lw=.6, ls='--')
            ax.axvspan(lead + i * slot, lead + i * slot + dur, color='c', alpha=.06)
        ax0.text(a, .37, kind, ha='center', va='top', fontsize=11, weight='bold')
        r = rows[i]
        ax0.text(a, .30, f'pk {r[2]:.2f}  {r[7]:+.1f}ms', ha='center', va='top', fontsize=8)
    ax1.set_xlabel('s  (dashed = center / accent, shaded = visual transition, dur 0.4 s)')
    fig.suptitle('trans_sfx demo - 13 transition kinds (dur 0.4 s, center 0.2 s)')
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    print('wrote', wav, png)
    return rows


if __name__ == '__main__':
    _demo()
