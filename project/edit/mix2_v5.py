"""v2 audio: python3 mix2.py A|B  ->  edit/mix_A.wav / mix_B.wav  (130 s)
A: «аллея кошмаров» under the intro and act I, a slow creaking smear into Mara (boom on 0:30).
B: Mara only; its first hits under the book, its wall under act I, the same creaking smear into the 0:30 boom.
No glitch clicks: cuts are marked by crushed, creaking 'pixel' kicks.
"""
import sys, subprocess, numpy as np
from scipy.signal import lfilter, fftconvolve, butter, sosfilt
sys.path.insert(0, '/home/claude/voynich/edit')
import os
VER = sys.argv[1] if len(sys.argv) > 1 else 'B'
os.environ['VOY_VER'] = VER
from timeline import LINES, A, DUR, MARA_BOOM, ACT2, T1_FILM0, B_MARA_OFF, TRANS0
import sync
import render as RN
import book as BK
shots = RN.build()
def shot_named(name):
    return next(s_ for s_ in shots if getattr(s_.fn, '__name__', '') == name)
UNDERWATER = (shot_named('s22').t0, shot_named('s22').t1)
CARD = shot_named('s7').t0
PAUSE = (shot_named('s32').t0, shot_named('s32').t1)
KNOCKS = [shot_named('s12').t0, shot_named('s13').t0, shot_named('s15').t0]   # three knocks = three cuts of the creak
SR = 48000
N = int(DUR * SR)
rng = np.random.default_rng(5)
tt = np.arange(N) / SR


def load(path, filt=None, start=None, end=None):
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    if end is not None:
        cmd += ['-to', str(end)]
    cmd += ['-i', path]
    if filt:
        cmd += ['-af', filt]
    cmd += ['-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.check_output(cmd), np.float32).reshape(-1, 2).copy()


def box(x, k):
    """fast centred moving average (same length)"""
    k = max(1, int(k))
    c = np.cumsum(np.concatenate([np.zeros(k // 2 + 1), x, np.full(k, x[-1])]))
    return ((c[k:k + len(x)] - c[:len(x)]) / k).astype(np.float32)


def place(buf, x, t, gain=1.0):
    i = int(round(t * SR))
    if i < 0:
        x = x[-i:]; i = 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[:j - i] * gain


def fade(x, a=0.01, b=0.03):
    na, nb = int(a * SR), int(b * SR)
    x = x.copy()
    if na:
        x[:na] *= np.linspace(0, 1, na)[:, None]
    if nb:
        x[-nb:] *= np.linspace(1, 0, nb)[:, None]
    return x


def env_curve(points):
    ts, vs = zip(*points)
    return np.interp(tt, ts, vs).astype(np.float32)


# ================================================================ voices (whole lines, never split)
NARR = ('highpass=f=70,lowpass=f=11000,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,'
        'acrusher=bits=12:mix=0.18:mode=log,aecho=0.8:0.6:45|90:0.18|0.10')
READER = (f'aresample={SR},asetrate={SR}*0.80,aresample={SR},atempo=1.25,'
          'acrusher=bits=8:mix=0.45:mode=log,aecho=0.8:0.7:60|130:0.3|0.2,acompressor=threshold=-20dB:ratio=5:makeup=4')
WHISP_LOW = f'aresample={SR},asetrate={SR}*0.75,aresample={SR},atempo=1.333,lowpass=f=5000'
DRY = 'highpass=f=70,acompressor=threshold=-24dB:ratio=3:makeup=5'

def reverb_ir(dur=2.8, seed=9):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    e = np.exp(-np.arange(n) / SR * 3.2)
    ir = np.stack([r.normal(0, 1, n) * e, r.normal(0, 1, n) * e], 1).astype(np.float32)
    return ir / np.sqrt((ir ** 2).sum(0)) * 0.6


def varispeed(x, speed):
    pos = np.cumsum(speed)
    pos = pos[pos < len(x) - 2]
    i = pos.astype(int); fr = (pos - i)[:, None]
    return x[i] * (1 - fr) + x[i + 1] * fr


vo = np.zeros((N, 2), np.float32)
cache = {}
def src(f, chain):
    k = (f, chain)
    if k not in cache:
        cache[k] = load(f'{A}/{f}', chain)
    return cache[k]

NIKTO = ('highpass=f=70,lowpass=f=11000,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,'
         'aecho=0.8:0.7:90|210|380:0.26|0.17|0.10')                 # v4: natural pitch, only a deeper echo


def speech_bounds(f, a, b):
    """last speech before a and first speech after b in the SOURCE file (any speech, used or not)"""
    x = src(f, None).mean(1)
    hop = SR // 100
    k = len(x) // hop
    lv = 20 * np.log10(np.sqrt((x[:k * hop].reshape(k, hop) ** 2).mean(1)) + 1e-9)
    on = lv > -42
    ia, ib = int(a * 100) - 5, int(b * 100) + 5
    while ia > 0 and on[ia - 1]:          # walk back over the syllable that is already sounding at a
        ia -= 1
    ia -= 8                               # and need a real gap of silence before it
    before = np.flatnonzero(on[:max(0, ia)])
    after = np.flatnonzero(on[ib:])
    return (before[-1] / 100 if len(before) else None), ((ib + after[0]) / 100 if len(after) else None)


def window(f, a, b):
    """cut a line from the middle of the silence before it to the middle of the silence after it: nothing gets clipped,
    nothing from a neighbouring (or unused) phrase leaks in"""
    prev_end, next_start = speech_bounds(f, a, b)
    dur = len(src(f, DRY)) / SR
    c0 = max(0.0, a - 0.3) if prev_end is None else max(a - 0.3, (prev_end + a) / 2)
    c1 = min(dur, b + 0.7) if next_start is None else min(b + 0.7, (b + next_start) / 2)
    return c0, c1


vsegs = []
for f, a, b, t, text, voice in LINES:
    if voice == 'whisper':
        sl = slice(int(a * SR), int(b * SR))
        w1, w2 = src(f, WHISP_LOW)[sl], src(f, 'highpass=f=150')[sl]
        m_ = min(len(w1), len(w2))
        wl = fade(w1[:m_], .01, .05) * 0.9 + fade(w2[:m_], .01, .05) * 0.55
        wet = fftconvolve(wl, reverb_ir(2.0, 31), axes=0)
        wl = np.pad(wl, ((0, len(wet) - len(wl)), (0, 0))) + wet.astype(np.float32) * 0.5
        place(vo, wl, t, 1.0)
        rev = src(f, 'areverse,aecho=0.8:0.9:120|260|420:0.5|0.35|0.25,areverse')
        r0 = max(0.0, a - 0.45)
        place(vo, fade(rev[int(r0 * SR):int(b * SR)], .2, .05), t - (a - r0), 0.45)
        vsegs.append((t, t + b - a, voice))
        continue
    c0, c1 = window(f, a, b)
    chain = {'narr': NARR, 'intro': NARR, 'final': DRY, 'reader': READER, 'nikto': NIKTO}[voice]
    seg = src(f, chain)[int(c0 * SR):int(c1 * SR)]
    if voice in ('final', 'intro'):
        # the source files stop dead on the last word on «книгой?» - let the last word ring out in a dark room instead
        seg = fade(seg, .02, .06)
        wet = fftconvolve(seg, reverb_ir(2.4, 21), axes=0)
        seg = np.pad(seg, ((0, len(wet) - len(seg)), (0, 0))) + wet.astype(np.float32) * 0.55
    g = {'final': 0.95}.get(voice, 1.0)
    place(vo, fade(seg, .02, .08), t - (a - c0), g)
    if voice == 'intro':
        # the intro file starts right on «Двести»: a breath of reversed room swells into it so it doesn't start hard
        head = seg[:int(0.5 * SR)][::-1]
        pre_ = fftconvolve(head, reverb_ir(1.2, 23), axes=0)[::-1][:int(1.2 * SR)]
        pre_ = pre_[:len(pre_) - int(0.5 * SR)] if len(pre_) > int(0.5 * SR) else pre_
        pre_ = pre_ / (np.abs(pre_).max() + 1e-9) * 0.12 * np.linspace(0, 1, len(pre_))[:, None] ** 2
        place(vo, pre_.astype(np.float32), t - (a - c0) - len(pre_) / SR, 1.0)
    vsegs.append((t, t + b - a, voice))


# ================================================================ music bed
def sweep_lowpass(x, f_from, f_to):
    """one-pole lowpass whose cutoff glides exponentially, processed in blocks"""
    out = np.zeros_like(x)
    nb = 512
    zi = np.zeros((1, 2))
    nblk = (len(x) + nb - 1) // nb
    for k in range(nblk):
        u = k / max(1, nblk - 1)
        fc = f_from * (f_to / f_from) ** u
        a = np.exp(-2 * np.pi * fc / SR)
        seg = x[k * nb:(k + 1) * nb]
        y, zf = lfilter([1 - a], [1, -a], seg, axis=0, zi=zi * np.ones((1, seg.shape[1])))
        out[k * nb:(k + 1) * nb] = y
        zi = zf
    return out


def creak(dur, seed, f_res=(820, 1350, 2300), rate=(22, 70), gain=0.3):
    """stick-slip creak: jittered impulses at a wandering rate through wooden resonances"""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    x = np.zeros(n, np.float32)
    lfo = np.interp(np.arange(n), np.linspace(0, n, 7), r.uniform(*rate, 7))
    pos = 0.0
    while pos < n:
        x[int(pos)] = r.uniform(.4, 1.0) * (1 if r.random() < .5 else -1)
        pos += SR / lfo[int(pos)] * r.uniform(.7, 1.3)
    k = np.arange(int(.012 * SR)) / SR
    ir = sum(np.sin(2 * np.pi * f * k) * np.exp(-k * r.uniform(350, 700)) * w for f, w in zip(f_res, (1, .6, .3)))
    y = np.convolve(x, ir)[:n]
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
    y = y * env
    y = y / (np.abs(y).max() + 1e-9) * gain
    return np.stack([y, np.roll(y, int(.004 * SR)) * .85], 1).astype(np.float32)


def crushed_kick(gain=.5, f0=115, f1=36, dur=.36, bits=5, hold=7, tail=True, seed=0):
    """'pixel' kick: pitch-dropping sine, saturated, sample-held and bit-reduced, with a short creak tail"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 16)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5)
    r = np.random.default_rng(seed)
    x += r.normal(0, 1, n) * np.exp(-t * 260) * .35
    x = np.tanh(x * 2.6)
    x = np.repeat(x[::hold], hold)[:n]
    step = 2 ** bits
    x = np.round(x * step) / step
    x = x * np.minimum(1, np.linspace(0, 40, n))      # declick
    y = np.stack([x, x], 1).astype(np.float32) * gain
    if tail:
        c = creak(.32, seed + 1, rate=(30, 55), gain=gain * .25)
        y[int(.04 * SR):int(.04 * SR) + len(c)] += c[:n - int(.04 * SR)]
    return y


mara = load(f'{A}/track2.mp3')
track1 = load(f'{A}/track1.mp3')
bed = np.zeros((N, 2), np.float32)
T_SMEAR = TRANS0          # 26.0: smear starts
T_GONE = 29.7            # smeared intro fully gone

if VER in ('A', 'C'):
    seg = track1[:int((T_SMEAR - T1_FILM0) * SR)] * 0.78
    place(bed, fade(seg, .02, .0), T1_FILM0)
    tail_src = track1[int((T_SMEAR - T1_FILM0) * SR):] * 0.78
else:
    off = B_MARA_OFF
    seg = mara[int(off * SR):int((T_SMEAR + off) * SR)]
    place(bed, fade(seg, 1.2, .0), 0.0)
    tail_src = mara[int((T_SMEAR + off) * SR):]

# the creaking smear: playback slows (pitch sinks), the top closes, a long reverb tail swallows it
ns = int((T_GONE - T_SMEAR) * SR) + int(1.0 * SR)
sp = np.linspace(0, 1, ns)
speed = 1 - 0.68 * sp ** 1.3
sm = varispeed(tail_src, speed)[:ns]
sm = sweep_lowpass(sm, 9000, 260)
g = np.interp(np.arange(len(sm)) / SR, [0, 1.6, T_GONE - T_SMEAR, T_GONE - T_SMEAR + .6], [1, .85, .25, 0])[:, None]
sm = sm * g
wet = fftconvolve(sm, reverb_ir(), axes=0)[:len(sm) + int(2.5 * SR)]
dry = np.pad(sm, ((0, len(wet) - len(sm)), (0, 0)))
place(bed, (dry * .8 + wet * .55).astype(np.float32), T_SMEAR)

# the creak itself: an old door / spine / floorboard, rising through the smear
place(bed, creak(3.6, 41, gain=.20), T_SMEAR + .5)
place(bed, creak(2.2, 42, f_res=(520, 980, 1700), rate=(12, 30), gain=.26), T_SMEAR + 2.0)

if VER == 'C':
    # «аллея кошмаров» only: after the smear the track drops back in on its own hit exactly at 0:30
    T1_BACK = sync.T1_BACK               # re-enter on a downbeat exactly at 0:30
    rest1 = track1[int(T1_BACK * SR):] * 0.78
    lvl = np.interp(np.arange(len(rest1)) / SR + ACT2,
                    [30, 31, 99.0, 100.5, 117.5, 119.0, 120.5, 126, 130], [.80, .62, .62, .62, .80, .80, .22, .12, 0])[:, None]
    place(bed, (rest1 * lvl).astype(np.float32), ACT2)
    rv_ = fftconvolve(rest1[:int(2.0 * SR)][::-1], reverb_ir(2.0, 3), axes=0)[::-1]
    suck = rv_[:int(2.0 * SR)]
    suck = suck / (np.abs(suck).max() + 1e-9) * .35 * np.linspace(0, 1, len(suck))[:, None] ** 2
    place(bed, suck.astype(np.float32), ACT2 - 2.0)

# Mara breathes in under the smear: its quiet pre-roll rises, a reversed-reverb suck lands on the 0:30 boom
if VER == 'C':
    mara = mara * 0.0                     # version C: no Mara at all
pre = mara[int((MARA_BOOM - (ACT2 - T_SMEAR)) * SR):int(MARA_BOOM * SR)]
pre = pre * np.linspace(0, 1, len(pre))[:, None] ** 2
place(bed, pre, T_SMEAR)
boom = mara[int(MARA_BOOM * SR):int((MARA_BOOM + 2.5) * SR)]
rv_ = fftconvolve(boom[::-1], reverb_ir(2.0, 3), axes=0)[::-1]
suck = rv_[:int(2.0 * SR)]                      # the reversed tail only, the boom itself comes from Mara
suck = suck / (np.abs(suck).max() + 1e-9) * .35 * np.linspace(0, 1, len(suck))[:, None] ** 2
place(bed, suck.astype(np.float32), ACT2 - len(suck) / SR)
rest = mara[int(MARA_BOOM * SR):]
place(bed, rest, ACT2)

np.save(f'/home/claude/voynich/edit/bed_{VER}.npy', bed[::3].mean(1).astype(np.float32))   # music stem (16 kHz) -> beat-synced motion
# ---- ducking under voice + muffled underwater shot + held breaths
from scipy.ndimage import maximum_filter1d
env = box(np.abs(vo).max(1), int(0.05 * SR))
env = maximum_filter1d(env, size=int(0.5 * SR), origin=-int(0.1 * SR))   # window [t-0.35, t+0.15]
env = box(env, int(0.12 * SR))
duck = 1.0 - 0.72 * np.clip(env / (env.max() * 0.15 + 1e-6), 0, 1)
bed_d = bed * duck[:, None]
m = np.zeros(N, np.float32)
m[int(UNDERWATER[0] * SR):int(UNDERWATER[1] * SR)] = 1
m = box(m, int(.2 * SR))
sos = butter(2, 420, 'low', fs=SR, output='sos')
uw = sosfilt(sos, bed, axis=0).astype(np.float32)
bed_d = bed_d * (1 - m[:, None]) + uw * duck[:, None] * m[:, None] * 1.2
bed_d[int(CARD * SR):int((CARD + 0.2) * SR)] *= 0.05                    # «НЕ ЧИТАЙ» card: music gone for 6 frames
gh = np.ones(N, np.float32); gh[int(PAUSE[0] * SR):int(PAUSE[1] * SR)] = 0.04   # held breath before «ты слишком долго смотришь»
gh = box(gh, int(.06 * SR))
bed_d *= gh[:, None]
import json as _json
from scipy.signal import butter as _bt, sosfilt as _sf
_sp = _bt(4, [300, 3400], 'band', fs=SR, output='sos')
vo_sp = _sf(_sp, vo.mean(1)).astype(np.float32)
_W = _json.load(open('/home/claude/voynich/edit/words.json'))
for _pass in range(3):          # every single word at least +8 dB over the music in the speech band
    bed_sp = _sf(_sp, bed_d.mean(1)).astype(np.float32)
    gmap = np.ones(N, np.float32)
    for w in _W:
        a, b = w['t0'], w['t1'] + 0.08
        tgt = 4.0 if w['voice'] == 'whisper' else 8.0
        i0, i1 = int(a * SR), int(b * SR)
        rv = np.sqrt((vo_sp[i0:i1] ** 2).mean()) * 1.5 + 1e-9
        rb = np.sqrt((bed_sp[i0:i1] ** 2).mean()) * 0.9 + 1e-9
        cur = 20 * np.log10(rv / rb)
        if cur < tgt:
            sl = slice(int((a - .12) * SR), int((b + .2) * SR))
            gmap[sl] = np.minimum(gmap[sl], 10 ** ((cur - tgt - .5) / 20))
    gmap = maximum_filter1d(-gmap, size=int(0.25 * SR))       # hold the dips, then smooth: no pumping inside a phrase
    gmap = box(-gmap, int(.2 * SR))
    bed_d *= gmap[:, None]

# ================================================================ sound design
fx = np.zeros((N, 2), np.float32)
def speaking(t):
    return any(w['t0'] - 0.1 <= t <= w['t1'] + 0.15 for w in _W)
cuts = sorted(set(round(s.t0, 4) for s in shots if s.t0 > 0.5))
_EV = sync.beat_events()
def music_at(t):
    """how hard the music really hits at t (0 = nothing there / not on the grid)"""
    best = None
    for g_, s_, h_ in _EV:
        if abs(g_ - t) < 0.025 and (best is None or s_ > best):
            best = s_
    return best or 0.0
for c in cuts:
    if 26.0 < c < 30.6 or abs(c - sync.DROP) < 0.1:
        continue                       # the creak has its knocks, the drop has its own kick
    m_ = music_at(c)
    if m_ < 0.45:
        continue                       # no kick where the music does not hit: every hit you hear is a real one
    g = 0.42 if VER == 'B' else 0.26   # the trailer track already has drums
    if speaking(c):
        g *= 0.4
    place(fx, crushed_kick(g, seed=int(c * 10)), c)
# every transition has its own sound, its accent exactly on the cut (= the beat)
try:
    import importlib, trans_sfx
    importlib.reload(trans_sfx)
    for s_ in shots:
        if not s_.trans or 26.0 < s_.t0 < 30.5:
            continue
        kind, n, pre = s_.trans
        y = trans_sfx.sfx(kind, n / 30, pre / 30, seed=int(s_.t0 * 7))
        if music_at(s_.t0) < 0.45:
            y = y * 0.4                    # off the music: a soft texture, not a hit
        place(fx, y.astype(np.float32), s_.t0 - pre / 30, 0.35 if speaking(s_.t0) else 0.8)
    print('transition sfx: on')
except Exception as e_:
    print('transition sfx: off', e_)
def knock(gain=.6, seed=0, f=(190, 420, 960)):
    """heavy knock on old wood: body thump + resonant knock, slightly crushed, in a dark room"""
    r = np.random.default_rng(seed)
    n = int(.5 * SR); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 28) * 1.2
    for fr, w, d in zip(f, (1, .6, .35), (38, 55, 90)):
        x += np.sin(2 * np.pi * fr * t + r.uniform(0, 6)) * np.exp(-t * d) * w
    x += r.normal(0, 1, n) * np.exp(-t * 400) * .8
    x = np.tanh(x * 1.8)
    x = np.repeat(x[::4], 4)[:n]
    x = np.round(x * 32) / 32
    y = np.stack([x, x * .92], 1).astype(np.float32)
    wet = fftconvolve(y, reverb_ir(1.6, seed + 5), axes=0)[:n + int(1.2 * SR)]
    return (np.pad(y, ((0, len(wet) - n), (0, 0))) + wet * .7) * gain

for i, kt in enumerate(KNOCKS):
    place(fx, knock(.75 if kt < 29 else .95, seed=11 + i), kt)
place(fx, creak(1.4, 51, f_res=(300, 640, 1150), rate=(6, 18), gain=.30), 28.75)     # the door gives way under «…не смог»

# the book: assembled thud, cover creak, pages
place(fx, crushed_kick(.5, f0=90, f1=30, dur=.6, seed=1), BK.ASM1)                # the pixels land
place(fx, creak(1.2, 7, f_res=(380, 760, 1300), rate=(10, 26), gain=.22), BK.OPEN0 - 0.15)
place(fx, crushed_kick(.38, seed=2), BK.OPEN0)                                     # the cover opens
import book as BK
def swish(d=0.5, g=0.25):
    n = int(d * SR); x = rng.normal(0, 1, n).astype(np.float32)
    x = np.convolve(x, np.ones(40) / 40, 'same')
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return np.stack([x * e * g, x * e * g * .8], 1)
for i, f in enumerate(BK.FLIPS):
    place(fx, swish(.45, .3), f)
    place(fx, creak(.35, 60 + i, gain=.10), f + .05)
    if f >= 8.3:
        place(fx, crushed_kick(.3 + .03 * i, seed=70 + i), f)
place(fx, crushed_kick(.55, f0=140, f1=30, dur=.8, bits=4, seed=99), sync.DROP)      # the drop into act I
# climax: an accelerating roll of crushed kicks into the collapse
S36, S37 = shot_named('s36').t0, shot_named('s37').t0
roll = []
for g_ in sync.grid(112.5, S37 + 0.01):          # on the beat: halves before the collapse, quarters during it
    sub = 2 if g_ < S36 else 4
    roll += [g_ + j * sync.PULSE / sub for j in range(sub)]
roll = [c for c in roll if c < S37 - 0.25]
for c in roll:
    place(fx, crushed_kick(.16 + (c - roll[0]) / (roll[-1] - roll[0] + 1e-9) * .22, hold=9, bits=4, tail=False, seed=int(c * 100)), c)
# sub drone 34 Hz: carries the silent start, swells through the creak, rises to the climax, gone for the clean end
drone = 0.5 * np.sin(2 * np.pi * 34 * tt) + 0.25 * np.sin(2 * np.pi * 51 * tt + 1)
amp = np.interp(tt, [0, 1.0, 5.3, 10.3, 10.4, 26, 29.9, 30.0, 70, PAUSE[0] - .1, PAUSE[0], PAUSE[1], 114, 118.5, 119, 130],
                [0, .07, .08, .06, .02, .02, .16, .05, .10, .12, .04, .14, .24, .26, 0, 0])
fx += (drone * amp)[:, None].astype(np.float32)
# whispers of nonsense (act II bed): the narrator reversed, pitched, filtered, panned
wh = load(f'{A}/vo_narrator.mp3', f'aresample={SR},areverse,asetrate={SR}*0.82,aresample={SR},highpass=f=600,lowpass=f=3500,aecho=0.8:0.8:90|200:0.4|0.3')
L = min(len(wh), int(54 * SR))
pan = 0.5 + 0.5 * np.sin(np.linspace(0, 9, L))
wseg = wh[:L].mean(1)
whis = np.stack([wseg * pan, wseg * (1 - pan)], 1) * np.interp(np.arange(L) / SR, [0, 6, 50, 54], [0, .16, .2, 0])[:, None]
place(fx, whis.astype(np.float32), 31.0)
# intro whisper under the book (very low)
place(fx, whis[int(10 * SR):int(16 * SR)] * np.linspace(0, 1, int(6 * SR))[:, None] * .5, 3.0)
rev = load(f'{A}/vo_whisper.mp3', 'areverse,aecho=0.8:0.9:120|260|420:0.5|0.35|0.25,areverse')
place(fx, fade(rev[:int(2.5 * SR)]), 40.0, 0.06)
place(fx, swish(), shot_named('s5').t0 - 0.25)                  # the swish peaks on the cut
place(fx, swish(), shot_named('s29').t0 - 0.25)
place(fx, swish(1.0, .10), shot_named('s26').t0 - 0.55)     # an inhale against the glass, just before «пахнут»

np.save(f'/home/claude/voynich/edit/vo_{VER}.npy', vo[::3].mean(1).astype(np.float32))   # voice stem at 16 kHz for the sync check
_bsp = _sf(_sp, bed_d.mean(1)).astype(np.float32)
_rat = []
for w in _W:
    i0, i1 = int(w['t0'] * SR), int((w['t1'] + 0.08) * SR)
    rv = np.sqrt((vo_sp[i0:i1] ** 2).mean()) * 1.5 + 1e-9
    rb = np.sqrt((_bsp[i0:i1] ** 2).mean()) * 0.9 + 1e-9
    _rat.append((round(20 * np.log10(rv / rb), 1), w['w'], round(w['t0'], 2)))
_rat.sort()
print('speech-band voice/music per word: worst', _rat[:6], 'median', _rat[len(_rat) // 2][0])
mix = bed_d * 0.9 + vo * 1.5 + fx
for a, b, v in vsegs:
    sl = slice(int(a * SR), int(b * SR))
    rv = np.sqrt((vo[sl] ** 2).mean()) * 1.5; rb = np.sqrt((bed_d[sl] ** 2).mean()) * .9
    print(f'{a:6.1f} {v:7s} voice/music dB {20 * np.log10(rv / (rb + 1e-9)):5.1f}')
mix[int(129.4 * SR):] *= np.linspace(1, 0, N - int(129.4 * SR))[:, None]
pk = np.abs(mix).max()
mix = np.tanh(mix / pk * 1.4) * 0.92
out = f'/home/claude/voynich/edit/mix_{VER}.wav'
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                      '-c:a', 'pcm_s16le', out], stdin=subprocess.PIPE)
p.stdin.write(mix.astype(np.float32).tobytes()); p.stdin.close(); p.wait()
print('ok', out)
