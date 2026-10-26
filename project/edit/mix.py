"""Audio for the 2:00 reel: music bed + processed voices + procedural sound design."""
import subprocess, json, numpy as np

SR = 48000
DUR = 120.0
A = '/home/claude/voynich/audio'
N = int(DUR * SR)


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


def place(buf, x, t, gain=1.0):
    i = int(t * SR)
    j = min(N, i + len(x))
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


# ---------------------------------------------------------------- voices
NARR = (f'asetrate={SR}*0.89,aresample={SR},atempo=1.1236,'
        'highpass=f=70,lowpass=f=9000,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,'
        'acrusher=bits=12:mix=0.18:mode=log,aecho=0.8:0.6:45|90:0.18|0.10')
READER = (f'asetrate={SR}*0.80,aresample={SR},atempo=1.25,'
          'acrusher=bits=8:mix=0.45:mode=log,aecho=0.8:0.7:60|130:0.3|0.2,acompressor=threshold=-20dB:ratio=5:makeup=4')
WHISP_LOW = f'asetrate={SR}*0.75,aresample={SR},atempo=1.333,lowpass=f=5000'
DRY = 'highpass=f=70,acompressor=threshold=-24dB:ratio=3:makeup=5'

NSEG = [  # (src start, src end, film start)
    (0.00, 1.15, 3.60), (1.41, 3.81, 5.55), (4.74, 8.07, 9.20), (8.51, 10.76, 12.50),
    (11.27, 12.03, 18.10), (12.17, 13.03, 20.05), (13.68, 20.54, 23.20), (21.07, 22.80, 30.40),
    (23.37, 26.67, 32.80), (27.11, 29.31, 37.60), (29.78, 31.99, 41.80), (32.58, 35.10, 46.20),
    (35.45, 38.58, 54.60), (39.18, 42.91, 59.40), (43.49, 46.39, 68.00), (46.99, 49.94, 75.60),
    (50.49, 52.77, 85.20),
]
FINAL = (53.31, 56.35, 111.20)

vo = np.zeros((N, 2), np.float32)
narr = load(f'{A}/vo_narrator.mp3', NARR)
narr_dry = load(f'{A}/vo_narrator.mp3', DRY)
for a, b, t in NSEG:
    seg = narr[int(a * SR):int((b + 0.25) * SR)]
    place(vo, fade(seg, .01, .2), t, 1.0)
a, b, t = FINAL
place(vo, fade(narr_dry[int(a * SR):int((b + .3) * SR)], .01, .25), t, 0.95)

reader = load(f'{A}/vo_reader.mp3', READER)[: int(2.75 * SR)]
place(vo, fade(reader, .01, .15), 95.0, 1.0)

# whisperer: low copy + original + reversed-reverb tail (echo before the sound)
w_raw = load(f'{A}/vo_whisper.mp3', 'highpass=f=150')
w_low = load(f'{A}/vo_whisper.mp3', WHISP_LOW)
rev = load(f'{A}/vo_whisper.mp3', 'areverse,aecho=0.8:0.9:120|260|420:0.5|0.35|0.25,areverse')
for seg_a, seg_b, t in [(0.0, 1.12, 100.6), (1.76, 2.51, 102.4)]:
    sl = slice(int(seg_a * SR), int(seg_b * SR))
    place(vo, fade(w_low[sl]), t, 0.9)
    place(vo, fade(w_raw[sl]), t, 0.55)
    pre = rev[max(0, int((seg_a - 0.45) * SR)):int(seg_b * SR)]
    place(vo, fade(pre, .2, .05), t - 0.45, 0.45)

# ---------------------------------------------------------------- music bed with ducking
bed = load(f'{A}/music_bed.wav')[:N]
bed = np.pad(bed, ((0, N - len(bed)), (0, 0)))
env = np.abs(vo).max(1)
k = int(0.25 * SR)
env = np.convolve(env, np.ones(k) / k, 'same')
duck = 1.0 - 0.72 * np.clip(env / (env.max() * 0.15 + 1e-6), 0, 1)
bed *= duck[:, None]
# underwater shot 50.0-54.3: muffle the bed
uw = load(f'{A}/music_bed.wav', 'lowpass=f=420')[:N]
uw = np.pad(uw, ((0, N - len(uw)), (0, 0)))
m = np.zeros(N, np.float32)
i0, i1 = int(50.0 * SR), int(54.3 * SR)
m[i0:i1] = 1
m = np.convolve(m, np.ones(int(.2 * SR)) / int(.2 * SR), 'same')
bed = bed * (1 - m[:, None]) + uw * duck[:, None] * m[:, None] * 1.2
# music break at "НЕ ЧИТАЙ" (8.05, 6 frames) and the pause 89.0-90.5
for t0, t1 in [(8.05, 8.25)]:
    bed[int(t0 * SR):int(t1 * SR)] *= 0.05
# the held breath before "ты слишком долго смотришь": music drops out 89.0-90.5
g = np.ones(N, np.float32)
g[int(89.0 * SR):int(90.5 * SR)] = 0.04
g = np.convolve(g, np.ones(int(.06 * SR)) / int(.06 * SR), 'same')
bed *= g[:, None]
# guarantee intelligibility: per voice segment, pull the bed down until voice sits >= target dB above it
segs = [(t, t + (b - a)) for a, b, t in NSEG] + [(95.0, 97.7, ), (100.6, 101.75), (102.4, 103.2)]
targets = [3.0] * len(NSEG) + [3.0, 0.0, 0.0]
gmap = np.ones(N, np.float32)
for (a, b), tgt in zip(segs, targets):
    sl = slice(int(a * SR), int(b * SR))
    rv = np.sqrt((vo[sl] ** 2).mean()) * 1.5
    rb = np.sqrt((bed[sl] ** 2).mean()) * 0.9 + 1e-9
    cur = 20 * np.log10(rv / rb)
    if cur < tgt:
        gmap[sl] = np.minimum(gmap[sl], 10 ** ((cur - tgt) / 20))
gmap = np.convolve(gmap, np.ones(int(.15 * SR)) / int(.15 * SR), 'same')
bed *= gmap[:, None]

# ---------------------------------------------------------------- sound design
rng = np.random.default_rng(5)
fx = np.zeros((N, 2), np.float32)
tt = np.arange(N) / SR


def burst(dur=0.06, crush=6, gain=0.35, hp=True):
    n = int(dur * SR)
    x = rng.normal(0, 1, n).astype(np.float32)
    step = 2 ** crush
    x = np.round(x * step) / step
    hold = int(rng.integers(4, 24))
    x = np.repeat(x[::hold], hold)[:n]
    e = np.exp(-np.linspace(0, 5, n))
    x = x * e * gain
    return np.stack([x, np.roll(x, 7)], 1)


cuts = [117.5, 119.75, 1.0, 2.5, 3.38, 5.08, 6.26, 8.05, 9.12, 10.9, 12.0, 14.32, 16.34, 18.0, 19.5, 28.57, 32.3, 34.8, 45.73,
        58.2, 71.44, 75.0, 79.7, 84.0, 90.5, 97.5, 99.0, 105.7]
for c in cuts:
    place(fx, burst(rng.uniform(.04, .09), int(rng.integers(3, 7)), .28), c)
for c in np.arange(5.08, 6.26, 4 / 30):      # strobe ticks
    place(fx, burst(.02, 4, .18), c)
for c in np.arange(14.32, 16.3, 0.11):       # whip-pan crackle
    if rng.random() < .6:
        place(fx, burst(.03, 5, .14), c)
for c in np.arange(104.2, 108.6, 0.07):      # climax digital storm
    place(fx, burst(.04, int(rng.integers(2, 6)), .12 + (c - 104.2) * .04), c)

# relay click at the card
click = np.zeros((int(.03 * SR), 2), np.float32); click[:30] = 0.6; click[30:60] = -0.4
place(fx, click, 8.05); place(fx, click, 8.40)

# sub drone 34 Hz, rising from 20s to 108s, gone for the clean ending
drone = 0.5 * np.sin(2 * np.pi * 34 * tt) + 0.25 * np.sin(2 * np.pi * 51 * tt + 1)
amp = np.interp(tt, [0, 19.9, 20.0, 60, 88.9, 89.0, 90.5, 104, 108.5, 109, 120], [0, 0, .05, .10, .12, .04, .14, .24, .26, 0, 0])
fx += (drone * amp)[:, None].astype(np.float32)

# whispers of nonsense: the narrator reversed, pitched, filtered, panned (act II bed)
wh = load(f'{A}/vo_narrator.mp3', f'areverse,asetrate={SR}*0.82,aresample={SR},highpass=f=600,lowpass=f=3500,aecho=0.8:0.8:90|200:0.4|0.3')
L = min(len(wh), int((75 - 21) * SR))
pan = 0.5 + 0.5 * np.sin(np.linspace(0, 9, L))
wseg = wh[:L].mean(1)
whis = np.stack([wseg * pan, wseg * (1 - pan)], 1) * np.interp(np.arange(L) / SR, [0, 6, 50, 54], [0, .16, .2, 0])[:, None]
place(fx, whis.astype(np.float32), 21.0)
# a short pre-echo of "она уже внутри" hidden at 0:30 (barely audible)
place(fx, fade(rev[:int(2.5 * SR)]), 30.0, 0.06)

# page turns: filtered noise swish, slowed
def swish(d=0.5, g=0.25):
    n = int(d * SR); x = rng.normal(0, 1, n).astype(np.float32)
    x = np.convolve(x, np.ones(40) / 40, 'same')
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return np.stack([x * e * g, x * e * g * .8], 1)
for c in [5.08, 75.2, 76.4, 77.2]:
    place(fx, swish(), c)
# inhale at the glass (62.87)
inh = swish(1.6, .12)
place(fx, inh, 63.4)

mix = bed * 0.9 + vo * 1.5 + fx
for a, b in [(3.6, 7.9), (23.2, 30), (59.4, 63), (85.2, 87.5), (95, 97.7), (100.6, 103.2), (111.2, 114.3)]:
    sl = slice(int(a * SR), int(b * SR))
    rv = np.sqrt((vo[sl] ** 2).mean()) * 1.5; rb = np.sqrt((bed[sl] ** 2).mean()) * .9
    print(f'{a:6.1f} voice/music dB {20 * np.log10(rv / (rb + 1e-9)):5.1f}')
# final silence after 119.4, soft limiter
mix[int(119.6 * SR):] *= np.linspace(1, 0, N - int(119.6 * SR))[:, None]
pk = np.abs(mix).max()
mix = np.tanh(mix / pk * 1.4) * 0.92
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                      '-c:a', 'pcm_s16le', '/home/claude/voynich/edit/mix.wav'], stdin=subprocess.PIPE)
p.stdin.write(mix.astype(np.float32).tobytes()); p.stdin.close(); p.wait()
print('ok', mix.shape)
