"""honest per-word speech-band clearance of the reel narrator from mix2 stems (vo, other): energy over voiced frames.
usage: python3 reel_words.py stems.npz [film_a film_b ...pairs]  (default: the film spans used by the episode)"""
import sys, json, numpy as np
from scipy.signal import butter, sosfiltfilt
SR = 48000; F = 480
d = np.load(sys.argv[1]); vo, ot = d['vo'], d['other']
sos = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
def fdb(x):
    m = len(x) // F
    return 10 * np.log10((x[:m * F].astype(np.float64).reshape(m, F) ** 2).mean(1) + 1e-14)
vf = fdb(sosfiltfilt(sos, vo.mean(1))); of = fdb(sosfiltfilt(sos, ot.mean(1)))
W = json.load(open('/home/claude/voynich/edit/words.json'))
spans = [(8.9, 9.85), (36.55, 38.55), (22.55, 25.15), (27.40, 29.55), (42.55, 46.6), (47.467, 50.05), (51.55, 54.25),
         (56.15, 62.55), (110.30, 113.45), (121.05, 124.40)]
if len(sys.argv) > 2:
    v = list(map(float, sys.argv[2:])); spans = list(zip(v[::2], v[1::2]))
res = []
for k, w in enumerate(W):
    end = w['t1'] + 0.08
    if k + 1 < len(W) and w['t0'] < W[k + 1]['t0'] < w['t1']:
        end = max(w['t0'] + 0.1, W[k + 1]['t0'] + 0.08)
    i0 = int(w['t0'] * 100); i1 = max(i0 + 3, int(end * 100))
    v, o = vf[i0:i1], of[i0:i1]; m = v > v.max() - 12
    snr = 10 * np.log10((10 ** (v[m] / 10)).mean()) - 10 * np.log10((10 ** (o[m] / 10)).mean())
    used = any(a <= w['t0'] < b for a, b in spans)
    res.append((snr, w['w'], w['t0'], used))
for tag, sel in (('ALL', [r for r in res]), ('USED IN EP', [r for r in res if r[3]])):
    s = sorted(sel)
    print(tag, 'n', len(s), 'worst:', ', '.join(f'{w}@{t:.2f}:{x:.1f}' for x, w, t, u in s[:8]), ' median %.1f' % s[len(s) // 2][0])
    print('   < 8 dB:', [(w, round(t, 2), round(x, 1)) for x, w, t, u in s if x < 8])
for x, w, t, u in res:
    if 51.7 < t < 54.1:
        print(f'   {w:10s} {t:6.2f} {x:5.1f}')
