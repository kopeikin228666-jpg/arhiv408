"""render the episode audio (make_ep1e.render_audio) to a scratch wav and verify it EXACTLY:
408's voice is known (ep_level LAST voice_out); the reel narrator is rebuilt from mix2 stems (VOY_STEMS npz, film time)
placed like render_audio places the reel and multiplied by ep_level's total gain on the rest (LAST rest_gain).
usage: cd /home/claude/voynich/edit && PART=3 python3 run_ep.py OUT.wav [stems.npz]"""
import os, sys, json, numpy as np
os.environ.setdefault('PART', '3')
out = sys.argv[1]
stems = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/voynich/team/agents/sound_r2/stems_clear.npz'
sys.argv = [sys.argv[0]]
sys.path.insert(0, '/home/claude/voynich/team/host'); sys.path.insert(0, '/home/claude/voynich/team/agents/sound')
import make_ep1e as M
if os.environ.get('REEL_DIR'):
    M.E = os.environ['REEL_DIR']
if not os.environ.get('NO_RENDER'):
    M.render_audio(out)
import ep_level as EL
from scipy.signal import butter, sosfiltfilt
SR = 48000; F = 480
x = M.load(out)
N = len(x)
if not os.environ.get('NO_RENDER'):
    L = EL.LAST
    np.savez(out + '.npz', voice_out=L['voice_out'][:N], rest_gain=L['rest_gain'][:N])
d = np.load(out + '.npz'); v408 = d['voice_out'][:N]; rg = d['rest_gain'][:N]
S = np.load(stems); svo = S['vo'] * S['scale']
rv = np.zeros((N + SR, 2), np.float32)
for sg in M.SEG:
    if sg['kind'] == 'reel':
        a, b = sg['src']
        M.place(rv, M.fade(svo[int(round(a * SR)):int(round(b * SR))], 0.015, 0.04), sg['t0'])
rv = rv[:N] * rg[:, None]
sos = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
def fdb(z):
    m = len(z) // F
    return 10 * np.log10((z[:m * F].astype(np.float64).reshape(m, F) ** 2).mean(1) + 1e-14)
xb = sosfiltfilt(sos, x.mean(1)); vb = sosfiltfilt(sos, v408.mean(1)); rb = sosfiltfilt(sos, rv.mean(1))
VF, RF = fdb(vb), fdb(rb)


def words_snr(ws, sf, of):
    res = []
    for k, w in enumerate(ws):
        end = w['t1'] + 0.08
        if k + 1 < len(ws) and w['t0'] < ws[k + 1]['t0'] < w['t1']:
            end = max(w['t0'] + 0.1, ws[k + 1]['t0'] + 0.08)
        end = min(end, w.get('lim', 1e9))
        i0 = int(w['t0'] * 100); i1 = max(i0 + 3, int(end * 100))
        s, o = sf[i0:i1], of[i0:i1]; m = s > s.max() - 12
        res.append((10 * np.log10((10 ** (s[m] / 10)).mean()) - 10 * np.log10((10 ** (o[m] / 10)).mean()), w['w'], w['t0']))
    return res


W408 = [dict(w=w['w'], t0=w['t0'], t1=w['t1']) for w in M.ALLW]
r408 = words_snr(W408, VF, fdb(xb - vb - rb))
WJ = json.load(open('/home/claude/voynich/edit/words.json'))
WR = []
for sg in M.SEG:
    if sg['kind'] != 'reel':
        continue
    fa, fb = sg['src']
    seg_ws = [w for w in WJ if fa <= w['t0'] < fb - 0.1]
    for k, w in enumerate(WJ):
        if fa <= w['t0'] < fb - 0.1:
            nxt = WJ[k + 1] if k + 1 < len(WJ) else None
            e = dict(w=w['w'], t0=sg['t0'] + w['t0'] - fa, t1=sg['t0'] + w['t1'] - fa, lim=sg['t1'], ft=w['t0'])
            WR.append(e)
rr = words_snr(WR, RF, fdb(xb - vb - rb))
print('408 words (300-3400 Hz, exact stems):  worst', ', '.join(f'{w}@{t:.2f}:{s:.1f}' for s, w, t in sorted(r408)[:5]),
      ' median %.1f' % sorted(r408)[len(r408) // 2][0])
print('reel words (exact stems):              worst', ', '.join(f'{w}@{t:.2f}:{s:.1f}' for s, w, t in sorted(rr)[:6]),
      ' median %.1f' % sorted(rr)[len(rr) // 2][0])
print('   «в» of «Женщины в зелёной воде»:', [f'{s:.1f}' for (s, w, t), e in zip(rr, WR) if abs(e['ft'] - 52.862) < 0.01])
print('   < 8 dB  408:', [(w, round(t, 2), round(s, 1)) for s, w, t in r408 if s < 8], ' reel:', [(w, round(t, 2), round(s, 1)) for s, w, t in rr if s < 8])
import meas
t, Mm, Ss, I, LRA, TP = meas.ebur(out)
print(f'integrated {I:.1f} LUFS  LRA {LRA:.1f} LU  true peak {TP:.1f} dBTP  | short-term (t>=3) {Ss[t >= 3].min():.1f}..{Ss[t >= 3].max():.1f}'
      f'  momentary max {Mm.max():.1f}')
for sg in M.SEG:
    s = (t > sg['t0'] + 0.2) & (t <= sg['t1'])
    if s.any():
        print(f"  {sg['kind']:6s} {sg['t0']:5.2f}-{sg['t1']:5.2f}  M max {Mm[s].max():6.1f}  S {Ss[s].min():6.1f}..{Ss[s].max():6.1f}")
print('seam: last 50 ms %.1f dB, last 5 ms %.1f dB, first 5 ms %.1f dB, |x[-1]-x[0]| %.3f' % (
    20 * np.log10(np.sqrt((x[-2400:] ** 2).mean()) + 1e-9), 20 * np.log10(np.sqrt((x[-240:] ** 2).mean()) + 1e-9),
    20 * np.log10(np.sqrt((x[:240] ** 2).mean()) + 1e-9), np.abs(x[-1] - x[0]).max()))
