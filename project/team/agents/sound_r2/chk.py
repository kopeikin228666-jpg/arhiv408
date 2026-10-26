"""round-2 sound check of an episode wav (default ep1e_p3/ep.wav), SEG timings from make_ep1e (imported, no render).
 - reel words: the reel narrator stem (edit/vo_B.npy, 16k -> 48k resample_poly 3,1, placed per reel segment exactly like
   render_audio) is least-squares fitted to the master IN THE WORD'S OWN WINDOW (300-4000 Hz); SNR = fitted voice energy /
   residual energy over the voiced 10 ms frames (stem > its window peak - 12 dB).  Also the director's snr3-style
   100 ms-window numbers (fit over -0.3..+0.4 s) for the director's listed windows.
 - 408 words: exact voice from ep_level.LAST (needs LAST npz written by run_chk), same band.
usage: cd /home/claude/voynich/edit && PART=3 python3 chk.py [ep.wav] [voice_npz]"""
import os, sys, json, subprocess, numpy as np
os.environ.setdefault('PART', '3')
wav = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/voynich/team/host/ep1e_p3/ep.wav'
vnpz = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = [sys.argv[0]]
import io, contextlib
sys.path.insert(0, '/home/claude/voynich/team/host'); sys.path.insert(0, '/home/claude/voynich/team/agents/sound')
with contextlib.redirect_stdout(io.StringIO()):
    import make_ep1e as M
from scipy.signal import butter, sosfiltfilt, resample_poly
SR = 48000; F = 480
x = np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', wav, '-f', 'f32le', '-']), np.float32).reshape(-1, 2)
xm = x.mean(1).astype(np.float64); N = len(xm)
stem = resample_poly(np.load('/home/claude/voynich/edit/vo_B.npy').astype(np.float64), 3, 1)
ref = np.zeros(N + SR)
for sg in M.SEG:
    if sg['kind'] == 'reel':
        a, b = sg['src']; s_ = stem[int(round(a * SR)):int(round(b * SR))].copy()
        n1, n2 = int(0.015 * SR), int(0.04 * SR)
        s_[:n1] *= np.linspace(0, 1, n1); s_[-n2:] *= np.linspace(1, 0, n2)
        i = int(round(sg['t0'] * SR)); ref[i:i + len(s_)] += s_
ref = ref[:N]
sos = butter(4, [300, 4000], 'bandpass', fs=SR, output='sos')
xb = sosfiltfilt(sos, xm); rb = sosfiltfilt(sos, ref)
v408 = None
if vnpz and os.path.exists(vnpz):
    v408 = np.load(vnpz)['voice_out'][:N].mean(1).astype(np.float64)
    vb408 = sosfiltfilt(sos, v408)


def fdb(z):
    m = len(z) // F
    return 10 * np.log10((z[:m * F].reshape(m, F) ** 2).mean(1) + 1e-14)


def win_snr(t0, t1, fit=None):
    a, b = int(t0 * SR), int(t1 * SR)
    fa, fb = (a, b) if fit is None else (int(fit[0] * SR), int(fit[1] * SR))
    r, e = rb[fa:fb], xb[fa:fb]
    g = np.dot(e, r) / (np.dot(r, r) + 1e-18)
    v = g * rb[a:b]; res = xb[a:b] - v
    vf, of = fdb(v), fdb(res); m = vf > vf.max() - 12
    return 10 * np.log10((10 ** (vf[m] / 10)).sum()) - 10 * np.log10((10 ** (of[m] / 10)).sum()), g


def ep_t(ft):
    for sg in M.SEG:
        if sg['kind'] == 'reel' and sg['src'][0] <= ft < sg['src'][1]:
            return sg['t0'] + ft - sg['src'][0]
    return None


LISTED = [('Растения(onset)', 47.59, 47.90), ('не-(правильные)', 48.72, 49.00), ('в', 52.86, 53.08), ('Их', 56.41, 56.68),
          ('не(те)', 58.13, 58.41), ('видеть(tail)', 62.00, 62.39)]
print('LISTED reel windows (film -> ep): word-window fit | snr3-style worst 100 ms (fit -0.3..+0.4 s)')
worst_listed = 99
for name, fa, fb in LISTED:
    t0 = ep_t(fa); t1 = t0 + fb - fa
    s, g = win_snr(t0, t1)
    s3 = []
    for k in np.arange(t0, t1 - 0.05, 0.05):
        sk, _ = win_snr(k, k + 0.1, (k - 0.3, k + 0.4))
        a_, b_ = int(k * SR), int((k + 0.1) * SR)
        if np.mean(rb[a_:b_] ** 2) > 0.05 * np.mean(rb[int(t0 * SR) - SR // 2:int(t1 * SR) + SR // 2] ** 2):
            s3.append(sk)
    worst_listed = min(worst_listed, s)
    print(f'  {name:16s} film {fa:6.2f}-{fb:6.2f}  ep {t0:6.2f}  {s:5.1f} dB  | snr3 min {min(s3) if s3 else float("nan"):5.1f}')
W = json.load(open('/home/claude/voynich/edit/words.json'))
rr = []
for k, w in enumerate(W):
    t0 = ep_t(w['t0'])
    if t0 is None:
        continue
    end = w['t1'] + 0.08
    if k + 1 < len(W) and w['t0'] < W[k + 1]['t0'] < w['t1']:
        end = max(w['t0'] + 0.1, W[k + 1]['t0'] + 0.08)
    sgx = [sg for sg in M.SEG if sg['kind'] == 'reel' and sg['src'][0] <= w['t0'] < sg['src'][1]][0]
    t1 = min(t0 + end - w['t0'], sgx['t1'])
    if t1 - t0 < 0.05:
        continue
    s, g = win_snr(t0, t1)
    rr.append((s, w['w'], w['t0'], t0))
rr.sort()
print('reel words: worst', ', '.join(f'{w}@{ft:.2f}(ep {t:.2f}):{s:.1f}' for s, w, ft, t in rr[:6]),
      ' median %.1f' % rr[len(rr) // 2][0], ' n', len(rr))
print('   < 8 dB:', [(w, round(ft, 2), round(s, 1)) for s, w, ft, t in rr if s < 8])
if v408 is not None:
    r4 = []
    ws = M.ALLW
    for k, w in enumerate(ws):
        end = w['t1'] + 0.08
        if k + 1 < len(ws) and w['t0'] < ws[k + 1]['t0'] < w['t1']:
            end = max(w['t0'] + 0.1, ws[k + 1]['t0'] + 0.08)
        a, b = int(w['t0'] * SR), int(end * SR)
        vf = fdb(vb408[a:b]); of = fdb(xb[a:b] - vb408[a:b]); m = vf > vf.max() - 12
        r4.append((10 * np.log10((10 ** (vf[m] / 10)).sum()) - 10 * np.log10((10 ** (of[m] / 10)).sum()), w['w'], w['t0']))
    r4.sort()
    print('408 words: worst', ', '.join(f'{w}@{t:.2f}:{s:.1f}' for s, w, t in r4[:5]), ' median %.1f' % r4[len(r4) // 2][0])
import meas
t, Mm, Ss, I, LRA, TP = meas.ebur(wav)
print(f'integrated {I:.1f} LUFS  LRA {LRA:.1f} LU  true peak {TP:.1f} dBTP  | short-term (t>=3) {Ss[t >= 3].min():.1f}..{Ss[t >= 3].max():.1f}  momentary max {Mm.max():.1f}')
for sg in M.SEG:
    s = (t > sg['t0'] + 0.2) & (t <= sg['t1'])
    if s.any():
        print(f"  {sg['kind']:6s} {sg['t0']:5.2f}-{sg['t1']:5.2f}  M max {Mm[s].max():6.1f}  S {Ss[s].min():6.1f}..{Ss[s].max():6.1f}")
q = lambda z: 20 * np.log10(np.sqrt((z ** 2).mean()) + 1e-9)
print('seam: last 250 ms %.1f dB, last 20 ms %.1f, first 20 ms %.1f dB, |x[-1]-x[0]| %.4f' % (
    q(x[-12000:]), q(x[-960:]), q(x[:960]), np.abs(x[-1] - x[0]).max()))
