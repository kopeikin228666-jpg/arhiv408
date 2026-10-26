"""per-50ms band power (300-4000) in the ep master of: reel narrator (exact stem), reel bed (exact stem), remainder
(episode sfx + tanh error).  usage: PART=3 python3 decomp.py ep.wav stems.npz film_a film_b [...]"""
import sys, os, io, contextlib, subprocess, numpy as np
wav, stems = sys.argv[1], sys.argv[2]; spans = list(map(float, sys.argv[3:])); sys.argv = [sys.argv[0]]
sys.path.insert(0, '/home/claude/voynich/team/host')
with contextlib.redirect_stdout(io.StringIO()):
    import make_ep1e as M
from scipy.signal import butter, sosfiltfilt
SR = 48000
x = np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', wav, '-ac', '1', '-f', 'f32le', '-']), np.float32).astype(np.float64)
d = np.load(stems); sc = float(d['scale'])
sos = butter(4, [300, 4000], 'bandpass', fs=SR, output='sos')
for fa, fb in zip(spans[::2], spans[1::2]):
    sg = [s for s in M.SEG if s['kind'] == 'reel' and s['src'][0] <= fa < s['src'][1]][0]
    off = sg['t0'] - sg['src'][0]
    a, b = fa - 0.6, fb + 0.6
    xs = sosfiltfilt(sos, x[int((a + off) * SR):int((b + off) * SR)])
    vs = sosfiltfilt(sos, d['vo'][int(a * SR):int(b * SR)].mean(1).astype(np.float64)) * sc
    os_ = sosfiltfilt(sos, d['other'][int(a * SR):int(b * SR)].mean(1).astype(np.float64)) * sc
    # local gain of the episode on this span (fit the stems to the master)
    A = np.stack([vs, os_], 1); coef = np.linalg.lstsq(A, xs, rcond=None)[0]
    rem = xs - A @ coef
    F = 2400; db = lambda z, k: 10 * np.log10((z[k * F:(k + 1) * F] ** 2).mean() + 1e-14)
    print(f'--- film {fa}-{fb}  fit: voice x{coef[0]:.2f} bed x{coef[1]:.2f}')
    for k in range(len(xs) // F):
        t = a + k * 0.05
        if fa - 0.2 <= t <= fb + 0.15:
            print(f'  {t:6.2f} voice {db(vs * coef[0], k):6.1f}  bed {db(os_ * coef[1], k):6.1f}  rem {db(rem, k):6.1f}')
