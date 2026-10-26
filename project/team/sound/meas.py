import json, numpy as np
from scipy.signal import butter, sosfilt
SR = 48000
D = '/home/claude/voynich/team/sound/'
W = json.load(open('/home/claude/voynich/edit/words.json'))
_sp = butter(4, [300, 3400], 'band', fs=SR, output='sos')
def load():
    return [np.load(D + f'st_{k}.npy') for k in ('bed', 'vo', 'fx')]
def words(vo, other):
    """per-word speech-band ratio, voice*1.5 vs other (already scaled as in the mix)"""
    v = sosfilt(_sp, vo.mean(1)); o = sosfilt(_sp, other.mean(1))
    r = []
    for w in W:
        i0, i1 = int(w['t0'] * SR), int((w['t1'] + .08) * SR)
        rv = np.sqrt((v[i0:i1] ** 2).mean()) * 1.5 + 1e-9
        ro = np.sqrt((o[i0:i1] ** 2).mean()) + 1e-9
        r.append((round(float(20 * np.log10(rv / ro)), 1), w['w'], round(w['t0'], 2), w['voice']))
    return r
def limiter(mix):
    pk = np.abs(mix).max()
    out = np.tanh(mix / pk * 1.4) * .92
    return pk, out
