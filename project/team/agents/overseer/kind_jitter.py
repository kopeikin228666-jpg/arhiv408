"""overseer: hood x/y shift frame by frame around a stressed word (close pose) - does the nod flip direction?"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/voynich/edit')
import host as HO
for t0 in (14.62, 9.76, 3.43):
    words = [dict(key=True, t0=t0)]
    base = None; row = []
    for f in range(int(t0 * 30) - 3, int(t0 * 30) + 6):
        t = f / 30; st = HO.stress_at(words, t)
        s = HO.sprite_img(t, 0.0, False, 0, 3, 'calm', 'close', sway=False, stress=st)
        m = s[..., 3] > 0
        top = m[:60]; ys, xs = np.nonzero(top)
        row.append(f'{t - t0:+.3f} st={st:.2f} kind={HO._stress_kind(t, st, True)} hoodX={xs.mean():.1f}')
    print(t0); print('\n'.join(row))
