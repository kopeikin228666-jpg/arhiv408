"""VFX agent preview: render single frames exactly like render.render() (minus encoding) and tile them.
usage: VOY_VER=B python3 preview.py OUT.png t1 t2 ... [--novfx] [--scale 2]"""
import sys, os, math, importlib
os.environ.setdefault('VOY_VER', 'B')
sys.path.insert(0, '/home/claude/voynich/edit')
import numpy as np, cv2
import render as R
import vfx
R.VFX = vfx

_SH = None
_BEATS = None
_FC, _LC = {}, {}


def frame(t, use_vfx=True, grid_only=False):
    global _SH, _BEATS
    if _SH is None:
        _SH = R.build()
        _BEATS = np.array(R.sync.beat_events(), dtype=np.float64)
    shots = _SH
    FPS = R.FPS
    f = int(round(t * FPS))
    t = f / FPS
    kn = R.knocks(shots)
    k = max(i for i, s in enumerate(shots) if s.t0 <= t + 1e-6)
    sh = shots[k]
    tq = sh.t0 + math.floor((t - sh.t0) * 15 + 1e-6) / 15
    tl = max(0.0, tq - sh.t0) + sh.toff
    u = min(1.0, tl / (sh.t1 - sh.t0 + sh.toff))
    rng = np.random.default_rng(int(t * 15) * 7 + 11)
    g = sh.fn(tl, u, rng, tq)
    for j in (k, k + 1):
        if j < 1 or j >= len(shots) or not shots[j].trans:
            continue
        kind, n, pre = shots[j].trans
        fc = int(round(shots[j].t0 * FPS))
        if not (fc - pre <= f < fc - pre + n):
            continue
        prog = (f - (fc - pre) + 1) / (n + 1)
        if f < fc:
            if j not in _FC:
                nx = shots[j]
                _FC[j] = nx.fn(nx.toff, 0.0, np.random.default_rng(j), nx.t0)
            a_, b_ = g, _FC[j]
        else:
            if j not in _LC:
                pv = shots[j - 1]
                _LC[j] = pv.fn(pv.t1 - pv.t0 - 1 / FPS + pv.toff, 1.0, np.random.default_rng(j + 99), pv.t1)
            a_, b_ = _LC[j], g
        if kind == 'tear' and f < fc:
            continue
        g = R.run_trans(kind, a_, b_, prog, j, rng)
        break
    if not sh.clean:
        g = R.beat_pulse(g, t, _BEATS, rng)
    if R.HOOK is not None and t < R.HOOK.END:
        hg = R.HOOK.frame(t, g, rng)
        if hg is not None:
            g = hg
    if use_vfx:
        g = vfx.post(g, t, sh.name, tl, u, rng)
    for kt in kn:
        dk = t - kt
        if 0 <= dk < 0.2:
            amp = int(round(5 * (1 - dk / 0.2)))
            if amp:
                g = np.roll(g, (int(rng.integers(-amp, amp + 1)), int(rng.integers(-amp, amp + 1))), axis=(0, 1))
    if grid_only:
        return g
    rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    ui = sh.ui
    if ui and 'track' in ui:
        _, c, bx = ui['track']
        boxes = []
        for b in bx:
            gx, gy = R.map_pt(u, c, *b['pt'])
            boxes.append(dict(r=(int(gx - 9), int(gy - 12), int(gx + 9), int(gy + 10)), label=b['label'], col='bone', corners=False))
        ui = dict(ui, boxes=boxes)
    rgb = R.draw_ui(rgb, t, ui)
    rgb = R.draw_caption(rgb, R.caption_at(t), t)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    args = sys.argv[1:]
    out = args.pop(0)
    use = '--novfx' not in args
    sc = 2
    if '--scale' in args:
        i = args.index('--scale'); sc = int(args[i + 1]); del args[i:i + 2]
    ts = [float(a) for a in args if not a.startswith('--')]
    tiles = []
    for t in ts:
        fr = frame(t, use)
        fr = cv2.resize(fr, (R.GW * sc, R.GH * sc), interpolation=cv2.INTER_NEAREST)
        cv2.putText(fr, f'{t:.2f}', (3, 12), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
        tiles.append(fr)
    while len(tiles) % 8 and len(tiles) > 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(out, np.vstack(rows))
