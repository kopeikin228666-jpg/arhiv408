"""preview single composited frames (shot fn + beat pulse + hook + HUD + captions), no transitions/scanlines"""
import sys, os, math
os.environ['VOY_VER'] = 'B'
sys.argv = [sys.argv[0]] + sys.argv[1:]
sys.path.insert(0, '/home/claude/voynich/edit')
import numpy as np, cv2
import render as R
import hook
out = sys.argv[1]; times = [float(x) for x in sys.argv[2:]]
shots = R.build(); beats = np.array(R.sync.beat_events(), dtype=np.float64)
tiles = []
for t in times:
    f = int(round(t * 30)); t = f / 30
    k = max(i for i, s in enumerate(shots) if s.t0 <= t + 1e-6); sh = shots[k]
    tq = sh.t0 + math.floor((t - sh.t0) * 15 + 1e-6) / 15
    tl = max(0.0, tq - sh.t0) + sh.toff
    u = min(1.0, tl / (sh.t1 - sh.t0 + sh.toff))
    rng = np.random.default_rng(int(t * 15) * 7 + 11)
    g = sh.fn(tl, u, rng, tq)
    if not sh.clean: g = R.beat_pulse(g, t, beats, rng)
    if t < hook.END:
        hg = hook.frame(t, g, rng)
        if hg is not None: g = hg
    rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    ui = sh.ui
    if ui and 'track' in ui:
        _, c, bx = ui['track']; boxes = []
        for b in bx:
            gx, gy = R.map_pt(u, c, *b['pt'])
            boxes.append(dict(r=(int(gx - 9), int(gy - 12), int(gx + 9), int(gy + 10)), label=b['label'], col='bone', corners=False))
        ui = dict(ui, boxes=boxes)
    rgb = R.draw_ui(rgb, t, ui); rgb = R.draw_caption(rgb, R.caption_at(t), t)
    im = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(im, f'{t:.3f}', (4, 636), 0, .5, (0, 255, 255), 1)
    tiles.append(im)
rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8]))) for i in range(0, len(tiles), 8)]
cv2.imwrite(out, np.vstack(rows))
