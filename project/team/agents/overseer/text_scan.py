"""overseer: scan the TEXT LAYER of every frame (no reels needed, cheap): safe zones, caption line count,
caption vs other text overlap, chunk endings, word-time sync.  EP=builder (default make_ep1e.py)  STEP=frames step
Zones in output px (1080x1920): top < 200, bottom > 1540 (380 px), icon column x > 900 for y 900..1700."""
import os, sys, importlib.util, numpy as np, cv2
os.environ['PART'] = '3'; sys.path.insert(0, '/home/claude/voynich/edit')
spec = importlib.util.spec_from_file_location('ep', os.environ.get('EP', '/home/claude/voynich/team/host/make_ep1e.py'))
ep = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(ep)
R = ep.R; S = 6; KEY = np.array(ep.KEY, np.uint8)
REC = {}
_dc = R.draw_caption
def dc(rgb, cap, t):
    REC['pre'] = rgb.copy(); out = _dc(rgb, cap, t); REC['out'] = out.copy(); REC['cap'] = cap; return out
R.draw_caption = dc
STEP = int(os.environ.get('STEP', '1'))
viol = {}; overl = []; nlines = {}; caps_seen = {}
def zone(mask):
    ys, xs = np.nonzero(mask); bad = []
    if not len(ys): return bad
    if ys.min() * S < 200: bad.append(f'TOP y={ys.min() * S}')
    if (ys.max() + 1) * S > 1540: bad.append(f'BOTTOM y={(ys.max() + 1) * S}')
    ic = mask[150:284, 150:]
    if ic.any():
        yy, xx = np.nonzero(ic); bad.append(f'ICONS x={(xx.max() + 151) * S} y={(yy.min() + 150) * S}..{(yy.max() + 151) * S}')
    return bad
for f in range(0, int(round(ep.T_END * ep.FPS)), STEP):
    t = f / ep.FPS; REC.clear()
    lay = ep.text_layer(t, f)
    if lay is None: continue
    m = ~(lay == KEY).all(2)
    i, sg = ep.seg_at(t)
    for b in zone(m):
        viol.setdefault((i, sg['kind'], b.split()[0]), []).append((round(t, 2), b))
    if 'pre' in REC and REC['cap']:
        pre = ~(REC['pre'] == KEY).all(2); cm = (REC['out'] != REC['pre']).any(2)
        if cm.any():
            ys, xs = np.nonzero(cm)
            box = np.zeros_like(cm); box[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 3):xs.max() + 4] = True
            ov = (box & pre).sum()
            if ov: overl.append((round(t, 2), i, sg['kind'], int(ov)))
            # line count: distinct text rows clusters
            rows = np.nonzero(cm.any(1))[0]; nl = 1 + int((np.diff(rows) > 3).sum())
            words = ' '.join(w[0] for w in REC['cap'][0])
            nlines[words] = max(nlines.get(words, 0), nl)
            caps_seen.setdefault(words, round(t, 2))
print('=== SAFE-ZONE VIOLATIONS (segment, kind, zone): first..last time, example')
for k, v in viol.items(): print(k, v[0][0], '..', v[-1][0], f'({len(v)} frames)', v[0][1])
print('=== CAPTION overlapped by other text (t, seg, kind, px):', len(overl)); print(overl[:30])
print('=== caption chunks (max lines) ===')
WEAK = ep._WEAK
for w, n in nlines.items():
    last = w.split()[-1].upper().strip('.,…!?')
    flag = (' <-- 3+ LINES' if n > 2 else '') + (' <-- ENDS ON WEAK' if last in WEAK else '')
    print(f'{caps_seen[w]:6.2f} [{n}] {w}{flag}')
