"""overseer: render single episode frames through the real builder.  TS=comma list  OUT=path  [EP=builder path]
also prints off-palette pixel count of each GRID (before finish) and saves grids at x3 for zoom."""
import os, sys, time, importlib.util, numpy as np, cv2
os.environ['PART'] = '3'
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/team/agents/overseer')
from pal import off_palette, up, label
t0 = time.time()
spec = importlib.util.spec_from_file_location('ep', os.environ.get('EP', '/home/claude/voynich/team/host/make_ep1e.py'))
ep = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(ep)
print(f'import {time.time() - t0:.1f}s', flush=True)
import subprocess
class LazyReel:
    '''REELS[i] stand-in: decodes only the frames actually asked for (one ffmpeg seek per frame)'''
    def __init__(self, a, b): self.a, self.n, self.c = a, int(round((b - a) * ep.FPS)), {}
    def __len__(self): return self.n
    def __getitem__(self, j):
        if j not in self.c:
            cmd = ['ffmpeg', '-v', 'error', '-ss', f'{self.a + j / ep.FPS:.3f}', '-i', ep.REEL_SRC, '-frames:v', '1',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
            buf = subprocess.run(cmd, capture_output=True).stdout
            fr = np.frombuffer(buf[:ep.OW * ep.OH * 3], np.uint8).reshape(ep.OH, ep.OW, 3)
            self.c[j] = fr[3::ep.S, 3::ep.S].copy()
        return self.c[j]
for i_, sg_ in enumerate(ep.SEG):
    if sg_['kind'] == 'reel': ep.REELS[i_] = LazyReel(*sg_['src'])
_cg = ep.clip_grids
def clip_lazy(film, dur):
    key = (film, dur)
    if key not in ep.CLIPS: ep.CLIPS[key] = LazyReel(film, film + dur + 0.05)
    return ep.CLIPS[key]
ep.clip_grids = clip_lazy
out = os.environ.get('OUT', '/home/claude/voynich/team/agents/overseer/ep_sheet.jpg')
ts = [float(x) for x in os.environ['TS'].split(',')]
tiles, grids = [], []
ZB = os.environ.get('ZONES', '1') == '1'
for t in ts:
    f = int(round(t * ep.FPS)); i, sg = ep.seg_at(t)
    if sg['kind'] == 'dive':
        fr = ep.dive_frame(i, t, f); g = np.zeros((320, 180, 3), np.uint8); z = 1.0; cx = cy = .5; rot = 0.0
    else:
        g, z, cx, cy, rot = ep.seg_frame(i, t, f)
    op = off_palette(g)
    if sg['kind'] != 'dive':
        fr = ep.finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, z, cx, cy, rot, drift=sg['kind'] != 'reel')
    if os.environ.get('ZONES', '1') == '1':     # draw the UI danger zones (top 200, bottom 380, icon column)
        fr = fr.copy(); fr[198:202] = (0, 255, 255); fr[1538:1542] = (0, 255, 255); fr[900:1700, 898:902] = (255, 0, 255)
    fr = ep.with_text(fr, t, f)
    if ZB: fr[198:202] = (0, 255, 255); fr[1538:1542] = (0, 255, 255); fr[900:1700, 898:902] = (255, 0, 255)
    if os.environ.get('FULL'):
        y0_, y1_ = [int(v) for v in os.environ['FULL'].split(':')]
        cv2.imwrite(out.replace('.jpg', f'_full_{t:.3f}.jpg'), cv2.cvtColor(fr[y0_:y1_], cv2.COLOR_RGB2BGR))
    sm = cv2.resize(fr, (360, 640), interpolation=cv2.INTER_AREA)
    sm = cv2.cvtColor(sm, cv2.COLOR_RGB2BGR)
    tiles.append(label(sm, f'{t:.3f} {sg["kind"]} z={z:.2f} off={op[0]}'))
    grids.append(label(up(g, 2), f'{t:.3f}'))
    print(f'{t:.3f} seg={i} {sg["kind"]} tl={t - sg["t0"]:.3f} zoom={z:.3f} offpal={op[0]} {op[1][:3]}', flush=True)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0])); grids.append(np.zeros_like(grids[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
cv2.imwrite(out.replace('.jpg', '_grid.png'), np.vstack([np.hstack(grids[k:k + 4]) for k in range(0, len(grids), 4)]))
