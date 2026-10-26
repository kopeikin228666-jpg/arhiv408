"""overseer: the loop seam - last frame of the episode vs frame 0: red-eye centroid / extent in output px (no text layer)"""
import os, sys, importlib.util, numpy as np, cv2
os.environ['PART'] = '3'; sys.path.insert(0, '/home/claude/voynich/edit')
spec = importlib.util.spec_from_file_location('ep', os.environ.get('EP', '/home/claude/voynich/team/host/make_ep1e.py'))
ep = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(ep)
n = int(round(ep.T_END * ep.FPS))
for f in (n - 2, n - 1, 0, 1, 3, 9):
    t = f / ep.FPS; i, sg = ep.seg_at(t)
    g, z, cx, cy, rot = ep.seg_frame(i, t, f)
    fr = ep.finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, z, cx, cy, rot, drift=sg['kind'] != 'reel').astype(int)
    red = (fr[..., 0] - fr[..., 1] > 70) & (fr[..., 0] > 120)
    ys, xs = np.nonzero(red[:, :])
    # two eyes: split at the x median
    L = xs < 540
    def c(m): return (int(xs[m].mean()), int(ys[m].mean()), int(xs[m].min()), int(xs[m].max()), int(ys[m].min()), int(ys[m].max())) if m.any() else None
    print(f'f={f:4d} t={t:.3f} {sg["kind"]} z={z:.3f} redpx={red.sum():6d} leftEye(cx,cy,x0,x1,y0,y1)={c(L)} rightEye={c(~L)}')
