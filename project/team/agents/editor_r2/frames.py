"""render single frames of PART 3 exactly like render_video does -> contact sheet.  usage: frames.py out.jpg t1 t2 ..."""
import os, sys, importlib.util, numpy as np, cv2
os.environ['PART'] = '3'
spec = importlib.util.spec_from_file_location('ep', '/home/claude/voynich/team/host/make_ep1e.py'); ep = importlib.util.module_from_spec(spec)
out_path = sys.argv[1]; ts = [float(x) for x in sys.argv[2:]]
sys.argv = ['x']; spec.loader.exec_module(ep)
R = ep.R


def need(i):
    if 0 <= i < len(ep.SEG) and ep.SEG[i]['kind'] == 'reel' and i not in ep.REELS:
        ep.REELS[i] = ep.reel_grids(*ep.SEG[i]['src'])


def frame_at(t):
    f = int(round(t * ep.FPS)); t = f / ep.FPS
    i, sg = ep.seg_at(t)
    for j in (i - 1, i, i + 1, i + 2):
        need(j)
    rng = np.random.default_rng(f); SEG = ep.SEG; frame = None
    for (j0, j1) in ((i - 1, i), (i, i + 1)):
        if j0 < 0 or j1 >= len(SEG) or not SEG[j0].get('trans_out'):
            continue
        kind, n = SEG[j0]['trans_out']
        fc = int(round(SEG[j1]['t0'] * ep.FPS)); pre = n // 2
        if not (fc - pre <= f < fc - pre + n):
            continue
        prog = (f - (fc - pre) + 1) / (n + 1)
        ga, za, cxa, cya, ra = ep.seg_frame(j0, min(t, SEG[j0]['t1'] - 1 / ep.FPS), f)
        gb, zb, cxb, cyb, rb = ep.seg_frame(j1, max(t, SEG[j1]['t0']), f)
        g = R.run_trans(kind, ga, gb, prog, j1, rng)
        frame = ep.finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, za if prog < .5 else zb, cxa if prog < .5 else cxb,
                          cya if prog < .5 else cyb, ra if prog < .5 else rb)
        break
    if frame is None:
        if sg['kind'] == 'dive':
            frame = ep.dive_frame(i, t, f)
        else:
            g, z, cx, cy, rot = ep.seg_frame(i, t, f)
            frame = ep.finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, z, cx, cy, rot, drift=sg['kind'] != 'reel')
    return ep.with_text(frame, t, f), sg['kind']


W, H = 270, 480
out = []
for t in ts:
    fr, k = frame_at(t)
    sm = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    for yy, col in ((200, (0, 120, 255)), (1920 - 380, (255, 0, 0))):   # top UI line / approx bottom UI line
        cv2.line(sm, (0, int(H * yy / 1920)), (W, int(H * yy / 1920)), col, 1)
    cv2.line(sm, (int(W * 900 / 1080), int(H * 0.45)), (int(W * 900 / 1080), H), (255, 0, 255), 1)  # right icon column
    cv2.putText(sm, f'{t:.2f} {k}', (4, 14), 0, 0.45, (0, 255, 0), 1)
    out.append(sm)
while len(out) % 6:
    out.append(np.zeros_like(out[0]))
rows = [np.hstack(out[k:k + 6]) for k in range(0, len(out), 6)]
cv2.imwrite(out_path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ok', out_path)
