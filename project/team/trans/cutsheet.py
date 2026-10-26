"""Extract frames around every transitioned cut from a rendered video -> one strip per cut, stacked sheets."""
import os, sys
os.environ.setdefault('VOY_VER', 'B')
sys.path.insert(0, '/home/claude/voynich/edit')
import numpy as np, cv2
import render

src = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs/voynich_v5_B.mp4'
out = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/voynich/team/trans/cuts_v5'
shots = render.build()
want = {}
info = []
for j, s in enumerate(shots):
    if j == 0:
        continue
    fc = int(round(s.t0 * 30))
    tr = s.trans
    if tr:
        kind, n, pre = tr
        f0, f1 = fc - pre - 1, fc - pre + n + 1
    else:
        kind, n, pre = 'cut', 0, 0
        f0, f1 = fc - 2, fc + 2
    fr = list(range(f0, f1 + 1))
    if len(fr) > 16:
        fr = sorted(set(np.linspace(f0, f1, 16).round().astype(int)))
    info.append((s.name, s.t0, kind, n, pre, fr, fc))
    for f in fr:
        want[f] = None
cap = cv2.VideoCapture(src)
f = 0
mx = max(want)
while f <= mx:
    ok = cap.grab()
    if not ok:
        break
    if f in want:
        _, im = cap.retrieve()
        want[f] = cv2.resize(im, (120, 213), interpolation=cv2.INTER_AREA)
    f += 1
os.makedirs(out, exist_ok=True)
rows = []
for name, t0, kind, n, pre, fr, fc in info:
    tiles = []
    for f in fr:
        im = want.get(f)
        if im is None:
            im = np.zeros((213, 120, 3), np.uint8)
        im = im.copy()
        if f == fc:
            cv2.rectangle(im, (0, 0), (119, 212), (0, 0, 255), 2)
        tiles.append(im)
    while len(tiles) < 16:
        tiles.append(np.zeros((213, 120, 3), np.uint8))
    lab = np.zeros((213, 110, 3), np.uint8)
    for i, txt in enumerate([name, f'{t0:.2f}', kind, f'{n}/{pre}']):
        cv2.putText(lab, txt, (4, 40 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 220, 230), 1)
    rows.append(np.hstack([lab] + tiles))
for k in range(0, len(rows), 8):
    cv2.imwrite(f'{out}/sheet_{k // 8}.jpg', np.vstack(rows[k:k + 8]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(rows), 'cuts')
