import sys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/claude/voynich/look')
from pixelize import quantize, PALETTE

rng = np.random.default_rng(11)
GW, GH, S = 180, 320, 6           # everything is drawn on a 180x320 grid, then x6 nearest
C = {k: tuple(int(v) for v in PALETTE[i]) for k, i in
     dict(void=0, void2=1, dark=2, tile=4, tile2=5, line=7, bone=9, red=17, red2=16, parch=15).items()}

# ---------- cut the seated Reader from the sheet
sheet = cv2.imread('/home/claude/voynich/assets/reader_sheet.jpg')
crop = sheet[180:768, 905:1376].astype(np.float32)
bg = np.median(np.concatenate([crop[:20, :20].reshape(-1, 3), crop[:20, -20:].reshape(-1, 3)]), axis=0)
mask = (np.linalg.norm(crop - bg, axis=2) > 30).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
mask = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 1500]).astype(np.uint8)
mask = cv2.erode(mask, np.ones((5, 5), np.uint8))

fw = 104                                   # figure width in grid cells
fh = int(crop.shape[0] * fw / crop.shape[1])
fig_small = cv2.resize(crop.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_AREA)
m_small = cv2.resize(mask.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA) > 0.5
fs = (255*((fig_small.astype(np.float32)/255)**0.72)).astype(np.uint8)
fig_q = quantize(fs, dither=0.12, contrast=1.3)

# ---------- canvas
yy_, xx_ = np.mgrid[0:GH, 0:GW]
fogv = 0.10 + 0.22*np.exp(-((yy_-200)/95.0)**2) * (1-0.5*np.abs(xx_-GW/2)/(GW/2))
fogimg = np.dstack([fogv*235, fogv*250, fogv*240]).clip(0,255).astype(np.uint8)
can = quantize(fogimg, dither=0.16, contrast=1.0)
# faint vertical gradient band


ox, oy = (GW - fw) // 2 + 2, 120
floor = oy + int(fh * 0.86)
# floating stone tiles (8x8 cells), dissolving away from centre, some falling
T = 9
for gy in range(-3, 6):
    for gx in range(0, GW // T + 1):
        x, y = gx * T, floor + gy * T
        dist = abs(gy - 0.5) * 1.0 + abs(x + T / 2 - GW / 2) / T * 0.45
        if rng.random() > np.clip(1.3 - dist * 0.24, 0, 1):
            continue
        drop = int(rng.integers(4, 40)) if rng.random() < 0.2 else 0
        yy = y + drop
        if yy < 0 or yy + T - 1 > GH:
            continue
        base = C['tile'] if (gx + gy) % 2 else C['tile2']
        can[yy:yy + T - 1, x:x + T - 1] = base
        # texture specks
        sp = rng.random((T - 1, T - 1)) < 0.18
        can[yy:yy + T - 1, x:x + T - 1][sp] = C['dark']
        if drop:
            can[yy:yy + T - 1, x:x + T - 1][::2, ::2] = C['void2']

# shadow under bench
cv2.ellipse(can, (GW // 2 + 2, floor - 2), (fw // 2 - 6, 3), 0, 0, 360, C['dark'], -1)

# ---------- paste figure; the right side dissolves into bigger blocks that drift up
region = can[oy:oy + fh, ox:ox + fw]
region[m_small] = fig_q[m_small]
cols = np.where(m_small.any(0))[0]
xr = cols.max()
for y in range(0, fh, 2):
    for x in range(0, fw, 2):
        if not m_small[y:y + 2, x:x + 2].any():
            continue
        t = (x - fw * 0.68) / (fw * 0.32)          # 0 at middle, 1 at right edge
        n_ = rng.random()
        if t > 0 and n_ < t * 0.9:
            col = fig_q[y:y + 2, x:x + 2].reshape(-1, 3).mean(0).astype(np.uint8)
            if n_ < t * 0.45:                     # detach and float upward / right
                region[y:y + 2, x:x + 2] = can[oy+y:oy+y+2, ox+x:ox+x+2][0,0]*0+np.array(C['void2'])
                dy = -int(rng.integers(2, 30)); dx = int(rng.integers(0, 14))
                ny, nx = oy + y + dy, ox + x + dx
                if 0 <= ny < GH - 2 and nx < GW - 2:
                    can[ny:ny + 2, nx:nx + 2] = col
            else:                                 # coarse 2x2 block in place
                region[y:y + 2, x:x + 2] = col

# ---------- pixel UI: tracking boxes, guides, labels (no anti-aliasing)
pil = Image.fromarray(cv2.cvtColor(can, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil); d.fontmode = '1'
f6 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 9)
f8 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 9)
fbig = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf', 15)
rgb = lambda k: tuple(int(v) for v in C[k][::-1])
ys, xs = np.where(m_small)
x0, y0, x1, y1 = ox + xs.min() - 3, oy + ys.min() - 3, ox + xs.max() + 3, oy + ys.max() + 2
d.rectangle([x0, y0, x1, y1], outline=rgb('bone'))
for (cx, cy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
    d.rectangle([cx - 1, cy - 1, cx + 1, cy + 1], fill=rgb('bone'))
d.rectangle([x0, y0 - 10, x0 + 70, y0 - 1], fill=rgb('bone'))
d.text((x0 + 2, y0 - 10), 'SUBJECT_01', font=f8, fill=rgb('void'))
d.text((x0 + 74, y0 - 10), '.97', font=f6, fill=rgb('line'))
# book box (red)
bx0, by0 = ox + int(fw * 0.30), oy + int(fh * 0.33)
d.rectangle([bx0, by0, bx0 + int(fw * 0.38), by0 + int(fh * 0.21)], outline=rgb('red'))
d.text((bx0, by0 + int(fh * 0.21) + 2), 'MS408', font=f6, fill=rgb('red'))
d.text((bx0, by0 + int(fh * 0.21) + 11), 'READ 0%', font=f6, fill=rgb('red'))
# guides
d.line([GW // 2, 18, GW // 2, GH - 60], fill=rgb('dark'))
d.line([6, y0 - 18, GW - 6, y0 - 18], fill=rgb('dark'))
for px in (6, GW // 2, GW - 7):
    d.rectangle([px - 1, y0 - 19, px + 1, y0 - 17], fill=rgb('red'))
d.text((8, 22), 'SURFACE_01', font=f6, fill=rgb('line'))
d.text((8, 30), 'SIGNAL --', font=f6, fill=rgb('tile2'))
d.text((GW - 50, 22), '0412', font=f6, fill=rgb('tile2'))
# caption in the "pixel stories" style: caps, key word in a red plate
cy = GH - 44
w1 = d.textlength('ОНА ', font=fbig); w2 = d.textlength('СМОТРИТ', font=fbig)
cx = int((GW - w1 - w2 - 4) / 2)
d.text((cx, cy), 'ОНА ', font=fbig, fill=rgb('bone'))
d.rectangle([cx + w1 - 1, cy + 1, cx + w1 + w2 + 2, cy + 17], fill=rgb('red'))
d.text((cx + w1, cy), 'СМОТРИТ', font=fbig, fill=rgb('bone'))

small = np.array(pil)
big = cv2.resize(small, (GW * S, GH * S), interpolation=cv2.INTER_NEAREST)
# very light scanline + grain at output res, keeps the pixel grid readable
big = big.astype(np.float32)
big[::S] *= 0.82
big += rng.normal(0, 4, big.shape[:2])[..., None]
cv2.imwrite('/home/claude/voynich/look/pixel_reader.png', cv2.cvtColor(np.clip(big, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print('ok')
