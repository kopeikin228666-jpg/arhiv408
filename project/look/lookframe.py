import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

rng = np.random.default_rng(7)
W, H = 1080, 1920
sheet = cv2.imread('/home/claude/voynich/assets/reader_sheet.jpg')

# --- cut out the seated figure (flat grey background -> key by colour distance)
crop = sheet[180:768, 905:1376].astype(np.float32)
bg = np.median(np.concatenate([crop[:20, :20].reshape(-1, 3), crop[:20, -20:].reshape(-1, 3)]), axis=0)
dist = np.linalg.norm(crop - bg, axis=2)
mask = (dist > 30).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
keep = np.zeros_like(mask)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 1500:
        keep[lab == i] = 1
mask = cv2.erode(keep, np.ones((5,5),np.uint8))
scale = 1.75
fig = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
fm = cv2.resize(mask.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
fm = cv2.GaussianBlur(fm, (3, 3), 0)
fh, fw = fm.shape

# --- canvas: dark void with floating concrete tiles
canvas = np.full((H, W, 3), (14, 16, 15), np.float32)
grad = np.linspace(0, 1, H)[:, None, None]
canvas += 10 * (1 - np.abs(grad - 0.55) * 2).clip(0)

def concrete(sz):
    t = rng.normal(0, 1, (sz // 4 + 2, sz // 4 + 2)).astype(np.float32)
    t = cv2.resize(t, (sz, sz), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0, 1, (sz, sz)).astype(np.float32)
    v = 92 + 14 * t + 6 * fine
    return np.dstack([v * 0.97, v, v * 0.99])

ts = 96
ox, oy = (W - fw) // 2, 640
floor_y0 = oy + int(fh * 0.80)
for gy in range(-4, 6):
    for gx in range(-1, W // ts + 1):
        x = gx * ts + 12
        y = floor_y0 + gy * ts
        d = abs(gy - 0.5) + abs((x + ts / 2 - W / 2) / ts) * 0.55
        p_keep = np.clip(1.25 - d * 0.22, 0, 1)
        if rng.random() > p_keep:
            continue
        t = concrete(ts - 4)
        drop = 0
        if rng.random() < 0.18:  # tile breaking away and falling
            drop = int(rng.integers(30, 260))
        yy = y + drop
        if yy < 0 or yy + ts > H or x < 0 or x + ts > W:
            continue
        a = 1.0 if drop == 0 else max(0.15, 1 - drop / 300)
        canvas[yy:yy + ts - 4, x:x + ts - 4] = canvas[yy:yy + ts - 4, x:x + ts - 4] * (1 - a) + t * a * 0.8

# --- composite the figure
region = canvas[oy:oy + fh, ox:ox + fw]
region[:] = region * (1 - fm[..., None]) + fig * fm[..., None]
# contact shadow
sh = np.zeros((H, W), np.float32)
cv2.ellipse(sh, (W // 2, oy + fh - 8), (fw // 2, 26), 0, 0, 360, 1, -1)
sh = cv2.GaussianBlur(sh, (0, 0), 18)
canvas *= (1 - 0.6 * sh[..., None])

# --- semi-pixel: blocks inside the figure where a noise field is high, some blocks falling off
full_m = np.zeros((H, W), np.float32)
full_m[oy:oy + fh, ox:ox + fw] = fm
b = 18
noise = cv2.resize(rng.random((H // 160 + 2, W // 160 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
yy_, xx_ = np.mgrid[0:H, 0:W]
side = ((xx_ - (ox + fw * 0.35)) / (fw * 0.5)).clip(0)
edge = 1 - cv2.erode(full_m, np.ones((61,61),np.uint8))
field = (noise * 0.5 + side * 0.55) * (0.35 + 0.9 * edge)
out = canvas.copy()
for y in range(oy, oy + fh, b):
    for x in range(ox, ox + fw, b):
        cell_m = full_m[y:y + b, x:x + b]
        if cell_m.mean() < 0.35:
            continue
        f = field[y + b // 2 if y + b // 2 < H else H - 1, x + b // 2 if x + b // 2 < W else W - 1]
        if f > 0.62:
            col = canvas[y:y + b, x:x + b].reshape(-1, 3).mean(0)
            out[y:y + b, x:x + b] = col
            if f > 0.78 and rng.random() < 0.35:  # detached pixel falling
                dy = int(rng.integers(20, 240)); dx = int(rng.integers(-30, 30))
                ny, nx = y + dy, x + dx
                if 0 <= nx and nx + b < W and ny + b < H:
                    out[ny:ny + b, nx:nx + b] = col * 0.9
                out[y:y + b, x:x + b] = canvas[y:y + b, x:x + b] * 0.25 + out[y:y + b, x:x + b] * 0.0 + np.array([14, 16, 15]) * 0.75
canvas = out

# --- grade: desaturate, teal-green shadows, lifted blacks, vignette, grain
img = canvas.clip(0, 255) / 255.0
lum = img @ np.array([0.114, 0.587, 0.299])
img = img * 0.38 + lum[..., None] * 0.62
shadow = (1 - lum)[..., None] ** 2
img = img + shadow * np.array([0.035, 0.03, -0.01])  # BGR: teal-green in shadows
img = 0.045 + img * 0.93
img = np.clip(img, 0, 1)
img = img ** 1.08
vy, vx = np.mgrid[0:H, 0:W]
vig = 1 - 0.38 * (((vx - W / 2) / (W / 2)) ** 2 + ((vy - H / 2) / (H / 1.6)) ** 2)
img *= vig.clip(0.3, 1)[..., None]
g1 = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
g2 = rng.normal(0, 1, (H, W)).astype(np.float32)
grain = (0.6 * g1 + 0.4 * g2) * 0.045 * (0.6 + 0.4 * (1 - lum))
img = np.clip(img + grain[..., None], 0, 1)
out8 = (img * 255).astype(np.uint8)

# --- CV tracking overlay (Synthetic-Urban style) + editorial labels (LILIUM style)
pil = Image.fromarray(cv2.cvtColor(out8, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil)
mono = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 19)
mono_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
ys, xs = np.where(full_m > 0.5)
x0, y0, x1, y1 = xs.min() - 14, ys.min() - 14, xs.max() + 14, ys.max() + 10
wcol = (225, 228, 222)
d.rectangle([x0, y0, x1, y1], outline=wcol, width=2)
d.rectangle([x0, y0 - 30, x0 + 300, y0], fill=wcol)
d.text((x0 + 8, y0 - 26), 'SUBJECT_01  READER  0.97', font=mono, fill=(15, 17, 16))
# box around the book
bx0, by0 = ox + int(fw * 0.30), oy + int(fh * 0.34)
d.rectangle([bx0, by0, bx0 + int(fw * 0.36), by0 + int(fh * 0.20)], outline=(160, 40, 36), width=2)
d.text((bx0, by0 + int(fh * 0.20) + 6), 'MS 408 / f.78r\nLANG: ---\nREAD: 0%', font=mono_s, fill=(190, 60, 54))
# guide lines
for gx in (W * 0.5,):
    d.line([gx, 160, gx, H - 160], fill=(80, 84, 82), width=1)
d.line([60, y0 - 60, W - 60, y0 - 60], fill=(70, 74, 72), width=1)
for (px, py) in [(W * 0.5, y0 - 60), (60, y0 - 60), (W - 60, y0 - 60)]:
    d.rectangle([px - 4, py - 4, px + 4, py + 4], fill=(160, 40, 36))
d.text((60, 210), 'SURFACE_01', font=mono_s, fill=(120, 124, 120))
d.text((60, 232), 'signal: unstable', font=mono_s, fill=(120, 124, 120))
d.text((W - 260, H - 260), 'frame 0412 / ????', font=mono_s, fill=(120, 124, 120))
pil.save('/home/claude/voynich/look/look_reader.png')
print('ok', x0, y0, x1, y1)
