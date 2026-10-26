"""Voynich reel — full 2:00 timeline renderer (approved pixel look).
Everything is drawn on a 180x320 grid, held at 15 fps for a sprite feel, upscaled x6 to 1080x1920 @30fps.
usage: python3 render.py OUT.mp4 [--from S] [--to S] [--sheet sheet.png]
"""
import sys, os, math, argparse, subprocess, functools
for _i, _a in enumerate(sys.argv):             # version must be known before sync / book are imported
    if _a == '--ver' and _i + 1 < len(sys.argv):
        os.environ['VOY_VER'] = sys.argv[_i + 1]
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/claude/voynich/look')
sys.path.insert(0, '/home/claude/voynich/pipe')
from pixelize import quantize, PALETTE
sys.path.insert(0, '/home/claude/voynich/edit')
import sync
try:
    import transitions as TRX
except Exception:                               # library not there yet -> fall back to the built-in ones
    TRX = None

R = '/home/claude/voynich'
GW, GH, S, FPS = 180, 320, 6, 30
OW, OH = GW * S, GH * S
P = {k: PALETTE[i].astype(np.uint8) for k, i in
     dict(void=0, void2=1, dark=2, d3=3, tile=4, tile2=5, mid=6, line=7, light=8, bone=9,
          g1=10, g2=11, g3=12, p1=13, p2=14, parch=15, red2=16, red=17).items()}  # BGR
RGB = {k: tuple(int(c) for c in v[::-1]) for k, v in P.items()}
FMONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
FCOND = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'
F9 = ImageFont.truetype(FMONO, 9)
F7 = ImageFont.truetype(FMONO, 7)
FCAP = ImageFont.truetype(FCOND, 15)
FCAP_S = ImageFont.truetype(FCOND, 13)
FBIG = ImageFont.truetype(FCOND, 30)
FMID = ImageFont.truetype(FMONO, 15)

# ------------------------------------------------------------------ sources
@functools.lru_cache(None)
def still(name):
    im = cv2.imread(f'{R}/kf/{name}')
    h, w = im.shape[:2]
    tgt = 9 / 16
    if w / h > tgt:
        nw = int(h * tgt); x0 = (w - nw) // 2; im = im[:, x0:x0 + nw]
    else:
        nh = int(w / tgt); y0 = (h - nh) // 2; im = im[y0:y0 + nh]
    return cv2.resize(im, (720, 1280), interpolation=cv2.INTER_AREA)


@functools.lru_cache(None)
def video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        tgt = 9 / 16
        if w / h > tgt:
            nw = int(h * tgt); x0 = (w - nw) // 2; f = f[:, x0:x0 + nw]
        else:
            nh = int(w / tgt); y0 = (h - nh) // 2; f = f[y0:y0 + nh]
        out.append(cv2.resize(f, (720, 1280), interpolation=cv2.INTER_AREA))
    return out, fps


@functools.lru_cache(None)
def gridvideo(path):
    """already-rendered approved-look clip (1080x1920): sample cell centres back to the grid"""
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[3::S, 3::S][:GH, :GW].copy())
    return out


def cam_M(u, c):
    e = 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, u)))  # ease in-out
    if c.get('lin'):
        e = u
    z = c.get('z0', 1) + (c.get('z1', 1) - c.get('z0', 1)) * e
    cx = c.get('x0', .5) + (c.get('x1', c.get('x0', .5)) - c.get('x0', .5)) * e
    cy = c.get('y0', .5) + (c.get('y1', c.get('y0', .5)) - c.get('y0', .5)) * e
    rot = c.get('r0', 0) + (c.get('r1', c.get('r0', 0)) - c.get('r0', 0)) * e
    s = z * 360 / 720
    M = cv2.getRotationMatrix2D((cx * 720, cy * 1280), rot, s)
    M[0, 2] += 180 - cx * 720
    M[1, 2] += 320 - cy * 1280
    return M


def cam(img, u, c):
    """img 720x1280, u in [0,1], c = dict(z0,z1,x0,y0,x1,y1,r0,r1) -> 360x640 view"""
    return cv2.warpAffine(img, cam_M(u, c), (360, 640), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def map_pt(u, c, x, y):
    """normalized image point -> grid coords under camera c at u"""
    M = cam_M(u, c)
    px, py = M @ np.array([x * 720, y * 1280, 1.0])
    return px / 2, py / 2


def to_grid(view, gamma=0.78, contrast=1.25, dither=0.12, lift=0.0, tint=None):
    sm = cv2.resize(view, (GW, GH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sm = np.clip(sm, 0, 1) ** gamma
    if tint is not None:
        lum = sm.mean(2, keepdims=True)
        sm = sm * (1 - tint[1]) + lum * np.array(tint[0], np.float32) * tint[1]
    return quantize((sm * 255).astype(np.uint8), dither=dither, contrast=contrast, lift=lift)


# ------------------------------------------------------------------ glyphs (procedural Voynich-like)
def glyph_sprite(seed, size=120):
    rng = np.random.default_rng(seed)
    im = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(im)
    w = max(3, size // 16)
    x = size * 0.2
    for k in range(rng.integers(2, 4)):
        kind = rng.integers(0, 4)
        if kind == 0:   # gallows: tall stroke with loop
            d.line([x, size * .15, x, size * .85], fill=255, width=w)
            d.arc([x - size * .05, size * .1, x + size * .25, size * .45], 200, 520, fill=255, width=w)
            x += size * .3
        elif kind == 1:  # o
            d.ellipse([x, size * .45, x + size * .22, size * .78], outline=255, width=w)
            x += size * .27
        elif kind == 2:  # 9-like
            d.ellipse([x, size * .35, x + size * .2, size * .6], outline=255, width=w)
            d.line([x + size * .2, size * .48, x + size * .14, size * .9], fill=255, width=w)
            x += size * .26
        else:           # c with tail
            d.arc([x, size * .4, x + size * .24, size * .8], 60, 300, fill=255, width=w)
            d.line([x + size * .02, size * .62, x - size * .06, size * .9], fill=255, width=w)
            x += size * .28
    a = np.array(im)
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return a


def stamp_glyph(g, seed, cx, cy, h, color='bone'):
    a = glyph_sprite(seed)
    sc = h / a.shape[0]
    a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(h))), interpolation=cv2.INTER_AREA) > 90
    y0, x0 = int(cy - a.shape[0] / 2), int(cx - a.shape[1] / 2)
    ys, xs = np.where(a)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < GH) & (xs >= 0) & (xs < GW)
    g[ys[ok], xs[ok]] = P[color]
    return g


# ------------------------------------------------------------------ effects on the grid (uint8 BGR GHxGWx3)
def tear(g, rng, bands=4, maxshift=24, hmin=3, hmax=18):
    out = g.copy()
    for _ in range(bands):
        y = int(rng.integers(0, GH)); h = int(rng.integers(hmin, hmax)); s = int(rng.integers(-maxshift, maxshift + 1))
        out[y:y + h] = np.roll(g[y:y + h], s, axis=1)
    return out


def rgbsplit(g, dx=2):
    out = g.copy()
    out[:, :, 2] = np.roll(g[:, :, 2], dx, axis=1)
    out[:, :, 0] = np.roll(g[:, :, 0], -dx, axis=1)
    return out


def pixsort(g, thr=0.45, frac=1.0, up=False, x0=0, x1=GW):
    lum = g.astype(np.float32).mean(2) / 255
    out = g.copy()
    ylim = int(GH * frac)
    for x in range(x0, x1):
        col = lum[:ylim, x]
        m = col > thr
        if not m.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(np.int8), [0]])))
        for a, b in zip(idx[::2], idx[1::2]):
            seg = out[a:b, x]
            order = np.argsort(lum[a:b, x])
            if not up:
                order = order[::-1]
            out[a:b, x] = seg[order]
    return out


def genloss(g, block=2, dither=0.2):
    sm = cv2.resize(g, (GW // block, GH // block), interpolation=cv2.INTER_AREA)
    sm = quantize(sm, dither=dither, contrast=1.1)
    return cv2.resize(sm, (GW, GH), interpolation=cv2.INTER_NEAREST)


def blockmix(a, b, prog, seed=1, blk=8):
    """dithered 8x8-cell wipe from a to b, prog 0..1"""
    rng = np.random.default_rng(seed)
    nb_y, nb_x = GH // blk, GW // blk + 1
    order = rng.random((nb_y + 1, nb_x))
    m = (order < prog)
    m = np.kron(m, np.ones((blk, blk), bool))[:GH, :GW]
    out = a.copy(); out[m] = b[m]
    return out


def melt(a, b, prog, seed=2):
    """datamosh-like: columns of a slide down and smear, revealing b"""
    rng = np.random.default_rng(seed)
    sp = rng.uniform(0.6, 1.6, GW)
    out = b.copy()
    for x in range(GW):
        d = int(prog * sp[x] * GH)
        if d < GH:
            out[d:, x] = a[:GH - d, x]
            if d > 0:
                out[max(0, d - 6):d, x] = a[0, x]
    return out


def pixelate(g, blk):
    blk = max(1, int(blk))
    if blk == 1:
        return g
    sm = cv2.resize(g, (max(1, GW // blk), max(1, GH // blk)), interpolation=cv2.INTER_AREA)
    return cv2.resize(sm, (GW, GH), interpolation=cv2.INTER_NEAREST)


def grow(a, b, prog):
    """pixels swell into huge blocks, the cut happens inside the blocks, they shrink back"""
    if prog < .5:
        return pixelate(a, 2 ** (prog / .5 * 4.3))
    return pixelate(b, 2 ** ((1 - prog) / .5 * 4.3))


def swirl_g(g, amt, cx=GW / 2, cy=GH * .45, rad=200.0):
    yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    ang = amt * np.clip(1 - r / rad, 0, 1) ** 2
    mx = cx + dx * np.cos(ang) - dy * np.sin(ang)
    my = cy + dx * np.sin(ang) + dy * np.cos(ang)
    return cv2.remap(g, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def swirl(a, b, prog, seed=3):
    """the old frame is wound into a whirlpool, the new one unwinds out of it"""
    k = math.sin(math.pi * prog)
    if prog < .5:
        g = swirl_g(a, k * 9)
    else:
        g = swirl_g(b, -k * 9)
    if .35 < prog < .65:
        g = blockmix(swirl_g(a, k * 9), g, (prog - .35) / .3, seed=seed, blk=4)
    return pixelate(g, 1 + int(k * 3))


def scatter(a, b, prog, seed=4):
    """2x2 blocks of the old frame fly outward from the centre and burn out, revealing the new one"""
    rng = np.random.default_rng(seed)
    out = b.copy()
    ys, xs = np.mgrid[0:GH:2, 0:GW:2]
    ys, xs = ys.ravel(), xs.ravel()
    vx, vy = xs - GW / 2 + rng.normal(0, 18, xs.size), ys - GH * .45 + rng.normal(0, 18, ys.size)
    life = rng.uniform(.25, 1.0, xs.size)
    alive = prog < life
    e = prog ** 2 * 2.2
    nx, ny = (xs + vx * e).astype(int), (ys + vy * e).astype(int)
    ok = alive & (nx >= 0) & (nx < GW - 1) & (ny >= 0) & (ny < GH - 1)
    for oy in (0, 1):
        for ox in (0, 1):
            out[ny[ok] + oy, nx[ok] + ox] = a[ys[ok] + oy, xs[ok] + ox]
    return out


def dissolve(g, mask, amount, seed, bg=None, up=True):
    """approved figure break-up: 2x2 blocks on the right side coarsen / detach and drift"""
    if mask is None or amount <= 0 or not mask.any():
        return g
    rng = np.random.default_rng(seed)
    out = g.copy()
    ys, xs = np.where(mask)
    x_lo, x_hi = xs.min(), xs.max()
    cut = x_lo + (x_hi - x_lo) * (1 - amount)
    for y in range(ys.min() // 2 * 2, ys.max() + 1, 2):
        for x in range(int(cut) // 2 * 2, x_hi + 1, 2):
            if not mask[y:y + 2, x:x + 2].any():
                continue
            e = (x - cut) / max(1.0, x_hi - cut)
            r = rng.random()
            if r < e * 0.9:
                c = g[y:y + 2, x:x + 2].reshape(-1, 3).mean(0)
                if r < e * 0.45:
                    out[y:y + 2, x:x + 2] = bg[y:y + 2, x:x + 2] if bg is not None else P['void2']
                    dy = -int(rng.integers(2, 26)) if up else int(rng.integers(2, 26))
                    dx = int(rng.integers(0, 12))
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < GH - 2 and nx < GW - 2:
                        out[ny:ny + 2, nx:nx + 2] = c
                else:
                    out[y:y + 2, x:x + 2] = c
    return out


def lummask(g, thr):
    m = g.astype(np.float32).mean(2) > thr
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return m


# ------------------------------------------------------------------ UI
def draw_ui(rgb, t, ui):
    if not ui:
        return rgb
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil); d.fontmode = '1'
    if ui.get('guides', True):
        gy = ui.get('gy', 100)
        gx = ui.get('gx', GW // 2)
        d.line([gx, 18, gx, GH - 60], fill=RGB['dark'])
        d.line([6, gy, GW - 6, gy], fill=RGB['dark'])
        for px in (6, gx, GW - 7):
            d.rectangle([px - 1, gy - 1, px + 1, gy + 1], fill=RGB['red'])
    if ui.get('hud', True):
        d.text((8, 22), ui.get('surface', 'SURFACE_01'), font=F9, fill=RGB['line'])
        if int(t * 7) % 11:
            d.text((8, 30), ui.get('signal', 'SIGNAL --'), font=F9, fill=RGB['tile2'])
        d.text((GW - 50, 22), f'{ui.get("c0", 412) + int(t * 30):04d}', font=F9, fill=RGB['tile2'])
    for box in ui.get('boxes', []):
        r = box['r'](t) if callable(box['r']) else box['r']
        if r is None:
            continue
        x0, y0, x1, y1 = r
        col = RGB[box.get('col', 'bone')]
        if box.get('t0', -1) > t or box.get('t1', 1e9) < t:
            continue
        d.rectangle([x0, y0, x1, y1], outline=col)
        if box.get('corners', True):
            for (cx, cy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                d.rectangle([cx - 1, cy - 1, cx + 1, cy + 1], fill=col)
        lab = box.get('label')
        lab = lab(t) if callable(lab) else lab
        if lab:
            if box.get('plate', True) and box.get('col', 'bone') == 'bone':
                lw = int(d.textlength(lab, font=F9)) + 4
                d.rectangle([x0, y0 - 10, x0 + lw, y0 - 1], fill=col)
                d.text((x0 + 2, y0 - 10), lab, font=F9, fill=RGB['void'])
                if box.get('conf'):
                    d.text((x0 + lw + 3, y0 - 10), box['conf'], font=F9, fill=RGB['line'])
            else:
                d.text((x0, y1 + 2), lab, font=F9, fill=col)
                if box.get('sub'):
                    d.text((x0, y1 + 11), box['sub'](t) if callable(box['sub']) else box['sub'], font=F9, fill=col)
    return np.array(pil)


def card(lines, font=FBIG, color='bone', bg='void', flip=False, red_line=None):
    pil = Image.new('RGB', (GW, GH), RGB[bg])
    d = ImageDraw.Draw(pil); d.fontmode = '1'
    hs = [font.getbbox(l)[3] for l in lines]
    y = (GH - sum(hs) - 8 * (len(lines) - 1)) // 2
    for i, l in enumerate(lines):
        w = d.textlength(l, font=font)
        col = RGB['red'] if red_line == i else RGB[color]
        d.text(((GW - w) // 2, y), l, font=font, fill=col)
        y += hs[i] + 8
    a = np.array(pil)
    if flip:
        a = a[::-1]
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


# ================================================================== the film
def faces():
    g = still('s10_grid.jpg')
    out = []
    for r in range(3):
        for c in range(2):
            cx = 178 if c == 0 else 544
            y0 = r * 427 + 8
            out.append(g[y0:y0 + 358, max(0, cx - 100):cx + 100])
    p = still('s17_profile.jpg'); out.append(p[300:1000, 120:513])
    w = still('s21_woman.jpg'); out.append(w[320:900, 180:506])
    return [cv2.resize(f, (720, 1280), interpolation=cv2.INTER_AREA) for f in out]


@functools.lru_cache(None)
def face_grids():
    return [to_grid(cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA), gamma=.8) for f in faces()]


def grid3x3(cells, t, glitch=0.0, rng=None):
    out = np.zeros((GH, GW, 3), np.uint8); out[:] = P['void']
    cw, ch = 58, 104
    k = 0
    for r in range(3):
        for c in range(3):
            x0, y0 = 2 + c * 60, 2 + r * 106
            cell = cells[k] if k < len(cells) else None
            k += 1
            if cell is None:
                continue
            sm = cv2.resize(cell, (cw, ch), interpolation=cv2.INTER_AREA)
            if rng is not None and glitch > 0 and rng.random() < glitch:
                sm = np.roll(sm, int(rng.integers(-8, 9)), axis=1)
            out[y0:y0 + ch, x0:x0 + cw] = sm
    return out


def flash(color='bone'):
    g = np.zeros((GH, GW, 3), np.uint8); g[:] = P[color]
    return g


def black():
    g = np.zeros((GH, GW, 3), np.uint8); g[:] = P['void']
    return g


def subl(name):
    return to_grid(cam(still(name), 0, dict(z0=1)), gamma=.8)


def bt(n):
    """n beats after the start of the shot, rounded to the 15 fps sprite clock"""
    return round(n * sync.PULSE * 15) / 15


class Shot:
    def __init__(self, t0, t1, fn, trans=None, ui=None, clean=False, toff=0.0):
        self.t0, self.t1, self.fn, self.trans, self.ui, self.clean, self.toff = t0, t1, fn, trans, ui, clean, toff


def still_fn(name, c, gamma=.78, contrast=1.25, dither=.12, fx=(), tint=None):
    def fn(tl, u, rng, tq):
        g = to_grid(cam(still(name), u, c), gamma=gamma, contrast=contrast, dither=dither, tint=tint)
        for f in fx:
            g = f(g, tl, u, rng)
        return g
    return fn


def video_fn(path, c, speed=1.0, offset=0.0, gamma=.78, contrast=1.25, fx=(), fallback=None):
    def fn(tl, u, rng, tq):
        if not os.path.exists(path):
            return fallback(tl, u, rng, tq)
        frames, fps = video(path)
        i = min(len(frames) - 1, int((offset + tl * speed) * fps))
        g = to_grid(cam(frames[i], u, c), gamma=gamma, contrast=contrast)
        for f in fx:
            g = f(g, tl, u, rng)
        return g
    return fn


# ---- small fx builders
def fx_when(a, b, f):
    def w(g, tl, u, rng):
        return f(g, tl, u, rng) if a <= tl < b else g
    return w


def fx_tear(bands=4, maxshift=20, p=1.0):
    return lambda g, tl, u, rng: tear(g, rng, bands, maxshift) if rng.random() < p else g


def fx_rgb(dx=2, p=1.0):
    return lambda g, tl, u, rng: rgbsplit(g, dx) if rng.random() < p else g


def fx_sort(thr=.5, frac=1.0, up=False):
    return lambda g, tl, u, rng: pixsort(g, thr, frac, up)


def fx_loss(block=2):
    return lambda g, tl, u, rng: genloss(g, block)


def fx_dissolve(thr, amt, up=True):
    return lambda g, tl, u, rng: dissolve(g, lummask(g, thr), amt, int(tl * 5), up=up)


# ---- captions: word by word, exactly when each word is spoken (edit/words.json from align.py)
import json
sys.path.insert(0, f'{R}/edit')
_W = json.load(open(f'{R}/edit/words.json'))
_LINES = {}
for _w in _W:
    _LINES.setdefault(_w['line'], []).append(_w)
_LINES = [_LINES[k] for k in sorted(_LINES)]
LEAD = 0.12          # a word lands on screen a hair before its first sound (reads as perfectly in sync)
_MEAS = ImageDraw.Draw(Image.new('RGB', (8, 8)))


def _fits(words):
    """True when these words wrap into at most 2 caption lines at the normal size"""
    lines, cur = 1, 0.0
    sp = _MEAS.textlength(' ', font=FCAP)
    for w in words:
        wl = _MEAS.textlength(w, font=FCAP)
        if cur and cur + sp + wl > GW - 12:
            lines += 1; cur = wl
        else:
            cur = cur + sp + wl if cur else wl
    return lines <= 2


def _split(idx, ws):
    """split a phrase into caption chunks: never in the middle of a thought, never leaving one orphan word"""
    words = [ws[i]['w'] for i in idx]
    if _fits(words) or len(idx) < 2:
        return [idx]
    total = sum(len(w) for w in words)
    best, bestk = None, None
    for k in range(1, len(idx)):
        left = sum(len(w) for w in words[:k])
        score = abs(left - total / 2) - (12 if words[k - 1].endswith(',') else 0)
        if min(k, len(idx) - k) < 2:
            score += 8                       # avoid a lonely word
        if not (_fits(words[:k]) and _fits(words[k:])):
            score += 100                     # prefer exactly two readable chunks
        if best is None or score < best:
            best, bestk = score, k
    return _split(idx[:bestk], ws) + _split(idx[bestk:], ws)


def _chunks(ws):
    groups, cur = [], []
    for i, w in enumerate(ws):
        cur.append(i)
        if w['w'][-1] in '….!?' and i != len(ws) - 1:
            groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        out += _split(g, ws)
    for c, idx in enumerate(out):
        for i in idx:
            ws[i]['chunk'] = c


for _ws in _LINES:
    _chunks(_ws)


CHUNK_LEAD = 0.20     # the whole phrase-chunk is already on screen 0.2 s before its first word
WORD_LEAD = 0.03      # each word lights up together with its first sound


def caption_at(t):
    for i, ws in enumerate(_LINES):
        start = ws[0]['t0'] - CHUNK_LEAD
        end = ws[-1]['t1'] + 0.9
        if i + 1 < len(_LINES):
            end = min(end, _LINES[i + 1][0]['t0'] - CHUNK_LEAD - 0.04)
        if start <= t < end:
            heads, tails = {}, {}
            for w in ws:
                heads.setdefault(w['chunk'], w['t0'])
                tails[w['chunk']] = w['t1']
            def shows_from(c):                   # next chunk waits until the previous one has been said
                h = heads[c] - CHUNK_LEAD
                if c - 1 in tails:
                    h = max(h, min(heads[c] - 0.03, tails[c - 1] + 0.08))
                return h
            ch = max([c for c in heads if shows_from(c) <= t] or [min(heads)])
            items = []
            for w in ws:
                if w['chunk'] == ch:
                    items.append((w['w'], w['key'], w['t0'] - WORD_LEAD <= t, 0 <= t - w['t0'] + WORD_LEAD < 2 / FPS))
            return items, ws[0]['voice']
    return None


def draw_caption(rgb, cap, t):
    """cap = ([(word, is_key, visible, fresh)], voice). Layout uses the whole chunk so words never jump."""
    if not cap:
        return rgb
    items, voice = cap
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil); d.fontmode = '1'
    font = FCAP
    def width(ws, f):
        return sum(d.textlength(w[0] + ' ', font=f) for w in ws)
    def wrap(f, lim):
        lines, cur = [], []
        for w in items:
            if cur and width(cur + [w], f) > lim:
                lines.append(cur); cur = []
            cur.append(w)
        if cur:
            lines.append(cur)
        return lines
    lines = wrap(font, GW - 12)
    if len(lines) > 2:
        font = FCAP_S
        lines = wrap(font, GW - 10)
    base = RGB['g3'] if voice == 'whisper' else RGB['bone']
    lh = 19
    y = GH - 44 - (len(lines) - 1) * lh
    for ln in lines:
        tw = width(ln, font) - d.textlength(' ', font=font)
        x = int((GW - tw) / 2)
        for w, is_key, said, fresh in ln:
            ww = d.textlength(w, font=font)
            yy = y - 1 if fresh else y
            if is_key:
                d.rectangle([x - 2, yy + 1, x + ww + 2, yy + 17], fill=RGB['red'] if said else RGB['red2'])
                d.text((x, yy), w, font=font, fill=RGB['bone'] if said else RGB['line'])
            else:
                for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
                    d.text((x + ox, yy + oy), w, font=font, fill=RGB['void'])
                d.text((x, yy), w, font=font, fill=base if said else RGB['mid'])
            x += ww + d.textlength(' ', font=font)
        y += lh
    return np.array(pil)


# ---- shots
def build():
    SH = []
    A = SH.append
    BENCH = f'{R}/clips/shot_bench_pix_nocap.mp4'
    V35 = f'{R}/clips/v35_whisperer.mp4'
    V21 = f'{R}/clips/v21_pools.mp4'

    # 1  0.0-1.0 black, tear, glyph
    def s1(tl, u, rng, tq):
        g = black()
        if 0.40 <= tl < 0.56:
            g = stamp_glyph(g, 7, GW / 2, GH / 2, 150)
            g = rgbsplit(g, 3)
            if tl > 0.48:
                g = pixsort(g, .3, 1.0)
        elif tl >= 0.56:                 # after the drop the glyph does not vanish: it bursts into rising pixels
            g = stamp_glyph(g, 7, GW / 2, GH / 2, 150)
            k = min(1.0, (tl - 0.56) / 0.45)
            g = dissolve(g, lummask(g, 60), 0.35 + 0.65 * k, int(tl * 15))
            g = (g.astype(np.float32) * (1 - 0.5 * k)).astype(np.uint8)
            return quantize(g, dither=.1, contrast=1.0)
        if tl < 0.4 or tl > 0.85:
            y = int(GH / 2 + rng.integers(-3, 4))
            g[y:y + 1, :] = P['bone'] if rng.random() < .5 else P['line']
            g[y, :int(rng.integers(20, GW))] = P['void2']
        return g
    A(Shot(0.0, 1.0, s1))
    # 2 parchment macro
    A(Shot(1.0, 2.5, still_fn('s02_parchment.jpg', dict(z0=2.1, z1=2.35, x0=.42, y0=.46), fx=(fx_sort(.62, .35),)), trans=('tear', 3)))
    # 3 eye
    def s3(tl, u, rng, tq):
        g = still_fn('s03_eye.jpg', dict(z0=1.3, z1=1.36, x0=.5, y0=.52), gamma=.82, fx=(fx_when(0, 9, fx_sort(.55, .55)),))(tl, u, rng, tq)
        if tl > 0.62:      # lid falls in square pixels
            k = int((tl - 0.62) / 0.26 * GH)
            g[:k] = P['void']
            g[max(0, k - 10):k, ::2] = P['dark']
        return g
    A(Shot(2.5, 3.38, s3, trans=('tear', 3)))
    # 4 museum
    museum_c = dict(z0=1.0, z1=1.16, x0=.5, y0=.56)
    A(Shot(3.38, 5.08, still_fn('s04_museum.jpg', museum_c, gamma=.75), trans=('grow', 5),
           ui=dict(surface='SURFACE_04', boxes=[dict(r=(63, 152, 132, 210), label='MS 408', col='red', sub='LANG: ---'),
                                              dict(r=(16, 120, 60, 230), label='SUBJECT_01', conf='.61')])))
    # 5 strobe pages
    pages = [('s02_parchment.jpg', dict(z0=1.6, x0=.35, y0=.5)), ('s11_wall.jpg', dict(z0=2.2, x0=.48, y0=.47)),
             ('s06_hand.jpg', dict(z0=1.7, x0=.7, y0=.62)), ('s23_scribe.jpg', dict(z0=1.8, x0=.45, y0=.5)),
             ('s27_sleep.jpg', dict(z0=1.5, x0=.3, y0=.3)), ('s16_hourglass.jpg', dict(z0=1.7, x0=.5, y0=.78))]
    def s5(tl, u, rng, tq):
        k = int(tl * FPS)
        if k % 4 == 3:
            return flash('bone' if k % 8 == 3 else 'g2')
        n, c = pages[(k // 4) % len(pages)]
        g = to_grid(cam(still(n), 0, c), gamma=.8)
        return melt(g, black(), 0.15 * (k % 4), seed=k)
    A(Shot(5.08, 6.26, s5))
    # 6 hand reaching: freeze + rgb + dissolve hand
    def s6(tl, u, rng, tq):
        c = dict(z0=1.0, z1=1.14, x0=.56, y0=.52)
        uu = min(u, 0.62)
        g = to_grid(cam(still('s06_hand.jpg'), uu, c), gamma=.8)
        if u >= 0.62:
            g = rgbsplit(g, 3)
            m = lummask(g, 120)
            m[:, :60] = False
            g = dissolve(g, m, min(1, (u - .62) / .38 * 1.4), int(tl * 15), up=False)
        return g
    A(Shot(6.26, 8.05, s6, trans=('tear', 2)))
    # 7 НЕ ЧИТАЙ
    def s7(tl, u, rng, tq):
        if tl < bt(1):
            return card(['НЕ', 'ЧИТАЙ'], red_line=1)
        if tl < bt(1) + 0.33:
            return card(['ЙАТИЧ', 'ЕН'], red_line=0, flip=True)
        g = card(['НЕ', 'ЧИТАЙ'], red_line=1)
        return tear(g, rng, 3, 30) if rng.random() < .5 else g
    A(Shot(8.05, 9.12, s7))
    # 8 archive photo + subliminal
    def s8(tl, u, rng, tq):
        if bt(2) <= tl < bt(2) + 1 / 15:
            return subl('s35_whisperer.jpg')
        return still_fn('s08_archive.jpg', dict(z0=1.0, z1=1.2, x0=.55, y0=.38), gamma=.85,
                        fx=(fx_when(0, 9, fx_tear(1, 6, .15)),))(tl, u, rng, tq)
    A(Shot(9.12, 10.90, s8, trans=('scatter', 6),
           ui=dict(surface='ARCHIVE_1912', boxes=[dict(r=(80, 70, 128, 135), label='UNKNOWN', conf='.12')])))
    # 9 tag
    A(Shot(10.90, 12.00, still_fn('s09_tag.jpg', dict(z0=1.25, z1=1.32, x0=.52, y0=.5), fx=(fx_tear(3, 8, .6), fx_rgb(1, .5))),
           trans=('tear', 2), ui=dict(surface='LOT_???', boxes=[dict(r=(50, 130, 150, 230), label='LOT', col='red', sub=lambda t: f'{np.random.default_rng(int(t*15)).integers(100,999)}')])))
    # 10 readers through the ages
    def s10(tl, u, rng, tq):
        fg = face_grids()
        k = int(tl / (sync.PULSE / 2))        # a new face on every half-beat
        if k >= len(fg):
            g = black()
            if (int(tl * FPS) // 3) % 2:
                g = cv2.addWeighted(g, 1, (fg[6] * 0.18).astype(np.uint8), 1, 0)
            return quantize(g, dither=.1, contrast=1.0)
        g = fg[k].copy()
        if k >= 3:
            g = genloss(g, 2 if k < 6 else 3)
        if k >= 5:
            g = pixsort(g, .4, .7)
        return g
    A(Shot(12.00, 14.32, s10, trans=('tear', 2), ui=dict(surface='READER_??', guides=False)))
    # 11 decipherer wall whip
    def s11(tl, u, rng, tq):
        ramp = u ** 0.35 if u < .5 else 0.785 + (u - .5) ** 2.2 * 1.0
        g = to_grid(cam(still('s11_wall.jpg'), min(1, ramp), dict(z0=1.7, z1=1.9, x0=.32, y0=.45, x1=.68, y1=.5, lin=True)), gamma=.8)
        if rng.random() < .55:
            g = tear(g, rng, 4, 26)
        if rng.random() < .3:
            g = rgbsplit(g, 2)
        return g
    A(Shot(14.32, 16.34, s11, trans=('tear', 2)))
    # 12 dive into the page
    def s12(tl, u, rng, tq):
        z = 1.0 * (9.0 ** u)
        g = to_grid(cam(still('s02_parchment.jpg'), 0, dict(z0=z, x0=.7, y0=.52)), gamma=.82)
        if 0.23 <= tl < 0.23 + 2 / FPS:
            return flash('bone')
        if u > .5:
            g = pixsort(g, .35, 1.0)
        return g
    A(Shot(16.34, 17.25, s12, trans=('swirl', 6)))
    # 13 3x3 heads turn
    def s13(tl, u, rng, tq):
        fg = face_grids()
        cells = fg[:4] + [None] + fg[4:8]
        g = grid3x3(cells, tl, .25 if tl < .6 else .05, rng)
        if tl > .6:       # the creak: the grid is wound into a slow whirlpool and sinks into the dark
            k = min(1.0, (tl - .6) / 1.25)
            g = swirl_g(g, k ** 1.5 * 10, GW / 2, GH * .5, 230)
            g = pixelate(g, 1 + int(k * 6))
            g = (g.astype(np.float32) * (1 - k ** 1.4)).astype(np.uint8)
            g = quantize(g, dither=.15, contrast=1.0)
        if tl > 1.85:
            g = black()
        return g
    A(Shot(17.25, 20.00, s13, trans=('tear', 2)))

    # ---------------- ACT II
    def s15(tl, u, rng, tq):
        g = still_fn('s15_reading.jpg', dict(z0=1.0, z1=1.12, x0=.47, y0=.6), gamma=.8)(tl, u, rng, tq)
        if tl < 1.5:
            g = blockmix(black(), g, tl / 1.5, seed=15, blk=4)
        return g
    A(Shot(20.00, 23.00, s15, ui=dict(surface='ROOM_03', hud=True, guides=False,
                                       boxes=[dict(r=(18, 190, 70, 280), label='EMPTY', conf='.98', t0=1.6)])))
    A(Shot(23.00, 28.57, still_fn('s16_hourglass.jpg', dict(z0=1.12, z1=1.3, x0=.46, y0=.5, x1=.54, y1=.52, r0=-2, r1=2.5)),
           trans=('grow', 10), ui=dict(surface='T_1404', boxes=[dict(r=(56, 92, 124, 235), label='T: ????', col='red')])))
    def s17(tl, u, rng, tq):
        c = dict(z0=1.05, z1=1.13, x0=.5, y0=.5)
        if bt(4) <= tl < bt(4) + 1 / 15:
            return subl('s35_whisperer.jpg')
        g = to_grid(cam(still('s17_profile.jpg'), u, c), gamma=.8)
        mx, my = map_pt(u, c, .56, .56)
        if tl > bt(3):
            for j in range(6):
                age = (tl - bt(3)) - j * sync.PULSE / 2
                if age < 0:
                    continue
                gx, gy = mx + 8 + age * 30, my - 4 - age * 14 + math.sin(age * 3 + j) * 3
                h = 18 + age * 8
                if gx < GW - 6:
                    g = stamp_glyph(g, 20 + j, gx, gy, h, 'bone' if age < .9 else 'mid')
        return dissolve(g, lummask(g, 55), .22, int(tl * 5))
    A(Shot(28.57, 32.30, s17, trans=('swirl', 12), ui=dict(surface='SUBJECT_01', guides=False,
                                                         boxes=[dict(r=(30, 70, 110, 190), label='SUBJECT_01', conf='.88')])))
    def s18(tl, u, rng, tq):
        g = to_grid(cam(still('s02_parchment.jpg'), 0, dict(z0=2.9, x0=.3, y0=.52)), gamma=.85, dither=.05)
        if bt(5) <= tl < bt(5) + 1 / 15:
            return black()
        if tl >= bt(5):
            rr = np.random.default_rng(18)
            bands = [g[i:i + 16] for i in range(0, GH, 16)]
            order = rr.permutation(len(bands))
            g = np.concatenate([bands[i] for i in order])[:GH]
        return g
    A(Shot(32.30, 37.16, s18, trans=('scatter', 10)))
    def s19(tl, u, rng, tq):
        g = still_fn('s19_water.jpg', dict(z0=1.18, z1=1.24, r0=0, r1=7, x0=.5, y0=.48), gamma=.82)(tl, u, rng, tq)
        return pixsort(g, .5, 1.0, up=True) if u > .3 else g
    A(Shot(37.16, 41.50, s19, trans=('melt', 12), ui=dict(surface='SURFACE_19', guides=False,
                                                         boxes=[dict(r=(55, 120, 120, 200), label='PLANT_?', col='red', sub='f.33v')])))
    pool_c = dict(z0=1.7, z1=1.05, x0=.5, y0=.66, x1=.5, y1=.52)
    def s20(tl, u, rng, tq):
        return to_grid(cam(still('s20_pools.jpg'), u, pool_c), gamma=.85)
    def pool_boxes():
        bx = []
        for i, (x, y) in enumerate([(.38, .72), (.5, .73), (.62, .69), (.45, .8)]):
            bx.append(dict(pt=(x, y), label=f'F_{i + 1:02d}'))
        return bx
    A(Shot(41.50, 45.73, s20, trans=('grow', 10), ui=dict(surface='POOL_20', guides=False, track=('pool', pool_c, pool_boxes()))))
    s21_still = still_fn('s21_woman.jpg', dict(z0=1.0, z1=1.1, x0=.45, y0=.45), gamma=.85)
    A(Shot(45.73, 49.00, video_fn(V21, dict(z0=1.0, z1=1.06), speed=0.93, fx=(fx_tear(2, 10, .25),), fallback=s21_still),
           trans=('scatter', 10), ui=dict(surface='POOL_21', boxes=[dict(r=(60, 120, 120, 250), label='SUBJECT_07', conf='.43')])))
    def s22(tl, u, rng, tq):
        g = still_fn('s22_underwater.jpg', dict(z0=1.28, z1=1.0, x0=.5, y0=.45))(tl, u, rng, tq)
        return genloss(g, 2 if u < .5 else 3)
    A(Shot(54.30, 58.20, s22, trans=('melt', 12)))
    def s23(tl, u, rng, tq):
        g = still_fn('s23_scribe.jpg', dict(z0=1.0, z1=1.14, x0=.45, y0=.53))(tl, u, rng, tq)
        if tl > bt(5):
            m = np.zeros((GH, GW), bool); m[150:230, 90:150] = True
            g = dissolve(g, m & lummask(g, 70), .8, int(tl * 5))
            g = pixsort(g, .55, 1.0, up=True)
        return g
    A(Shot(49.00, 54.30, s23, trans=('swirl', 14), ui=dict(surface='SCRIBE_1438', boxes=[dict(r=(50, 140, 110, 210), label='SUBJECT_01', conf='.97', t0=1.0)])))
    def s24(tl, u, rng, tq):
        g = to_grid(cam(still('s25_handprint.jpg'), 0, dict(z0=1.0)), gamma=.8)
        k = int(u * GH * 1.1)
        water = to_grid(cam(still('s19_water.jpg'), 0, dict(z0=1.3, r0=20)), gamma=.9)
        g[:k] = water[:k]
        return pixsort(g, .3, min(1, u * 1.2))
    A(Shot(58.20, 59.00, s24))
    def s25(tl, u, rng, tq):
        g = still_fn('s25_handprint.jpg', dict(z0=1.0, z1=1.2, x0=.5, y0=.45), gamma=.8)(tl, u, rng, tq)
        if int(tl * FPS) % 37 in (0, 1):
            g = rgbsplit(g, 2)
        return g
    A(Shot(61.65, 67.00, s25, trans=('grow', 8), ui=dict(surface='GLASS_INNER', boxes=[dict(r=(60, 90, 120, 160), label='PRINT', col='red', sub='SIDE: IN')])))
    def s26(tl, u, rng, tq):
        g = to_grid(cam(still('s26_glass.jpg'), u, dict(z0=1.0, z1=1.08, x0=.5, y0=.45)), gamma=.82)
        m = lummask(g, 60); m[:, :70] = False
        sh = int(u * 10)
        if sh:
            moved = np.roll(np.where(m[..., None], g, 0), -sh, axis=1)
            g = np.where(np.roll(m, -sh, axis=1)[..., None], moved, g)
        return dissolve(g, m, .3 + u * .3, int(tl * 5))
    A(Shot(59.00, 61.65, s26, trans=('scatter', 10)))
    def s27(tl, u, rng, tq):
        # loop-jitter: every 2s jump back 1s
        per = 4 * sync.PULSE                  # loop-jitter: every bar jump back half a bar
        tt = tl - (int(tl // per) * per / 2)
        uu = tt / (71.44 - 67.0)
        return still_fn('s27_sleep.jpg', dict(z0=1.12, z1=1.25, r0=0, r1=9, x0=.5, y0=.42), gamma=.82)(tt, uu, rng, tq)
    A(Shot(67.00, 71.44, s27, trans=('melt', 12)))
    def s28(tl, u, rng, tq):
        rep = int(tl / (2 * sync.PULSE))
        tt = tl % (2 * sync.PULSE)
        g = to_grid(cam(still('s27_sleep.jpg'), tt / (2 * sync.PULSE) * .3, dict(z0=1.25, z1=1.3, r0=9, r1=10, x0=.5, y0=.42)), gamma=.82)
        g = genloss(g, 2 + rep)
        if rep >= 2:
            corner = to_grid(cam(still('s28_corner.jpg'), 0, dict(z0=1.1, x0=.55, y0=.45)), gamma=.85)
            m = np.zeros((GH, GW), bool); m[20:200, 100:180] = True
            g[m] = corner[m]
            if tl > 3.3:
                g = tear(g, rng, 8, 40)
        return g
    A(Shot(71.44, 75.00, s28, trans=('grow', 6)))

    # ---------------- ACT III
    def s29(tl, u, rng, tq):
        g = still_fn('s29_bookopen.jpg', dict(z0=1.0, z1=1.16, x0=.5, y0=.52), gamma=.8)(tl, u, rng, tq)
        gt = pixsort(np.ascontiguousarray(g.transpose(1, 0, 2)), .35, 1.0)
        return np.ascontiguousarray(gt.transpose(1, 0, 2)) if u > .25 else g
    A(Shot(75.00, 79.10, s29, trans=('swirl', 12), ui=dict(surface='MS408', boxes=[dict(r=(30, 140, 150, 205), label='MS 408', col='red', sub='OPEN')])))
    def s30(tl, u, rng, tq):
        g = still_fn('s30_pov.jpg', dict(z0=1.05, z1=1.16, x0=.5, y0=.42), gamma=.82)(tl, u, rng, tq)
        if bt(3) <= tl < bt(3) + 2 / 15:
            g = g[:, ::-1].copy()
        return g
    A(Shot(80.30, 84.30, s30, trans=('scatter', 10), ui=dict(surface='INSIDE', boxes=[dict(r=(60, 108, 130, 140), label='READING', col='red', sub='YOU')])))
    def s31(tl, u, rng, tq):
        g = to_grid(cam(still('s31_lips.jpg'), 0, dict(z0=1.05, x0=.5, y0=.5)), gamma=.9, dither=.22)
        if rng.random() < .5:
            m = lummask(g, 60)
            g = np.where(m[..., None], np.roll(g, int(rng.integers(-1, 2)), axis=1), g)
        return g
    A(Shot(79.10, 80.30, s31, trans=('grow', 8)))
    A(Shot(84.30, 89.00, s31, trans=('grow', 8)))
    def s32(tl, u, rng, tq):
        if bt(1) <= tl < bt(1) + 1 / 15:
            return subl('s37_final.jpg')
        n = (np.random.default_rng(int(tl * 15)).random((GH, GW, 1)) * 40).astype(np.uint8)
        return quantize(np.repeat(n, 3, 2), dither=.2, contrast=1)
    A(Shot(89.00, 90.50, s32))
    def s33(tl, u, rng, tq):
        fr = gridvideo(BENCH)
        return fr[min(len(fr) - 1, 90 + int(tl * FPS))].copy()
    A(Shot(90.50, 97.50, s33, trans=('grow', 10), ui=None))
    def s34(tl, u, rng, tq):
        fg = face_grids()
        center = to_grid(cam(still('s30_pov.jpg'), 0, dict(z0=1.9, x0=.5, y0=.4)), gamma=.85)
        g = grid3x3(fg[:4] + [center] + fg[4:8], tl, .4, rng)
        if tl > 3 / FPS:
            water = flash('g2')
            g = blockmix(g, quantize(cv2.addWeighted(water, .7, g, .3, 0), dither=.2, contrast=1), (tl - .1) / 1.4, seed=34)
        return g
    A(Shot(97.50, 99.00, s34))
    s35_still = still_fn('s35_whisperer.jpg', dict(z0=1.0, z1=1.35, x0=.5, y0=.45), gamma=.82)
    def s35(tl, u, rng, tq):
        base = video_fn(V35, dict(z0=1.0, z1=1.12), speed=0.6, fallback=s35_still)(tl, u, rng, tq)
        g = base
        if tl > bt(4):   # droste: the frame inside her chest, a new level on every second beat
            lvl = min(4, 1 + int((tl - bt(4)) / (2 * sync.PULSE)))
            cx, cy, s = GW / 2, GH * .44, 0.34
            for L in range(lvl):
                w, h = int(GW * s ** (L + 1)), int(GH * s ** (L + 1))
                if w < 4:
                    break
                sm = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
                if L >= 1:
                    sm = rgbsplit(tear(sm, rng, 2, 4), 1)
                x0, y0 = int(cx - w / 2), int(cy - h / 2)
                g[y0:y0 + h, x0:x0 + w] = sm
        if tl > bt(8):
            g = tear(g, rng, 3 + int((tl - bt(8)) * 4), 18)
        return g
    A(Shot(99.00, 105.70, s35, trans=('swirl', 14), ui=dict(surface='SUBJECT_02', boxes=[dict(r=(40, 40, 140, 300), label='SUBJECT_02', conf='?.??')])))
    def s36(tl, u, rng, tq):
        if u > 0.985:
            return black()
        g = s35(6.7 + tl, 1, rng, tq)
        z = 1 + (u * 1.1) ** 3 * 30
        M = cv2.getRotationMatrix2D((GW / 2, GH * .44), u * 40, z)
        g = cv2.warpAffine(g, M, (GW, GH), flags=cv2.INTER_NEAREST, borderValue=tuple(int(c) for c in P['void']))
        return pixsort(tear(g, rng, 6, 30), .3, 1.0)
    A(Shot(105.70, 109.00, s36))
    def s37(tl, u, rng, tq):
        g = to_grid(cam(still('s37_final.jpg'), 0, dict(z0=1.0)), gamma=.8, dither=.08)
        if tl < 0.6:
            g = blockmix(black(), g, tl / 0.6, seed=37, blk=4)
        return g
    A(Shot(109.00, 117.50, s37, clean=True))
    def s38(tl, u, rng, tq):
        if tl < 1 / FPS:
            return to_grid(cam(still('s29_bookopen.jpg'), 0, dict(z0=1.0)), gamma=.8)
        if tl > 2.5 - 6 / FPS:      # loop: shot 1's opening line
            g = black(); y = GH // 2
            g[y, :int(rng.integers(20, GW))] = P['line']
            return g
        l1 = 'ты дочитал.' if bt(0.5) <= tl < bt(4) else ''
        l2 = 'она — нет.' if bt(1.5) <= tl < bt(3.5) else ''
        g = card([l1 or ' ', l2 or ' '], font=FMID, red_line=1)
        if bt(1.5) <= tl < bt(1.5) + 2 / 15:
            g = rgbsplit(g, 2)
        return g
    A(Shot(117.50, 120.00, s38, clean=True))
    # v2: everything moves 10 s later, the book intro takes 0.0-10.4 and hands over to shot 1's glyph on the drop
    for s_ in SH:
        s_.t0 += 10.0; s_.t1 += 10.0
    SH.sort(key=lambda s_: s_.t0)
    SH[0].t0, SH[0].toff = sync.DROP, 0.4
    import book
    SH.insert(0, Shot(0.0, sync.DROP, book.book_fn, ui=book.book_ui()))
    NAMES = ['book', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's15', 's16', 's17', 's18',
             's19', 's20', 'v21', 's23', 's22', 's24', 's26', 's25', 's27', 's28', 's29', 's31', 's30', 's31b', 's32', 's33', 's34', 's35',
             's36', 's37', 's38']
    assert len(SH) == len(NAMES), len(SH)
    for s_, nm in zip(SH, NAMES):
        s_.name = nm
    _snap_to_music(SH)
    # every cut gets its own transition, chosen for the moment; (kind, frames, frames before the beat)
    TR = {'s2': ('tear', 3, 0), 's3': ('zoom', 4, 1), 's4': ('grow', 6, 3), 's5': None, 's6': ('slice', 4, 1), 's7': ('flash', 3, 0),
          's8': ('scatter', 6, 2), 's9': ('tear', 3, 0), 's10': ('slice', 5, 1), 's11': ('zoom', 5, 2), 's12': ('flash', 3, 0),
          's13': ('glyph', 8, 3), 's16': ('sortwipe', 12, 6), 's17': ('swirl', 14, 7), 's18': ('glyph', 14, 7), 's19': ('melt', 12, 5),
          's20': ('iris', 12, 6), 'v21': ('scatter', 10, 5), 's23': ('shutter', 10, 5), 's22': ('melt', 14, 6), 's25': ('shutter', 8, 3),
          's26': ('iris', 10, 5), 's27': ('swirl', 14, 7), 's28': ('grow', 8, 4), 's29': ('glyph', 14, 7), 's31': ('slice', 6, 2),
          's30': ('zoom', 8, 3), 's31b': ('glyph', 12, 6), 's33': ('grow', 10, 5), 's34': ('tear', 3, 0), 's35': ('swirl', 14, 7)}
    for s_ in SH:
        if s_.name in TR:
            s_.trans = TR[s_.name]
    return SH


def _snap_to_music(SH):
    """move every cut onto the music of this version.  Priorities: 1) never put a cut inside a spoken word,
    2) every word stays on the picture it was written for, 3) land on a beat (big hit for long shots)."""
    words = [(w['t0'], w['t1'] + 0.07) for ws in _LINES for w in ws]       # +70 ms: the audible end of the word
    grid, strong = sync.grid(0, 131), sync.strong(0, 131)
    for i in range(2, len(SH)):
        b = SH[i].t0
        if SH[i].name in ('s30', 's12', 's13'):   # pinned: «с другой стороны» on the man behind the book; the creak
            continue
        prev_len, next_len = b - SH[i - 1].t0, SH[i].t1 - b
        lo, hi = SH[i - 1].t0 + 0.45, SH[i].t1 - 0.45
        for w0, w1 in words:
            if w1 <= b + 1e-6:
                lo = max(lo, w0 + 0.02)          # a word that was before the cut may not move after it
            elif w0 >= b - 1e-6:
                hi = min(hi, w1 - 0.02)
        major = prev_len >= 2.5 and next_len >= 2.5
        best, bx = None, b
        for x in [b] + [g for g in grid if abs(g - b) <= 0.6]:
            if not lo <= x <= hi:
                continue
            inside = any(w0 - 0.03 < x < w1 for w0, w1 in words)
            on_grid = any(abs(x - g) < 0.002 for g in grid)
            cost = abs(x - b) + (0 if on_grid else 0.45) + (0.8 if inside else 0)
            if major and sync.is_strong(x):
                cost -= 0.45
            if best is None or cost < best:
                best, bx = cost, x
        SH[i - 1].t1 = bx
        SH[i].t0 = bx
    for i in range(1, len(SH)):                  # cuts sit exactly on a frame: picture and kick land together
        q = round(SH[i].t0 * FPS) / FPS
        SH[i].t0 = q
        SH[i - 1].t1 = q


def knocks(shots):
    s12 = next(s_ for s_ in shots if s_.name == 's12')
    s13 = next(s_ for s_ in shots if s_.name == 's13')
    s15 = next(s_ for s_ in shots if s_.name == 's15')
    return [s12.t0, s13.t0, s15.t0]


def run_trans(kind, a, b, prog, seed, rng):
    if kind == 'tear':
        return rgbsplit(tear(b, rng, 6, 34), 2)
    if kind == 'blocks':
        return blockmix(a, b, prog, seed=seed)
    if kind == 'melt':
        return melt(a, b, prog, seed=seed)
    if kind == 'grow':
        return grow(a, b, prog)
    if kind == 'swirl':
        return swirl(a, b, prog, seed=seed)
    if kind == 'scatter':
        return scatter(a, b, prog, seed=seed)
    if TRX is not None and kind in TRX.ALL:
        return TRX.ALL[kind](a, b, prog, seed)
    fb = {'slice': 'blocks', 'zoom': 'grow', 'glyph': 'swirl', 'iris': 'blocks', 'sortwipe': 'melt', 'flash': 'tear', 'shutter': 'blocks'}
    return run_trans(fb.get(kind, 'blocks'), a, b, prog, seed, rng)


def beat_pulse(g, t, beats, rng):
    """the picture breathes with the music: a punch-in on every beat (as strong as the beat really is), glitch on the big hits"""
    i = np.searchsorted(beats[:, 0], t + 1e-6) - 1
    if i < 0:
        return g
    bt_, s, isst = beats[i]
    dt = t - bt_
    if dt > 0.4:
        return g
    talking = any(w0 - 0.05 <= t <= w1 + 0.1 for w0, w1 in _WSPANS)
    if isst:
        amp = 0.05 + 0.03 * s
    elif talking or s < 0.35:
        return g
    else:
        amp = 0.008 + 0.022 * s
    amp *= math.exp(-dt / 0.1)
    if amp > 0.004:
        z = 1 + amp
        w, h = int(round(GW / z)), int(round(GH / z))
        x0, y0 = (GW - w) // 2, int((GH - h) * 0.45)
        g = cv2.resize(g[y0:y0 + h, x0:x0 + w], (GW, GH), interpolation=cv2.INTER_NEAREST)
    if isst and dt < 1 / FPS:
        g = rgbsplit(tear(g, rng, 2, 8), 2)
    elif isst and dt < 2 / FPS:
        g = rgbsplit(g, 1)
    elif s > 0.6 and dt < 1 / FPS:
        g = tear(g, rng, 1, 5)
    return g


_WSPANS = [(ws[0]['t0'], ws[-1]['t1']) for ws in _LINES]


# ------------------------------------------------------------------ main loop
def render(out, t_from=0.0, t_to=130.0, sheet=None):
    shots = build()
    beats = np.array(sync.beat_events(), dtype=np.float64)
    kn = knocks(shots)
    first_cache, last_cache = {}, {}
    noise_bank = [np.random.default_rng(k).normal(0, 4.0, (OH, OW, 1)).astype(np.float32) for k in range(8)]
    scan = np.ones((OH, 1, 1), np.float32); scan[::S] = 0.82
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_frames = []
    f0, f1 = int(t_from * FPS), int(t_to * FPS)
    for f in range(f0, f1):
        t = f / FPS
        k = max(i for i, s in enumerate(shots) if s.t0 <= t + 1e-6)
        sh = shots[k]
        tq = sh.t0 + math.floor((t - sh.t0) * 15 + 1e-6) / 15     # sprite hold, counted from the cut
        tl = max(0.0, tq - sh.t0) + sh.toff
        u = min(1.0, tl / (sh.t1 - sh.t0 + sh.toff))
        rng = np.random.default_rng(int(t * 15) * 7 + 11)
        g = sh.fn(tl, u, rng, tq)
        # transition around the cut: the switch lands on the beat (cut time), part of it happens before
        for j in (k, k + 1):
            if j < 1 or j >= len(shots) or not shots[j].trans:
                continue
            kind, n, pre = shots[j].trans
            fc = int(round(shots[j].t0 * FPS))
            if not (fc - pre <= f < fc - pre + n):
                continue
            prog = (f - (fc - pre) + 1) / (n + 1)
            if f < fc:                            # still in the outgoing shot: incoming = its first frame
                if j not in first_cache:
                    nx = shots[j]
                    first_cache[j] = nx.fn(nx.toff, 0.0, np.random.default_rng(j), nx.t0)
                a_, b_ = g, first_cache[j]
            else:                                 # in the incoming shot: outgoing = its last frame
                if j not in last_cache:
                    pv = shots[j - 1]
                    last_cache[j] = pv.fn(pv.t1 - pv.t0 - 1 / FPS + pv.toff, 1.0, np.random.default_rng(j + 99), pv.t1)
                a_, b_ = last_cache[j], g
            if kind == 'tear' and f < fc:
                continue
            g = run_trans(kind, a_, b_, prog, j, rng)
            break
        if not sh.clean:
            g = beat_pulse(g, t, beats, rng)
        for kt in kn:                          # the knocks: the whole picture jolts, one frame flares
            dk = t - kt
            if 0 <= dk < 0.2:
                amp = int(round(5 * (1 - dk / 0.2)))
                if amp:
                    g = np.roll(g, (int(rng.integers(-amp, amp + 1)), int(rng.integers(-amp, amp + 1))), axis=(0, 1))
                if dk < 1 / FPS:
                    g = quantize(np.clip(g.astype(np.int16) + 45, 0, 255).astype(np.uint8), dither=.1, contrast=1.0)
        rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        ui = sh.ui
        if ui and 'track' in ui:
            _, c, bx = ui['track']
            boxes = []
            for b in bx:
                gx, gy = map_pt(u, c, *b['pt'])
                boxes.append(dict(r=(int(gx - 9), int(gy - 12), int(gx + 9), int(gy + 10)), label=b['label'], col='bone', corners=False))
            ui = dict(ui, boxes=boxes)
        rgb = draw_ui(rgb, t, ui)
        rgb = draw_caption(rgb, caption_at(t), t)
        big = cv2.resize(rgb, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if not sh.clean:
            big *= scan
            big += noise_bank[f % 8]
        else:
            big += noise_bank[f % 8] * 0.35
        frame = np.clip(big, 0, 255).astype(np.uint8)
        p.stdin.write(frame.tobytes())
        if sheet and f % 30 == 0:
            sheet_frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (135, 240), interpolation=cv2.INTER_AREA))
        if f % 300 == 0:
            print('t', round(t, 1), flush=True)
    p.stdin.close(); p.wait()
    if sheet and sheet_frames:
        while len(sheet_frames) % 12:
            sheet_frames.append(np.zeros_like(sheet_frames[0]))
        rows = [np.hstack(sheet_frames[i:i + 12]) for i in range(0, len(sheet_frames), 12)]
        cv2.imwrite(sheet, np.vstack(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--from', dest='t_from', type=float, default=0)
    ap.add_argument('--to', dest='t_to', type=float, default=130)
    ap.add_argument('--sheet', default=None)
    ap.add_argument('--ver', default=os.environ.get('VOY_VER', 'B'))
    a = ap.parse_args()
    render(a.out, a.t_from, a.t_to, a.sheet)
