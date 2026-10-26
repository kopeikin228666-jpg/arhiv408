"""Procedural Voynich book for the intro (0.0-10.4): pixels assemble into a closed book, the cover opens,
pages flip faster and faster, glyphs rise off the vellum, the camera dives into the last page.
Scene is drawn at 720x1280 (like a keyframe) and goes through the same pixel pipeline as every other shot.
"""
import math, functools, numpy as np, cv2
import render as R

W, H = 330, 470            # page size in scene space
SX, SY = 360, 640          # spine centre
PY0 = SY - H // 2
import sync
FLIPS = list(sync.FLIPS)               # page turns land on the music of this version
OPEN0, OPEN1 = sync.OPEN0, 5.30
ASM0, ASM1 = 0.35, sync.ASM1
DIVE0, END = 9.95, sync.DROP

PARCH = np.array([128, 176, 200], np.float32)   # BGR
INK = np.array([38, 50, 62], np.float32)
LEATHER = np.array([34, 40, 46], np.float32)

ILL = ['s19_water.jpg', 's20_pools.jpg', 's21_woman.jpg', 's16_hourglass.jpg', 's22_underwater.jpg', 's23_scribe.jpg',
       's27_sleep.jpg', 's03_eye.jpg', 's25_handprint.jpg', 's31_lips.jpg', 's28_corner.jpg', 's35_whisperer.jpg']
# spreads: (left, right); 'T' = text page, 'P' = pastedown, int = illustration index
SPREADS = [('P', 'T'), ('T', 0), (1, 'T'), ('T', 2), (3, 4), (5, 'T'), (6, 7), ('T', 8), (9, 'T'), (10, 'T'), ('T', 2), (7, 'T'), ('T', 11)]


def _noise(shape, seed, k):
    r = np.random.default_rng(seed).random((shape[0] // k + 2, shape[1] // k + 2)).astype(np.float32)
    return cv2.resize(r, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _glyph_rows(img, seed, y0, y1, x0, x1, h=15, ink=INK):
    rng = np.random.default_rng(seed)
    y = y0
    while y + h < y1:
        x = x0 + int(rng.integers(0, 12))
        while x < x1 - 20:
            a = R.glyph_sprite(int(rng.integers(0, 10 ** 6)))
            sc = h / a.shape[0]
            a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if x + a.shape[1] >= x1:
                break
            reg = img[y:y + h, x:x + a.shape[1]]
            img[y:y + h, x:x + a.shape[1]] = reg * (1 - a[..., None] * .85) + ink * a[..., None] * .85
            x += a.shape[1] + int(rng.integers(6, 16))
        y += h + 9
    return img


@functools.lru_cache(None)
def page(kind, seed):
    rng = np.random.default_rng(seed)
    n = _noise((H, W), seed, 40) * 0.6 + _noise((H, W), seed + 1, 8) * 0.4
    base = PARCH[None, None] * (0.78 + 0.28 * n[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    edge = np.minimum.reduce([xx, W - xx, yy, H - yy]) / 40
    base *= (0.55 + 0.45 * np.clip(edge, 0, 1))[..., None]
    for _ in range(3):   # stains
        cx, cy, r = rng.integers(0, W), rng.integers(0, H), rng.integers(20, 70)
        m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        base *= (1 - 0.25 * m)[..., None]
    img = base
    if kind == 'P':      # inside of the cover: leather lining + label
        img = np.clip(LEATHER[None, None] * (0.8 + 0.4 * n[..., None]), 0, 255).astype(np.uint8)
        cv2.rectangle(img, (95, 190), (235, 260), tuple(int(c) for c in PARCH * .8), -1)
        cv2.putText(img, 'MS 408', (112, 236), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tuple(int(c) for c in INK), 2, cv2.LINE_AA)
        return img
    if kind == 'T':
        img = _glyph_rows(img, seed, 34, H - 40, 28, W - 26)
        a = R.glyph_sprite(seed * 7 + 3, 160).astype(np.float32) / 255     # big initial
        a = cv2.resize(a, (int(a.shape[1] * 52 / a.shape[0]), 52))
        img[30:82, 26:26 + a.shape[1]] = img[30:82, 26:26 + a.shape[1]] * (1 - a[..., None]) + np.array([40, 50, 150], np.float32) * a[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)
    # illustration: keyframe "drawn" onto vellum, fading into the page, a few lines of text under it
    src = R.still(ILL[kind])
    crop = cv2.resize(src[160:1120, 40:680], (W - 40, 300), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    L = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    L = np.clip((L - .5) * 1.5 + .55, 0, 1)
    wash = crop * np.array([0.9, 1.15, 0.9], np.float32)           # keep the greens of the water
    ill = PARCH * (0.25 + 0.75 * L[..., None]) * 0.6 + wash * 255 * 0.4
    m = np.zeros((300, W - 40), np.float32)
    cv2.ellipse(m, ((W - 40) // 2, 150), ((W - 40) // 2 - 6, 140), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 14)[..., None]
    reg = img[30:330, 20:W - 20]
    img[30:330, 20:W - 20] = reg * (1 - m) + ill * m
    img = _glyph_rows(img, seed + 9, 345, H - 36, 28, W - 26, h=13)
    return np.clip(img, 0, 255).astype(np.uint8)


def spread_img(i, side):
    k = SPREADS[i % len(SPREADS)][0 if side == 'L' else 1]
    return page(k, 100 + i * 2 + (0 if side == 'L' else 1))


@functools.lru_cache(None)
def cover():
    n = _noise((H + 16, W + 12), 77, 6) * .5 + _noise((H + 16, W + 12), 78, 30) * .5
    c = LEATHER[None, None] * (0.7 + 0.6 * n[..., None])
    c = np.ascontiguousarray(np.clip(c, 0, 255).astype(np.uint8)).astype(np.float32)
    hh, ww = c.shape[:2]
    cv2.rectangle(c, (16, 16), (ww - 17, hh - 17), (70, 82, 86), 2)
    cv2.rectangle(c, (26, 26), (ww - 27, hh - 27), (52, 60, 64), 1)
    a = R.glyph_sprite(408, 200).astype(np.float32) / 255
    a = cv2.resize(a, (int(a.shape[1] * 150 / a.shape[0]), 150))
    y0, x0 = hh // 2 - 75, ww // 2 - a.shape[1] // 2
    c[y0:y0 + 150, x0:x0 + a.shape[1]] = c[y0:y0 + 150, x0:x0 + a.shape[1]] * (1 - a[..., None]) + np.array([60, 140, 175], np.float32) * a[..., None]
    for cx, cy in [(0, 0), (ww - 40, 0), (0, hh - 40), (ww - 40, hh - 40)]:     # corner metal
        cv2.rectangle(c, (cx + 4, cy + 4), (cx + 36, cy + 36), (110, 118, 112), -1)
    return np.clip(c, 0, 255).astype(np.uint8)


def warp_onto(scene, img, quad, shade=1.0):
    h, w = img.shape[:2]
    srcq = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(srcq, np.float32(quad))
    out = cv2.warpPerspective(img, M, (720, 1280), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (720, 1280), flags=cv2.INTER_LINEAR)
    m = (m.astype(np.float32) / 255)[..., None]
    scene[:] = (scene * (1 - m) + out.astype(np.float32) * shade * m)


def flip_quad(p, side_w, lift=0.16):
    """page hinged at the spine; p 0 = flat right, 1 = flat left"""
    c = math.cos(math.pi * p)
    s = math.sin(math.pi * p)
    xo = SX + side_w * c
    grow = 1 + lift * s
    y0o, y1o = SY - H / 2 * grow - 18 * s, SY + H / 2 * grow - 18 * s
    if c >= 0:
        return [[SX, PY0], [xo, y0o], [xo, y1o], [SX, PY0 + H]]
    return [[xo, y0o], [SX, PY0], [SX, PY0 + H], [xo, y1o]]


@functools.lru_cache(None)
def background():
    yy, xx = np.mgrid[0:1280, 0:720].astype(np.float32)
    r = np.sqrt(((xx - 360) / 360) ** 2 + ((yy - 610) / 560) ** 2)
    glow = np.clip(1 - r, 0, 1) ** 1.6
    n = _noise((1280, 720), 5, 60)
    bg = np.array([12, 14, 13], np.float32) + glow[..., None] * np.array([34, 44, 46], np.float32) * (0.8 + 0.4 * n[..., None])
    return bg


def scene(t):
    """720x1280 BGR float scene of the book at intro time t"""
    sc = background().copy()
    # shadow under the book
    cv2.ellipse(sc, (SX + 10, SY + H // 2 + 6), (360, 40), 0, 0, 360, (8, 9, 9), -1)
    if t < OPEN0:
        # closed: page block edges, then the cover
        sc[PY0 + 6:PY0 + H + 10, SX + 4:SX + W + 10] = PARCH * .55
        for k in range(0, 10, 3):
            sc[PY0 + H + k // 3 * 2 + 4, SX + 6:SX + W + 8] = PARCH * .8
        c = cover()
        sc[PY0 - 8:PY0 - 8 + c.shape[0], SX - 6:SX - 6 + c.shape[1]] = c
        return sc
    # open book: page block thickness under the spread
    sc[PY0 + 4:PY0 + H + 10, SX - W - 8:SX + W + 8] = PARCH * .5
    if t < OPEN1:
        p = (t - OPEN0) / (OPEN1 - OPEN0)
        p = 0.5 - 0.5 * math.cos(math.pi * p)
        sc[PY0:PY0 + H, SX:SX + W] = spread_img(0, 'R')
        sc[PY0:PY0 + H, SX - W:SX] = background()[PY0:PY0 + H, SX - W:SX] * 0.6
        c = cover()
        img = c if p < 0.5 else spread_img(0, 'L')
        shade = 0.55 + 0.45 * abs(math.cos(math.pi * p))
        warp_onto(sc, img, flip_quad(p, W + 6, lift=0.22), shade)
        return sc
    # page flips
    k = sum(1 for f in FLIPS if f <= t)          # flips started
    cur = k - 1
    if cur >= 0:
        f0 = FLIPS[cur]
        nxt = FLIPS[cur + 1] if cur + 1 < len(FLIPS) else f0 + 0.5
        dur = min(0.7, (nxt - f0) * 0.92)
        p = (t - f0) / dur
    else:
        p = 1.0
    if p >= 1.0:             # resting on spread k
        sc[PY0:PY0 + H, SX - W:SX] = spread_img(k, 'L')
        sc[PY0:PY0 + H, SX:SX + W] = spread_img(k, 'R')
    else:
        p = 0.5 - 0.5 * math.cos(math.pi * p)
        sc[PY0:PY0 + H, SX - W:SX] = spread_img(cur, 'L')
        sc[PY0:PY0 + H, SX:SX + W] = spread_img(cur + 1, 'R')
        img = spread_img(cur, 'R') if p < 0.5 else spread_img(cur + 1, 'L')
        shade = 0.5 + 0.5 * abs(math.cos(math.pi * p))
        warp_onto(sc, img, flip_quad(p, W), shade)
    # gutter shadow
    g = np.exp(-((np.arange(-40, 40) / 16.0) ** 2)).astype(np.float32)
    sc[PY0:PY0 + H, SX - 40:SX + 40] *= (1 - 0.55 * g)[None, :, None]
    return sc


def camera(t):
    if t < OPEN0:
        u = t / OPEN0
        return dict(z0=1.30, z1=1.42, x0=0.70, y0=0.5, lin=True), u
    if t < OPEN1 + 0.4:
        u = (t - OPEN0) / (OPEN1 + 0.4 - OPEN0)
        return dict(z0=1.42, z1=1.0, x0=0.70, y0=0.5, x1=0.5, y1=0.5), u
    if t < DIVE0:
        u = (t - OPEN1 - 0.4) / (DIVE0 - OPEN1 - 0.4)
        return dict(z0=1.0, z1=1.12, x0=0.5, y0=0.5, x1=0.52, y1=0.49, lin=True), u
    u = (t - DIVE0) / (END - DIVE0)
    return dict(z0=1.12, z1=7.0, x0=0.52, y0=0.49, x1=0.70, y1=0.42), u ** 1.6


def frame_grid(t):
    c, u = camera(t)
    sc = np.clip(scene(t), 0, 255).astype(np.uint8)
    return R.to_grid(R.cam(sc, u, c), gamma=.82, contrast=1.3, dither=.12)


@functools.lru_cache(None)
def assembled():
    """target grid (closed book at the end of the assembly) + bg grid + per-pixel motion params"""
    tgt = frame_grid(ASM1)
    bg = R.to_grid(R.cam(np.clip(background(), 0, 255).astype(np.uint8), 0, camera(ASM1)[0]), gamma=.82, contrast=1.3, dither=.12)
    diff = np.abs(tgt.astype(int) - bg.astype(int)).sum(2) > 24
    ys, xs = np.where(diff)
    rng = np.random.default_rng(408)
    ang = rng.uniform(0, 2 * np.pi, len(ys))
    rad = rng.uniform(70, 260, len(ys))
    sx = xs + np.cos(ang) * rad
    sy = ys + np.sin(ang) * rad
    delay = 0.55 * (1 - ys / R.GH) * 0.6 + rng.uniform(0, 0.4, len(ys))
    spin = rng.uniform(1.5, 3.5, len(ys)) * np.where(rng.random(len(ys)) < .5, -1, 1)
    return tgt, bg, ys, xs, sy, sx, delay, spin


def assemble(t):
    tgt, bg, ys, xs, sy, sx, delay, spin = assembled()
    s = (t - ASM0) / (ASM1 - ASM0)
    p = np.clip((s - delay * 0.9) / 0.45, 0, 1)
    e = 1 - (1 - p) ** 3
    th = spin * (1 - e)
    dx, dy = (sx - xs) * (1 - e), (sy - ys) * (1 - e)
    px = xs + dx * np.cos(th) - dy * np.sin(th)
    py = ys + dx * np.sin(th) + dy * np.cos(th)
    out = bg.copy()
    vis = p > 0
    px, py = np.round(px[vis]).astype(int), np.round(py[vis]).astype(int)
    ok = (px >= 0) & (px < R.GW) & (py >= 0) & (py < R.GH)
    out[py[ok], px[ok]] = tgt[ys[vis][ok], xs[vis][ok]]
    return out


def rising_glyphs(g, t, rng):
    if t < 6.0:
        return g
    n = int(2 + (t - 6.0) * 5)
    for j in range(n):
        birth = 6.0 + j * 0.19
        age = t - birth
        if age < 0 or age > 1.6:
            continue
        r = np.random.default_rng(j * 31 + 5)
        x0 = r.uniform(30, 150)
        y0 = r.uniform(150, 230)
        gx = x0 + math.sin(age * 2 + j) * 6 + r.uniform(-10, 10) * age
        gy = y0 - age * r.uniform(40, 80)
        col = 'bone' if age < .7 else ('mid' if age < 1.2 else 'tile')
        g = R.stamp_glyph(g, 500 + j, gx, gy, 8 + age * 10, col)
    return g


def swirl(g, amt, cx=R.GW / 2, cy=R.GH / 2):
    yy, xx = np.mgrid[0:R.GH, 0:R.GW].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    a = amt * np.clip(1 - r / 190.0, 0, 1) ** 2
    mx = cx + dx * np.cos(a) - dy * np.sin(a)
    my = cy + dx * np.sin(a) + dy * np.cos(a)
    return cv2.remap(g, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def book_fn(tl, u, rng, tq):
    t = tl
    if t < ASM0:
        g = R.black()
        y = int(R.GH / 2 + rng.integers(-2, 3))
        g[y, :int(rng.integers(10, R.GW))] = R.P['line'] if rng.random() < .6 else R.P['tile']
        return g
    if t < ASM1:
        return assemble(t)
    g = frame_grid(t)
    if ASM1 <= t < ASM1 + 2 / 15:          # arrival: one rgb split frame
        g = R.rgbsplit(g, 2)
    g = rising_glyphs(g, t, rng)
    if t >= DIVE0:
        k = (t - DIVE0) / (END - DIVE0)
        g = swirl(g, k ** 2 * 7)
        g = R.pixsort(g, .4, 1.0) if k > .5 else g
    elif any(f <= t < f + 1 / 15 for f in FLIPS[4:]):
        g = R.tear(g, rng, 3, 10)
    return g


def book_ui():
    def cover_box(t):
        if t < ASM1 + .3 or t > OPEN0 + .3:
            return None
        return (40, 76, 160, 246)
    def label(t):
        s = 'MS 408'
        n = int((t - ASM1 - .3) / 0.09)
        return s[:max(0, min(len(s), n))] + ('_' if n < len(s) and int(t * 8) % 2 else '')
    def scan(t):
        return f'SCAN {min(100, int((t - ASM1) / 1.6 * 100)):3d}%'
    def page_box(t):
        if t < OPEN1 + .3 or t > DIVE0:
            return None
        return (16, 92, 164, 236)
    def page_label(t):
        k = sum(1 for f in FLIPS if f <= t)
        return f'f.{k * 7 + 1}r'
    def page_sub(t):
        n = np.random.default_rng(int(t * 15)).integers(100, 999)
        return f'LANG: --- {n}'
    return dict(surface='SURFACE_00', signal='SIGNAL --', guides=True, gy=60, c0=0,
                boxes=[dict(r=cover_box, label=label, col='red', sub=scan),
                       dict(r=page_box, label=page_label, col='bone', conf='?.??', sub=page_sub)])
