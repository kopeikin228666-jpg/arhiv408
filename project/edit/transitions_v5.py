"""Pixel-art transitions for the Voynich reel (dark / glitch / occult).

Frames are uint8 BGR arrays of shape (320, 180, 3) on the fixed 18-colour palette of
look/pixelize.py (later upscaled x6 nearest-neighbour to 1080x1920).

API:   fn(a, b, prog, seed=0) -> uint8 (320, 180, 3)
       a = outgoing frame, b = incoming frame, prog in [0, 1] (0 = pure a, 1 = pure b),
       the decisive "switch" happens at prog ~ 0.5 so a cut can be centred on a beat.

    slice     VHS shutter: horizontal bands of b push a out sideways, staggered
    zoom      zoom-through punch: a falls in (growing blocks), shatters into b, b settles
    glyph     b revealed through a growing procedural Voynich glyph with a red rim
    iris      blocky 4x4 circle of b with bone/red ring and pixels flying off (cx, cy kwargs)
    sortwipe  top->bottom wipe, a pixel-sorted into dripping streaks under the line
    flash     a burns up to bone (2x2 dither), one bone frame, b lands with a settling RGB split
    shutter   6-px vertical blinds flip a->b sweeping from the centre outwards

Everything is deterministic for a given seed; per-seed layouts are cached.
Colour snapping uses a 6-bit LUT that reproduces pixelize.quantize's weighted
nearest-colour metric (quantize itself costs ~85 ms/frame, far over budget).
No dependency on render.py (would be circular) - only numpy, cv2, PIL, pixelize.
"""
import sys
import math
import functools

import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, '/home/claude/voynich/look')
from pixelize import PALETTE, quantize  # noqa: E402

H, W = 320, 180
CX, CY = W / 2.0, H / 2.0
PAL = np.clip(np.round(PALETTE), 0, 255).astype(np.uint8)          # (18, 3) BGR
NAMES = ['void', 'void2', 'dark', 'd3', 'tile', 'tile2', 'mid', 'line', 'light', 'bone',
         'g1', 'g2', 'g3', 'p1', 'p2', 'parch', 'red2', 'red']
IDX = {n: i for i, n in enumerate(NAMES)}
C = {n: PAL[i] for i, n in enumerate(NAMES)}


# ----------------------------------------------------------------------------- palette helpers
def _keys(img):
    q = (img >> 2).astype(np.int32)
    return (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]


def _build_lut():
    v = np.arange(64, dtype=np.float32) * 4 + 1.5                    # bin centres
    bb, gg, rr = np.meshgrid(v, v, v, indexing='ij')
    cols = np.stack([bb.ravel(), gg.ravel(), rr.ravel()], 1)
    wts = np.array([0.8, 1.2, 1.0], np.float32)                      # same metric as quantize()
    best = np.zeros(cols.shape[0], np.uint8)
    bestd = np.full(cols.shape[0], np.inf, np.float32)
    for i, p in enumerate(PALETTE):
        d = (((cols - p) * wts) ** 2).sum(1)
        m = d < bestd
        bestd[m] = d[m]
        best[m] = i
    best[_keys(PAL)] = np.arange(len(PAL), dtype=np.uint8)           # exact colours map to themselves
    return best


_LUT = _build_lut()


def _bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m.astype(np.float32) + 0.5) / m.size - 0.5


_TH8 = np.tile(_bayer(8), (H // 8 + 1, W // 8 + 1))[:H, :W]          # [-0.5, 0.5)
_B2 = np.tile(np.array([[0, 2], [3, 1]], np.float32) / 4, (H // 2, W // 2))  # 2x2 ordered dither, [0, 1)


def to_idx(img):
    """palette index map of an (H, W, 3) uint8 image (nearest palette colour)"""
    return _LUT[_keys(img)]


def snap(img, dither=0.0):
    """fast LUT equivalent of pixelize.quantize(img, dither, contrast=1.0)"""
    if dither > 0:
        x = img.astype(np.int16) + (_TH8 * (dither * 255.0)).astype(np.int16)[..., None]
        img = np.clip(x, 0, 255).astype(np.uint8)
    return PAL[to_idx(img)]


# brightness ramp along each hue family (greys -> bone, greens/parchment/red climb to bone)
_NEXT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 7, 14, 15, 9, 17, 15], np.uint8)
_PREV = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 10, 11, 3, 13, 14, 1, 16], np.uint8)
_LIFT = [np.arange(18, dtype=np.uint8)]
for _k in range(12):
    _LIFT.append(_NEXT[_LIFT[-1]])


def lift(img, level, idx=None):
    """push every pixel `level` steps up its colour ramp; fractional part as 2x2 ordered dither"""
    if idx is None:
        idx = to_idx(img)
    level = max(0.0, float(level))
    n = min(int(level), 11)
    f = level - int(level)
    out = _LIFT[n][idx]
    if f > 1e-3:
        out = np.where(_B2 < f, _LIFT[n + 1][idx], out)
    return PAL[out]


def _c01(x):
    return min(1.0, max(0.0, float(x)))


def _ss(e0, e1, x):
    t = _c01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def _rng(seed, salt):
    return np.random.default_rng([int(seed) & 0x7fffffff, salt])


def _ends(a, b, p):
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    return None


_K3 = np.ones((3, 3), np.uint8)
_KX = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


# ============================================================================ 1. slice
@functools.lru_cache(maxsize=32)
def _slice_plan(seed):
    rng = _rng(seed, 11)
    bands, y = [], 0
    while y < H:
        h = int(rng.integers(6, 15))
        if H - (y + h) < 6:
            h = H - y
        bands.append((y, y + h))
        y += h
    n = len(bands)
    rank = rng.permutation(n).astype(np.float64) / max(1, n - 1)
    start = 0.05 + rank * 0.62 + rng.uniform(-0.02, 0.02, n)
    dur = np.minimum(rng.uniform(0.17, 0.27, n), 0.97 - start)
    d0 = 1 if rng.random() < 0.5 else -1
    dirs = [d0 if i % 2 == 0 else -d0 for i in range(n)]
    red = rng.random(n) < 0.3
    jolt = rng.choice([-3, -2, 2, 3], n)
    return bands, start, dur, dirs, red, jolt


def slice(a, b, prog, seed=0):  # noqa: A001  (name required by the API)
    """VHS shutter: bands (6-14 px) of b push a out sideways, alternating L/R, staggered."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    bands, start, dur, dirs, red, jolt = _slice_plan(seed)
    out = a.copy()
    bone, void, redc = C['bone'], C['void'], C['red']
    for i, (y0, y1) in enumerate(bands):
        t = (p - start[i]) / dur[i]
        if t >= 1.0:
            out[y0:y1] = b[y0:y1]
            continue
        if t <= 0.0:
            if p > start[i] - 0.045:                                    # tracking jolt before it goes
                out[y0:y1] = np.roll(a[y0:y1], int(jolt[i]), axis=1)
                out[y0, :] = C['dark']
            continue
        k = 1.0 - (1.0 - t) ** 3                                        # ease-out snap
        off = int(round((1.0 - k) * W))
        if off <= 0:
            out[y0:y1] = b[y0:y1]
            continue
        if off >= W:
            continue
        if dirs[i] > 0:                                                 # enters from the left
            out[y0:y1, :W - off] = b[y0:y1, off:]
            out[y0:y1, W - off:] = a[y0:y1, :off]
            ex = W - off - 1                                            # b's leading column
            out[y0:y1, min(W - 1, ex + 1):min(W, ex + 3)] = void
        else:                                                           # enters from the right
            out[y0:y1, off:] = b[y0:y1, :W - off]
            out[y0:y1, :off] = a[y0:y1, W - off:]
            ex = off
            out[y0:y1, max(0, ex - 2):ex] = void
        out[y0:y1, ex] = redc if red[i] else bone
        out[y0, :] = void                                               # scanline seam on moving bands
    return out


# ============================================================================ 2. zoom
def _zgrid(n, c, z, blk):
    i = np.arange(n, dtype=np.float32) + 0.5
    if blk > 1:
        i = np.floor((i - c) / blk) * blk + blk * 0.5 + c
    return np.clip(np.floor((i - c) / z + c), 0, n - 1).astype(np.intp)


def _zoomed(img, z, blk):
    return img[_zgrid(H, CY, z, blk)[:, None], _zgrid(W, CX, z, blk)[None, :]]


@functools.lru_cache(maxsize=16)
def _shards(seed):
    rng = _rng(seed, 22)
    n = 34
    rr = 200.0 * rng.random(n) ** 1.6 + 6
    th = rng.uniform(0, 2 * np.pi, n)
    px, py = CX + rr * np.cos(th) * 0.75, CY + rr * np.sin(th)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = (xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2
    lab = d.argmin(-1).astype(np.intp)
    crack = np.zeros((H, W), bool)
    crack[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    crack[:-1, :] |= lab[:-1, :] != lab[1:, :]
    mx = np.bincount(lab.ravel(), xx.ravel(), n) / np.maximum(1, np.bincount(lab.ravel(), None, n))
    my = np.bincount(lab.ravel(), yy.ravel(), n) / np.maximum(1, np.bincount(lab.ravel(), None, n))
    vx, vy = mx - CX, my - CY
    rad = np.hypot(vx, vy) + 1e-3
    vx, vy = vx / rad, vy / rad
    thr = 0.43 + 0.15 * np.clip(rad / 170.0, 0, 1) + rng.uniform(-0.02, 0.02, n)
    spd = rng.uniform(0.6, 1.4, n) * (0.5 + np.clip(rad / 120.0, 0, 1.2))
    glint = rng.random((H, W)) < 0.22
    return lab, crack, glint & crack, mx, my, thr, vx * spd, vy * spd


def _za(u):
    return 1.0 + 6.0 * u ** 3, 1 + int(round(9.0 * u ** 2))


_YY = np.arange(H, dtype=np.float32)[:, None] + 0.5
_XX = np.arange(W, dtype=np.float32)[None, :] + 0.5


def zoom(a, b, prog, seed=0):
    """Zoom-through punch: a rushes in with growing blocks, shatters, b lands zoomed and settles."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    v = _c01((p - 0.5) / 0.5)
    zb, bb = 1.0 + 4.0 * (1 - v) ** 3, 1 + int(round(8.0 * (1 - v) ** 2))
    if p < 0.37:
        za, ba = _za(p / 0.5)
        out = _zoomed(a, za, ba)
        if p > 0.27:                                                    # hairline cracks spread
            lab, crack, glint, cx_, cy_, thr, fx, fy = _shards(seed)
            k = (p - 0.27) / 0.10
            cm = crack & (_TH8 + 0.5 < 0.15 + k * 0.85)
            out[cm] = C['void']
            out[cm & glint] = C['bone']
        return out
    if p > 0.6:
        return _zoomed(b, zb, bb)
    # ---- shatter window: shards shrink and fly outward, b shows through the gaps
    lab, crack, glint, cx_, cy_, thr, fx, fy = _shards(seed)
    za, ba = _za(p / 0.5)
    amt = _ss(0.37, 0.6, p)
    disp = amt ** 1.4 * 60.0
    sc = 1.0 - 0.3 * amt
    k = lab
    qy = cy_[k] + (_YY - cy_[k] - fy[k] * disp) / sc                    # inverse of the shard motion
    qx = cx_[k] + (_XX - cx_[k] - fx[k] * disp) / sc
    iy = np.clip(qy.astype(np.intp), 0, H - 1)
    ix = np.clip(qx.astype(np.intp), 0, W - 1)
    own = (lab[iy, ix] == k) & (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W) & (p < thr[k])
    if ba > 1:
        qy = np.floor((qy - CY) / ba) * ba + ba * 0.5 + CY
        qx = np.floor((qx - CX) / ba) * ba + ba * 0.5 + CX
    ry = np.clip(np.floor((qy - CY) / za + CY), 0, H - 1).astype(np.intp)
    rx = np.clip(np.floor((qx - CX) / za + CX), 0, W - 1).astype(np.intp)
    m = own.astype(np.uint8)
    out = np.where(own[..., None], a[ry, rx], _zoomed(b, zb, bb))
    edge = own & (cv2.erode(m, _KX) == 0)
    out[edge] = C['void']
    out[edge & (_TH8 > 0.32)] = C['bone']
    halo = (cv2.dilate(m, _KX) > 0) & ~own                             # dark lip on b around each shard
    out[halo] = C['void']
    return out


# ============================================================================ 3. glyph
def _bez(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
    return [tuple(q) for q in pts]


def _cub(p0, p1, p2, p3, n=32):
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
           + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3))
    return [tuple(q) for q in pts]


@functools.lru_cache(maxsize=16)
def _glyph_field(seed):
    """procedural Voynich-like word (e.g. o + gallows + 9) -> distance field from its pen strokes"""
    rng = _rng(seed, 33)
    N = 512
    im = Image.new('L', (N, N), 0)
    d = ImageDraw.Draw(im)
    u = 44.0                      # x-height in sprite px
    yb = 300.0                    # baseline
    lw = 4
    x = 0.0
    strokes = []                  # deferred so the word can be centred afterwards

    def ell(x0, y0, x1, y1):
        strokes.append(('ell', (x0, y0, x1, y1)))

    def poly(pts):
        strokes.append(('line', pts))

    first = rng.choice(['o', 'c', 'a', 'q'])
    last = rng.choice(['y', 'y', 'd', 'n'])
    plume = rng.random() < 0.5
    bench = rng.random() < 0.45
    lean = rng.uniform(0.1, 0.35) * u
    loop = rng.uniform(0.9, 1.25)

    # --- first character
    if first == 'q':                                               # '4'-like
        poly([(x + 0.7 * u, yb + 0.15 * u), (x + 0.7 * u, yb - 1.4 * u), (x, yb - 0.35 * u),
              (x + 1.0 * u, yb - 0.35 * u)])
        x += 1.15 * u
        ell(x, yb - u, x + 0.85 * u, yb)
        x += 1.05 * u
    elif first == 'c':
        strokes.append(('arc', (x, yb - u, x + 0.85 * u, yb), 55, 305))
        x += 0.9 * u
    else:
        ell(x, yb - u, x + 0.85 * u, yb)
        if first == 'a':
            poly(_bez((x + 0.85 * u, yb - 0.9 * u), (x + 0.9 * u, yb), (x + 1.15 * u, yb - 0.1 * u), 12))
            x += 1.25 * u
        else:
            x += 1.05 * u
    # --- gallows (k / t / p / f family)
    gx = x + 0.15 * u
    top = (gx + lean, yb - 2.9 * u)
    poly([(gx, yb + 0.05 * u), top])                                               # tall stem
    rx_ = gx + 1.15 * u * loop
    loop_end = (rx_, yb - 1.55 * u)
    poly(_cub(top, (top[0] + 1.3 * u * loop, top[1] - 0.9 * u),
              (rx_ + 0.75 * u, yb - 2.2 * u), loop_end))                          # the loop
    poly(_bez(loop_end, (rx_ - 0.2 * u, yb - 0.6 * u), (rx_ + 0.05 * u, yb)))      # second stem
    poly(_bez((rx_ + 0.05 * u, yb), (rx_ + 0.3 * u, yb + 0.15 * u), (rx_ + 0.55 * u, yb - 0.1 * u), 10))
    if plume:                                                                       # 't'-type flourish
        poly(_bez(top, (gx - 0.7 * u, yb - 3.3 * u), (gx - 0.9 * u, yb - 2.3 * u)))
    if bench:                                                                       # pedestal 'ch' bar
        poly([(gx - 0.55 * u, yb - 1.0 * u), (rx_ + 0.6 * u, yb - 1.0 * u)])
        strokes.append(('arc', (gx - 0.75 * u, yb - u, gx + 0.1 * u, yb), 60, 300))
    stem_pt = (gx + lean * 0.45, yb - 1.3 * u)
    x = rx_ + 0.75 * u
    # --- last character
    if last == 'y':                                                 # '9' with descending hook
        ell(x, yb - u, x + 0.85 * u, yb)
        poly(_cub((x + 0.85 * u, yb - 0.55 * u), (x + 0.95 * u, yb + 0.5 * u),
                  (x + 0.6 * u, yb + 1.25 * u), (x - 0.25 * u, yb + 1.0 * u)))
        x += 1.0 * u
    elif last == 'd':                                               # '8'-like with tall back stroke
        ell(x, yb - u, x + 0.85 * u, yb)
        poly(_cub((x + 0.8 * u, yb - 0.7 * u), (x + 1.1 * u, yb - 1.6 * u),
                  (x + 0.6 * u, yb - 2.1 * u), (x + 0.05 * u, yb - 1.7 * u)))
        x += 1.05 * u
    else:                                                           # 'in' minims with hook
        poly([(x + 0.1 * u, yb), (x + 0.15 * u, yb - 0.9 * u)])
        poly(_cub((x + 0.55 * u, yb), (x + 0.55 * u, yb - 1.0 * u),
                  (x + 0.95 * u, yb - 1.1 * u), (x + 1.25 * u, yb - 0.6 * u)))
        x += 1.3 * u

    # centre the word on the canvas
    xs, ys = [], []
    for s in strokes:
        if s[0] == 'line':
            xs += [q[0] for q in s[1]]
            ys += [q[1] for q in s[1]]
        else:
            xs += [s[1][0], s[1][2]]
            ys += [s[1][1], s[1][3]]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    ox, oy = N / 2 - (bx0 + bx1) / 2, N / 2 - (by0 + by1) / 2
    for s in strokes:
        if s[0] == 'line':
            d.line([(q[0] + ox, q[1] + oy) for q in s[1]], fill=255, width=lw, joint='curve')
        elif s[0] == 'ell':
            x0, y0, x1, y1 = s[1]
            d.ellipse([x0 + ox, y0 + oy, x1 + ox, y1 + oy], outline=255, width=lw)
        else:
            x0, y0, x1, y1 = s[1]
            d.arc([x0 + ox, y0 + oy, x1 + ox, y1 + oy], s[2], s[3], fill=255, width=lw)
    ink = np.array(im) > 0
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    gh = by1 - by0
    centre = (N / 2.0, N / 2.0)
    anchor = (stem_pt[0] + ox, stem_pt[1] + oy)
    return dist, gh, centre, anchor


@functools.lru_cache(maxsize=16)
def _glyph_noise(seed):
    rng = _rng(seed, 34)
    n = cv2.resize(rng.random((12, 7)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    n += 0.5 * cv2.resize(rng.random((40, 23)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-6)


def glyph(a, b, prog, seed=0):
    """b shows through a Voynich gallows-word that grows from the centre past the screen."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    dist, gh, centre, anchor = _glyph_field(seed)
    N = dist.shape[0]
    h = 46.0 * math.exp(3.7 * p)                                    # glyph height on screen
    s = h / gh                                                      # screen px per sprite px
    m = _ss(0.18, 0.62, p)
    ax = centre[0] + (anchor[0] - centre[0]) * m
    ay = centre[1] + (anchor[1] - centre[1]) * m
    iy = np.clip(((np.arange(H) + 0.5 - CY) / s + ay).astype(np.intp), 0, N - 1)
    ix = np.clip(((np.arange(W) + 0.5 - CX) / s + ax).astype(np.intp), 0, N - 1)
    D = dist[iy[:, None], ix[None, :]]
    tb = (2.2 + 6.0 * p * p) * min(1.0, p / 0.06)                   # stroke half-width (sprite px)
    k = _ss(0.5, 0.88, p) ** 1.3
    T = tb + (float(D.max()) + 1.0 - tb) * k                        # strokes swell until nothing is left
    mask = (D < T * (1.0 + 0.45 * _ss(0.2, 0.6, p) * _glyph_noise(seed))).astype(np.uint8)
    out = np.where(mask[..., None] > 0, b, a)
    d1 = cv2.dilate(mask, _KX)
    ring = (d1 > 0) & (mask == 0)
    halo = (cv2.dilate(d1, _KX) > 0) & (d1 == 0)
    out[halo] = C['void']
    out[ring] = C['red']
    return out


# ============================================================================ 4. iris
@functools.lru_cache(maxsize=32)
def _iris_geo(cx, cy):
    by = np.arange(0, H, 4, dtype=np.float32) + 2.0
    bx = np.arange(0, W, 4, dtype=np.float32) + 2.0
    dist = np.hypot(by[:, None] - cy, (bx[None, :] - cx))           # per 4x4 block
    ang = np.arctan2(np.arange(H)[:, None] + 0.5 - cy, np.arange(W)[None, :] + 0.5 - cx)
    rmax = max(math.hypot(x - cx, y - cy) for x in (0, W) for y in (0, H)) + 6.0
    return dist, ang.astype(np.float32), rmax


@functools.lru_cache(maxsize=32)
def _iris_sparks(seed):
    rng = _rng(seed, 44)
    n = 40
    return (rng.uniform(0, 2 * np.pi, n), rng.uniform(0.04, 0.7, n), rng.uniform(0.6, 1.6, n),
            rng.integers(2, 5, n), rng.uniform(0.18, 0.4, n), rng.integers(0, 4, n))


def iris(a, b, prog, seed=0, cx=None, cy=None):
    """b opens in a blocky (4x4-step) circle with a bone/red rim and pixels flung outward."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    cx = CX if cx is None else float(cx)
    cy = CY if cy is None else float(cy)
    dist, ang, rmax = _iris_geo(round(cx, 1), round(cy, 1))
    rp = lambda q: rmax * _c01((q - 0.04) / 0.86) ** 1.25            # noqa: E731
    r = rp(p)
    mb = (dist < r).astype(np.uint8)
    mask = np.repeat(np.repeat(mb, 4, 0), 4, 1)[:H, :W]
    out = np.where(mask[..., None] > 0, b, a)
    if r < rmax:
        d1 = cv2.dilate(mask, _K3)
        d2 = cv2.dilate(d1, _K3)
        r1 = (d1 > 0) & (mask == 0)
        r2 = (d2 > 0) & (d1 == 0)
        seg = (np.floor((ang + p * 2.2) / (2 * np.pi) * 28) % 2) == 0   # rotating dashes
        out[r2] = C['void']
        out[r2 & seg] = C['red']
        out[r1] = C['bone']
        out[r1 & ~seg & (_B2 > 0.6)] = C['red']
    th, t0, sp, sz, life, col = _iris_sparks(seed)
    if 4 < r < rmax:                                               # occult tick ring, counter-rotating
        for t in range(24):
            ang_t = t * 2 * math.pi / 24 - p * 3.0
            rr_ = r + 9 + (3 if t % 3 == 0 else 0)
            x = int(cx + math.cos(ang_t) * rr_)
            y = int(cy + math.sin(ang_t) * rr_)
            if 0 <= x < W - 1 and 0 <= y < H - 1:
                out[y:y + 2, x:x + 2] = C['red'] if t % 3 == 0 else C['red2']
    cols = (C['bone'], C['red'], C['parch'], None)
    for i in range(len(th)):
        age = p - t0[i]
        if age < 0 or age > life[i]:
            continue
        rad = rp(t0[i]) + 6 + sp[i] * age * 260.0 * (1 + age)
        x = int(round(cx + math.cos(th[i]) * rad))
        y = int(round(cy + math.sin(th[i]) * rad))
        q = int(sz[i])
        if age > life[i] * 0.6:
            q = max(1, q - 1)
        if 0 <= x < W - q and 0 <= y < H - q:
            c_ = cols[col[i]]
            if c_ is None:                                         # a chunk of b flung out
                sx_ = int(cx + math.cos(th[i]) * rp(t0[i])); sy_ = int(cy + math.sin(th[i]) * rp(t0[i]))
                c_ = b[min(H - 1, max(0, sy_)), min(W - 1, max(0, sx_))]
            out[y:y + q, x:x + q] = c_
    return out


# ============================================================================ 5. sortwipe
@functools.lru_cache(maxsize=16)
def _drips(seed):
    rng = _rng(seed, 55)
    L = np.empty(W, np.int32)
    x = 0
    while x < W:
        run = int(rng.integers(1, 5))
        L[x:x + run] = int(np.clip(rng.gamma(2.4, 12.0) + 16, 14, 60))
        x += run
    sparks = rng.random(W) < 0.12
    shift = rng.integers(-5, 6, 4)
    return L, sparks, shift


def sortwipe(a, b, prog, seed=0):
    """Wipe line runs down; under it a melts into vertical brightness-sorted drips; b above."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    L, sparks, shift = _drips(seed)
    LM = 60
    yw = int(round(-LM + (H + LM) * p))                             # rows < yw are b
    out = a.copy()
    if yw > 0:
        out[:min(yw, H)] = b[:min(yw, H)]
        for j in range(4):                                         # torn rows just above the line
            yy = yw - 1 - j * 2
            if 0 <= yy < H and shift[j]:
                out[max(0, yy - 1):yy + 1] = np.roll(out[max(0, yy - 1):yy + 1], int(shift[j]), axis=1)
    y0, y1 = max(yw, 0), min(yw + LM, H)
    if y1 > y0:
        src0 = max(0, yw - 24)                                     # drag some of the covered a down too
        seg = a[src0:y1]
        lum = seg.astype(np.int16).sum(2)
        rows = np.arange(src0, y1)[:, None]
        inside = rows < yw + L[None, :]
        key = np.where(inside, -lum, 10000 + rows)                 # sorted run first, rest untouched
        order = np.argsort(key, axis=0, kind='stable')
        srt = np.take_along_axis(seg, order[..., None], axis=0)
        cnt = np.maximum(1, inside.sum(0))                         # sorted px per column
        yd = np.arange(y0, y1)[:, None]
        pos = (yd - yw) / L[None, :]                               # 0..1 along each drip
        src_r = np.clip((pos * cnt[None, :]).astype(np.intp), 0, cnt[None, :] - 1)
        band = srt[src_r, np.arange(W)[None, :]]
        glow = np.clip(2.6 * (1.0 - pos) - 0.4, 0, None)            # hot near the line, fading down
        lv = (glow.astype(np.int32) + (_B2[y0:y1] < (glow % 1.0))).clip(0, 3)
        bi = to_idx(band)
        band = PAL[np.choose(lv, [_LIFT[0][bi], _LIFT[1][bi], _LIFT[2][bi], _LIFT[3][bi]])]
        out[y0:y1] = np.where((pos < 1.0)[..., None], band, a[y0:y1])
        # drip tips
        tip = yw + L - 1
        ok = (tip >= 0) & (tip < H) & sparks
        xs_ = np.flatnonzero(ok)
        out[tip[ok], xs_] = C['red']
        ok2 = ok & (tip + 1 < H)
        out[tip[ok2] + 1, np.flatnonzero(ok2)] = C['red2']
    if 0 <= yw < H:
        out[yw] = C['bone']
        if yw + 1 < H:
            out[yw + 1, ::3] = C['void']
    return out


# ============================================================================ 6. flash
def flash(a, b, prog, seed=0):
    """a burns up to bone in 2x2 dither, a hard bone frame, then b lands with a settling RGB split."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    if p < 0.30:
        return lift(a, 9.0 * (p / 0.30) ** 1.6)
    if p < 0.45:
        return np.broadcast_to(C['bone'], (H, W, 3)).copy()
    v = (p - 0.45) / 0.55
    g = lift(b, 3.0 * (1 - v) ** 4) if v < 0.6 else b.copy()
    dx = int(round(4.0 * (1 - v) ** 1.8))
    if dx > 0:
        sgn = 1 if (seed % 2 == 0) else -1
        sp = g.copy()
        sp[:, :, 2] = np.roll(g[:, :, 2], sgn * dx, axis=1)
        sp[:, :, 0] = np.roll(g[:, :, 0], -sgn * dx, axis=1)
        if v < 0.25:                                               # a couple of torn rows on impact
            rng = _rng(seed, 66 + int(p * 64))
            for _ in range(3):
                y = int(rng.integers(0, H - 8))
                hgt = int(rng.integers(2, 8))
                sp[y:y + hgt] = np.roll(sp[y:y + hgt], int(rng.integers(-10, 11)), axis=1)
        g = snap(sp)
    return g


# ============================================================================ 7. shutter
@functools.lru_cache(maxsize=16)
def _shutter_plan(seed):
    rng = _rng(seed, 77)
    n = W // 6
    dc = np.abs(np.arange(n) + 0.5 - n / 2) / (n / 2)              # 0 centre .. 1 edge
    start = 0.1 + 0.52 * dc + rng.uniform(-0.035, 0.035, n)
    dur = 0.24
    return start, dur


def shutter(a, b, prog, seed=0):
    """6-px vertical blinds flip a->b in a sweep from the centre outward, each showing a dark edge."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    start, dur = _shutter_plan(seed)
    n = len(start)
    srccol = np.arange(W)
    sel = np.zeros(W, np.int8)                                     # 0 a, 1 b, 2 gap, 3 edge, 4 glint
    vsc = np.ones(W, np.float32)
    shade = np.zeros(W, np.float32)
    for j in range(n):
        x0 = j * 6
        f = _c01((p - start[j]) / dur)
        if f <= 0:
            continue
        if f >= 1:
            sel[x0:x0 + 6] = 1
            continue
        c = abs(math.cos(math.pi * f))
        w = int(round(6 * c))
        src = 1 if f >= 0.5 else 0
        sel[x0:x0 + 6] = 2
        side = 1 if f < 0.5 else -1                                # blade turns away to the right, back from the left
        l = x0 + (6 - w) // 2
        for k in range(w):
            sx = x0 + min(5, int((k + 0.5) * 6 / w))
            sel[l + k] = src
            srccol[l + k] = sx
            shade[l + k] = 3.2 * math.sin(math.pi * f) ** 1.5
            vsc[l + k] = 1.0 + 0.18 * math.sin(math.pi * f) * ((k + 0.5) / w - 0.5) * side * 2
        if w < 6:
            ex = l + w if side > 0 else l - 1                      # blade thickness = dark edge
            if x0 <= ex < x0 + 6:
                sel[ex] = 3
            gx = l if side > 0 else l + w - 1
            if w >= 2 and 0.2 < f < 0.8:
                sel[gx] = 4
    yy = np.arange(H, dtype=np.float32)[:, None] + 0.5
    ry = np.clip(np.floor((yy - CY) / vsc[None, :] + CY), 0, H - 1).astype(np.intp)
    cc = srccol[None, :]
    out = np.where((sel == 1)[None, :, None], b[ry, cc], a[ry, cc])
    if (shade > 0).any():                                          # turning slats fall into shadow
        oi = to_idx(out)
        lv = np.broadcast_to(shade[None, :], (H, W))
        lo = lv.astype(np.int32)
        dk = np.where(_B2 < (lv - lo), lo + 1, lo).clip(0, 4)
        res = oi
        for _ in range(4):
            res = np.where(dk > 0, _PREV[res], res)
            dk = dk - 1
        out = PAL[res]
    out[:, sel == 2] = C['void']
    out[:, sel == 3] = C['dark']
    gm = sel == 4
    if gm.any():
        gl = out[:, gm]
        out[:, gm] = lift(gl.reshape(H, -1, 3) if gl.ndim == 3 else gl, 2.0)
    return out


ALL = {
    'slice': slice,
    'zoom': zoom,
    'glyph': glyph,
    'iris': iris,
    'sortwipe': sortwipe,
    'flash': flash,
    'shutter': shutter,
}


# ============================================================================ preview / QA
def _load_kf(path):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    tgt = 9 / 16
    if w / h > tgt:
        nw = int(h * tgt)
        x0 = (w - nw) // 2
        im = im[:, x0:x0 + nw]
    else:
        nh = int(w / tgt)
        y0 = (h - nh) // 2
        im = im[y0:y0 + nh]
    return quantize(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), dither=0.10)


def preview(out_path='/home/claude/voynich/qa/transitions_sheet.png',
            ka='/home/claude/voynich/kf/s16_hourglass.jpg', kb='/home/claude/voynich/kf/s19_water.jpg',
            progs=(0, .15, .3, .45, .55, .7, .85, 1), seed=0):
    a, b = _load_kf(ka), _load_kf(kb)
    gap, lab = 2, 64
    sheet = np.zeros((len(ALL) * (H + gap) + 18, lab + len(progs) * (W + gap), 3), np.uint8)
    sheet[:] = C['void2']
    for j, pr in enumerate(progs):
        cv2.putText(sheet, f'{pr:g}', (lab + j * (W + gap) + 78, 13), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    tuple(int(v) for v in C['light']), 1, cv2.LINE_AA)
    for i, (name, fn) in enumerate(ALL.items()):
        y = 18 + i * (H + gap)
        cv2.putText(sheet, name, (4, y + H // 2), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    tuple(int(v) for v in C['bone']), 1, cv2.LINE_AA)
        for j, pr in enumerate(progs):
            x = lab + j * (W + gap)
            sheet[y:y + H, x:x + W] = fn(a, b, pr, seed=seed)
    cv2.imwrite(out_path, sheet)
    return out_path


def check(seed=0, n=48):
    """palette / endpoint / determinism / timing check; returns {name: ms per call}"""
    import time
    a = _load_kf('/home/claude/voynich/kf/s16_hourglass.jpg')
    b = _load_kf('/home/claude/voynich/kf/s19_water.jpg')
    pal = set(map(tuple, PAL.tolist()))
    res = {}
    for name, fn in ALL.items():
        assert np.array_equal(fn(a, b, 0.0, seed), a), name + ' prog 0 != a'
        assert np.array_equal(fn(a, b, 1.0, seed), b), name + ' prog 1 != b'
        for pr in np.linspace(0, 1, 21):
            o = fn(a, b, pr, seed)
            assert o.shape == (H, W, 3) and o.dtype == np.uint8, name
            assert set(map(tuple, np.unique(o.reshape(-1, 3), axis=0).tolist())) <= pal, f'{name} off-palette @{pr}'
            assert np.array_equal(o, fn(a, b, pr, seed)), name + ' not deterministic'
        progs = np.linspace(0.02, 0.98, n)
        t = time.perf_counter()
        for pr in progs:
            fn(a, b, pr, seed)
        res[name] = (time.perf_counter() - t) / n * 1000
    return res


if __name__ == '__main__':
    print(preview())
    for k, v in check().items():
        print(f'{k:9s} {v:6.2f} ms/call')
