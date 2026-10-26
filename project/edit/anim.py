"""In-shot animation (Animation agent).  animate(g, shot_name, tl, u, t, rng) -> grid.

Objects inside the stills move, in pixel-art style and on the music of version B:
sand falling in the hourglass, a lamp that flickers, a page that turns by itself, dust in light beams,
leaves and roots swaying under water, ripples, mist, a candle flame, a quill writing, breath on glass,
condensation drips, a sleeper breathing, glyphs lifting off the book, whispering lips, a swinging tag ...

Two techniques:
  * source-space patches: the keyframe itself is modified (sand level, tag rotation, a local warp of the chest / lips /
    quill) and only a small window is re-projected through the shot's own camera and re-quantised, so the motion follows
    the camera push, rotation, palette and dither exactly; only pixels that actually changed are pasted back.
  * grid-space sprites: particles, rings, drips, flames, glints, mist and light flicker drawn on the 180x320 grid with
    palette colours (or by stepping the underlying palette colour lighter / darker).
Everything is a pure function of (shot, film time): no state, deterministic, temporally coherent.
"""
import os, sys, math, json, functools
import numpy as np, cv2

sys.path.insert(0, '/home/claude/voynich/look')
sys.path.insert(0, '/home/claude/voynich/edit')
from pixelize import PALETTE, bayer
import sync

R = '/home/claude/voynich'
GW, GH = 180, 320
PAL = PALETTE.astype(np.uint8)                       # (18,3) BGR
NP = len(PAL)
# palette index names
VOID, VOID2, DARK, D3, TILE, TILE2, MID, LINE, LIGHT, BONE, G1, G2, G3, P1, P2, PARCH, RED2, RED = range(18)

# ------------------------------------------------------------------ palette machinery
_KEY = np.full(1 << 24, 255, np.uint8)
_k = (PAL[:, 0].astype(np.int64) << 16) | (PAL[:, 1].astype(np.int64) << 8) | PAL[:, 2].astype(np.int64)
_KEY[_k] = np.arange(NP, dtype=np.uint8)
PAL_X = np.vstack([PAL, np.zeros((256 - NP, 3), np.uint8)])          # index 255 (unknown) never used for writing


def pidx(g):
    """palette index of every pixel (255 for colours that are not in the palette, e.g. after an RGB split)"""
    g = g.astype(np.int32)
    return _KEY[(g[..., 0] << 16) | (g[..., 1] << 8) | g[..., 2]]


def _lut(pairs):
    a = np.arange(256, dtype=np.uint8)
    for i, j in pairs.items():
        a[i] = j
    return a


# one step lighter / darker inside each colour family
UP = _lut({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 9, 10: 11, 11: 12, 12: 7, 13: 14, 14: 15, 15: 9, 16: 17, 17: 15})
DN = _lut({0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 1, 11: 10, 12: 11, 13: 3, 14: 13, 15: 14, 16: 2, 17: 16})
UP2 = UP[UP]
DN2 = DN[DN]
# warm lamp light: greys warm up toward the parchment family
WARM = _lut({0: 1, 1: 2, 2: 13, 3: 13, 4: 13, 5: 14, 6: 14, 7: 15, 8: 15, 9: 9, 10: 13, 11: 13, 12: 14, 13: 14, 14: 15, 15: 9})
DIMW = _lut({0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 15, 9: 15, 10: 1, 11: 10, 12: 11, 13: 3, 14: 13, 15: 14, 16: 2, 17: 16})
LUM = PAL.astype(np.float32).mean(1)


def apply_lut(g, mask, lut, ys=None, xs=None):
    """step the palette colour of the masked pixels through lut (unknown colours stay as they are)"""
    if ys is None:
        ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return g
    i = pidx(g[ys, xs])
    j = lut[i]
    ok = i != 255
    g[ys[ok], xs[ok]] = PAL_X[j[ok]]
    return g


def put(g, xs, ys, col=None, lut=None):
    """draw single pixels (float coords ok) with a palette colour or a palette step of what is underneath"""
    xs = np.asarray(xs).astype(np.int32).ravel(); ys = np.asarray(ys).astype(np.int32).ravel()
    ok = (xs >= 0) & (xs < GW) & (ys >= 0) & (ys < GH)
    xs, ys = xs[ok], ys[ok]
    if len(xs) == 0:
        return g
    if lut is not None:
        return apply_lut(g, None, lut, ys, xs)
    if np.ndim(col) == 0:
        g[ys, xs] = PAL[col]
    else:
        g[ys, xs] = PAL[np.asarray(col).ravel()[ok]]
    return g


# ------------------------------------------------------------------ fast quantiser (same maths as pixelize.quantize, 7-bit LUT)
def _build_qlut():
    pal = PALETTE.astype(np.float32)
    w = np.array([0.8, 1.2, 1.0], np.float32)
    c = np.arange(128, dtype=np.float32) * 2 + 1.0
    out = np.empty((128, 128, 128), np.uint8)
    gg, rr = np.meshgrid(c, c, indexing='ij')                          # axes: b fixed per slice, then g, r
    for bi in range(128):
        flat = np.stack([np.full(gg.size, c[bi], np.float32), gg.ravel(), rr.ravel()], 1)
        d = (((flat[:, None, :] - pal[None]) * w) ** 2).sum(-1)
        out[bi] = d.argmin(1).reshape(128, 128).astype(np.uint8)
    return out


_QL = None
_B8 = bayer(8)


def quantize_fast(small8, dither, contrast, lift=0.0):
    """uint8 BGR (h,w,3) whose origin sits on a multiple of 8 of the grid -> palette indices"""
    global _QL
    if _QL is None:
        _QL = _build_qlut()
    img = small8.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * contrast + 0.5 + lift, 0, 1)
    h, w = img.shape[:2]
    if dither > 0:
        th = np.tile(_B8, (h // 8 + 1, w // 8 + 1))[:h, :w]
        img = img + th[..., None] * dither
    v = np.clip(img * 255.0, 0, 255).astype(np.int32) >> 1
    return _QL[v[..., 0], v[..., 1], v[..., 2]]


# ------------------------------------------------------------------ the shots' sources and cameras (copied from render.py)
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


def cam_M(u, c):
    e = 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, u)))
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


def zoom_of(u, c):
    e = 0.5 - 0.5 * math.cos(math.pi * min(1, max(0, u)))
    return c.get('z0', 1) + (c.get('z1', 1) - c.get('z0', 1)) * e


def gpt(M, x, y):
    """normalised source point(s) -> grid coords under view matrix M"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    px = M[0, 0] * x * 720 + M[0, 1] * y * 1280 + M[0, 2]
    py = M[1, 0] * x * 720 + M[1, 1] * y * 1280 + M[1, 2]
    return px / 2, py / 2


CAM = {
    's4': ('s04_museum.jpg', dict(z0=1.0, z1=1.16, x0=.5, y0=.56), dict(gamma=.75)),
    's8': ('s08_archive.jpg', dict(z0=1.0, z1=1.2, x0=.55, y0=.38), dict(gamma=.85)),
    's9': ('s09_tag.jpg', dict(z0=1.25, z1=1.32, x0=.52, y0=.5), dict()),
    's15': ('s15_reading.jpg', dict(z0=1.0, z1=1.12, x0=.47, y0=.6), dict(gamma=.8)),
    's16': ('s16_hourglass.jpg', dict(z0=1.12, z1=1.3, x0=.46, y0=.5, x1=.54, y1=.52, r0=-2, r1=2.5), dict()),
    's19': ('s19_water.jpg', dict(z0=1.18, z1=1.24, r0=0, r1=7, x0=.5, y0=.48), dict(gamma=.82)),
    's20': ('s20_pools.jpg', dict(z0=1.7, z1=1.05, x0=.5, y0=.66, x1=.5, y1=.52), dict(gamma=.85)),
    's23': ('s23_scribe.jpg', dict(z0=1.0, z1=1.14, x0=.45, y0=.53), dict()),
    's22': ('s22_underwater.jpg', dict(z0=1.28, z1=1.0, x0=.5, y0=.45), dict()),
    's26': ('s26_glass.jpg', dict(z0=1.0, z1=1.08, x0=.5, y0=.45), dict(gamma=.82)),
    's25': ('s25_handprint.jpg', dict(z0=1.0, z1=1.2, x0=.5, y0=.45), dict(gamma=.8)),
    's27': ('s27_sleep.jpg', dict(z0=1.12, z1=1.25, r0=0, r1=9, x0=.5, y0=.42), dict(gamma=.82)),
    's29': ('s29_bookopen.jpg', dict(z0=1.0, z1=1.16, x0=.5, y0=.52), dict(gamma=.8)),
    's30': ('s30_pov.jpg', dict(z0=1.05, z1=1.16, x0=.5, y0=.42), dict(gamma=.82)),
    's31': ('s31_lips.jpg', dict(z0=1.05, x0=.5, y0=.5), dict(gamma=.9, dither=.22)),
    's37': ('s37_final.jpg', dict(z0=1.0), dict(gamma=.8, dither=.08)),
}
CAM['s31b'] = CAM['s31']


def view_M(name, u):
    return cam_M(u, CAM[name][1])


def patch(g, name, u, gbox, mod, guard=False, M=None):
    """re-project a window of the shot's keyframe after mod(crop, ox, oy) changed it; paste only the changed pixels.
    gbox = (x0, y0, x1, y1) on the grid.  guard: keep pixels the shot's own fx already altered."""
    img_name, c, tone = CAM[name]
    S = still(img_name)
    if M is None:
        M = cam_M(u, c)
    x0, y0, x1, y1 = gbox
    x0 = max(0, int(x0) // 8 * 8); y0 = max(0, int(y0) // 8 * 8)
    x1 = min(GW, int(math.ceil(x1))); y1 = min(GH, int(math.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return g
    vw, vh = (x1 - x0) * 2, (y1 - y0) * 2
    M2 = M.copy(); M2[0, 2] -= 2 * x0; M2[1, 2] -= 2 * y0
    Mi = cv2.invertAffineTransform(M2)
    cs = np.array([[0, 0, 1], [vw, 0, 1], [0, vh, 1], [vw, vh, 1]], np.float64) @ Mi.T
    sx0 = int(max(0, cs[:, 0].min() - 6)); sy0 = int(max(0, cs[:, 1].min() - 6))
    sx1 = int(min(720, cs[:, 0].max() + 7)); sy1 = int(min(1280, cs[:, 1].max() + 7))
    if sx1 - sx0 < 2 or sy1 - sy0 < 2:
        return g
    crop = S[sy0:sy1, sx0:sx1]
    new = mod(crop.copy(), sx0, sy0)
    if new is None:
        return g
    M3 = M2.copy(); M3[:, 2] += M2[:, :2] @ np.array([sx0, sy0], np.float64)
    border = cv2.BORDER_REFLECT
    gam, con, dit, tint = tone.get('gamma', .78), tone.get('contrast', 1.25), tone.get('dither', .12), tone.get('tint')

    def grid_of(src):
        v = cv2.warpAffine(src, M3, (vw, vh), flags=cv2.INTER_AREA, borderMode=border)
        sm = cv2.resize(v, (vw // 2, vh // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        sm = np.clip(sm, 0, 1) ** gam
        return quantize_fast((sm * 255).astype(np.uint8), dit, con)
    p0, p1 = grid_of(crop), grid_of(new)
    ch = p0 != p1
    if guard:
        ch &= (pidx(g[y0:y1, x0:x1]) == p0)
    if ch.any():
        sub = g[y0:y1, x0:x1]
        sub[ch] = PAL[p1[ch]]
    return g


def local_warp(crop, ox, oy, fields):
    """fields: list of (cx, cy, rx, ry, dx, dy) in source pixels: a gaussian blob around (cx,cy) is pushed by (dx,dy)
    (scalars, or callables f(X, Y) -> arrays for non-rigid motion).  Smooth, seam-free."""
    h, w = crop.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) + ox, np.arange(h, dtype=np.float32) + oy)
    mx, my = X.copy(), Y.copy()
    for cx, cy, rx, ry, dx, dy in fields:
        wgt = np.exp(-(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2))
        ddx = dx(X, Y) if callable(dx) else dx
        ddy = dy(X, Y) if callable(dy) else dy
        mx -= wgt * ddx
        my -= wgt * ddy
    return cv2.remap(crop, (mx - ox).astype(np.float32), (my - oy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ music
_PULSES = np.array(sync.grid(0, 140))
_STRONG = np.array(sync.strong(0, 140))
_EV = None


def _events():
    global _EV
    if _EV is None:
        try:
            ev = sync.beat_events()
            _EV = (np.array([e[0] for e in ev]), np.array([e[1] for e in ev]))
        except Exception:
            _EV = (_PULSES, np.full(len(_PULSES), .5))
    return _EV


def since(t, pts=None):
    """(time since the last point, index of it); pts default = every pulse"""
    pts = _PULSES if pts is None else pts
    i = int(np.searchsorted(pts, t + 1e-6)) - 1
    if i < 0:
        return 1e9, -1
    return t - pts[i], i


def strength(i):
    tt, ss = _events()
    if i < 0 or i >= len(ss):
        return 0.0
    return float(ss[i])


def hit_env(t, tau=0.12, pts=None):
    dt, i = since(t, pts)
    return math.exp(-dt / tau) if dt < 8 * tau else 0.0


# ------------------------------------------------------------------ captions: keep the bottom calm while they are on
_W = json.load(open(f'{R}/edit/words.json'))
_SPANS = {}
for _w in _W:
    a, b = _SPANS.get(_w['line'], (1e9, -1e9))
    _SPANS[_w['line']] = (min(a, _w['t0']), max(b, _w['t1']))
_SPANS = sorted(_SPANS.values())


def cap_on(t):
    return any(a - 0.25 <= t <= b + 0.95 for a, b in _SPANS)


def calm_y(t):
    """lowest grid row where loose particles may go"""
    return 236 if cap_on(t) else GH


# ------------------------------------------------------------------ small helpers
def h01(*k):
    """deterministic hash -> [0,1)"""
    x = 0
    for v in k:
        x = (x * 1000003 + int(v) * 7919 + 12345) & 0xFFFFFFFF
    x ^= x >> 13; x = (x * 0x5bd1e995) & 0xFFFFFFFF; x ^= x >> 15
    return (x & 0xFFFFFF) / float(1 << 24)


@functools.lru_cache(None)
def _seeds(name, n, k=8):
    return np.random.default_rng(sum(ord(ch) * (i + 1) for i, ch in enumerate(name)) * 131 + n).random((n, k))


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def noise_field(seed, w, h, cell):
    """smooth value noise (h, w) in [0,1], tileable horizontally by construction of the caller"""
    rr = np.random.default_rng(seed)
    sm = rr.random((h // cell + 3, w // cell + 3)).astype(np.float32)
    return cv2.resize(sm, ((w // cell + 3) * cell, (h // cell + 3) * cell), interpolation=cv2.INTER_CUBIC)[:h, :w]


@functools.lru_cache(None)
def _mist_tex(seed):
    a = noise_field(seed, 360, 200, 24) * 0.65 + noise_field(seed + 1, 360, 200, 9) * 0.35
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return np.concatenate([a, a[:, ::-1]], 1)                 # mirrored -> seamless when scrolled


def mist(g, t, y0, y1, amount, speed=6.0, seed=3, lut=None, fade_bottom=True):
    """drifting mist: ordered-dither lighten of the underlying colours where a scrolling noise is dense"""
    tex = _mist_tex(seed)
    h = y1 - y0
    off = int(t * speed) % (tex.shape[1] - GW)
    n = tex[:h, off:off + GW].copy()
    n2 = tex[50:50 + h, (off // 2 + 97) % (tex.shape[1] - GW):][:, :GW]
    if n2.shape[1] < GW:
        n2 = tex[50:50 + h, :GW]
    n = 0.6 * n + 0.4 * n2
    if fade_bottom:
        n *= np.linspace(1, 0.15, h, dtype=np.float32)[:, None]
    th = np.tile(_B8, (h // 8 + 1, GW // 8 + 1))[:h, :GW] + 0.5
    m = (n * amount) > (th * 0.8 + 0.25)
    ys, xs = np.nonzero(m)
    return apply_lut(g, None, UP if lut is None else lut, ys + y0, xs)


def ring(g, cx, cy, r, aspect, lut, gap=None, ang0=0.0):
    """1-px ellipse outline drawn by stepping the colours underneath"""
    if r < 0.6:
        return put(g, [cx], [cy], lut=lut)
    n = max(8, int(2 * math.pi * r * 1.2))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + ang0
    if gap is not None:
        a = a[(np.cos(a - gap[0]) < math.cos(gap[1]))]
    xs = np.round(cx + r * np.cos(a)).astype(np.int32)
    ys = np.round(cy + r * aspect * np.sin(a)).astype(np.int32)
    k = np.unique(ys * 1000 + xs)
    return put(g, k % 1000, k // 1000, lut=lut)


# ------------------------------------------------------------------ glyphs (as in render.py)
@functools.lru_cache(None)
def glyph_mask(seed, h):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    size = 120
    im = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(im)
    w = max(3, size // 9)
    x = size * 0.2
    for k in range(rng.integers(1, 3)):
        kind = rng.integers(0, 4)
        if kind == 0:
            d.line([x, size * .15, x, size * .85], fill=255, width=w)
            d.arc([x - size * .05, size * .1, x + size * .25, size * .45], 200, 520, fill=255, width=w)
            x += size * .3
        elif kind == 1:
            d.ellipse([x, size * .45, x + size * .22, size * .78], outline=255, width=w)
            x += size * .27
        elif kind == 2:
            d.ellipse([x, size * .35, x + size * .2, size * .6], outline=255, width=w)
            d.line([x + size * .2, size * .48, x + size * .14, size * .9], fill=255, width=w)
            x += size * .26
        else:
            d.arc([x, size * .4, x + size * .24, size * .8], 60, 300, fill=255, width=w)
            d.line([x + size * .02, size * .62, x - size * .06, size * .9], fill=255, width=w)
            x += size * .28
    a = np.array(im)
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = h / a.shape[0]
    return cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(h))), interpolation=cv2.INTER_AREA) > 70


def stamp(g, m, cx, cy, col=None, lut=None):
    ys, xs = np.nonzero(m)
    return put(g, xs + int(cx - m.shape[1] / 2), ys + int(cy - m.shape[0] / 2), col=col, lut=lut)


# ================================================================== the shots
SHOT = {}


def shot(*names):
    def d(f):
        for n in names:
            SHOT[n] = f
        return f
    return d


def animate(g, shot_name, tl, u, t, rng):
    f = SHOT.get(shot_name)
    out = g
    if f is not None:
        out = f(g.copy(), tl, u, t)
    if 30.0 <= t < 85.2 and shot_name not in ('s24',):
        out = ash(out if out is not g else g.copy(), t, shot_name)
    return out


# ---- global: very sparse ash / dust floating through act II
def ash(g, t, name):
    S = _seeds('ash', 14, 6)
    lim = calm_y(t)
    xs, ys, cols = [], [], []
    for k in range(len(S)):
        a = S[k]
        life = 5 + a[0] * 6
        ph = (t + a[1] * life) / life
        cyc = int(ph); f = ph - cyc
        x0 = h01(k, cyc) * GW
        y0 = h01(k, cyc, 7) * 260
        x = x0 + math.sin(t * (0.5 + a[2]) + k) * 6 + (a[3] - .5) * 10 * f * life * .3
        y = y0 + f * life * (2 + a[4] * 3) - 6
        if y >= lim or f < .08 or f > .92:
            continue
        xs.append(x); ys.append(y)
    return put(g, xs, ys, lut=UP2)


# ---- s16 hourglass: the top level sinks, a sand stream falls in grains, the pile grows — one tick per pulse
HG_NECK, HG_TOP, HG_BASE, HG_APEX = (361.0, 603.0), 503.0, 790.0, 740.0


def _hg_progress(t, t0=32.7, t1=38.6):
    """sand drained 0..1, advancing a little on every pulse (eased), so the hourglass ticks with the music"""
    ps = [p for p in _PULSES if t0 - 0.6 < p < t1]
    n = len(ps)
    dt, i = since(t)
    k = sum(1 for p in ps if p <= t + 1e-6)
    step = smooth(min(1.0, dt / 0.32)) if k else 0.0
    return min(1.0, (max(0, k - 1) + step) / max(1, n))


def _hg_mod(p):
    L = HG_TOP + 40 * p                            # top sand surface sinks
    s = 1.0 + 0.55 * p                             # pile grows
    cx, ny = HG_NECK

    def hw(y):                                     # half width of the top bulb's sand cone at row y
        return np.clip(84 * (ny - y) / (ny - HG_TOP), 4, 84)

    def mod(crop, ox, oy):
        h, w = crop.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32) + ox, np.arange(h, dtype=np.float32) + oy)
        mx, my = X.copy(), Y.copy()
        # top: the sand cone L..neck is the old cone HG_TOP..neck, squeezed
        top = (Y >= HG_TOP - 7) & (Y < ny)
        inside_old = top & (np.abs(X - cx) <= hw(Y) + 3)
        yy = HG_TOP + (Y - L) * (ny - HG_TOP) / (ny - L)
        sand = inside_old & (Y >= L)
        mx = np.where(sand, cx + (X - cx) * hw(yy) / hw(Y), mx)
        my = np.where(sand, yy, my)
        empty = inside_old & (Y < L)               # what used to be sand is now empty glass (texture from above the old level)
        my = np.where(empty, HG_TOP - 15 - np.mod(Y - HG_TOP + 7, 17) * 0.8, my)
        # bottom: vertical stretch of the pile about its base, faded in at the sides and above
        wx = np.clip(1 - (np.abs(X - cx) / 78) ** 4, 0, 1)
        wy = np.clip((Y - 625) / 70, 0, 1) * (Y <= HG_BASE + 4)
        sy = HG_BASE - (HG_BASE - Y) / s
        my = np.where(Y > 620, Y + wx * wy * (sy - Y), my)
        return cv2.remap(crop, (mx - ox).astype(np.float32), (my - oy).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return mod, s


@shot('s16')
def a_s16(g, tl, u, t):
    p = _hg_progress(t)
    mod, s = _hg_mod(p)
    M = view_M('s16', u)
    bx, by = gpt(M, np.array([270, 452, 270, 452]) / 720, np.array([470, 470, 800, 800]) / 1280)
    g = patch(g, 's16', u, (bx.min() - 2, by.min() - 2, bx.max() + 3, by.max() + 3), mod, M=M)
    # the falling stream: grains scrolling down, a clump on every pulse, splashes on the pile
    apex = HG_BASE - (HG_BASE - HG_APEX) * s
    nx, ny = gpt(M, HG_NECK[0] / 720, (HG_NECK[1] + 4) / 1280)
    ax, ay = gpt(M, 359 / 720, (apex + 2) / 1280)
    n = int(abs(ay - ny)) + 1
    ys = np.linspace(ny, ay, n); xs = np.linspace(nx, ax, n)
    ph = np.floor((np.arange(n) - t * 70) / 2).astype(np.int64)
    bright = np.array([h01(v, 16) for v in ph]) > 0.42
    g = put(g, xs[bright], ys[bright], col=LIGHT)
    g = put(g, xs[~bright], ys[~bright], col=MID)
    dt, i = since(t)
    if dt < 0.5:                                   # a clump of sand falls from the neck on the pulse
        fall = min(1.0, (dt / 0.3) ** 1.6)
        cy = ny + (ay - ny) * fall
        k = np.arange(4)
        g = put(g, nx + (ax - nx) * fall + 0 * k, cy - k, col=BONE)
        if fall >= 1.0:                            # splash: grains bounce off the top of the pile
            e = (dt - 0.3) / 0.2
            for j in range(4):
                sx = (h01(i, j) - .5) * 2
                x = ax + sx * (2 + 6 * e)
                y = ay - 4 * math.sin(math.pi * min(1, e)) * (0.6 + h01(i, j, 3)) + 2 * e
                g = put(g, [x], [y], col=LIGHT)
    # loose grains rolling down the pile's flanks
    for j in range(5):
        a = (t * 1.3 + j * 0.37) % 1.0
        sd = -1 if j % 2 else 1
        x = ax + sd * a * 14 * (0.7 + 0.3 * h01(j, 5))
        y = ay + a * 9
        g = put(g, [x], [y], col=LIGHT if a < .7 else MID)
    return g


# ---- generic: dust motes floating in a light cone (source coords), brighter in the light
def motes(g, t, M, name, n, apex, base_l, base_r, lim=GH, lut=UP2, speed=1.0, dim_lut=UP, top_w=0.0):
    """apex = (x, y) top centre of the beam (source px, top_w = its width), base_l / base_r = bottom corners.
    motes drift slowly and twinkle"""
    S = _seeds(name + 'm', n, 6)
    ax, ay = apex
    xs, ys, xd, yd = [], [], [], []
    for k in range(n):
        a = S[k]
        v = (a[0] * 0.8 + 0.2) * speed
        f = (a[1] + t * 0.035 * v) % 1.0                       # height inside the beam (slowly sinking)
        w = (a[2] + 0.18 * math.sin(t * (0.4 + a[3]) + 6 * a[4])) % 1.0
        lx = ax - top_w / 2 + (base_l[0] - ax + top_w / 2) * f; rx = ax + top_w / 2 + (base_r[0] - ax - top_w / 2) * f
        y = ay + (base_l[1] - ay) * f
        x = lx + (rx - lx) * w + 3 * math.sin(t * 1.1 + k)
        tw = math.sin(t * (2 + 3 * a[5]) + k * 1.7)
        if tw < -0.55:
            continue
        (xs if tw > 0.1 else xd).append(x / 720); (ys if tw > 0.1 else yd).append(y / 1280)
    if xs:
        gx, gy = gpt(M, xs, ys); ok = gy < lim
        g = put(g, gx[ok], gy[ok], lut=lut)
    if xd:
        gx, gy = gpt(M, xd, yd); ok = gy < lim
        g = put(g, gx[ok], gy[ok], lut=dim_lut)
    return g


def page_flap(g, M, G1, G2, R1, R2, L1, L2, th, lift=0.95, curl=0.25, front=(BONE, PARCH), back=(P2, PARCH), edge=P1):
    """a page turning about the gutter G1-G2 (source px) from the right page (R vectors, th=0) to the left (L, th=pi)"""
    if th <= 0.01 or th >= math.pi - 0.01:
        return g
    c, s = math.cos(th), math.sin(th)
    pts = []
    for G, Rv, Lv in ((G1, R1, L1), (G2, R2, L2)):
        Rv = np.array(Rv, float); Lv = np.array(Lv, float); G = np.array(G, float)
        ln = np.linalg.norm(Rv)
        P = G + Rv * (1 + c) / 2 + Lv * (1 - c) / 2 + np.array([0, -ln * lift * s])
        # a slight curl: the middle of the page lags behind the edge
        c2, s2 = math.cos(th * (1 - curl)), math.sin(th * (1 - curl))
        Mi = G + 0.5 * (Rv * (1 + c2) / 2 + Lv * (1 - c2) / 2 + np.array([0, -ln * lift * s2]))
        pts.append((G, Mi, P))
    poly = [pts[0][0], pts[0][1], pts[0][2], pts[1][2], pts[1][1], pts[1][0]]
    px, py = gpt(M, [p[0] / 720 for p in poly], [p[1] / 1280 for p in poly])
    P = np.stack([px, py], 1)
    # shading: the face rising toward the lamp goes from bone to warm parchment; the back of the page is in shade
    rise = s
    if c >= 0:
        (ca, cb), fr = front, min(1.0, rise * 1.3)
    else:
        (ca, cb), fr = back, min(1.0, (1 - rise) * 1.2 + 0.15)
    idx = np.zeros((GH, GW), np.uint8)
    cv2.fillPoly(idx, [np.round(P).astype(np.int32)], 1)
    ys, xs = np.nonzero(idx)
    gx0, gy0 = P[0]
    gx1, gy1 = (P[2] + P[3]) / 2
    d = ((xs - gx0) * (gx1 - gx0) + (ys - gy0) * (gy1 - gy0)) / max(1.0, (gx1 - gx0) ** 2 + (gy1 - gy0) ** 2)
    fpx = np.clip(fr + 0.35 * (0.5 - d), 0, 1)                 # the gutter side a bit darker
    th8 = _B8[ys % 8, xs % 8] + 0.5
    g[ys, xs] = np.where((fpx > th8)[:, None], PAL[cb], PAL[ca])
    if edge is not None:
        e = np.round(P[2:4]).astype(np.int32)
        cv2.line(idx, tuple(int(v) for v in e[0]), tuple(int(v) for v in e[1]), 2, 1)
        g[idx == 2] = PAL[edge]
    return g


def flicker_lut(t, seed, base=None, dim=DN):
    """lamp/candle flicker: which palette step to apply this frame (None, DN, DN2, UP) — dips on the pulses"""
    dt, i = since(t)
    st = strength(i)
    f = int(round(t * 15))
    if dt < 1 / 15 + 1e-3:
        return dim[dim] if st > 0.6 else dim
    if dt < 2 / 15 + 1e-3 and h01(i, seed) < 0.5:
        return None
    if dt < 3 / 15 + 1e-3 and h01(i, seed, 2) < 0.45:
        return dim
    if h01(f, seed, 9) < 0.05:
        return dim
    return base


# ---- s15 reading room: the lamp flickers on the pulses, dust in its light, a page turns by itself
@shot('s15')
def a_s15(g, tl, u, t):
    M = view_M('s15', u)
    lit_from = 1.2                                   # the shot builds out of black blocks first
    if tl > lit_from:
        g = motes(g, t, M, 's15', 22, (545, 664), (440, 800), (665, 800), lim=calm_y(t))
    # the page: lifts at 31.62 (pulse), falls to the left on 32.16, a second one starts to rise at the end
    ga, gb = (510, 780), (437, 807)
    ra, rb = (85, 11), (93, 13)
    la, lb = (-68, -18), (-67, -22)
    p1 = 30.016 + 3 * 0.53564
    for start, dur, top in ((p1 - 0.25, 0.80, math.pi), (p1 + 0.8, 0.9, 0.5)):
        k = (t - start) / dur
        if 0 < k:
            th = top * smooth(min(1.0, k)) if top < math.pi else math.pi * (k ** 1.6 if k < 1 else 1)
            th += 0.06 * math.sin(t * 19) * (1 - min(1, k))
            if k < 1 or top < math.pi:
                g = flap(g, 's15', u, M, ga, gb, ra, rb, la, lb, th, guard=tl < 1.5)
    # the lamp: dips on every pulse (a stutter now and then)
    if tl > 0.5:
        lut = flicker_lut(t, 15, dim=DIMW)
        if lut is not None:
            ys, xs = np.mgrid[80:250, 40:180]
            sub = pidx(g[80:250, 40:180])
            m = np.isin(sub, [P1, P2, PARCH, BONE, LIGHT, G2, G3, LINE, MID])
            g = apply_lut(g, None, lut, ys[m], xs[m])
    return g


# ---- s4 museum vitrine: dust motes sinking through the beam, a page of the manuscript lifts and falls back on the hit
@shot('s4')
def a_s4(g, tl, u, t):
    M = view_M('s4', u)
    g = motes(g, t, M, 's4', 34, (378, 120), (268, 690), (492, 690), top_w=110, lim=calm_y(t))
    lift0, snap = 13.72, 14.79                          # rises on a pulse, slaps down on Mara's hit
    if t >= lift0:
        if t < snap:
            k = (t - lift0) / (snap - lift0)
            th = 1.1 * smooth(k * 1.3) + 0.12 * math.sin(t * 13) * min(1, k * 3)
        else:
            k = (t - snap) / 0.2
            th = 1.1 * max(0.0, 1 - k) ** 2 + (0.08 * math.sin(k * 6) * math.exp(-k) if k < 3 else 0)
        g = flap(g, 's4', u, M, (377, 718), (377, 806), (101, -21), (101, -24), (-97, -25), (-97, -22), th,
                 lift=0.18, curl=0.3, shade_front=0.3)
    return g


def flap_quads(G1, G2, R1, R2, L1, L2, th, lift, curl):
    """source-space corner points of the turning page: two strips (gutter->middle, middle->edge)"""
    c, sn = math.cos(th), math.sin(th)
    c2, s2 = math.cos(th * (1 - curl)), math.sin(th * (1 - curl))
    out = []
    for G, Rv, Lv in ((G1, R1, L1), (G2, R2, L2)):
        G = np.array(G, float); Rv = np.array(Rv, float); Lv = np.array(Lv, float)
        ln = np.linalg.norm(Rv)
        P = G + Rv * (1 + c) / 2 + Lv * (1 - c) / 2 + np.array([0, -ln * lift * sn])
        Mi = G + 0.5 * (Rv * (1 + c2) / 2 + Lv * (1 - c2) / 2 + np.array([0, -ln * lift * s2]))
        out.append((G, Mi, P))
    return out


def flap_mod(G1, G2, R1, R2, L1, L2, th, lift=0.95, curl=0.25, shade_front=0.35, shade_back=0.45):
    """mod() for patch(): the page's own picture is carried on the turning leaf (front = right page, back = left page)"""
    q = flap_quads(G1, G2, R1, R2, L1, L2, th, lift, curl)
    back = math.cos(th) < 0
    V = L1, L2 if back else None
    src_v = (L1, L2) if back else (R1, R2)
    sn = math.sin(th)
    shade = 1 - (shade_back * (0.6 + 0.4 * sn) if back else shade_front * sn)

    def mod(crop, ox, oy):
        o = np.array([ox, oy], float)
        out = crop.copy()
        for a, b in ((0, 1), (1, 2)):                         # two strips: gutter-mid, mid-edge
            fa, fb = a * 0.5, b * 0.5
            srcq = np.array([np.array(G1) + np.array(src_v[0]) * fa, np.array(G1) + np.array(src_v[0]) * fb,
                             np.array(G2) + np.array(src_v[1]) * fb, np.array(G2) + np.array(src_v[1]) * fa]) - o
            dstq = np.array([q[0][a], q[0][b], q[1][b], q[1][a]]) - o
            if back:                                          # the left page lands mirrored: gutter stays the gutter
                pass
            H = cv2.getPerspectiveTransform(srcq.astype(np.float32), dstq.astype(np.float32))
            w = cv2.warpPerspective(crop, H, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            m = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillConvexPoly(m, np.round(dstq).astype(np.int32), 1)
            sh = shade * (0.92 if a == 0 else 1.0)
            out[m > 0] = np.clip(w[m > 0].astype(np.float32) * sh, 0, 255).astype(np.uint8)
        # the leaf's outer edge catches the light
        e = np.round(np.array([q[0][2], q[1][2]]) - o).astype(np.int32)
        cv2.line(out, tuple(int(v) for v in e[0]), tuple(int(v) for v in e[1]), (200, 214, 214), 2)
        return out
    return mod, q


def flap(g, name, u, M, G1, G2, R1, R2, L1, L2, th, guard=False, **kw):
    if th <= 0.02 or th >= math.pi - 0.02:
        return g
    mod, q = flap_mod(G1, G2, R1, R2, L1, L2, th, **kw)
    pts = np.array([p for side in q for p in side] + [np.array(G1) + R1, np.array(G2) + R2, np.array(G1) + L1,
                                                       np.array(G2) + L2])
    gx, gy = gpt(M, pts[:, 0] / 720, pts[:, 1] / 1280)
    return patch(g, name, u, (gx.min() - 2, gy.min() - 2, gx.max() + 3, gy.max() + 3), mod, guard=guard, M=M)
