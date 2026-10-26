"""Cold-open hook + book identity (version B).  Owner: hook agent.

frame(t, g, rng) -> grid | None   (applied after shot / transition / beat pulse, before HUD + captions)
design(ver, shots, words, sync, SR, N) -> (N, 2) float32 audio layer

Beats (film seconds, version B):
  0.000-0.167  flash-forward: the whisperer (s35) stares at you, red tracking box on her eyes, turns red, tears away
  0.167-0.90   in the dark a Voynich glyph (the cover emblem, seed 408) fades up; an eye opens inside its loop,
               glances aside, then snaps to look straight at the viewer; a red tracking box closes in and locks
               («ВИЖУ ТЕБЯ»), the glyph then bursts into the pixel storm that builds the book
  0.62-1.22    the tracking box jumps up and frames the title: Voynich glyphs decode letter by letter into
               «РУКОПИСЬ / ВОЙНИЧА» (red plate) — fully readable before the first word (1.21)
  1.93         title slams with the book's arrival (Mara's first hit)
  2.10-3.35    grounding lines typed on the cover: «ЙЕЛЬСКИЙ УНИВЕРСИТЕТ» / «MS 408 · XV ВЕК» / «НЕ РАСШИФРОВАНА»
  3.55         stamp «ЭТО НАСТОЯЩАЯ КНИГА» slams across the emblem (Mara pulse)
  4.06-4.55    the cover opens: the whole label block breaks into rising pixels
  4.45-9.95    small HUD tag «РУКОПИСЬ ВОЙНИЧА / MS 408 · ЙЕЛЬ» top-left while the pages flip
  19.25-21.73  s8/s9 «В 1912-м её купили»: name plate «УИЛФРИД ВОЙНИЧ» + «АНТИКВАР» (out before the s10 faces)
  34.9-38.55   s16 hourglass: «АВТОР: НЕИЗВЕСТЕН», «РАДИОУГЛЕРОД:» + rolling digits locking on «1404–1438» at «назад»
  119.6-127.3  s37 clean museum ending: a quiet museum label under the vitrine
"""
import sys, math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/claude/voynich/look')
from pixelize import PALETTE, quantize

END = 127.5
GW, GH, FPS = 180, 320, 30
FCOND = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'
FMONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
P = {k: PALETTE[i].astype(np.uint8) for k, i in
     dict(void=0, void2=1, dark=2, d3=3, tile=4, tile2=5, mid=6, line=7, light=8, bone=9,
          g1=10, g2=11, g3=12, p1=13, p2=14, parch=15, red2=16, red=17).items()}   # BGR

ASM1, OPEN0 = 1.93, 4.06          # book.py (B): the book lands / the cover opens
HIT = 1.946                       # Mara's first hit (attack); pulses every 0.5352 s
GUIDE_X = 90                      # book_ui draws a dark vertical guide at x = 90 over everything: keep it in a letter gap


def _R():
    """the render module (as __main__ when render.py runs, or imported as 'render')"""
    for name in ('render', '__main__'):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, 'glyph_sprite') and hasattr(m, 'still'):
            return m
    import render as m
    return m


# ------------------------------------------------------------------ pixel text / glyph masks
@functools.lru_cache(None)
def _font(path, size):
    return ImageFont.truetype(path, size)


@functools.lru_cache(4096)
def _tmask(text, path, size):
    f = _font(path, size)
    w = int(math.ceil(f.getlength(text))) + 3
    im = Image.new('L', (max(1, w), size + 8), 0)
    d = ImageDraw.Draw(im); d.fontmode = '1'
    d.text((0, 0), text, font=f, fill=255)
    return np.array(im) > 127


def _tlen(text, path, size):
    return _font(path, size).getlength(text)


@functools.lru_cache(512)
def _gmask(seed, w, h):
    a = _R().glyph_sprite(seed)
    return cv2.resize(a, (max(1, w), max(1, h)), interpolation=cv2.INTER_AREA) > 90


def _outline(m, k=1):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0


class Layer:
    """overlay canvas: colour + coverage, composited onto the grid at the end"""
    def __init__(self):
        self.c = np.zeros((GH, GW, 3), np.uint8)
        self.m = np.zeros((GH, GW), bool)

    def paint(self, mask, x, y, col):
        x, y = int(round(x)), int(round(y))
        h, w = mask.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(GW, x + w), min(GH, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        self.c[y0:y1, x0:x1][sub] = P[col]
        self.m[y0:y1, x0:x1] |= sub

    def text(self, s, x, y, col='bone', font=(FMONO, 9), outline='void'):
        m = _tmask(s, *font)
        if outline:
            self._ol(m, x, y, outline)
        self.paint(m, x, y, col)
        return _tlen(s, *font)

    def _ol(self, m, x, y, col):
        pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
        pad[1:-1, 1:-1] = m
        self.paint(_outline(pad), x - 1, y - 1, col)

    def rect(self, x0, y0, x1, y1, col, fill=False):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        if fill:
            m[:] = True
        else:
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
        self.paint(m, x0, y0, col)

    def box(self, x0, y0, x1, y1, col='red', corners='bone', label=None, lab_col='bone'):
        self.rect(x0, y0, x1, y1, col)
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.rect(cx - 1, cy - 1, cx + 1, cy + 1, corners or col, fill=True)
        if label:
            lw = _tlen(label, FMONO, 9)
            self.rect(x0, y0 - 10, x0 + lw + 4, y0 - 1, col, fill=True)
            self.text(label, x0 + 2, y0 - 10, lab_col, outline=None)

    def comp(self, g):
        g[self.m] = self.c[self.m]
        return g


def _typed(s, t, t0, cps=48.0):
    """typewriter: the visible prefix of s at time t (whole characters), '' before t0"""
    if t < t0:
        return ''
    return s[:min(len(s), int((t - t0) * cps) + 1)]


def _sprite_t(t):
    return math.floor(t * 15 + 1e-6) / 15


# ------------------------------------------------------------------ the title «РУКОПИСЬ / ВОЙНИЧА»
T_SIZE = 22
T_Y1, T_Y2 = 85, 111
DEC0, DEC_STEP = 0.70, 0.034          # first letter resolves at DEC0, then one every DEC_STEP (15 letters -> 1.18)


@functools.lru_cache(None)
def _title_layout():
    out = []
    for li, (s, y) in enumerate((('РУКОПИСЬ', T_Y1), ('ВОЙНИЧА', T_Y2))):
        w = _tlen(s, FCOND, T_SIZE)
        best = None
        for dx in range(-6, 7):                       # nudge so the HUD guide (x = 90) falls in a gap between letters
            x0 = int(round(92 - w / 2)) + dx
            m = _tmask(s, FCOND, T_SIZE)
            col = GUIDE_X - x0
            hit = m[:, col].sum() if 0 <= col < m.shape[1] else 0
            score = hit * 10 + abs(dx)
            if best is None or score < best[0]:
                best = (score, x0)
        x0 = best[1]
        letters = []
        for i, ch in enumerate(s):
            lx = x0 + _tlen(s[:i], FCOND, T_SIZE)
            lw = max(4, int(round(_tlen(ch, FCOND, T_SIZE))) - 1)
            letters.append((ch, lx, lw))
        out.append((s, x0, y, w, letters))
    order = np.random.default_rng(408).permutation(sum(len(l[0]) for l in out))
    return out, order


def _draw_title(L, t):
    lay, order = _title_layout()
    st = _sprite_t(t)
    k = 0
    for li, (s, x0, y, w, letters) in enumerate(lay):
        n_res = 0
        for i, (ch, lx, lw) in enumerate(letters):
            tr = DEC0 + order[k] * DEC_STEP
            appear = DEC0 - 0.12 + (k % 5) * 0.02
            k += 1
            if t < appear:
                continue
            if t < tr:                                # still an unknown glyph, reshuffling on the sprite clock
                gm = _gmask(3000 + 37 * k + int(st * 15) % 7, lw, T_SIZE - 4)
                L.paint(_outline(np.pad(gm, 1)), lx - 1, y + 2 - 1, 'void')
                L.paint(gm, lx, y + 3, 'mid' if (int(st * 15) + k) % 3 else 'g3')
                continue
            n_res += 1
            fresh = t - tr < 1 / 15
            if li == 1:
                continue                               # line 2 is drawn on its plate below
            L.text(ch, lx, y, 'red' if fresh else 'bone', (FCOND, T_SIZE))
        if li == 1 and n_res:
            # the red plate grows under «ВОЙНИЧА» as its letters resolve
            frac = n_res / len(letters)
            px1 = x0 - 3 + (w + 6) * frac
            L.rect(x0 - 3, y + 1, px1, y + T_SIZE + 3, 'red', fill=True)
            for i, (ch, lx, lw) in enumerate(letters):
                kk = sum(len(l[0]) for l in lay[:1]) + i
                tr = DEC0 + order[kk] * DEC_STEP
                if t >= tr:
                    L.text(ch, lx, y, 'void' if t - tr < 1 / 15 else 'bone', (FCOND, T_SIZE), outline=None)


def _title_done():
    lay, order = _title_layout()
    return DEC0 + (len(order) - 1) * DEC_STEP


# ------------------------------------------------------------------ grounding lines + stamp on the cover
G_LINES = [('ЙЕЛЬСКИЙ УНИВЕРСИТЕТ', 2.08, 'bone'),
           ('MS 408 · XV ВЕК', HIT + 0.5352 + 0.02, 'bone'),
           ('НЕ РАСШИФРОВАНА', HIT + 2 * 0.5352 + 0.02, 'red')]
G_Y0, G_DY = 193, 11
STAMP = 'ЭТО НАСТОЯЩАЯ КНИГА'
T_STAMP = HIT + 3 * 0.5352            # 3.552, a Mara pulse
STAMP_Y = 160


def _draw_cover_block(L, t):
    _draw_title(L, t)
    for i, (s, t0, col) in enumerate(G_LINES):
        vis = _typed(s, t, t0, 55)
        if not vis:
            continue
        w = _tlen(s, FMONO, 9)
        x = int(round(92 - w / 2))
        L.text(vis, x, G_Y0 + i * G_DY, col)
        if len(vis) < len(s) and int(t * 15) % 2 == 0:
            cx = x + _tlen(vis, FMONO, 9) + 1
            L.rect(cx, G_Y0 + i * G_DY + 1, cx + 4, G_Y0 + i * G_DY + 8, 'bone', fill=True)
    if t >= T_STAMP:
        f = (FCOND, 11)
        w = _tlen(STAMP, *f)
        x = int(round(84 - w / 2))                    # right end clear of the TikTok button rail
        y = STAMP_Y
        first = t - T_STAMP < 1 / 15
        jx = 0 if t - T_STAMP > 2 / 15 else (1 if int(t * 30) % 2 else -1)
        L.rect(x - 4 + jx, y - 1, x + w + 3 + jx, y + 14, 'void', fill=True)
        L.rect(x - 3 + jx, y, x + w + 2 + jx, y + 13, 'bone' if first else 'red', fill=True)
        L.text(STAMP, x + jx, y - 1, 'void' if first else 'bone', f, outline=None)


@functools.lru_cache(None)
def _cover_final():
    L = Layer()
    _draw_cover_block(L, OPEN0 - 0.01)
    return L


def _burst(L, t, t0, dur=0.45, seed=9):
    """the finished label block breaks into pixels that rise and fade (the cover opening beneath it)"""
    ys, xs = np.where(L.m)
    cols = L.c[ys, xs]
    r = np.random.default_rng(seed)
    sp = r.uniform(30, 120, len(ys))
    dr = r.uniform(-6, 6, len(ys))
    dl = (ys - 80) / 160 * 0.12 + r.uniform(0, 0.12, len(ys))
    k = np.clip((t - t0 - dl) / dur, 0, 1)
    keep = r.random(len(ys)) > k ** 0.8
    ny = np.round(ys - sp * k ** 1.4).astype(int)
    nx = np.round(xs + dr * k + np.sin(ys * .3 + k * 6) * 2 * k).astype(int)
    ok = keep & (ny >= 0) & (ny < GH) & (nx >= 0) & (nx < GW)
    out = Layer()
    c = cols[ok].copy()
    kk = k[ok]
    redish = c[:, 2].astype(int) - c[:, 0].astype(int) > 60
    c[(kk > 0.3) & redish] = P['red2']
    c[(kk > 0.3) & ~redish & (c.astype(int).sum(1) > 300)] = P['line']
    c[kk > 0.6] = P['tile2']
    out.c[ny[ok], nx[ok]] = c
    out.m[ny[ok], nx[ok]] = True
    return out


# ------------------------------------------------------------------ the cold open
@functools.lru_cache(None)
def _whisper(z, red=False):
    R = _R()
    c = dict(z0=z, x0=.5, y0=.135)
    if red:
        return R.to_grid(R.cam(R.still('s35_whisperer.jpg'), 0, c), gamma=.7, contrast=1.6, tint=((0.2, 0.25, 1.0), .7))
    return R.to_grid(R.cam(R.still('s35_whisperer.jpg'), 0, c), gamma=.72, contrast=1.5)


EYE_G_H = 128                         # the big glyph of the cold open
EYE_GC = (88, 176)


@functools.lru_cache(None)
def _eye_glyph():
    a = _R().glyph_sprite(408)
    sc = EYE_G_H / a.shape[0]
    w = max(1, int(a.shape[1] * sc))
    m = cv2.resize(a, (w, EYE_G_H), interpolation=cv2.INTER_AREA) > 90
    # the loop that becomes an eye: largest background region not touching the border
    bg = (~m).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bg, connectivity=4)
    best = None
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if x == 0 or y == 0 or x + ww >= m.shape[1] or y + hh >= m.shape[0]:
            continue
        if best is None or area > best[4]:
            best = (x, y, ww, hh, area)
    x0, y0 = EYE_GC[0] - w // 2, EYE_GC[1] - EYE_G_H // 2
    if best is None:
        ex, ey, ew, eh = w // 2, EYE_G_H // 4, 20, 20
    else:
        ex, ey, ew, eh = best[0] + best[2] / 2, best[1] + best[3] / 2, best[2], best[3]
    return m, x0, y0, (x0 + ex, y0 + ey, ew, eh)


def _draw_eye(L, cx, cy, a, b_open, look):
    """almond eye, a = half width, b_open = half height now, look = pupil x offset"""
    if b_open < 0.6:
        L.rect(cx - a, cy, cx + a, cy, 'bone', fill=True)
        return
    yy, xx = np.mgrid[-12:13, -16:17].astype(np.float32)
    almond = (xx / a) ** 2 + (yy / b_open) ** 2 <= 1.0
    lid = _outline(almond)
    ir = max(2.6, a * 0.42)
    iris = ((xx - look) ** 2 + yy ** 2 <= ir * ir) & almond
    pupil = ((xx - look) ** 2 + yy ** 2 <= (ir * 0.5) ** 2) & almond
    glint = (np.abs(xx - look + 1) < 0.5) & (np.abs(yy + 1) < 0.5) & almond
    ox, oy = int(round(cx)) - 16, int(round(cy)) - 12
    L.paint(lid, ox, oy, 'void')
    L.paint(almond, ox, oy, 'bone')
    L.paint(iris, ox, oy, 'red')
    L.paint(pupil, ox, oy, 'void')
    L.paint(glint, ox, oy, 'bone')


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _cold_open(g, t, rng):
    st = _sprite_t(t)
    f = int(round(t * FPS))
    # ---- 0..4 (30 fps frames): the flash-forward
    if f <= 4:
        if f <= 1:
            g = _whisper(3.7).copy()
        elif f == 2:
            g = _whisper(4.6).copy()
        elif f == 3:
            g = _R().rgbsplit(_whisper(5.6, red=True), 3)
        else:
            g = _R().tear(_R().pixsort(_whisper(5.6, red=True), .3, 1.0), np.random.default_rng(4), 6, 40)
            g = (g.astype(np.float32) * 0.45).astype(np.uint8)
            g = quantize(g, dither=.1, contrast=1.0)
        if f <= 3:
            L = Layer()
            # red tracking box on her eyes (eyes ~ row 160 at these zooms)
            zz = [3.7, 3.7, 4.6, 5.6][f]
            hw = int(22 * zz / 3.7); hh = int(8 * zz / 3.7)
            L.box(90 - hw, 158 - hh, 90 + hw, 158 + hh, 'red', 'bone', label='ВИЖУ ТЕБЯ' if f >= 1 else None)
            g = L.comp(g)
        return g
    L = Layer()
    m, gx0, gy0, (ex, ey, ew, eh) = _eye_glyph()
    # ---- the glyph in the dark (0.167 .. ~0.95)
    if t < 0.98:
        lvl = st - 0.167
        col = 'tile2' if lvl < 1 / 15 else ('line' if lvl < 2 / 15 else 'bone')
        gm = m
        jx = int(rng.integers(-1, 2)) if t < 0.3 else 0
        if t >= 0.62:                                # it bursts into the pixel storm that builds the book
            k = min(1.0, (t - 0.62) / 0.34)
            ys, xs = np.where(m)
            r = np.random.default_rng(77)
            keep = r.random(len(ys)) > k ** 0.7
            ang = r.uniform(0, 2 * np.pi, len(ys)); spd = r.uniform(10, 90, len(ys))
            ny = np.round(ys + np.sin(ang) * spd * k ** 1.3 - 20 * k).astype(int) + gy0
            nx = np.round(xs + np.cos(ang) * spd * k ** 1.3).astype(int) + gx0
            ok = keep & (ny >= 0) & (ny < GH) & (nx >= 0) & (nx < GW)
            L.c[ny[ok], nx[ok]] = P['bone' if k < .35 else ('line' if k < .7 else 'mid')]
            L.m[ny[ok], nx[ok]] = True
        else:
            L.paint(_outline(np.pad(m, 1)), gx0 - 1 + jx, gy0 - 1, 'void')
            L.paint(m, gx0 + jx, gy0, col)
        # the eye opens in the loop at 0.27, glances aside, snaps to the viewer at 0.40
        if 0.27 <= t < 0.80:
            a = max(5.0, min(ew * 0.55, 13))
            bmax = max(3.0, min(eh * 0.32, a * 0.6))
            op = _ease((st - 0.27) / 0.13)
            if 0.62 <= t:                            # it stays open the longest, then closes as everything scatters
                op = 1 - _ease((st - 0.66) / 0.12)
            look = -a * 0.45 if st < 0.40 else 0.0
            _draw_eye(L, ex, ey, a, bmax * op, look)
    # ---- the tracking box: closes in on the eye, locks, then jumps up to frame the title
    if 0.27 <= t < 0.62:
        k = _ease((st - 0.27) / 0.20)
        tx0, ty0, tx1, ty1 = ex - 17, ey - 12, ex + 17, ey + 12
        bx = [4 + (tx0 - 4) * k, 18 + (ty0 - 18) * k, 175 + (tx1 - 175) * k, 300 + (ty1 - 300) * k]
        locked = st >= 0.47
        flash = 0.47 <= st < 0.47 + 1 / 15
        L.box(*bx, 'bone' if flash else 'red', 'bone', label='ВИЖУ ТЕБЯ' if locked else None,
              lab_col='void' if flash else 'bone')
    elif 0.62 <= t < ASM1:
        lay, _ = _title_layout()
        x0 = min(l[1] for l in lay) - 7; x1 = max(l[1] + l[3] for l in lay) + 6
        y0, y1 = T_Y1 - 4, T_Y2 + T_SIZE + 7
        k = _ease((st - 0.62) / 0.13)
        tx0, ty0, tx1, ty1 = ex - 17, ey - 12, ex + 17, ey + 12
        bx = [tx0 + (x0 - tx0) * k, ty0 + (y0 - ty0) * k, tx1 + (x1 - tx1) * k, ty1 + (y1 - ty1) * k]
        done = st >= _title_done() + 0.05
        if not (done and st > 1.45 and int(st * 15) % 2):    # after decoding the box blinks out by 1.6
            if st < 1.6:
                L.box(*bx, 'red', 'bone')
    # ---- the title decoding
    if t >= DEC0 - 0.12:
        _draw_title(L, t)
    return L.comp(g)


# ------------------------------------------------------------------ later identity beats
def _dossier(g, t):
    """s8/s9 «В 1912-м её купили»"""
    T0, T1 = 19.25, 21.58                               # gone before the s10 cut (21.77): the next faces are not Voynich
    L = Layer()
    name = 'УИЛФРИД ВОЙНИЧ'
    f = (FCOND, 13)
    w = _tlen(name, *f)
    n = len(name)
    k = min(n, int((t - T0) / 0.035) + 1)
    x, y = 8, 41
    if t < T1:
        frac = min(1.0, (t - T0) / 0.25)
        L.rect(x - 2, y, x - 2 + (w + 5) * frac, y + 16, 'red', fill=True)
        st = _sprite_t(t)
        for i, ch in enumerate(name):
            lx = x + _tlen(name[:i], *f)
            if i < k:
                L.text(ch, lx, y - 1, 'bone', f, outline=None)
            elif ch != ' ' and frac >= 1:
                gm = _gmask(5000 + i * 13 + int(st * 15) % 5, max(4, int(_tlen(ch, *f)) - 1), 11)
                L.paint(gm, lx, y + 3, 'red2')
        sub = _typed('АНТИКВАР', t, T0 + 0.55, 30)
        if sub:
            L.text(sub, x, y + 19, 'bone')
        return L.comp(g)
    k = (t - T1) / 0.15                                 # out: the plate drops out in scanline bands
    L.rect(x - 2, y, x + w + 3, y + 16, 'red', fill=True)
    L.text(name, x, y - 1, 'bone', f, outline=None)
    L.text('АНТИКВАР', x, y + 19, 'bone')
    rows = np.arange(GH)
    L.m &= (((rows * 7) % 10) / 10 >= k)[:, None]
    return L.comp(g)


def _radiocarbon(g, t):
    """s16 hourglass «Она написана… неизвестно кем. Шестьсот лет назад.»"""
    TA, TB, TC, TLOCK, TOUT = 34.94, 36.69, 36.86, 37.76, 38.32
    L = Layer()
    x = 8
    s1 = _typed('АВТОР: НЕИЗВЕСТЕН', t, TA, 40)
    if s1:
        L.rect(x - 2, 43, x + _tlen(s1, FMONO, 9) + 2, 53, 'void', fill=True)
        L.text(s1, x, 43, 'bone', outline=None)
    s2 = _typed('РАДИОУГЛЕРОД:', t, TB, 45)
    if s2:
        L.rect(x - 2, 56, x + _tlen(s2, FMONO, 9) + 2, 66, 'void', fill=True)
        L.text(s2, x, 56, 'line', outline=None)
    if t >= TC:
        f = (FMONO, 15)
        tgt = '1404–1438'
        if t < TLOCK:
            r = np.random.default_rng(int(t * 15))
            s = ''.join(ch if ch == '–' else str(r.integers(0, 10)) for ch in tgt)
            s = '1' + s[1:5] + '1' + s[6:]                # the centuries are already plausible while the rest spins
        else:
            s = tgt
        w = _tlen(tgt, *f)
        fresh = TLOCK <= t < TLOCK + 1 / 15
        L.rect(x - 3, 68, x + w + 3, 85, 'bone' if fresh else ('red' if t >= TLOCK else 'red2'), fill=True)
        L.text(s, x, 69, 'void' if fresh else 'bone', f, outline=None)
    if t >= TOUT:
        k = (t - TOUT) / 0.25
        rows = np.arange(GH)
        L.m &= (((rows * 7) % 10) / 10 >= k)[:, None]
    return L.comp(g)


def _plaque(g, t):
    """s37 clean museum ending: a quiet museum label under the vitrine"""
    T0, T1 = 119.6, 126.9
    lines = ['MS 408 · БИБЛИОТЕКА БЕЙНЕКЕ', 'ЙЕЛЬСКИЙ УНИВЕРСИТЕТ · США']
    col = 'line'
    if t > T1:
        col = 'tile2' if t < T1 + 0.2 else ('tile' if t < T1 + 0.4 else None)
    if col is None:
        return None
    L = Layer()
    for i, s in enumerate(lines):
        vis = _typed(s, t, T0 + i * 0.75, 18)
        if not vis:
            continue
        w = _tlen(s, FMONO, 7)
        L.text(vis, int(round(90 - w / 2)), 203 + i * 10, col, (FMONO, 7), outline=None)
    if not L.m.any():
        return None
    return L.comp(g)


def _tag(g, t):
    """small persistent HUD name tag while the pages flip"""
    L = Layer()
    T0 = 4.45
    a = _typed('РУКОПИСЬ ВОЙНИЧА', t, T0, 50)
    b = _typed('MS 408 · ЙЕЛЬ', t, T0 + 0.25, 50)
    if t > 9.80 and int(t * 30) % 2:                   # flickers away into the dive
        return None
    if a:
        L.text(a, 8, 41, 'bone')
    if b:
        L.text(b, 8, 50, 'line', outline=None)
    return L.comp(g)


# ------------------------------------------------------------------ plug-in entry points
def frame(t, g, rng):
    if t < 4.06:
        g = g.copy()
        if t < ASM1:
            return _cold_open(g, t, rng)
        L = Layer()
        _draw_cover_block(L, t)
        if ASM1 - 0.01 <= t < ASM1 + 2 / 30:             # the slam: colour fringes for two frames on Mara's hit
            sh = Layer(); sh.c[:] = 0
            ys, xs = np.where(L.m)
            for dx, col in ((-2, 'red'), (2, 'g3')):
                nx = np.clip(xs + dx, 0, GW - 1)
                sh.c[ys, nx] = P[col]; sh.m[ys, nx] = True
            g = sh.comp(g)
        return L.comp(g)
    if t < 4.70:
        out = _burst(_cover_final(), t, OPEN0)
        g = out.comp(g.copy())
        if t >= 4.45:
            g2 = _tag(g, t)
            g = g2 if g2 is not None else g
        return g
    if t < 9.95:
        return _tag(g.copy(), t)
    if 19.25 <= t < 21.78:
        return _dossier(g.copy(), t)
    if 34.94 <= t < 38.57:
        return _radiocarbon(g.copy(), t)
    if 119.6 <= t < END:
        return _plaque(g.copy(), t)
    return None


# ------------------------------------------------------------------ audio
def design(ver, shots, words, sync, SR, N):
    """cold-open sound: the flash-forward stinger, a reversed whisper in the left ear, the lock thump,
    a low thud under the stamp.  Everything audible in the speech band is over before the first word (1.21 s)."""
    import subprocess
    out = np.zeros((N, 2), np.float32)
    if ver != 'B':
        return out

    def put(x, t, gain=1.0):
        i = int(round(t * SR)); j = min(N, i + len(x))
        if j > i:
            out[i:j] += x[:j - i] * gain

    def thump(dur, f0, f1, decay, seed, crush=True):
        n = int(dur * SR); tt = np.arange(n) / SR
        f = f1 + (f0 - f1) * np.exp(-tt * 14)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * decay)
        if crush:
            x = np.tanh(x * 2.2)
            x = np.repeat(x[::6], 6)[:n]
            x = np.round(x * 24) / 24
            # keep it out of the speech band: a gentle one-pole low-pass
            from scipy.signal import butter, sosfilt
            x = sosfilt(butter(4, 220, 'low', fs=SR, output='sos'), x)
        x *= np.minimum(1, np.linspace(0, dur * 1000 / 6, n))      # 6 ms attack: no click
        return np.stack([x, x], 1).astype(np.float32)

    # 1) the flash-forward: a sub drop + a short crushed metallic screech (0.0 - 0.2), before any word
    put(thump(1.0, 140, 32, 4.0, 1), 0.0, 0.42)
    n = int(0.22 * SR); tt = np.arange(n) / SR
    r = np.random.default_rng(3)
    scr = sum(np.sin(2 * np.pi * (f + 40 * np.sin(2 * np.pi * 23 * tt)) * tt + r.uniform(0, 6)) for f in (1870, 1985, 2650, 3310))
    scr = np.tanh(scr * 1.5) * np.exp(-tt * 14) * np.minimum(1, tt / 0.004)
    scr = np.round(np.repeat(scr[::4], 4)[:n] * 12) / 12
    put(np.stack([scr * 0.9, scr], 1).astype(np.float32), 0.0, 0.10)

    # 2) a reversed whisper creeping into the left ear (0.17 - 1.12), gone before «Двести»
    try:
        cmd = ['ffmpeg', '-v', 'error', '-i', '/home/claude/voynich/audio/vo_whisper.mp3', '-af',
               f'aresample={SR},areverse,asetrate={SR}*0.86,aresample={SR},highpass=f=250,lowpass=f=6000,'
               'aecho=0.8:0.7:70|160:0.35|0.2', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
        w = np.frombuffer(subprocess.check_output(cmd), np.float32).copy()
        # the loudest stretch of the reversed take, 0.95 s of it
        L_ = int(0.95 * SR)
        e = np.convolve(np.abs(w), np.ones(2400) / 2400, 'same')
        i0 = int(np.argmax(np.convolve(e, np.ones(L_) / L_, 'valid'))) if len(w) > L_ else 0
        seg = w[i0:i0 + L_]
        seg = seg / (np.abs(seg).max() + 1e-9)
        env = np.interp(np.arange(len(seg)) / SR, [0, .25, .7, .88, .95], [0, .7, 1.0, .3, 0])
        seg = seg * env
        d = int(0.011 * SR)
        right = np.concatenate([np.zeros(d), seg[:-d]]) * 0.22
        put(np.stack([seg, right], 1).astype(np.float32), 0.17, 0.26)
    except Exception as ex:                              # never break the mix over a garnish
        print('hook: whisper skipped', ex)

    # 3) the lock on the viewer (0.47) and the stamp (3.55): low thuds only
    put(thump(0.45, 95, 38, 9.0, 2), 0.47, 0.30)
    put(thump(0.5, 80, 34, 8.0, 3), T_STAMP, 0.26)
    pk = np.abs(out).max()
    if pk > 0.5:
        out *= 0.5 / pk
    return out
