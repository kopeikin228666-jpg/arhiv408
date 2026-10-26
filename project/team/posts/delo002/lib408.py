"""shared kit for the ДЕЛО 002 carousel directions (A/B/C).
art layer: 180x225 grid, painted in float RGB, quantized (Bayer-dithered) to the 18-colour palette, x6 -> 1080x1350
text layer: 360x450 grid, aliased DejaVu in palette colours, x3 -> 1080x1350
"""
import sys, math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path[:0] = ['/home/claude/voynich/edit', '/home/claude/voynich/look']
import pixelize as P
import host as HO

AW, AH, AK = 180, 225, 6
TW, TH, TK = 360, 450, 3
OW, OH = 1080, 1350
RGB = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in P.PALETTE_HEX]
(VOID, VOID2, DARK, D3, TILE, TILE2, MID, LINE, LIGHT, BONE, G1, G2, G3, SEPIA, PARCH, PARCH2, RED2, RED) = RGB
PALSET = set(RGB)

FD = '/usr/share/fonts/truetype/dejavu/'
_FC = {}
def font(kind, size):
    p = {'cond': 'DejaVuSansCondensed-Bold.ttf', 'mono': 'DejaVuSansMono-Bold.ttf', 'monor': 'DejaVuSansMono.ttf',
         'sans': 'DejaVuSans-Bold.ttf', 'serif': 'DejaVuSerifCondensed-Bold.ttf', 'condr': 'DejaVuSansCondensed.ttf'}[kind]
    k = (p, size)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(FD + p, size)
    return _FC[k]


# ------------------------------------------------------------------ art painting (float RGB, AH x AW)
_YY, _XX = np.mgrid[0:AH, 0:AW].astype(np.float32)

def canvas(c=VOID, h=AH, w=AW):
    a = np.zeros((h, w, 3), np.float32); a[:] = c; return a

def rect(a, x0, y0, x1, y1, c, al=1.0):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    s = a[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    s[:] = s * (1 - al) + np.array(c, np.float32) * al

def vgrad(a, y0, y1, c0, c1, x0=0, x1=None):
    x1 = a.shape[1] if x1 is None else x1
    for y in range(max(0, int(y0)), min(a.shape[0], int(y1))):
        t = (y - y0) / max(1, (y1 - y0 - 1))
        a[y, x0:x1] = np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t

def glow(a, cx, cy, rx, c, k=1.0, ry=None, mode='add'):
    """mode 'add' = light added; 'lerp' = coloured light that replaces the hue (red light stays red, not brown)"""
    ry = rx if ry is None else ry
    h, w = a.shape[:2]
    yy, xx = (_YY, _XX) if (h, w) == (AH, AW) else np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 0.5) * k
    if mode == 'lerp':
        g = np.clip(g, 0, 1)[..., None]
        a[:] = a * (1 - g) + np.array(c, np.float32) * g
    else:
        a += g[..., None] * np.array(c, np.float32)

def poly(a, pts, c, al=1.0):
    if al >= 1:
        cv2.fillPoly(a, [np.round(np.array(pts)).astype(np.int32)], tuple(float(v) for v in c))
    else:
        m = np.zeros(a.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
        a[m > 0] = a[m > 0] * (1 - al) + np.array(c, np.float32) * al

def line(a, p0, p1, c, w=1):
    cv2.line(a, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), tuple(float(v) for v in c), w)

def px(a, x, y, c):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < a.shape[1] and 0 <= y < a.shape[0]:
        a[y, x] = c

def aura(a, cx, cy, rx, ry=None, k=0.9):
    """darkness that eats the light around him"""
    ry = rx if ry is None else ry
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 1.5) * k
    a *= (1 - g)[..., None]

def vignette(a, k=0.55, p=2.2):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
    a *= (1 - k * d ** p)[..., None]

def dust(a, n, seed, c=MID):
    r = np.random.default_rng(seed)
    for _ in range(n):
        px(a, r.integers(0, a.shape[1]), r.integers(0, a.shape[0]), c)

PAL_SCENE = np.array([P.PALETTE[i] for i in range(18) if i not in (13, 14, 15)], np.float32)   # no parchment: red light must not go brown
PAL_NOGREEN = np.array([P.PALETTE[i] for i in range(18) if i not in (10, 11, 12, 13, 14, 15)], np.float32)

def finish(a, dither=0.13, contrast=1.0, pal=None):
    """float RGB -> palette RGB uint8 (Bayer-dithered)"""
    b = np.clip(a, 0, 255).astype(np.uint8)[..., ::-1]
    q = P.quantize(b, palette=PAL_SCENE if pal is None else pal, dither=dither, contrast=contrast)
    return np.ascontiguousarray(q[..., ::-1])

def onebit(a, ink=VOID, paper=LIGHT, keep_red=True, dither=0.7, gamma=1.0, lift=0.0):
    """2-tone photocopy treatment (direction C): luminance -> Bayer threshold, red stays red"""
    lum = (0.3 * a[..., 0] + 0.55 * a[..., 1] + 0.15 * a[..., 2]) / 255.0
    lo, hi = np.percentile(lum, 3), np.percentile(lum, 99.5)
    lum = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)          # stretch: the photocopier finds its blacks and whites
    lum = np.clip(lum + lift, 0, 1) ** gamma
    h, w = lum.shape
    m = P.bayer(4)
    th = np.tile(m, (h // 4 + 1, w // 4 + 1))[:h, :w]
    b = (lum + th * dither) > 0.5
    out = np.zeros((h, w, 3), np.uint8); out[:] = ink; out[b] = paper
    if keep_red:
        red = (a[..., 0] > 70) & (a[..., 0] > a[..., 1] * 1.8) & (a[..., 0] > a[..., 2] * 1.8)
        hot = red & (a[..., 0] > 150)
        out[red] = RED2; out[hot] = RED
    return out


# ------------------------------------------------------------------ simple perspective
class Cam:
    def __init__(s, cx=90, cy=100, f=110):
        s.cx, s.cy, s.f = cx, cy, f
    def __call__(s, X, Y, Z):
        return (s.cx + s.f * X / Z, s.cy + s.f * Y / Z)
    def quad(s, a, P4, c, al=1.0):
        poly(a, [s(*p) for p in P4], c, al)


def fogc(c, z, fog=VOID, k=0.12):
    t = 1 - math.exp(-k * z)
    return tuple(c[i] * (1 - t) + fog[i] * t for i in range(3))


# ------------------------------------------------------------------ host sprites onto a quantized art grid (RGB uint8)
def host_rgba(pose='front', t=1.3, mood='scary', glow_=0.5, stare=0.0, h=None):
    s = HO.sprite_img(t, 0.0, False, 0, 3, mood, pose, sway=False, glow=glow_, stare=stare)
    s = np.ascontiguousarray(s[..., [2, 1, 0, 3]])
    if h is not None and h != s.shape[0]:
        w = max(1, int(round(s.shape[1] * h / s.shape[0])))
        if h >= s.shape[0] * 0.8:
            s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
        else:                                   # clean downscale: area-average, then snap to the palette
            al = s[..., 3:4].astype(np.float32) / 255
            pm = cv2.resize(s[..., :3].astype(np.float32) * al, (w, h), interpolation=cv2.INTER_AREA)
            aa = cv2.resize(al[..., 0], (w, h), interpolation=cv2.INTER_AREA)
            col = pm / np.maximum(aa[..., None], 1e-3)
            # keep the eyes hot: the red survives the averaging
            col = np.clip(col, 0, 255).astype(np.uint8)
            q = P.quantize(col[..., ::-1], dither=0.0, contrast=1.0)[..., ::-1]
            s = np.zeros((h, w, 4), np.uint8); s[..., :3] = q; s[..., 3] = (aa > 0.5) * 255
    return s

def paste_sprite(q, s, x, y, rim=None, rim_dir=(0, -1), rim2=None):
    """paste RGBA sprite s at top-left (x, y) into RGB uint8 q; rim: colour of the 1px edge facing the light"""
    h, w = s.shape[:2]
    m = s[..., 3] > 0
    if rim is not None:
        dx, dy = rim_dir
        sh = np.zeros_like(m)
        H, W = m.shape
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        sh[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = m[ys0:ys1, xs0:xs1]
        edge = m & ~sh
        s = s.copy(); s[edge, :3] = rim
    for yy in range(h):
        Y = y + yy
        if not (0 <= Y < q.shape[0]):
            continue
        row = m[yy]
        xs = np.nonzero(row)[0]
        for xx in xs:
            X = x + xx
            if 0 <= X < q.shape[1]:
                q[Y, X] = s[yy, xx, :3]
    return q

def shadow_under(a, cx, y, rx, k=0.7):
    """darken an ellipse on the float canvas (contact shadow)"""
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - y) / (rx * 0.18)) ** 2)) * k
    a *= (1 - g)[..., None]


# ------------------------------------------------------------------ text layer
class Txt:
    def __init__(s, w=TW, h=TH):
        s.im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        s.d = ImageDraw.Draw(s.im); s.d.fontmode = '1'

    def t(s, x, y, txt, f, fill=BONE, stroke=0, sfill=VOID, anchor='la'):
        s.d.text((x, y), txt, font=f, fill=tuple(fill) + (255,), stroke_width=stroke, stroke_fill=tuple(sfill) + (255,), anchor=anchor)
        b = s.d.textbbox((x, y), txt, font=f, anchor=anchor, stroke_width=stroke)
        return b

    def fit(s, txt, kind='cond', maxs=40, width=316, mins=10):
        """largest font of this kind (<= maxs) whose longest line fits the width"""
        for sz in range(maxs, mins - 1, -1):
            f = font(kind, sz)
            if max(s.d.textlength(l, font=f) for l in txt.split('\n')) <= width:
                return f
        return font(kind, mins)

    def lines(s, x, y, txt, kind='cond', maxs=30, fill=BONE, stroke=2, width=316, lhk=1.12, anchor='la'):
        """explicit \n lines, one size that fits them all; returns the y under the block"""
        f = s.fit(txt, kind, maxs, width - 2 * stroke - 2)
        for l in txt.split('\n'):
            s.t(x, y, l, f, fill, stroke, VOID, anchor)
            y += int(f.size * lhk)
        return y

    def w(s, txt, f):
        return s.d.textlength(txt, font=f)

    def box(s, x0, y0, x1, y1, fill=None, outline=None, width=1):
        s.d.rectangle([x0, y0, x1, y1], fill=None if fill is None else tuple(fill) + (255,),
                      outline=None if outline is None else tuple(outline) + (255,), width=width)

    def hl(s, x, y, txt, f, fg=BONE, bg=RED, padx=5, padt=2, padb=4, anchor='la'):
        """text on a solid block; returns bbox of the block"""
        b = s.d.textbbox((x, y), txt, font=f, anchor=anchor)
        s.box(b[0] - padx, b[1] - padt - 1, b[2] + padx - 1, b[3] + padb, fill=bg)
        s.d.text((x, y), txt, font=f, fill=tuple(fg) + (255,), anchor=anchor)
        return (b[0] - padx, b[1] - padt - 1, b[2] + padx - 1, b[3] + padb)

    def wrap(s, txt, f, width):
        out = []
        for para in txt.split('\n'):
            cur = ''
            for word in para.split(' '):
                tt = (cur + ' ' + word).strip()
                if s.d.textlength(tt, font=f) <= width:
                    cur = tt
                else:
                    out.append(cur); cur = word
            out.append(cur)
        return out

    def para(s, x, y, txt, f, fill=BONE, width=300, lh=None, stroke=0, sfill=VOID, anchor='la'):
        lh = lh or int(f.size * 1.18)
        for row in s.wrap(txt, f, width):
            s.t(x, y, row, f, fill, stroke, sfill, anchor)
            y += lh
        return y

    def rich(s, x, y, parts, f, stroke=0, sfill=VOID):
        """parts: [(text, colour), ...] on one line"""
        for txt, c in parts:
            s.t(x, y, txt, f, c, stroke, sfill)
            x += s.d.textlength(txt, font=f)
        return x

    def stamp(s, cx, cy, txt, f, col=RED, angle=-8, pad=6, border=3, seed=1, wear=0.18):
        tw = int(s.d.textlength(txt, font=f)) + 2 * pad + 2 * border
        bb = s.d.textbbox((0, 0), txt, font=f)
        th = bb[3] - bb[1] + 2 * pad + 2 * border
        im = Image.new('RGBA', (tw + 4, th + 4), (0, 0, 0, 0)); d = ImageDraw.Draw(im); d.fontmode = '1'
        d.rectangle([2, 2, tw + 1, th + 1], outline=tuple(col) + (255,), width=border)
        d.text((2 + border + pad, 2 + border + pad - bb[1]), txt, font=f, fill=tuple(col) + (255,))
        a = np.array(im)
        r = np.random.default_rng(seed)
        holes = r.random(a.shape[:2]) < wear
        holes = cv2.dilate(holes.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        holes &= r.random(a.shape[:2]) < 0.6
        a[holes, 3] = 0
        im = Image.fromarray(a).rotate(angle, resample=Image.NEAREST, expand=True)
        s.im.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

    def paste(s, im, x, y):
        s.im.alpha_composite(im, (int(x), int(y)))


# ------------------------------------------------------------------ chrome + output
def chrome(T, n, tag, left='АРХИВ 408 · ДЕЛО 002', col=LINE, y=22):
    f = font('mono', 11)
    T.t(22, y, left, f, col)
    T.t(338, y, f'{n}/7', f, RED, anchor='ra')
    if tag:
        T.t(338 - T.w(f'{n}/7', f) - 10, y, tag, f, col, anchor='ra')

def compose(q, T, path=None):
    bb = T.im.getbbox()
    if bb and (bb[0] < 20 or bb[1] < 20 or bb[2] > 340 or bb[3] > 430):
        print('  SAFE-ZONE', path, bb)
    big = Image.fromarray(q).resize((OW, OH), Image.NEAREST).convert('RGBA')
    a = np.array(T.im)
    a[..., 3] = np.where(a[..., 3] > 127, 255, 0)
    pal = np.array(RGB, np.int32)
    flat = a[..., :3].reshape(-1, 3).astype(np.int32)
    idx = ((flat[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
    a[..., :3] = pal[idx].reshape(a.shape[:2] + (3,)).astype(np.uint8)
    tl = Image.fromarray(a).resize((OW, OH), Image.NEAREST)
    big.alpha_composite(tl)
    out = big.convert('RGB')
    if path:
        out.save(path, optimize=False, compress_level=6)
    return out

def offpal(img):
    a = np.array(img).reshape(-1, 3)
    u = np.unique(a, axis=0)
    return [tuple(int(v) for v in c) for c in u if tuple(int(v) for v in c) not in PALSET]

def sheet(paths, out, cols=4, w=360, gap=8, bg=VOID2):
    ims = [Image.open(p).convert('RGB').resize((w, int(w * 1.25)), Image.LANCZOS) for p in paths]
    rows = (len(ims) + cols - 1) // cols
    h = int(w * 1.25)
    S = Image.new('RGB', (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), bg)
    for i, im in enumerate(ims):
        S.paste(im, (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
    S.save(out, quality=88)
    return S
