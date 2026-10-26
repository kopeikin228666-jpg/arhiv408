"""VFX / visual-horror layer for the Voynich reel (version B).  Owner: VFX agent.

post(g, t, shot_name, tl, u, rng) -> grid   (uint8 BGR 320x180, palette colours only, deterministic, ~1-3 ms)
Called on every 30 fps frame after transitions + beat pulse (+ hook), before HUD and captions.

Layers, in order of application (all keyed to film time t and the musical grid in sync.py):
  1. datamosh smear after the big hits of act II / the wall (blocks of the picture drag along motion vectors)
  2. breathing dithered vignette (darkness closes in with Mara's envelope)
  3. VHS tracking band (act III, rolling one screen per 8 pulses)
  4. Voynich glyphs infesting the frame edges (act II -> wall), stepping on every pulse
  5. eyes in the dark (s4 figure, s15 «EMPTY» box, s31b darkness, s37 hood - one subtle glint)
  6. the face that forms in the static (s32 held breath)
  7. HUD corruption: SUBJECT_01 -> SUBJECT_YOU (own pixel overlay)
  8. subliminal frames (1/15 s) on the words
The bottom ~84 px (y >= 236) are never touched while a caption is on screen.
"""
import os, sys, json, math, functools
import numpy as np, cv2

sys.path.insert(0, '/home/claude/voynich/look')
sys.path.insert(0, '/home/claude/voynich/edit')
from pixelize import PALETTE
import sync

E = '/home/claude/voynich/edit'
KF = '/home/claude/voynich/kf'
GW, GH, FPS = 180, 320, 30
CALM_Y = 236                                   # captions live in y 255..300; keep everything below this calm
P = {k: PALETTE[i].astype(np.uint8) for k, i in
     dict(void=0, void2=1, dark=2, d3=3, tile=4, tile2=5, mid=6, line=7, light=8, bone=9,
          g1=10, g2=11, g3=12, p1=13, p2=14, parch=15, red2=16, red=17).items()}
ON = os.environ.get('VOY_VER', 'B') == 'B'     # everything here is timed on Mara (version B)


# ------------------------------------------------------------------ timing data
@functools.lru_cache(None)
def _words():
    W = json.load(open(f'{E}/words.json'))
    lines = {}
    for w in W:
        lines.setdefault(w['line'], []).append(w)
    lines = [lines[k] for k in sorted(lines)]
    spans = []                                   # when a caption is on screen (same rule as render.caption_at)
    for i, ws in enumerate(lines):
        s0, s1 = ws[0]['t0'] - 0.20, ws[-1]['t1'] + 0.9
        if i + 1 < len(lines):
            s1 = min(s1, lines[i + 1][0]['t0'] - 0.24)
        spans.append((s0 - 0.05, s1 + 0.05))
    return spans


def calm(t):
    return any(a <= t < b for a, b in _words())


@functools.lru_cache(None)
def _beats():
    ev = sync.beat_events()
    bt = np.array([e[0] for e in ev])
    st = [(e[0], e[1]) for e in ev if e[2]]
    return bt, st


def _last_beat(t):
    bt, _ = _beats()
    i = int(np.searchsorted(bt, t + 1e-6)) - 1
    return i, (t - bt[i] if i >= 0 else 9.0)


def _last_hit(t, t_min=29.9):
    _, st = _beats()
    best = None
    for k, (h, s) in enumerate(st):
        if t_min <= h <= t + 1e-6:
            best = (k, h, s)
    return best


@functools.lru_cache(None)
def _env():
    """Mara's loudness at 30 fps, slow (0.6 s) and normalised 0..1 over the film"""
    b = np.load(f'{E}/bed_B.npy')
    n = len(b) // 533
    r = np.sqrt((b[:n * 533].astype(np.float64).reshape(n, 533) ** 2).mean(1))
    db = 20 * np.log10(r + 1e-6)
    k = np.ones(18) / 18
    db = np.convolve(db, k, mode='same')
    return np.clip((db + 30) / 26, 0, 1)


def env(t):
    e = _env()
    return float(e[min(len(e) - 1, max(0, int(t * FPS)))])


def tick(t):
    return int(t * 15 + 1e-6)                     # the 15 fps sprite clock


def hrand(*k):
    """deterministic hash -> [0,1)"""
    h = 2166136261
    for v in k:
        h = ((h ^ (int(v) & 0xffffffff)) * 16777619) & 0xffffffff
    h ^= h >> 13; h = (h * 0x5bd1e995) & 0xffffffff; h ^= h >> 15
    return h / 4294967296.0


# ------------------------------------------------------------------ sprites
def glyph_sprite(seed, size=120):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    im = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(im)
    w = max(3, size // 14)
    x = size * 0.2
    for k in range(rng.integers(2, 4)):
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
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


@functools.lru_cache(None)
def glyph_mask(seed, h, orient):
    a = glyph_sprite(seed)
    w = max(2, int(round(a.shape[1] * h / a.shape[0])))
    m = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA) > 70
    if orient == 0:          # left edge: lying on its side, feet to the wall
        m = np.rot90(m, -1)
    elif orient == 1:        # right edge
        m = np.rot90(m, 1)
    elif orient == 2:        # top edge: hanging upside down
        m = m[::-1]
    return np.ascontiguousarray(m)


@functools.lru_cache(None)
def text_mask(s):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 9)
    im = Image.new('L', (8 * len(s) + 8, 12), 0)
    d = ImageDraw.Draw(im); d.fontmode = '1'
    d.text((0, 0), s, font=f, fill=255)
    a = np.array(im) > 0
    xs = np.where(a.any(0))[0]
    return a[:, :xs.max() + 1]


def stamp(g, m, x, y, col, ymax=GH):
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(GW, x + w), min(ymax, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    g[y0:y1, x0:x1][sub] = col


@functools.lru_cache(None)
def _bayer():
    m = np.array([[0]])
    while m.shape[0] < 8:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    m = (m.astype(np.float32) + 0.5) / m.size
    return np.tile(m, (GH // 8 + 1, GW // 8 + 1))[:GH, :GW]


def _lum(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255


def tones(L, cols, gain=1.0, bias=0.0):
    """luminance 0..1 -> ordered-dither onto a short ramp of palette colours (pixel-art two/three-tone)"""
    n = len(cols)
    q = np.clip(np.floor((np.clip(L * gain + bias, 0, 1)) * (n - 1) + _bayer()[:L.shape[0], :L.shape[1]]), 0, n - 1).astype(np.int32)
    lut = np.array([P[c] for c in cols], np.uint8)
    return lut[q]


def _cover(img, w, h):
    ih, iw = img.shape[:2]
    if iw / ih > w / h:
        nw = int(ih * w / h); x0 = (iw - nw) // 2; img = img[:, x0:x0 + nw]
    else:
        nh = int(iw * h / w); y0 = (ih - nh) // 2; img = img[y0:y0 + nh]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _contrast(L, lo=2, hi=98, gamma=1.0):
    a, b = np.percentile(L, lo), np.percentile(L, hi)
    return np.clip((L - a) / (b - a + 1e-6), 0, 1) ** gamma


@functools.lru_cache(None)
def _skull():
    """the Whisperer's skull face from the s35 clip (frame 36), 720x1280 source"""
    cap = cv2.VideoCapture('/home/claude/voynich/clips/v35_whisperer.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 36)
    ok, f = cap.read()
    if not ok:
        f = cv2.resize(cv2.imread(f'{KF}/s35_whisperer.jpg'), (720, 1280))
    return f


@functools.lru_cache(None)
def img_sub(name):
    """subliminal frames, pre-rendered once"""
    if name == 'eye_red':               # the eye from the cold open, staring - blood two-tone
        im = cv2.imread(f'{KF}/s03_eye.jpg')
        im = im[250:1150, 0:768]
        L = _contrast(_lum(_cover(im, GW, GH)), 3, 99, 1.3)
        return tones(L, ['void', 'red2', 'red', 'bone'], 1.05)
    if name == 'whisper_neg':           # what the artist saw: the Whisperer, negative, red on black
        im = cv2.imread(f'{KF}/s35_whisperer.jpg')
        im = im[40:1300, 60:700]
        L = 1 - _contrast(_lum(_cover(im, GW, GH)), 2, 99.5, 0.8)
        L = L * (_contrast(_lum(_cover(im, GW, GH)), 2, 99, 1) > 0.08)
        return tones(L, ['void', 'void', 'red2', 'red', 'bone'], 1.0)
    if name == 'whisper_face':          # «шепчет»: her skull face, too close
        im = _skull()[150:440, 255:465]
        L = _contrast(_lum(_cover(im, GW, GH)), 2, 99.5, 1.2)
        return tones(L, ['void', 'g1', 'g2', 'g3', 'light'], 1.0)
    if name == 'reader_face':           # «смотришь»: the hooded reader, staring into the lens
        im = cv2.imread(f'{KF}/s30_pov.jpg')
        im = im[420:720, 255:525]
        L = _contrast(_lum(_cover(im, GW, GH)), 2, 99.5, 1.25)
        return tones(L, ['void', 'dark', 'tile', 'line', 'bone'], 1.0)
    if name == 'cell_face':             # the face in the empty cell of the 3x3 readers (58x104)
        im = _skull()[110:520, 245:475]
        L = _contrast(_lum(_cover(im, 58, 104)), 2, 99.5, 1.1)
        return tones(L, ['void', 'dark', 'tile2', 'mid', 'light'], 0.95)
    if name == 'static_face':           # luminance mask of her face for the static (0..1)
        im = _skull()[120:560, 215:505]
        L = _contrast(_lum(_cover(im, GW, GH)), 5, 99, 1.0)
        L = cv2.GaussianBlur(L, (0, 0), 1.2)
        return L
    raise KeyError(name)


# ------------------------------------------------------------------ 1. datamosh after the big hits
@functools.lru_cache(None)
def _mosh_plan(k):
    r = np.random.default_rng(5000 + k)
    bw, bh = 9, 8
    nb_y, nb_x = GH // bh, GW // bw
    val = r.random((nb_y, nb_x))
    # blocks cluster in a few horizontal "slices" (looks like broken P-frames, not confetti)
    rowboost = (r.random(nb_y) < 0.35) * 0.35
    val = np.clip(val - rowboost[:, None], 0, 1)
    vid = r.integers(0, 3, (nb_y, nb_x))
    vecs = []
    for _ in range(3):
        if r.random() < 0.6:
            vecs.append((int(r.choice([-1, 1]) * r.integers(1, 3)), 0))     # vertical drag (melting)
        else:
            vecs.append((0, int(r.choice([-1, 1]) * r.integers(1, 3))))
    return val, vid, vecs, bw, bh


def mosh(g, t, quiet):
    if t < 29.9 or t >= 118.9:
        return g
    hit = _last_hit(t)
    if hit is None:
        return g
    k, h, s = hit
    dt = t - h
    dur = 0.30 if h < 90 else 0.42
    if dt >= dur:
        return g
    wall = h >= 89.9
    amp = (0.22 if not wall else 0.34) * (1 - dt / dur) ** 0.7
    val, vid, vecs, bw, bh = _mosh_plan(k)
    sel = val < amp
    if quiet:
        sel = sel.copy(); sel[CALM_Y // bh:] = False
    if not sel.any():
        return g
    out = g.copy()
    grow = 1 + int(dt * 30) * (3 if wall else 2)          # the drag grows frame by frame
    for vi, (dy, dx) in enumerate(vecs):
        m = sel & (vid == vi)
        if not m.any():
            continue
        M = np.repeat(np.repeat(m, bh, 0), bw, 1)
        M = np.pad(M, ((0, GH - M.shape[0]), (0, GW - M.shape[1])))
        src = np.roll(g, (dy * grow, dx * grow), axis=(0, 1))
        out[M] = src[M]
    return out


# ------------------------------------------------------------------ 2. breathing vignette
@functools.lru_cache(None)
def _vig_field():
    yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
    dx = (xx - GW / 2) / (GW / 2)
    dy = (yy - GH * 0.46) / (GH / 2)
    return np.sqrt(dx ** 2 * 1.0 + dy ** 2 * 0.85)


def vignette(g, t, name):
    if t < 10.4 or name in ('s37', 's38'):
        return g
    if t < 30:
        base = 0.04
    elif t < 90:
        base = 0.08 + 0.10 * (t - 30) / 60
    else:
        base = 0.24 if t < 116 else 0.14
    e = env(t)
    # the dark breathes on Mara: slow 2-bar inhale over the envelope
    breath = 0.5 - 0.5 * math.cos(2 * math.pi * t / (16 * 0.53564))
    s = base + 0.10 * e + 0.06 * breath * (t > 30)
    d = _vig_field()
    thr = np.clip((d - (1.12 - s)) / 0.35, 0, 1)
    m = thr > _bayer()
    g = g.copy()
    g[m] = P['void']
    return g


# ------------------------------------------------------------------ 3. VHS tracking band
@functools.lru_cache(None)
def _rowidx():
    return np.arange(GW)


def vhs(g, t, quiet, name):
    if t < 64.3 or t >= 118.9 or name in ('s32', 's36'):
        return g
    per = 8 * 0.53564
    ph = ((t - 64.3) / per) % 1.0
    inten = 0.4 + 0.6 * min(1, (t - 64.3) / 40)
    y_c = int(-20 + ph * (GH + 40))
    hgt = 5 + int(6 * inten)
    tk = tick(t)
    out = None
    for y in range(max(0, y_c - hgt), min(GH, y_c + hgt)):
        if quiet and y >= CALM_Y:
            break
        if out is None:
            out = g.copy()
        a = 1 - abs(y - y_c) / hgt
        sh = int(round((hrand(y, tk, 3) - 0.5) * 8 * a * inten))
        if sh:
            out[y] = np.roll(g[y], sh, axis=0)
        if hrand(y, tk, 9) < 0.35 * a * inten:                # dropout specks
            x0 = int(hrand(y, tk, 10) * GW); ln = 3 + int(hrand(y, tk, 11) * 18)
            out[y, x0:x0 + ln] = P['light'] if hrand(y, tk) < .5 else P['void2']
    if out is None:
        return g
    yl = y_c + hgt                                             # the bright head-switch line
    if 0 <= yl < GH and not (quiet and yl >= CALM_Y):
        xs = (np.arange(GW) + tk * 7) % 11 < 7
        out[yl, xs] = P['tile2']
    return out


# ------------------------------------------------------------------ 4. glyph infestation
N_CRAWL = 34
_RAMP_T = [30.0, 38.6, 47.2, 55.7, 64.3, 72.9, 81.4, 90.0, 99.0, 104.0, 108.8, 113.0]
_RAMP_N = [0, 1, 2, 4, 6, 8, 10, 14, 18, 23, 29, 34]


@functools.lru_cache(None)
def _crawlers():
    r = np.random.default_rng(408)
    cs = []
    for i in range(N_CRAWL):
        ts = float(np.interp(i + 0.5, _RAMP_N, _RAMP_T))
        e = int(r.choice([0, 1, 2], p=[.42, .30, .28]))
        cs.append(dict(ts=ts, e=e, seed=int(r.integers(1, 9999)), h=int(r.integers(14, 23)),
                       p0=float(r.random()), v=float(r.choice([-1, 1]) * r.uniform(1.0, 3.5)),
                       d=int(r.integers(0, 9)), red=r.random() < 0.18, ink=r.random()))
    return cs


def glyphs(g, t, quiet, name):
    if t < 30 or t >= 118.9 or name in ('s32', 's36', 's37', 's38'):
        return g
    bi, bdt = _last_beat(t)
    step = bi + min(1.0, bdt / 0.09)                # each glyph lunges on the pulse, then freezes
    hit = _last_hit(t)
    flare = hit is not None and t - hit[1] < 2 / FPS
    tk = tick(t)
    out = g.copy()
    ymax = CALM_Y if quiet else GH
    for i, c in enumerate(_crawlers()):
        if t < c['ts']:
            break
        age_b = (t - c['ts']) / 0.53564                # age in pulses
        m = glyph_mask(c['seed'], c['h'], c['e'])
        mh, mw = m.shape
        enter = min(1.0, age_b / 4)                    # slides in from outside the frame over 4 pulses
        tw = (hrand(i, tk) < 0.25) * (1 if hrand(i, tk, 1) < .5 else -1)
        if c['e'] == 2:                                # top edge: crawls along x
            L = GW + mw
            pos = (c['p0'] * L + c['v'] * step) % (2 * L)
            pos = pos if pos < L else 2 * L - pos
            x = int(pos - mw) + tw
            y = int(-mh + (mh + c['d']) * enter)       # hanging from the top edge
        else:                                          # left / right edges: crawl along y, 40..300
            lo, hi = 40, 300
            L = hi - lo
            pos = (c['p0'] * L + c['v'] * step) % (2 * L)
            pos = pos if pos < L else 2 * L - pos
            y = int(lo + pos - mh / 2) + tw
            dep = int(-mw + (mw + c['d']) * enter)
            x = dep if c['e'] == 0 else GW - mw - dep
        if quiet and y + mh > CALM_Y:
            continue
        if flare or (c['red'] and t > 90):
            col = P['red'] if flare else P['red2']
        elif c['ink'] < 0.7:
            col = P['p1']
        else:
            col = P['tile2']
        stamp(out, m, x, y, col, ymax)
    return out


# ------------------------------------------------------------------ 5. eyes in the dark
def eye_pair(g, x, y, sep, col, w=1, ymax=GH):
    for ex in (x, x + sep):
        if 0 <= y < ymax and 0 <= ex and ex + w <= GW:
            g[y, ex:ex + w] = col


def _bright_center(g, x0, y0, x1, y1, thr=0.33):
    L = g[y0:y1, x0:x1].astype(np.float32).mean(2) / 255
    m = L > thr
    if m.sum() < 4:
        return None
    ys, xs = np.nonzero(m)
    return x0 + xs.mean(), y0 + ys.min()


_DARK_EYES = None


def _dark_eyes():
    """s31b: the readers in the dark - one pair opens on every pulse around the lips"""
    global _DARK_EYES
    if _DARK_EYES is None:
        r = np.random.default_rng(31)
        ev = []
        t0 = 95.10
        pos = []
        while len(pos) < 13:
            x = int(r.integers(6, 166)); y = int(r.integers(40, 250))
            if 70 <= x <= 104:
                continue
            if any(abs(x - a) < 16 and abs(y - b) < 12 for a, b in pos):
                continue
            pos.append((x, y))
        for j, (x, y) in enumerate(pos):
            ts = t0 + j * 0.53564 * (0.5 if j > 5 else 1.0)
            w = 2 if r.random() < .6 else 1
            sep = 4 + w * 2 + int(r.integers(0, 3))
            col = 'red' if j in (3, 9) else ('bone' if r.random() < .5 else 'light')
            ev.append((ts, x, y, sep, w, col, int(r.integers(0, 9999))))
        _DARK_EYES = ev
    return _DARK_EYES


def eyes(g, t, name, tl, quiet):
    ymax = CALM_Y if quiet else GH
    if name == 's4' and t >= 14.79:
        # the figure behind the case (SUBJECT_01 .61) - his eyes catch light: he is looking at you
        c = _bright_center(g, 18, 84, 56, 125, 0.30)
        if c is not None and not (15.06 <= t < 15.10):
            cx, cy = c
            g = g.copy()
            eye_pair(g, int(round(cx)) - 3, int(cy) + 5, 5, P['red'], 1, ymax)
        return g
    if name == 's15' and 32.16 <= t < 32.62:
        # inside the box the tracker calls «EMPTY .98»
        if 32.40 <= t < 32.44:                      # one blink
            return g
        g = g.copy()
        eye_pair(g, 39, 226, 6, P['light'], 2, ymax)
        return g
    if name == 's31b':
        g = g.copy()
        for k, (ts, x, y, sep, w, col, sd) in enumerate(_dark_eyes()):
            if t < ts:
                continue
            if 98.58 <= t < 98.58 + 2 / FPS:        # the wall hit: they all blink at once
                continue
            blink = hrand(sd, tick(t)) < 0.04
            if blink:
                continue
            dx = 1 if (t > 97.6 and k % 2 == 0) else 0        # ...and then turn to the lips
            eye_pair(g, x + dx, y, sep, P[col], w, ymax)
        return g
    if name == 's37' and 125.62 <= t < 126.42:
        # the single subtle detail: the hood behind the glass opens its eyes once, after the last word
        if 126.02 <= t < 126.09:
            return g
        g = g.copy()
        eye_pair(g, 87, 131, 5, P['tile2'], 1, ymax)
        return g
    return g


# ------------------------------------------------------------------ 6. the face in the static
def static_face(g, t, tl):
    # s32: 99.1-100.73.  render puts its own 1/15 s subliminal at bt(1) ~ tl 0.53 - leave that frame alone
    if 0.52 <= tl < 0.61:
        return g
    k = float(np.clip((tl - 0.15) / 1.20, 0, 1)) ** 1.6
    reveal = tl >= 1.40
    M = img_sub('static_face')
    r = np.random.default_rng(9900 + tick(t)).random((GH, GW)).astype(np.float32)
    out = g.copy()
    if reveal:                                   # for two sprite frames she is fully there, then the cut
        f = tones(M, ['void', 'void2', 'dark', 'tile', 'line', 'light'], 1.0)
        n = r < 0.35
        f[n] = g[n]
        return f
    a = M ** 1.4 * k
    out[r < a * 0.55] = P['tile']
    out[r < a * 0.28] = P['line']
    out[r < (a ** 2) * 0.12] = P['light']
    return out


# ------------------------------------------------------------------ 7. HUD corruption
def bracket_box(g, x0, y0, x1, y1, col, L=10):
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        xa, xb = sorted((x, x + sx * L)); ya, yb = sorted((y, y + sy * L))
        g[y, xa:xb + 1] = col
        g[ya:yb + 1, x] = col


def plate(g, x, y, text, bg, fg, conf=None, conf_col=None):
    m = text_mask(text)
    h, w = m.shape
    g[y:y + 10, x:x + w + 4] = P[bg]
    stamp(g, m[:10], x + 2, y, P[fg])
    if conf:
        cm = text_mask(conf)
        stamp(g, cm[:10], x + w + 7, y, P[conf_col or 'line'])


def hud(g, t, name, quiet):
    if name == 's17' and 41.37 <= t < 41.37 + 2 / FPS:
        # «НИЧЕЙ»: for 2 frames the tracker's guess under its plate reads the viewer
        g = g.copy()
        plate(g, 31, 72, 'SUBJECT_YOU', 'red', 'bone')
        return g
    if name == 's33' and t >= 105.12 - 0.03:
        # «ТЫ слишком долго смотришь»: the tracker locks onto the one watching - the whole frame is the box
        dt = t - 105.09
        if dt < 0.25 and int(dt * 30) % 2 == 1:      # stutters in
            return g
        g = g.copy()
        col = P['red']
        bracket_box(g, 4, 46, GW - 5, CALM_Y - 4, col, 12)
        plate(g, 4, 48, 'SUBJECT_YOU', 'red', 'bone', '1.00', 'red')
        if t >= 106.97:                              # «смотришь»
            m = text_mask('EYES: OPEN')
            if int(t * 15) % 4 != 3:
                stamp(g, m[:10], 6, 60, P['red'])
        return g
    if name == 's35' and 112.43 <= t < 113.11:
        # «ВНУТРИ»: the lock flickers back - she is inside the subject
        g = g.copy()
        if tick(t) % 3 != 2:
            bracket_box(g, 4, 46, GW - 5, CALM_Y - 4, P['red'], 12)
            plate(g, 4, 48, 'HOST: YOU', 'red', 'bone', '1.00', 'red')
        return g
    return g


# ------------------------------------------------------------------ 8. subliminals
# (start, frames@30fps, image, shot guard)
SUBS = [
    (27.733, 2, 'cell', 's13'),          # «НИКТО…»: a face fills the empty cell of the 3x3 readers
    (45.467, 2, 'eye_red', 's18'),       # «…НЕ СМОТРИТ.»: something does
    (61.400, 2, 'whisper_neg', 's23'),   # «ТО, ЧЕГО НЕЛЬЗЯ ВИДЕТЬ»: what the artist saw
    (89.467, 2, 'whisper_face', 's31'),  # «…ШЕПЧЕТ»: her face, too close
    (107.000, 2, 'reader_face', 's33'),  # «…СМОТРИШЬ»: he looks back
]


def subliminal(g, t, name, quiet):
    for t0, nf, img, guard in SUBS:
        f0, fn = int(round(t0 * FPS)), int(round(t * FPS))
        if name == guard and f0 <= fn < f0 + nf:
            if img == 'cell':
                g = g.copy()
                g[108:212, 62:120] = img_sub('cell_face')
                return g
            f = img_sub(img).copy()
            if quiet:                                 # the caption band stays as it was
                f[CALM_Y:] = g[CALM_Y:]
                f[CALM_Y - 1] = P['void']
            return f
    return g


# ------------------------------------------------------------------ entry point
def post(g, t, shot_name, tl, u, rng):
    if not ON or t < 10.4 or shot_name in ('book', 's38'):
        return g
    quiet = calm(t)
    if shot_name == 's32':
        return static_face(g, t, tl)
    g = mosh(g, t, quiet)
    g = vignette(g, t, shot_name)
    g = vhs(g, t, quiet, shot_name)
    g = glyphs(g, t, quiet, shot_name)
    g = eyes(g, t, shot_name, tl, quiet)
    g = hud(g, t, shot_name, quiet)
    g = subliminal(g, t, shot_name, quiet)
    return g
