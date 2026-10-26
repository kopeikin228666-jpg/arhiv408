# ============================================================================ 8. inkbleed
def _smooth_noise(rng, gy, gx, interp=cv2.INTER_CUBIC):
    n = cv2.resize(rng.random((gy, gx)).astype(np.float32), (W, H), interpolation=interp)
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-6)


@functools.lru_cache(maxsize=16)
def _ink_plan(seed):
    """Voynich words (procedural gallows glyphs) + spatter drops stamped as ink seeds -> bleed field F
    (distance from the ink, warped by blot noise and horizontal vellum fibres) and its quantiles."""
    rng = _rng(seed, 88)
    ink = np.zeros((H, W), np.uint8)
    rows = 4
    for r in range(rows):                                       # one word per band, alternating sides
        dist, gh, centre, anchor = _glyph_field(int(seed) * 7 + r + 1)
        m = (dist < 2.6).astype(np.uint8) * 255
        ys, xs = np.nonzero(m)
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h = int(rng.integers(44, 66))
        w = max(4, int(m.shape[1] * h / m.shape[0]))
        if w > W - 6:
            h = int(h * (W - 6) / w); w = W - 6
        g = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 60
        cy = int(H * (r + 0.5) / rows + rng.integers(-14, 15))
        cx = int(W * (0.36 if r % 2 == 0 else 0.64) + rng.integers(-12, 13))
        y0, x0 = cy - h // 2, cx - w // 2
        yy0, xx0 = max(0, y0), max(0, x0)
        yy1, xx1 = min(H, y0 + h), min(W, x0 + w)
        ink[yy0:yy1, xx0:xx1] |= g[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0]
    for _ in range(26):                                         # spatter: drops flicked off the pen
        y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
        rr = int(rng.choice([1, 1, 1, 2, 2, 3]))
        cv2.circle(ink, (x, y), rr, 1, -1)
    F = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    blot = _smooth_noise(rng, 14, 8)
    fib = cv2.resize(rng.random((H // 2, W // 10)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    fib = (fib - fib.mean()) * 2.0
    F = F * (1.0 + 0.45 * blot) + 4.0 * fib * np.clip(F / 6.0, 0, 1)
    F[ink > 0] = -1.0
    q = np.quantile(F, np.linspace(0, 1, 513)).astype(np.float32)
    return F, q


def _cov_thr(q, c):
    return float(np.interp(c * (len(q) - 1), np.arange(len(q)), q))


def inkbleed(a, b, prog, seed=0):
    """Voynich words are written onto a in wet iron-gall ink, the ink bleeds along the vellum fibres into
    blots, b soaks through from the heart of each stroke until the page is drowned in it."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    F, q = _ink_plan(seed)
    cb = _ss(0.10, 0.93, p)                                     # b coverage: 0.5 at p ~ 0.51
    ci = min(1.0, cb + 0.04 + 0.30 * math.sin(math.pi * min(1.0, p / 0.93)) ** 1.3)
    Ti = _cov_thr(q, ci) if p > 0.015 else -0.5
    mi = F < Ti
    out = a.copy()
    out[mi] = C['void']
    wet = mi & (F > Ti - 2.5) & (_B2 < 0.5)                     # wet sheen inside the ink edge
    out[wet] = C['red2']
    mi8 = mi.astype(np.uint8)
    rim = (cv2.dilate(mi8, _KX) > 0) & ~mi
    out[rim] = C['red2']
    soak = (cv2.dilate(mi8, _K3, iterations=2) > 0) & ~mi & ~rim   # the vellum darkens around the blot
    if soak.any():
        out[soak] = PAL[_PREV[_PREV[to_idx(out[soak][None])[0]]]]
    if cb > 0:
        Tb = _cov_thr(q, cb)
        mb = F < Tb
        out[mb] = b[mb]
        edge = (cv2.dilate(mb.astype(np.uint8), _KX) > 0) & ~mb & mi
        out[edge] = C['p1']
    return out


# ============================================================================ 9. pageturn
@functools.lru_cache(maxsize=16)
def _page_geo(seed):
    rng = _rng(seed, 99)
    phi = math.radians(float(rng.uniform(24, 40)))
    up = (int(seed) // 2) % 2 == 1                              # peel from the bottom-right or the top-right corner
    nx, ny = math.cos(phi), (-math.sin(phi) if up else math.sin(phi))
    s = _XX * nx + _YY * ny
    corners = [x * nx + y * ny for x in (0, W) for y in (0, H)]
    kappa = float(rng.uniform(0.74, 0.86))                     # curl: the flap shows a little less than it turned
    return nx, ny, s.astype(np.float32), min(corners), max(corners), kappa


def pageturn(a, b, prog, seed=0):
    """a is a vellum leaf: its corner lifts, the leaf turns over (back side = parchment with a's image
    showing through in reverse), casting a shadow on b, and lands off the left edge.  The book is the spine."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    nx, ny, s, smin, smax, kap = _page_geo(seed)
    ez = 0.45 * p + 0.55 * _ss(0, 1, p)                         # symmetric ease -> half the page turned at p=0.5
    c = smax + 4 - (smax - smin + 8) * ez                       # fold line position (signed distance along n)
    turned = s > c
    out = np.where(turned[..., None], b, a)
    # shadow the lifted leaf throws on b, right next to the fold
    sh = turned & (s < c + 9)
    if sh.any():
        lv = np.clip((c + 9 - s) / 9 * 2.6, 0, 2.6)
        bi = to_idx(out)
        k = np.where(sh, lv.astype(np.int32) + (_B2 < lv % 1), 0)
        for _ in range(3):
            bi = np.where(k > 0, _PREV[bi], bi)
            k = k - 1
        out = PAL[bi]
    # the flap: points on the unturned side whose mirror (compressed by the curl) lies on the turned part
    d = c - s                                                   # >= 0 on the unturned side
    ds = d / kap
    sx = _XX + nx * (d + ds)
    sy = _YY + ny * (d + ds)
    flap = (~turned) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & (ds < smax - c + 1)
    if flap.any():
        ix = np.clip(sx.astype(np.intp), 0, W - 1)
        iy = np.clip(sy.astype(np.intp), 0, H - 1)
        fl = flap.astype(np.uint8)
        drop = (cv2.dilate(fl, _K3, iterations=2) > 0) & ~flap & ~turned      # flap shadow on a
        if drop.any():
            out[drop] = PAL[_PREV[_PREV[to_idx(out[drop][None])[0]]]]
        lum = a[iy[flap], ix[flap]].astype(np.int32).sum(1)                 # a seen through the vellum, reversed
        back = np.full(lum.shape, IDX['parch'], np.uint8)
        dth = _B2[flap]
        back[lum < 300] = IDX['p2']
        back[(lum < 170) & (dth < 0.5)] = IDX['p1']
        dd = d[flap]
        back[dd < 2.0] = IDX['bone']                                       # the curl catches the light
        back[(dd >= 2.0) & (dd < 5.0) & (dth < 0.5)] = IDX['parch']
        far = ds[flap] > (smax - c) - 3.0                                  # the leaf's own (dark) edge
        back[far] = IDX['p1']
        out[flap] = PAL[back]
        edge = (cv2.dilate(fl, _KX) > 0) & ~flap & ~turned
        out[edge] = C['void']
    return out


# ============================================================================ 10. ripple
@functools.lru_cache(maxsize=16)
def _ripple_geo(cx, cy):
    yb = (np.floor(_YY / 2) * 2 + 1.0)                          # 2x2-stepped radius: blocky pixel rings
    xb = (np.floor(_XX / 2) * 2 + 1.0)
    dx, dy = np.broadcast_to(xb - cx, (H, W)), np.broadcast_to(yb - cy, (H, W))
    r = np.hypot(dx, dy).astype(np.float32)
    ux, uy = (dx / (r + 1e-3)).astype(np.float32), (dy / (r + 1e-3)).astype(np.float32)
    q = np.quantile(r, np.linspace(0, 1, 513)).astype(np.float32)
    return r, ux, uy, q


def ripple(a, b, prog, seed=0, cx=None, cy=None):
    """A drop hits the water: pixel rings run out from it, the frame wobbles like a reflection,
    behind the leading ring the water shows b.  Crests glint, troughs darken."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    rng = _rng(seed, 101)
    if cx is None:
        cx = W * float(rng.uniform(0.38, 0.62))
    if cy is None:
        cy = H * float(rng.uniform(0.42, 0.58))
    r, ux, uy, q = _ripple_geo(round(float(cx), 1), round(float(cy), 1))
    R = _cov_thr(q, _ss(0.08, 0.92, p))                         # b covers half the screen at p = 0.5
    amp = 5.5 * math.sin(math.pi * p) ** 0.6
    lam = 15.0
    ph = (r - R) * (2 * math.pi / lam)
    env = np.exp(-np.abs(r - R) / 34.0) + 0.25 * math.sin(math.pi * p)    # the whole frame wobbles a little
    w = np.sin(ph - p * 9.0) * env
    disp = amp * w
    sx = np.clip((_XX + ux * disp).astype(np.intp), 0, W - 1)
    sy = np.clip((_YY + uy * disp).astype(np.intp), 0, H - 1)
    inside = r < R
    out = np.where(inside[..., None], b[sy, sx], a[sy, sx])
    # crest glints / trough shadows near the front, as 2x2 dither
    oi = to_idx(out)
    crest = (w > 0.55) & (_B2 < (w - 0.55) * 2.4 * min(1.0, amp / 3))
    trough = (w < -0.6) & (_B2 < (-w - 0.6) * 2.4)
    oi = np.where(crest, _NEXT[_NEXT[oi]], oi)
    oi = np.where(trough, _PREV[oi], oi)
    out = PAL[oi]
    if R > 2:
        front = (np.abs(r - R) < 1.0) & (_B2 < 0.75)
        out[front] = C['light'] if p < 0.5 else C['bone']
    if p < 0.16:                                                # the drop itself
        k = p / 0.16
        y = int(cy - (1 - k) * 60)
        x = int(cx)
        if 0 <= y < H - 3:
            out[max(0, y - 4):y, x:x + 2] = C['line']
            out[y:y + 3, x:x + 2] = C['bone']
    return out


# ============================================================================ 11. pixelrain
@functools.lru_cache(maxsize=16)
def _rain_plan(seed):
    rng = _rng(seed, 111)
    by, bx = np.mgrid[0:H:2, 0:W:2]
    by, bx = by.ravel(), bx.ravel()
    col = cv2.resize(rng.random((1, 9)).astype(np.float32), (W // 2, 1), interpolation=cv2.INTER_CUBIC)[0]
    col = (col - col.min()) / (col.max() - col.min() + 1e-6)
    tr = 0.03 + 0.20 * col[bx // 2] + 0.42 * (1 - by / H) ** 1.2 + rng.uniform(0, 0.07, by.size)
    G = 7200.0                                                  # px per prog^2
    land = tr + np.sqrt((H - by) / G)
    nc = W // 2
    order = []                                                  # per block-column: blocks in landing order
    for c in range(nc):
        ids = np.flatnonzero(bx == c * 2)
        order.append(ids[np.argsort(land[ids], kind='stable')])
    order = np.stack(order, 1)                                  # (H//2, W//2)
    land_sorted = land[order]
    xo = rng.integers(-1, 2, by.size)
    return by, bx, tr, land, order, land_sorted, G, xo


def pixelrain(a, b, prog, seed=0):
    """a crumbles from the bottom up: 2x2 pixels fall like sand, pile up in heaps on the floor
    (in the order they land), b shows behind; then the heap drains through the floor."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    by, bx, tr, land, order, land_sorted, G, xo = _rain_plan(seed)
    comp = 0.34                                                 # sand packs tighter than the picture
    out = b.copy()
    still = p < tr
    ys, xs = by[still], bx[still]
    for oy in (0, 1):
        for ox in (0, 1):
            out[ys + oy, xs + ox] = a[ys + oy, xs + ox]
    # heap: per 2px column, number of landed blocks -> height
    nland = (land_sorted <= p).sum(0)                           # (W//2,)
    hgt = np.floor(nland * 2 * comp).astype(np.int32)
    sink = int(round(_ss(0.66, 1.0, p) ** 1.5 * (hgt.max() + 4))) if hgt.max() > 0 else 0
    if hgt.max() > 0:
        hmax = int(hgt.max())
        k = np.arange(hmax)[:, None]                            # 0 = bottom row of the heap
        bi = np.minimum((k / (2 * comp)).astype(np.intp), order.shape[0] - 1)
        ids = order[bi, np.arange(order.shape[1])[None, :]]
        colr = a[by[ids], bx[ids]]                              # (hmax, W//2, 3)
        colr = np.repeat(colr, 2, axis=1)
        hh = np.repeat(hgt, 2)
        rows = H - 1 - k + sink                                 # screen row of heap layer k
        okm = (k < hh[None, :]) & (rows < H)
        rr = np.broadcast_to(rows, okm.shape)[okm]
        cc = np.broadcast_to(np.arange(W)[None, :], okm.shape)[okm]
        out[rr, cc] = colr[okm]
        top = H - hh + sink                                     # bone glints on the heap's surface
        tm = (hh > 0) & (top >= 0) & (top < H) & (_B2[0, :] < 0.5)
        out[top[tm], np.flatnonzero(tm)] = C['light']
    # falling grains, with a short dark trail
    fall = (~still) & (p < land)
    dt = p - tr[fall]
    dy = (G * dt * dt).astype(np.int32)
    fy = by[fall] + dy
    fx = np.clip(bx[fall] + xo[fall] * (dy > 30), 0, W - 2)
    hcol = np.repeat(hgt, 2)
    ok = (fy < H - 1 - hcol[fx] + sink) & (fy >= 0)
    sy, sx = by[fall][ok], bx[fall][ok]
    fy, fx, dy = fy[ok], fx[ok], dy[ok]
    tr_ok = dy > 8
    out[np.maximum(fy[tr_ok] - 2, 0), fx[tr_ok]] = PAL[_PREV[to_idx(a[sy[tr_ok], sx[tr_ok]][None])[0]]]
    for oy in (0, 1):
        for ox in (0, 1):
            out[np.minimum(fy + oy, H - 1), fx + ox] = a[sy + oy, sx + ox]
    return out


# ============================================================================ 12. blink
@functools.lru_cache(maxsize=16)
def _blink_plan(seed):
    rng = _rng(seed, 121)
    lash = np.zeros(W, np.int32)
    xs = np.arange(1, W, 3) + rng.integers(0, 2, len(range(1, W, 3)))
    xs = xs[xs < W]
    lash[xs] = rng.integers(2, 6, len(xs))
    return lash, float(rng.uniform(0.46, 0.54))


def _lid_closure(p):
    if p < 0.36:
        return _ss(0.0, 0.36, p) ** 0.8                         # lids drop fast
    if p < 0.62:
        return 1.0
    return 1.0 - _ss(0.62, 1.0, p) ** 0.7                       # and open a little slower


def blink(a, b, prog, seed=0):
    """block eyelids (with lashes and a crease) close over a; while they are shut the light
    glows red through the skin; they open on b, which comes into focus."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    lash, cyf = _blink_plan(seed)
    cl = _lid_closure(p)
    cy = H * cyf
    xn = (np.arange(W) + 0.5 - W / 2) / (W * 0.58)
    arch = np.clip(1 - xn * xn, 0, 1) ** 0.75                   # almond-shaped opening
    hu = (1 - cl) * (H * 0.95) + 0.0                            # upper lid travels far
    hl = (1 - cl) * (H * 0.62)
    yt = np.floor((cy - hu * arch) / 2) * 2                     # 2px steps
    yb = np.floor((cy + 2 + hl * arch) / 2) * 2
    if cl >= 1.0:
        yt = np.full(W, cy, np.float64); yb = yt.copy()
    rows = np.arange(H)[:, None]
    src = a if p < 0.5 else b
    if p > 0.62:                                                # the new view swims into focus
        blk = int(round(1 + 3 * (1 - _ss(0.62, 0.9, p))))
        if blk > 1:
            sm = cv2.resize(src, (W // blk, H // blk), interpolation=cv2.INTER_AREA)
            src = PAL[to_idx(cv2.resize(sm, (W, H), interpolation=cv2.INTER_NEAREST))]
    out = src.copy()
    up = rows < yt[None, :]
    lo = rows >= yb[None, :]
    lid = up | lo
    # skin: void2 base, lighter towards the lid margin; shut lids glow red (light through skin)
    dist_u = yt[None, :] - rows
    dist_l = rows - yb[None, :]
    dm = np.where(up, dist_u, np.where(lo, dist_l, 999))
    skin = np.full((H, W), IDX['void2'], np.uint8)
    skin[(dm < 7) & (_B2 < 0.5)] = IDX['dark']
    skin[dm < 3] = IDX['d3']
    glow = cl ** 3 * np.clip(1 - np.abs(rows - cy) / (H * 0.42), 0, 1) * np.clip(1 - np.abs(xn) * 0.7, 0, 1)
    skin[(glow > 0.15) & (_TH8 + 0.5 < glow * 0.55)] = IDX['red2']
    crease = up & (np.abs(dist_u - 12 - 6 * (1 - arch[None, :])) < 1.0)
    skin[crease] = IDX['void']
    margin = (dm < 1.0) & lid
    skin[margin & up] = IDX['red2'] if cl < 1 else IDX['void']  # the wet rim of the lid
    out[lid] = PAL[skin[lid]]
    if cl < 1.0:
        # shadow of the lids on the eye
        sh = (~lid) & (((rows - yt[None, :]) < 4) | ((yb[None, :] - rows) < 3))
        if sh.any():
            out[sh] = PAL[_PREV[_PREV[to_idx(out[sh][None])[0]]]]
        # lashes hang from the upper lid, slanting away from the centre
        for x in np.flatnonzero(lash):
            L = int(lash[x] * (0.6 + 0.4 * arch[x]))
            y0 = int(yt[x])
            sl = 1 if x > W / 2 else -1
            for k in range(L):
                yy = y0 + k
                xx = x + sl * (k // 2)
                if 0 <= yy < H and 0 <= xx < W:
                    out[yy, xx] = C['void']
    else:
        mid = int(cy)
        out[max(0, mid - 1):mid + 1, :] = C['void']               # the shut seam
    return out


# ============================================================================ 13. textdissolve
_GLYPHS_TXT = [
    # o            a              y (9)          d (8)          k (gallows)    t (gallows)
    ['.....', '.....', '.###.', '#...#', '#...#', '.###.', '.....'],
    ['.....', '.....', '.###.', '#..#.', '#..#.', '.##.#', '.....'],
    ['.....', '.###.', '#...#', '.####', '....#', '...#.', '###..'],
    ['..##.', '.#..#', '..##.', '.#..#', '#...#', '.###.', '.....'],
    ['.#...', '.#.#.', '.##.#', '.#.##', '.#..#', '##..#', '.....'],
    ['#.##.', '.#..#', '.#.#.', '##.##', '.#..#', '.#..#', '.....'],
    # e (c)        ch             n              r              q (4)          l
    ['.....', '.....', '.##..', '#....', '#....', '.##..', '.....'],
    ['.....', '..#..', '#####', '#.#.#', '#.#.#', '.#.#.', '.....'],
    ['.....', '.....', '.#...', '.#.#.', '.#.##', '.#.#.', '.....'],
    ['.....', '.....', '..#..', '.##..', '#.#..', '..#.#', '.....'],
    ['...#.', '..##.', '.#.#.', '#####', '...#.', '...#.', '.....'],
    ['.....', '...#.', '..#..', '.###.', '#..#.', '...#.', '.....'],
    # p (gallows)    f
    ['.#.#.', '####.', '.#..#', '.#.#.', '.#..#', '##...', '.....'],
    ['#.#..', '####.', '.#.##', '.#..#', '.#.#.', '##...', '.....'],
]
CW, CH = 6, 8
NCX, NCY = W // CW, H // CH


def _glyph_bank():
    g = np.zeros((len(_GLYPHS_TXT) + 1, CH, CW), bool)           # last = blank (word space)
    for i, rows in enumerate(_GLYPHS_TXT):
        for y, r in enumerate(rows):
            for x, ch in enumerate(r):
                g[i, y + 1, x] = ch == '#'
    return g


_GBANK = _glyph_bank()
_GDENS = _GBANK[:-1].sum((1, 2))
_GBY_D = np.argsort(_GDENS, kind='stable')                         # glyphs sorted sparse -> dense
_CYI = (np.arange(H) // CH)[:, None]
_CXI = (np.arange(W) // CW)[None, :]
_OYI = (np.arange(H) % CH)[:, None]
_OXI = (np.arange(W) % CW)[None, :]


@functools.lru_cache(maxsize=16)
def _text_plan(seed):
    rng = _rng(seed, 131)
    space = np.zeros((NCY, NCX), bool)
    for r in range(NCY):                                         # words of 2-7 glyphs, single-cell spaces
        x = int(rng.integers(0, 3))
        while x < NCX:
            x += int(rng.integers(2, 8))
            if x < NCX:
                space[r, x] = True
            x += 1
    tau = (np.arange(NCY)[:, None] / NCY) * 0.62 + (np.arange(NCX)[None, :] / NCX) * 0.38
    tau = tau + rng.uniform(-0.03, 0.03, tau.shape)              # reading order: line by line, left to right
    pick = rng.random((8, NCY, NCX))                             # 8 scramble states
    wid = np.cumsum(space, 1)                                    # word id per row
    red = rng.random((8, NCY, NCX + 1)) < 0.035                  # a few words flare red (the key-word plate)
    return space, tau.astype(np.float32), pick, wid, red


def textdissolve(a, b, prog, seed=0):
    """The picture is transcribed: line by line, left to right, cells of a turn into Voynich glyphs drawn
    in a's own colours; the script changes hand to b's colours, then b is read back out of the glyphs."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    space, tau, pick, wid, red = _text_plan(seed)
    sa = 0.04 + 0.30 * tau                                       # a -> glyphs(a)
    sb = 0.33 + 0.34 * tau                                       # glyphs(a) -> glyphs(b): half the cells at p=.5
    sc = 0.62 + 0.33 * tau                                       # glyphs(b) -> b
    use_b = p >= sb
    st = int(p * 22) % 8                                         # the script keeps rewriting itself
    ca = to_idx(cv2.resize(a, (NCX, NCY), interpolation=cv2.INTER_AREA))
    cbi = to_idx(cv2.resize(b, (NCX, NCY), interpolation=cv2.INTER_AREA))
    ci = np.where(use_b, cbi, ca)
    lum = PAL[ci].astype(np.int32).sum(-1)
    dens = np.clip(lum / 520.0, 0, 0.999)                        # bright cells get dense glyphs
    gi = _GBY_D[np.clip((dens * 0.6 + pick[st] * 0.4) * len(_GBY_D), 0, len(_GBY_D) - 1).astype(np.intp)]
    gi = np.where(space, len(_GLYPHS_TXT), gi)
    fg = _NEXT[ci]                                               # ink a step brighter than the cell
    fg = np.where(lum < 140, IDX['tile'], fg)                    # even black cells write in faint ink
    rw = red[st][np.arange(NCY)[:, None], wid] & ~space & (np.abs(p - 0.5) < 0.22)
    fg = np.where(rw, IDX['red'], fg)
    bg = _PREV[_PREV[ci]]
    mask = _GBANK[gi[_CYI, _CXI], _OYI, _OXI]
    txt = np.where(mask, fg[_CYI, _CXI], bg[_CYI, _CXI])
    out = PAL[txt]
    keep_a = (p < sa)[_CYI, _CXI]
    keep_b = (p >= sc)[_CYI, _CXI]
    out = np.where(keep_a[..., None], a, out)
    out = np.where(keep_b[..., None], b, out)
    # the pen: a bone caret at the writing position of each line in transition
    for thr in (sa, sb, sc):
        cur = np.abs(p - thr) < 0.012
        if cur.any():
            cy_, cx_ = np.nonzero(cur)
            out[cy_ * CH + CH - 1, cx_ * CW] = C['bone']
            out[cy_ * CH + CH - 1, cx_ * CW + 1] = C['bone']
    return out


# ============================================================================ 14. mirror
def _inkblot(img, k):
    """Rorschach: dithered blend (amount k) of img towards a two-ink card (void / bone, red where mid)"""
    if k <= 0:
        return img
    idx = to_idx(img)
    lum = PAL[idx].astype(np.int32).sum(-1)
    med = float(np.median(lum))
    card = np.where(lum < med * 0.8, IDX['void'], np.where(lum < med * 1.15, IDX['red2'], IDX['bone']))
    sel = (_TH8 + 0.5) < k
    return PAL[np.where(sel, card, idx)]


def mirror(a, b, prog, seed=0):
    """Rorschach fold: a mirror plane slides in from the right until a is perfectly symmetric, the
    picture curdles into a two-ink blot, flips to b's symmetric blot on the cut, and b unfolds from the left."""
    p = _c01(prog)
    e = _ends(a, b, p)
    if e is not None:
        return e
    cols = np.arange(W)
    if p < 0.5:
        m = W - 1 - (W / 2.0 - 0.5) * _ss(0.0, 0.4, p)           # plane: right edge -> centre
        src = np.where(cols > m, np.round(2 * m - cols), cols).astype(np.intp)
        img = a[:, np.clip(src, 0, W - 1)]
    else:
        m = (W / 2.0 - 0.5) * (1 - _ss(0.6, 1.0, p))              # plane: centre -> left edge
        src = np.where(cols < m, np.round(2 * m - cols), cols).astype(np.intp)
        img = b[:, np.clip(src, 0, W - 1)]
    k = _c01(1.0 - abs(p - 0.5) / 0.22) ** 0.8
    out = _inkblot(img, k * 0.9)
    mx = int(round(m))
    if 0 <= mx < W and 0.02 < p < 0.98:                          # the fold line
        out[:, mx] = C['red']
        out[::3, mx] = C['bone']
    if 0.44 < p < 0.56:                                          # on the cut the card splits open for an instant
        g = int(round(10 * (1 - abs(p - 0.5) / 0.06)))
        if g > 0:
            x0 = W // 2 - g // 2
            out[:, x0:x0 + g] = b[:, x0:x0 + g] if p >= 0.5 else C['void']
    return out


