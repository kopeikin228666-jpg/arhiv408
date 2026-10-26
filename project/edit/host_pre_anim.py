"""The channel host «408» - a small Death in a black ragged hooded cloak.  No face - only darkness, two glowing red
eyes, and a mouth that is a digital sound wave while he speaks.  Wide black trousers and heavy black boots show
under the parted hem.

Drawn at full grid resolution (1 sprite pixel = 1 grid pixel = 6 output pixels), so he has the same detail as the scene.
All geometry is written in "units" (1 unit = R sprite pixels) and sampled per sprite pixel.

sprite(t, amp, blink, look, wave_seed, mood, pose) -> (H, W, 4) uint8 BGRA, palette colours only
draw(g, cx, by, t, amp, scale=1, ...) pastes it on a 180x320 BGR grid (cx = centre x, by = bottom y)
"""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/voynich/look')
import pixelize as P

BGR = np.array(P.PALETTE, np.uint8).reshape(-1, 3)          # the grids are BGR, so is the palette
VOID, VOID2, DARK, D3, TILE, TILE2, MID, LINE, LIGHT, BONE = range(10)
G1, G2, G3, SEPIA, PARCH, PARCH2, RED2, RED = 10, 11, 12, 13, 14, 15, 16, 17

R = 2                                   # sprite pixels per unit
UW, UH = 52, 74                         # size in units
W, H = UW * R, UH * R
C = 26.0                                # centre line (units)
_B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 - 0.47
_yy, _xx = np.mgrid[0:H, 0:W]
X = (_xx + 0.5) / R                      # unit coordinates of every sprite pixel
Y = (_yy + 0.5) / R
BAY = _B4[_yy % 4, _xx % 4]
_NOISE = np.random.default_rng(408).random((H, W))


def _ell(cx, cy, rx, ry):
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0


def _box(x0, y0, x1, y1):
    return (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)


def _tone(t, levels=(VOID, VOID2, DARK, D3), cuts=(-0.35, 0.12, 0.55), dither=0.22):
    """map a shading value to palette levels with ordered dithering (pixel-art fabric)"""
    v = t + BAY * dither
    out = np.full(t.shape, levels[0], np.int16)
    for lv, c in zip(levels[1:], cuts):
        out[v >= c] = lv
    return out


def _smooth_noise(n, seed, k=3):
    r = np.random.default_rng(seed).random(n + 2 * k)
    return np.convolve(r, np.ones(2 * k + 1) / (2 * k + 1), 'valid')[:n]


def sprite(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='stand'):
    sit = pose == 'sit'
    idx = np.full((H, W), -1, np.int16)
    sw_t = t * 2 * math.pi / 1.07
    # ================================================================ shapes
    # hood: deep, softly pointed, the cowl falls over the shoulders and ends in a V on the chest
    hood = _ell(C, 11.5, 10.0, 10.5) | _ell(C + 1.5, 2.6, 5.5, 2.4)
    half = 10.0 + (Y - 12) * 0.32
    hood |= (Y >= 12) & (Y < 24) & (np.abs(X - C) <= half)
    half = 9 - (Y - 24) * 2.6
    hood |= (Y >= 24) & (Y < 28) & (np.abs(X - C) <= half)
    # cloak: broad shoulders, flares to a long hem, parts in an inverted V at the front
    hem = 50.0 if sit else 58.0
    half = np.where(Y <= 23, 10.5 + (Y - 19) * 1.0, 14.5 + (Y - 23) / (hem - 23) * 7.5)
    body = (Y >= 19) & (Y < hem) & (np.abs(X - C) <= half)
    # tattered hem: per pixel column a ragged length, swinging
    lens = 1.5 + 7.5 * _smooth_noise(W, 11, 2) ** 1.5
    swing = np.sin(sw_t + np.arange(W) * 0.35) * 0.8
    yh = int(hem * R) - 1
    for x in range(W):
        if not body[yh, x]:
            continue
        L = lens[x]
        xs = min(W - 1, max(0, x + int(round(swing[x])) if L > 3 else x))
        body[int(hem * R):min(H, int((hem + L) * R)), xs] = True
    v0 = 99.0                                                          # no front slit: one closed cloak
    holes = body & (Y > hem - 8) & (Y < hem) & (_NOISE > 0.985)        # moth holes near the hem
    body &= ~((Y > hem + 1.5) & (_NOISE < np.clip((Y - hem - 1.5) / 7, 0, 0.8)))   # the rags crumble into darkness
    # bell sleeves
    sleeves = np.zeros((H, W), bool)
    hands = np.zeros((H, W), bool)
    for sgn in (-1, 1):
        k = np.clip((Y - 22) / 22, 0, 1)
        xc = C + sgn * (15.0 + k * 4.5)
        hw = 2.5 + k * 3.5
        cuff = 44 + 1.6 * _smooth_noise(W, 6 + sgn, 1)[_xx]
        sleeves |= (Y >= 22) & (Y < cuff) & (np.abs(X - xc) <= hw)
        hx = C + sgn * 19.5
        hands |= _ell(hx, 46.4, 1.9, 2.3) | _ell(hx - sgn * 1.7, 45.6, 0.8, 1.1)
    # legs + boots
    LEGS = ((C - 9, C - 2), (C + 2, C + 9))
    legs = np.zeros((H, W), bool); boots = np.zeros((H, W), bool)
    geo = []
    for j, (x0, x1) in enumerate(LEGS):
        out = -1 if j == 0 else 1
        if sit:
            s_ = int(round(0.6 + 0.6 * math.sin(sw_t + j * math.pi)))
            ly1 = 61 + s_
            legs |= _box(x0, 44, x1, ly1)
            by0, by1 = ly1 - 1, ly1 + 8
            boots |= _box(x0 - 1, by0, x1 + 1, by1) | _box(x0 - (2 if out < 0 else 1), by0 + 3, x1 + (1 if out < 0 else 2), by1)
        else:
            ly1 = 64
            legs |= _box(x0, 46, x1, ly1) | _box(x0 - (1 if out < 0 else 0), 58, x1 + (1 if out > 0 else 0), ly1)
            by0, by1 = 62, 73
            boots |= _box(x0 - 1, by0, x1 + 1, by0 + 5)
            fx0, fx1 = x0 - (4 if out < 0 else 1), x1 + (4 if out > 0 else 1)
            foot = _box(fx0, by0 + 5, fx1, by1)
            tipx = fx0 if out < 0 else fx1
            cut = ((X - tipx) ** 2 / 9 + (Y - (by0 + 5)) ** 2 / 4 < 1)
            foot &= ~cut                                              # rounded toe
            boots |= foot
        geo.append((x0, x1, ly1, by0, by1))
    shape = hood | body | sleeves | hands | legs | boots
    # ================================================================ shading
    light = -(X - C) / 22 - (Y - 30) / 70
    # cloak fabric: vertical folds, lit from the upper left
    fold = np.sin((X - C) * 0.95 + 0.6 * np.sin(Y * 0.21)) * np.clip((Y - 24) / 14, 0, 1)
    cl = _tone(0.3 * fold + 0.35 * light - 0.42)
    idx[body] = cl[body]
    idx[holes] = VOID
    # sleeves: their own folds, a dark inside of the bell, a dark edge against the cloak
    st = _tone(0.3 * np.sin(X * 1.6 + Y * 0.15) + 0.35 * light - 0.38)
    idx[sleeves] = st[sleeves]
    for sgn in (-1, 1):
        idx[sleeves & _ell(C + sgn * 19.5, 44.4, 5.0, 1.5)] = VOID
        e = sleeves & ~np.roll(sleeves, sgn, 1) & (np.abs(X - C) < 19)
        idx[e] = VOID
    # hood: rounded, lit upper-left, a stitched seam along its front edge
    ht = _tone(-(X - C) / 15 - (Y - 6) / 20 - 0.45)
    idx[hood] = ht[hood]
    seam_line = hood & _ell(C, 13.5, 9.6, 10.5) & ~_ell(C, 13.5, 9.0, 9.9)
    idx[seam_line & ((_xx + _yy) % 3 == 0)] = DARK
    # torn notches in the hood's edge
    for (nx, ny) in ((C - 9.6, 17.0), (C + 9.9, 20.5), (C - 9.2, 22.5)):
        nm = _ell(nx, ny, 0.9, 0.6) & hood
        idx[nm & ~(body | sleeves)] = -1
        idx[nm & (body | sleeves)] = VOID
    # shadow of the hood on the shoulders
    under = (body | sleeves) & ~hood & np.roll(hood, R, 0)
    idx[under] = VOID
    # rope belt with a knot and hanging ends
    by_ = 36.5 if not sit else 35.0
    hb = 14.5 + (by_ - 23) / (hem - 23) * 7.5
    belt = (np.abs(Y - by_ - 0.25 * np.sin((X - C) * 0.5)) < 0.75) & (np.abs(X - C) < hb - 1.5) & body
    idx[belt] = DARK
    idx[belt & ((_xx + _yy) % 4 < 2)] = D3                     # twisted cord
    knot = _ell(C + 3.5, by_, 1.3, 1.1) & body
    idx[knot] = D3
    idx[knot & ((_xx + _yy) % 3 == 0)] = TILE
    for k, dx in enumerate((3.0, 4.6)):
        cord = (np.abs(X - (C + dx + 0.15 * (Y - by_))) < 0.5) & (Y > by_ + 1) & (Y < by_ + 8 + k * 2)
        idx[cord & ~hands] = DARK
        idx[_ell(C + dx + 0.15 * (8 + k * 2), by_ + 8.3 + k * 2, 0.7, 0.9)] = D3   # tassels
    # an old iron key hangs from the belt (the archive's key)
    kx, ky = C - 5.0, by_ + 1.0
    ring = _ell(kx, ky + 1.2, 1.2, 1.2) & ~_ell(kx, ky + 1.2, 0.55, 0.55)
    shaft = (np.abs(X - kx) < 0.35) & (Y > ky + 2.2) & (Y < ky + 7.5)
    bit = _box(kx, ky + 6.0, kx + 1.4, ky + 6.6) | _box(kx, ky + 7.0, kx + 1.1, ky + 7.5)
    key = ring | shaft | bit
    idx[key] = D3
    idx[key & (X > kx + 0.2)] = DARK
    idx[ring & (Y < ky + 0.8)] = TILE
    # coarse cloth: sparse dark speckles, a few lighter threads
    idx[body & (_NOISE > 0.93) & (_NOISE < 0.96)] = VOID
    idx[body & (_NOISE > 0.996)] = DARK
    # shoulder seams with crude stitches
    for sgn in (-1, 1):
        sx = C + sgn * 11.0
        seam = body & ~hood & (np.abs(X - sx - sgn * 0.25 * (Y - 22)) < 0.3) & (Y > 21) & (Y < 31)
        idx[seam] = VOID
        st_ = body & ~hood & (np.abs(X - sx - sgn * 0.25 * (Y - 22)) < 0.9) & (Y > 21) & (Y < 31) & (_yy % 3 == 0) & ~seam
        idx[st_] = D3
    # a patch sewn on the cloak (lower left) with big cross stitches
    patch = _box(C - 12.5, 47.5, C - 7.0, 53.0) & body
    pt = _tone(-(X - C) / 22 - 0.2 + 0.15 * np.sin(Y * 2.0), cuts=(-0.3, 0.1, 0.6))
    idx[patch] = pt[patch]
    pe = patch & ~(np.roll(patch, 1, 0) & np.roll(patch, -1, 0) & np.roll(patch, 1, 1) & np.roll(patch, -1, 1))
    idx[pe] = VOID
    for k in range(4):
        for (cx_, cy_) in ((C - 12.5, 48.5 + k * 1.4), (C - 7.0, 48.5 + k * 1.4)):
            idx[_ell(cx_, cy_, 0.45, 0.25) & body] = TILE
    # frayed threads hanging from the hem
    for x in range(4, W - 4, 7):
        if body[int(hem * R) - 1, x]:
            L = int(3 + (x * 13) % 5)
            y0 = int((hem + lens[x]) * R)
            for k in range(L):
                yy_ = min(H - 1, y0 + k)
                xx_ = min(W - 1, x + (k // 3) * (1 if x % 2 else -1))
                if idx[yy_, xx_] < 0:
                    idx[yy_, xx_] = DARK if k < L - 1 else VOID2
    # clasp: a red stone on a short chain at the throat
    chain = (np.abs(Y - 26.6 - 0.12 * np.abs(X - C)) < 0.35) & (np.abs(X - C) < 6) & hood
    idx[chain & (_xx % 2 == 0)] = D3
    stone = _ell(C, 27.4, 1.25, 1.25)
    idx[stone] = RED2
    idx[_ell(C - 0.3, 27.1, 0.35, 0.35)] = RED
    # gloves: black, separate fingers, light on the knuckles
    idx[hands] = VOID2
    for sgn in (-1, 1):
        hx = C + sgn * 19.5
        for f in (-0.9, 0.0, 0.9):
            idx[(np.abs(X - (hx + f)) < 0.25) & (Y > 46.6) & hands] = VOID
        idx[hands & (np.abs(Y - 45.2) < 0.3) & ((_xx % 2) == 0)] = D3
    # trousers: wide, black, folds, thin rim
    lg = legs & ~body
    trt = _tone(0.3 * np.sin(X * 2.1 + Y * 0.6) - (X - C) / 30 - 0.4)
    idx[lg] = trt[lg]
    for (x0, x1, ly1, by0, by1) in geo:
        col = lg & (X >= x0 - 1) & (X < x1 + 1)
        idx[col & (np.abs(X - (x0 + 0.25)) < 0.3)] = D3          # lit edge
        idx[col & (Y > ly1 - 1.2)] = VOID                        # hem falls over the boot
    # boots: heavy black leather, criss-cross laces, an ankle strap with a buckle, a stitched sole, scuffed toe
    bt = boots & ~body
    btt = _tone(-(X - C) / 26 - (Y - 62) / 30 - 0.45, cuts=(-0.3, 0.2, 0.7))
    idx[bt] = btt[bt]
    for (x0, x1, ly1, by0, by1) in geo:
        bj = bt & (X >= x0 - 5) & (X < x1 + 5)
        cx = (x0 + x1) / 2
        for k in range(4):
            yl = by0 + 0.8 + k * 1.1
            lace = bj & (np.abs(Y - yl - 0.3 * (X - cx) * (1 if k % 2 else -1)) < 0.3) & (np.abs(X - cx) < 1.3)
            idx[lace] = D3
        idx[bj & (np.abs(Y - (by0 + 4.6)) < 0.45)] = DARK                     # strap
        idx[bj & _ell(x1 - 0.4, by0 + 4.6, 0.6, 0.6)] = TILE2               # buckle
        idx[bj & (Y > by1 - 1.0)] = VOID                                    # sole
        idx[bj & (np.abs(Y - (by1 - 1.4)) < 0.25) & (_xx % 3 == 0)] = D3     # stitch
        out = -1 if x0 < C else 1
        tx = (x0 - 2.6) if out < 0 else (x1 + 1.6)
        idx[bj & _ell(tx, by1 - 3.0, 0.8, 0.45)] = TILE                     # toe shine
    # ================================================================ outline / rim
    filled = idx >= 0
    edge = filled & ~(np.roll(filled, 1, 0) & np.roll(filled, -1, 0) & np.roll(filled, 1, 1) & np.roll(filled, -1, 1))
    lit = (X - C) * 0.6 + (Y - 20) * 0.4
    keep = (idx == RED) | (idx == RED2) | (idx == BONE)
    idx[edge & (lit < -4) & ~keep & ((_xx + _yy) % 2 == 0)] = DARK
    idx[edge & (lit < -12) & ~keep] = D3
    # ================================================================ face
    face = _ell(C, 13.8, 6.9, 8.0)
    inner = _ell(C, 13.6, 8.4, 9.3) & ~face & hood
    it = _tone(-(X - C) / 10 - (Y - 8) / 7 - 0.75, levels=(VOID, VOID2, DARK, DARK))
    idx[inner] = it[inner]
    idx[face] = VOID
    ex = [C - 2.6 + look, C + 2.6 + look]
    ey = 13.8
    if mood == 'scary' and not blink:
        for j, e in enumerate(ex):
            sg = 1 if j == 0 else -1                            # the inner ends sit lower: angry
            slit = (np.abs((Y - ey) - sg * 0.4 * (X - e)) < 0.3) & (np.abs(X - e) < 1.3) & face
            idx[slit] = RED
            glow = face & _ell(e, ey + 0.6, 1.4, 0.8) & ~slit & ((_xx + _yy) % 2 == 0)
            idx[glow] = RED2
    else:
        for e in ex:
            if blink:
                idx[face & (np.abs(Y - ey) < 0.26) & (np.abs(X - e) < 0.8)] = RED2
                continue
            # a faint ember-glow around, sparser further out
            flare = (t % 4.3) < 0.2
            core = _ell(e, ey, 0.55, 0.75)
            ring = face & _ell(e, ey, 1.6 if flare else 1.15, 1.8 if flare else 1.3) & ~core & ((_xx + _yy) % 2 == 0)
            idx[ring] = RED2                                      # a faint ember-glow
            idx[core] = RED if flare or amp > 0.05 else RED2      # smoulders dark red, burns when he speaks
            idx[_ell(e, ey, 0.3, 0.45)] = RED
    # mouth: a digital sound wave, 1-px bars with gaps on a faint baseline
    yb = int(18.4 * R)
    xs = list(range(int((C - 4) * R), int((C + 4) * R) + 1))
    if amp < 0.04:
        pass                                                      # silent: only darkness
    else:
        ph = np.random.default_rng(wave_seed).random(64) * 6.28
        n = len(xs)
        for i, x in enumerate(xs):
            if i % 2:
                continue
            env = 0.3 + 0.7 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.2 + 0.8 * abs(math.sin(t * 19 + ph[(i // 2) % 64] + i * 0.45)))
            hgt = int(round(v * 4.0))
            idx[yb - hgt:yb + hgt + 1, x] = RED2
            if hgt >= 3:
                idx[yb - 1:yb + 2, x] = RED
    out = np.zeros((H, W, 4), np.uint8)
    m = idx >= 0
    out[m, :3] = BGR[idx[m]]
    out[m, 3] = 255
    return out


# ------------------------------------------------------------------ the generated look (pixelised to the palette)
TH = '/home/claude/voynich/team/host'
POSES = {
    # name: sprite file, eye centres (y, x), mouth (y, x0, x1), face ellipse (cy, cx, ry, rx), core ellipse, eye scale, waist row
    'front': dict(path=f'{TH}/gen1_sprite.png', eyes=((43, 48), (43, 62)), mouth=(53, 51, 61),
                  face=(49, 55, 8.5, 13.5), core=(48.5, 55, 7.5, 12.0), k=1, waist=72),
    'point': dict(path=f'{TH}/gen_point_sprite.png', eyes=((37, 44), (37, 55)), mouth=(43, 45, 55),
                  face=(38.5, 50, 6.5, 10.5), core=(38.5, 50, 5.5, 9.0), k=1, waist=70),
    'close': dict(path=f'{TH}/gen_close_sprite.png', eyes=((88, 61), (88, 88)), mouth=(104, 63, 86),
                  face=(92, 74.5, 15.5, 25.0), core=(91, 74.5, 13.5, 22.5), k=2, waist=150, extend=80),
}
IMG_PATH = POSES['front']['path']
_IMGS = {}


def _load_img(pose='front'):
    if pose not in _IMGS:
        import cv2
        P_ = POSES[pose]
        s = cv2.imread(P_['path'], cv2.IMREAD_UNCHANGED)
        q = s[..., :3].copy(); m = s[..., 3] > 0
        # clean the matte: drop the light-grey fringe left by the old background
        er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ring = m & ~er
        lum = q.astype(int).sum(2)
        m &= ~(ring & (lum > 3 * 70))
        ring = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        q[ring & (lum > 3 * 55)] = BGR[DARK]
        warm = m & (q[..., 2].astype(int) - q[..., 0] > 25) & (q[..., 2] < 150)
        q[warm] = BGR[DARK]
        # erase the baked-in eyes and mouth: the face is pure darkness, we draw our own
        yy, xx = np.mgrid[0:q.shape[0], 0:q.shape[1]]
        cy, cx, ry, rx = P_['face']
        face = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1) & m
        redish = q[..., 2].astype(int) - q[..., 1] > 40
        q[face & ((lum < 3 * 60) | redish)] = BGR[VOID]
        cy, cx, ry, rx = P_['core']
        q[(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1) & m] = BGR[VOID]
        ext = P_.get('extend', 0)
        if ext:                                   # the portrait runs out of the frame: continue the cloak downward
            tile_q, tile_m = q[-24:], m[-24:]
            reps = int(math.ceil(ext / 24))
            q = np.concatenate([q] + [tile_q] * reps)[:q.shape[0] + ext]
            m = np.concatenate([m] + [tile_m] * reps)[:m.shape[0] + ext]
        _IMGS[pose] = (q, m)
    return _IMGS[pose]


def _slit(q, ey, ex, sg, k, mood, lit):
    """one eye: a heavy half-shut slit (calm) or a narrow burning slash (scary); k = pixel scale"""
    def px(y, x, c):
        q[y * 1:y + k, x:x + k] = BGR[c] if k == 1 else BGR[c]
    def put(dy, dx, c):
        y, x = ey + dy * k, ex + dx * k
        q[y:y + k, x:x + k] = BGR[c]
    if mood == 'scary':
        for dx in range(-3, 4):
            put((1 if dx * sg > 1 else 0) - (1 if dx * sg < -1 else 0), dx, RED)
        for dx in (-1, 1):
            put(2, dx, RED2)
        return
    for dx in range(-2, 3):
        put(1 if dx * sg >= 1 else 0, dx, RED2)
    put(0, -sg, RED)
    if lit:
        put(0, 0, RED)
        put(0, -2 * sg, RED)


def sprite_img(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='front', sway=True):
    P_ = POSES[pose]
    q, m = _load_img(pose)
    q = q.copy(); m = m.copy()
    h, w = m.shape
    k = P_['k']
    flare = (t % 4.3) < 0.15
    lit = amp > 0.05 or flare
    if not blink:
        for j, (ey, ex) in enumerate(P_['eyes']):
            _slit(q, int(ey), int(round(ex + look * k)), 1 if j == 0 else -1, k, mood, lit)
    if amp >= 0.04:
        # mouth: a broken signal - dim, low bars, some columns drop out
        my, mx0, mx1 = P_['mouth']
        r = np.random.default_rng(wave_seed)
        ph = r.random(64) * 6.28
        xs = list(range(mx0, mx1 + 1, k))
        n = len(xs)
        for i, x in enumerate(xs):
            if i % 2 or r.random() < 0.25:
                continue
            env = 0.3 + 0.7 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.2 + 0.8 * abs(math.sin(t * 19 + ph[(i // 2) % 64] + i * 0.45)))
            hgt = int(round(v * 3.0)) * k
            q[my + 1 - hgt:my + 1 + k + hgt, x:x + k] = BGR[RED2]
            if hgt >= 2 * k:
                q[my + 1:my + 1 + k, x:x + k] = BGR[RED]
    if sway:
        # the cloak moves: rows below the waist swing sideways, more toward the hem, rags flutter
        wy = P_['waist']
        for y in range(wy, h):
            d = (y - wy) / max(1, h - wy)
            sh = int(round(math.sin(t * 2 * math.pi / 2.14 + y * 0.09) * 2.2 * d + math.sin(t * 9 + y * 0.7) * 0.6 * d * d))
            if sh:
                q[y] = np.roll(q[y], sh, 0); m[y] = np.roll(m[y], sh, 0)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = q
    out[..., 3] = m * 255
    return out


def _darker(c):
    """one palette step toward black, staying in the same hue family"""
    order = {BONE: LIGHT, LIGHT: LINE, LINE: MID, MID: TILE2, TILE2: TILE, TILE: D3, D3: DARK, DARK: VOID2, VOID2: VOID, VOID: VOID,
             G3: G2, G2: G1, G1: DARK, PARCH2: PARCH, PARCH: SEPIA, SEPIA: D3, RED: RED2, RED2: VOID2}
    return order.get(c, VOID2)


_LUT = None


def _pal_index(g):
    global _LUT
    if _LUT is None:
        _LUT = {tuple(int(v) for v in BGR[i]): i for i in range(len(BGR))}
    flat = g.reshape(-1, 3)
    keys = flat[:, 0].astype(np.int32) << 16 | flat[:, 1].astype(np.int32) << 8 | flat[:, 2].astype(np.int32)
    out = np.full(len(keys), VOID2, np.int16)
    for (b_, g_, r_), i in _LUT.items():
        out[keys == (b_ << 16 | g_ << 8 | r_)] = i
    return out.reshape(g.shape[:2])


def draw(g, cx, by, t, amp, scale=1, blink=False, look=0, bob=0, wave_seed=0, mood='calm', pose='front', aura=True, look_img=True, flip=False, smoke=46,
         sx=1.0, sy=1.0, scale_f=1.0, tilt=0):
    if pose in ('stand', 'sit'):
        pose = 'front'
    s = sprite_img(t, amp, blink, look, wave_seed, mood, pose) if look_img else sprite(t, amp, blink, look, wave_seed, mood, pose)
    if flip:
        s = s[:, ::-1].copy()
    # body language: squash for a turn (sx < 1), lean toward the camera (scale > 1), tilt (shear, px at the top)
    if sx != 1.0 or scale_f != 1.0 or sy != 1.0:
        import cv2
        h0, w0 = s.shape[:2]
        nw, nh = max(2, int(round(w0 * sx * scale_f))), max(2, int(round(h0 * sy * scale_f)))
        s = cv2.resize(s, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if tilt:
        h0 = s.shape[0]
        s = s.copy()
        for y in range(h0):
            d = int(round(tilt * (1 - y / h0) ** 1.3))
            if d:
                s[y] = np.roll(s[y], d, 0)
    if scale != 1:
        s = np.repeat(np.repeat(s, scale, 0), scale, 1)
    h, w = s.shape[:2]
    x0, y0 = int(cx - w // 2), int(by - h + bob)
    if aura:
        # darkness bleeds out of him: a dithered shadow halo that darkens the room around his silhouette
        pad = 10
        m = np.zeros((h + 2 * pad, w + 2 * pad), bool)
        m[pad:pad + h, pad:pad + w] = s[..., 3] > 0
        import cv2
        d1 = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        d2 = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        yy, xx = np.mgrid[0:h + 2 * pad, 0:w + 2 * pad]
        bay = _B4[yy % 4, xx % 4] + 0.47
        ring1 = d1 & ~m
        ring2 = d2 & ~d1 & (bay < 0.5)
        X0, Y0 = x0 - pad, y0 - pad
        gx0, gy0 = max(0, X0), max(0, Y0)
        gx1, gy1 = min(g.shape[1], X0 + w + 2 * pad), min(g.shape[0], Y0 + h + 2 * pad)
        sub = g[gy0:gy1, gx0:gx1]
        pi = _pal_index(sub)
        r1 = ring1[gy0 - Y0:gy1 - Y0, gx0 - X0:gx1 - X0]
        r2 = ring2[gy0 - Y0:gy1 - Y0, gx0 - X0:gx1 - X0]
        lut = np.array([_darker(i) for i in range(18)], np.int16)
        pi2 = pi.copy()
        pi2[r1] = lut[lut[pi[r1]]]
        pi2[r2] = lut[pi[r2]]
        g[gy0:gy1, gx0:gx1] = BGR[pi2]
        # black smoke rising off the hood and shoulders, drifting and thinning out
        rng = np.random.default_rng(408)
        tq = math.floor(t * 15) / 15
        for k in range(smoke):
            sx = cx + (rng.random() - 0.5) * w * 0.85
            life = 2.2 + rng.random() * 1.8
            ph = (tq / life + rng.random()) % 1.0
            top = y0 + 6 + abs(sx - cx) * 0.6
            py = top + 18 - ph * 52
            px = sx + math.sin(tq * 1.3 + k) * 3 + (sx - cx) * ph * 0.4
            if ph < 0.85 and 0 <= int(py) < g.shape[0] and 0 <= int(px) < g.shape[1]:
                size = 2 if ph < 0.4 else 1
                col = BGR[VOID] if ph < 0.5 else BGR[VOID2]
                g[int(py):int(py) + size, int(px):int(px) + size] = col
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(g.shape[1], x0 + w), min(g.shape[0], y0 + h)
    if gx1 <= gx0 or gy1 <= gy0:
        return g
    sub = s[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    m = sub[..., 3] > 0
    g[gy0:gy1, gx0:gx1][m] = sub[..., :3][m]
    return g


if __name__ == '__main__':
    import cv2
    tiles = []
    for amp, bl, lk, md, po in [(0, False, 0, 'calm', 'stand'), (0.7, False, 0, 'calm', 'stand'), (0.5, False, 0, 'scary', 'stand'),
                                (0, False, 0, 'calm', 'sit')]:
        s_ = sprite(0.3, amp, bl, lk, 3, md, po)
        bg = np.zeros((H + 8, W + 8, 3), np.uint8); bg[:] = BGR[D3]
        bg[(np.indices(bg.shape[:2]).sum(0) // 8) % 2 == 0] = BGR[DARK]
        a_ = s_[..., 3:] / 255
        bg[4:4 + H, 4:4 + W] = (s_[..., :3] * a_ + bg[4:4 + H, 4:4 + W] * (1 - a_)).astype(np.uint8)
        tiles.append(cv2.resize(bg, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite('/home/claude/voynich/team/host_sheet.png', np.hstack(tiles))
