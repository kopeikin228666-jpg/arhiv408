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
# motion rig (sprite px): head = (bottom row, x0, x1) of the box that nods / tilts with the voice, tip = rows of the hood
# point that bend in the lamp's draught, arm = (y0, y1, x0) of the pointing arm that beats on stressed words
POSES['front'].update(head=(64, 22, 90), tip=26)
POSES['point'].update(head=(56, 12, 68), tip=22, arm=(36, 64, 70), pad=10)
POSES['close'].update(head=(140, 0, 149), tip=50)


# ------------------------------------------------------------------ motion helpers (pure functions of time)
def lamp_wind(t):
    """the slow draught of the swinging lamp (room.py swings it with a 4.28 s period): -1..1"""
    return 0.8 * math.cos(t * 2 * math.pi / 4.28) + 0.2 * math.sin(t * 2 * math.pi / 1.63 + 0.7)


def heartbeat(t, period=0.92):
    """lub-dub: 1.0 on the first thump, 0.6 on the second, else 0 (wide enough to land on a 15 fps frame)"""
    u = (t % period) / period
    return 1.0 if u < 0.095 else (0.6 if 0.24 <= u < 0.33 else 0.0)


def speech_beat(t, amp):
    """stressed-syllable pulse: on only while he is loud, with the cadence of speech (~2.3 beats/s)"""
    if amp < 0.6:
        return 0.0
    s = math.sin(t * 2 * math.pi * 2.3)
    return min(1.0, (amp - 0.6) / 0.35) if s > 0.15 else 0.0


def _hash(n, salt=0):
    """a deterministic pseudo-random 0..1 from an integer (frame index)"""
    return ((int(n) * 2654435761 + salt * 40503) % 4294967296) / 4294967296.0


def breath(t, mood='calm'):
    """slow breathing: 0 = full (chest up), 1 = exhaled (shoulders and hood sunk).  scary: slower, deeper, with a
    held breath at the top - a predator's calm.  -> (depth 0..1, exhaling bool)"""
    per = 5.4 if mood == 'scary' else 4.1
    u = ((t + 0.9) % per) / per
    if u < 0.38:                                            # inhale: rises
        return 0.5 + 0.5 * math.cos(math.pi * u / 0.38), False
    if u < 0.55 if mood == 'scary' else u < 0.45:            # hold at the top
        return 0.0, False
    a = 0.55 if mood == 'scary' else 0.45
    return 0.5 - 0.5 * math.cos(math.pi * (u - a) / (1 - a)), True


STRESS_TAU = 0.16


def stress_at(words, t, tau=STRESS_TAU):
    """stress envelope of his stressed words (the '*' key words of make_ep1e: dicts with 'key' and 't0') at time t:
    a short anticipation just before the attack, a hit of 1.0 on it, an exponential release.  -> 0..1
    (pass it to draw(..., stress=...) so his eyes and hood react exactly on the stressed words)"""
    s = 0.0
    for w in words or ():
        if not w.get('key'):
            continue
        d = t - w['t0']
        if -0.08 <= d < 0:
            s = max(s, 0.3 * (1 + d / 0.08))
        elif 0 <= d < 0.6:
            s = max(s, math.exp(-d / tau))
    return s


def _stress_kind(t, st, given):
    """which reaction this stressed word gets (0 nod-left, 1 nod-right, 2 recoil), constant over one word:
    recover the word's attack time from the exponential envelope"""
    if st <= 0.05:
        return 0
    if given and st < 0.999:
        t0 = t + STRESS_TAU * math.log(max(st, 1e-6))
    else:
        t0 = math.floor(t * 2.3) / 2.3
    return int(_hash(int(round(t0 * 4)), 77) * 3)
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


def _slit(q, ey, ex, sg, k, mood, lit, open_=0, flare=0, stare=0.0):
    """one eye: a heavy half-shut slit (calm) or a narrow burning slash (scary); k = pixel scale
    open_ 1: the slit widens (loud voice) ; flare 1: it burns hot with a sparse ember halo (stressed syllable / heartbeat)
    stare 0..1: the lids snap open - a wide burning almond with a black pinprick in it (the last beat)"""
    def put(dy, dx, c):
        y, x = ey + dy * k, ex + dx * k
        if y >= 0 and x >= 0:
            q[y:y + k, x:x + k] = BGR[c]
    if stare >= 0.34:
        wide = stare >= 0.67
        # an almond torn open: (row, from, to) in "inner-positive" units (u = dx * sg), the inner corner lower
        spans = ((-2, -1, 0), (-1, -3, 1), (0, -4, 3), (1, -3, 4), (2, 0, 2)) if wide else ((0, -4, 3), (1, -2, 3))
        for (dy, u0, u1) in spans:
            for u in range(u0, u1 + 1):
                edge = u in (u0, u1) or dy in (-2, 2)
                put(dy, u * sg, RED2 if edge else RED)
        if wide:
            put(0, 0, VOID); put(1, 0, VOID2)                   # a pinprick: it is looking at you
        return
    if mood == 'scary':
        # narrowed to a thin burning slash, the inner end dropped; hot (flare): longer, the outer tail cuts upward
        L = 4 if flare else 3
        for dx in range(-L, L + 1):
            dy = (1 if dx * sg > 1 else 0) - (1 if dx * sg < -1 else 0) - (1 if dx * sg < -3 else 0)
            put(dy, dx, RED if abs(dx) <= 2 or flare else RED2)
        for dx in (-1, 1):
            put(2, dx, RED2)
        if flare:                                   # the heartbeat: the slash flares, embers above and below
            if k > 1:
                put(-2, -2 * sg, RED2); put(3, 0, RED2); put(-1, sg, RED2)
        return
    # calm: a heavy lid over a smouldering slit - two rows, visibly more open than the scary slash
    for dx in range(-2, 3):
        put(1 if dx * sg >= 1 else 0, dx, RED2)
    if k > 1:
        for dx in (-1, 0):
            put(1, dx * sg, RED2)                   # the lower lid: the slit has some height
    put(0, -sg, RED)
    if lit:
        put(0, 0, RED)
        put(0, -2 * sg, RED)
    if open_:                                       # the lid lifts: a second row under the outer half
        put(1, 0, RED2); put(1, -sg, RED2)
    if flare:                                       # hot: the whole slit burns, a longer tail, a faint halo
        for dx in range(-2, 2):
            put(1 if dx * sg >= 1 else 0, dx, RED)
        put(0, -3 * sg, RED2)
        if k > 1:                                   # the halo only where there is room for it (close-ups)
            put(2, sg, RED)
            put(-1, -sg, RED2); put(2, -2 * sg, RED2)


def _face_smoke(q, P_, t, k, mood):
    """thin curls of smoke drifting across the void of the face (painted only on the void, under the eyes)"""
    cy, cx, ry, rx = P_['face']
    ry, rx = ry * 1.5, rx * 1.15
    void = BGR[VOID]
    n = 5 if mood == 'scary' else 3
    h, w = q.shape[:2]
    eye_y = P_['eyes'][0][0]
    for i in range(n):
        per = 3.1 + i * 0.83
        u = (t / per + i * 0.37) % 1.0
        dirn = 1 if i % 2 == 0 else -1
        xc = cx + dirn * (u * 2 - 1) * rx * 1.1
        yc = cy + (i - (n - 1) / 2) * ry * 0.38 + math.sin(t * 0.9 + i) * ry * 0.12
        L = int(rx * 0.75)
        fade_ = math.sin(math.pi * u)                     # a curl is born, crosses and thins out
        for j in range(0, L, k):
            e = math.sin(math.pi * j / L) * fade_
            if e < 0.25 or ((j // k + i + int(t * 10)) % 3 == 0 and e < 0.6):
                continue
            x = int(xc - L / 2 + j)
            y = int(yc + ry * 0.16 * math.sin(j * 0.22 / k + t * 2.1 + i * 1.7))
            y -= y % k; x -= x % k
            if abs(y - eye_y) <= 3 * k:                     # the smoke never crosses the eyes themselves
                continue
            if 0 <= y < h - k and 0 <= x < w - k and (q[y, x] == void).all():
                hot = e > (0.55 if mood == 'scary' else 0.7)
                c = (D3 if hot else DARK) if k > 1 else (DARK if hot else VOID2)
                q[y:y + k, x:x + k] = BGR[c]


def _eye_glow(q, m, eyes, k, amt, rim=0.0):
    """the eyes light the darkness around them: a sparse dithered red halo on the void of the face (1-px dither, so
    it reads as glow, not as a shape) and, with rim > 0, a faint red light on the inside edge of the hood"""
    if amt <= 0.02 and rim <= 0.02:
        return
    h, w = m.shape
    void = BGR[VOID]
    (ey0, ex0), (ey1, ex1) = eyes
    for (ey, ex) in eyes:
        if amt <= 0.02:
            break
        rx, ry = (4.0 + 3.0 * amt) * k, (1.7 + 2.0 * amt) * k
        y0, y1 = max(0, int(ey - ry)), min(h, int(ey + ry + 1) + k)
        x0, x1 = max(0, int(ex - rx)), min(w, int(ex + rx + 1) + k)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = ((xx - ex - k / 2) / rx) ** 2 + ((yy - ey - k / 2) / ry) ** 2
        bay = _B4[yy % 4, xx % 4] + 0.47
        sub = q[y0:y1, x0:x1]
        isv = (sub == void).all(2) & m[y0:y1, x0:x1]
        sel = isv & (d < 1) & (bay < amt * (1 - d) * 0.62)
        sub[sel] = BGR[RED2]
    if rim > 0.02:
        # the red light reaches the hood's inner edge (the lit pixels nearest the face)
        cy, cx = (ey0 + ey1) / 2 + 2 * k, (ex0 + ex1) / 2 + k / 2
        rx, ry = abs(ex1 - ex0) * 1.35 + 8 * k, 13 * k
        y0, y1 = max(0, int(cy - ry)), min(h, int(cy + ry))
        x0, x1 = max(0, int(cx - rx)), min(w, int(cx + rx))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        sub = q[y0:y1, x0:x1]
        lum = sub.astype(np.int16).sum(2)
        cloth = m[y0:y1, x0:x1] & (lum > 3 * 18) & (lum < 3 * 70) & (sub[..., 2] < 100)
        bay = _B4[yy % 4, xx % 4] + 0.47
        sel = cloth & (d < 1) & (bay < rim * (1 - d) * 0.9)
        sub[sel] = BGR[RED2]


def _sink(q, m, y_end, B, top=0):
    """breathing: everything above y_end sinks by B px (exhale), the rows just above y_end compress; the hood
    keeps its shape (the whole box above y_end - span moves rigidly)"""
    if B <= 0:
        return
    span = max(1, (y_end - top) // 3)
    ys = np.arange(top, y_end)
    d = np.clip((y_end - ys) / span, 0, 1) * B
    src = np.clip(np.round(ys - d).astype(int), 0, m.shape[0] - 1)
    q[top:y_end] = q[src]; m[top:y_end] = m[src]
    m[top:top + int(math.ceil(B))] = False


def _shift_box(q, m, y0, y1, x0, x1, dy, dx, clear=False):
    """move a box of the sprite by (dy, dx) px (the hood nodding, an arm beating).  Inside the box the moved
    content replaces what was there; what spills over its edge is added; the strip it vacates keeps the old pixels
    (cloak behind) or, with clear=True, becomes empty (the hood point drops)"""
    h, w = m.shape
    y0, y1, x0, x1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    ty0, tx0 = y0 + dy, x0 + dx
    a0, b0, a1, b1 = max(0, ty0), max(0, tx0), min(h, ty0 + y1 - y0), min(w, tx0 + x1 - x0)
    if a1 <= a0 or b1 <= b0:
        return
    bq = np.zeros_like(q); bm = np.zeros_like(m); cov = np.zeros_like(m)
    bq[a0:a1, b0:b1] = q[a0 - dy:a1 - dy, b0 - dx:b1 - dx]
    bm[a0:a1, b0:b1] = m[a0 - dy:a1 - dy, b0 - dx:b1 - dx]
    cov[a0:a1, b0:b1] = True
    inbox = np.zeros_like(m); inbox[y0:y1, x0:x1] = True
    r = cov & inbox
    q[r] = bq[r]; m[r] = bm[r]
    o = cov & ~inbox & bm
    q[o] = bq[o]; m[o] = True
    if clear:
        m[inbox & ~cov] = False


def _anim_motion(q, m, P_, t, amp, mood, pose, k, sway, st=None, kind=0, sink=0):
    """the living layer: hood nods / tilts / recoils on stressed words, slow idle sway, the hood point and rags move
    in the lamp's draught, the pointing arm beats on stressed words and a red signal runs to the finger"""
    h, w = m.shape
    wind = lamp_wind(t)
    beat = speech_beat(t, amp) if st is None else st
    hy1, hx0, hx1 = P_['head']
    # --- breathing: the hood and shoulders sink on the exhale (done first: everything else rides on it)
    if sink:
        _sink(q, m, P_['waist'] + 6 * k, sink)
    # --- head: on a stressed word a sharp nod down + a jab to one side, or a recoil (the hood jerks back up);
    #     the size follows the stress envelope, so it hits on the attack and settles
    tilt = 0.0
    if beat > 0.12:
        a = min(1.0, beat)
        if kind == 2:
            _shift_box(q, m, 0, hy1, hx0, hx1, -int(round(a * 1.4)) * k, 0)
            tilt = 0.0
        else:
            _shift_box(q, m, 0, hy1, hx0, hx1, int(round(a * 2.0)) * k, 0, clear=True)
            tilt = (1 if kind == 0 else -1) * a * 2.2 * k
    elif amp > 0.3:
        tilt = (1 if math.sin(t * 0.83 + 0.4) > 0 else -1) * k
    # idle: the hood turns very slowly, as if he were listening to something behind the screen
    tilt += math.sin(t * 2 * math.pi / 7.3 + 1.1) * 1.3 * k * (1.0 if amp < 0.05 else 0.4)
    tip = P_['tip']
    for y in range(0, hy1):
        d = 0.0
        if tilt:
            d += tilt * (1 - y / hy1)
        if y < tip:                                        # the hood point bends in the draught, the very tip flutters
            f = (1 - y / tip) ** 1.6
            d += wind * 1.8 * k * f + (0.8 * k * f if _hash(int(t * 12), y // 3) > 0.7 else 0)
        d = int(round(d))
        if d:
            q[y, hx0:hx1] = np.roll(q[y, hx0:hx1], d, 0); m[y, hx0:hx1] = np.roll(m[y, hx0:hx1], d, 0)
    # --- ragged edges flutter downwind (whole little strips of cloth, 12 fps)
    if sway:
        sgn = 1 if wind >= 0 else -1
        f = int(t * 12)
        rows = np.array([_hash(f, (yy // (2 * k)) * 7 + 3) for yy in range(h)])
        zone = np.zeros(h, bool)
        zone[int(tip * 0.4):h - 22 * k] = True
        p = 0.16 + 0.22 * abs(wind) + (0.12 if mood == 'scary' else 0)
        on = (rows < p) & zone
        dw = np.roll(m, sgn * k, 1) & ~m & on[:, None]
        q[dw] = BGR[VOID2]; m |= dw
        nib = m & ~np.roll(m, -sgn, 1) & (rows > 1 - p * 0.6)[:, None] & zone[:, None]
        m &= ~nib                                          # and the upwind edge frays a little
    # --- the pointing arm
    if 'arm' in P_:
        ay0, ay1, ax0 = P_['arm']
        if beat > 0.25:
            _shift_box(q, m, ay0, ay1, ax0, w, -1, 1)
        hand = m.copy(); hand[:ay0 - 2] = False; hand[ay1:] = False; hand[:, :ax0 + 22] = False
        ys, xs = np.nonzero(hand)
        if len(xs):
            i = int(np.argmax(xs)); fy, fx = int(ys[i]), int(xs[i])
            fi = int(t * 15)
            # a faint red signal leaves the fingertip: two dashes travelling outward, flickering
            for j in range(2):
                dx_ = 2 + ((fi + j * 3) % 6) * 1
                if _hash(fi, j + 11) < 0.75:
                    x_ = fx + dx_
                    if x_ + 1 < w:
                        q[fy, x_:x_ + 2 - j] = BGR[RED2 if j else RED2]; m[fy, x_:x_ + 2 - j] = True
            # embers run along the top of the sleeve toward the hand while he speaks
            if amp > 0.05:
                top = m & ~np.roll(m, 1, 0)
                top[:ay0] = False; top[ay1:] = False; top[:, :ax0 + 4] = False; top[:, fx - 1:] = False
                ty, tx = np.nonzero(top)
                sel = ((tx - fi * 2) % 9) == 0
                q[ty[sel], tx[sel]] = BGR[RED2]
            # glitch: on a stressed word a slice of the glove jumps forward with a red fringe
            if beat > 0.25 or _hash(fi, 5) < 0.12:
                gy = fy - 2 + int(_hash(fi, 6) * 5)
                gy0, gy1 = max(0, gy), min(h, gy + 2)
                xa = max(0, fx - 12)
                for yy in range(gy0, gy1):
                    row_q = q[yy, xa:fx + 1].copy(); row_m = m[yy, xa:fx + 1].copy()
                    q[yy, xa + 2:fx + 3][row_m] = row_q[row_m]; m[yy, xa + 2:fx + 3] |= row_m
                    lead = np.nonzero(row_m)[0]
                    if len(lead):
                        q[yy, xa + 2 + lead[-1]] = BGR[RED]
                        if xa + lead[0] < w:
                            q[yy, xa + lead[0]] = BGR[RED2]


def sprite_img(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='front', sway=True, anim=True,
               stress=None, stare=0.0, glow=0.0):
    """anim=False gives the old, quieter figure (no face smoke, nods, flares, draught, arm beat, breathing)
    stress 0..1: how hard the current word is stressed (stress_at); None = guessed from the voice loudness
    stare 0..1: the eyes snap wide open ; glow 0..1: the eyes light the hood from inside (leaning in, the dark beats)"""
    P_ = POSES[pose]
    q, m = _load_img(pose)
    q = q.copy(); m = m.copy()
    k = P_['k']
    pad = P_.get('pad', 0) if anim else 0
    if pad:                                                # room around the pointing hand for its red signal
        q = np.pad(q, ((0, 0), (pad, pad), (0, 0))); m = np.pad(m, ((0, 0), (pad, pad)))
        P_ = dict(P_, eyes=tuple((y, x + pad) for y, x in P_['eyes']), mouth=(P_['mouth'][0], P_['mouth'][1] + pad, P_['mouth'][2] + pad),
                  head=(P_['head'][0], P_['head'][1] + pad, P_['head'][2] + pad), arm=(P_['arm'][0], P_['arm'][1], P_['arm'][2] + pad),
                  face=(P_['face'][0], P_['face'][1] + pad, P_['face'][2], P_['face'][3]))
    h, w = m.shape
    flare = (t % 4.3) < 0.15
    lit = amp > 0.05 or flare
    open_ = fl = 0
    st, kind, sink, heat = 0.0, 0, 0, 0.0
    if anim:
        st = speech_beat(t, amp) if stress is None else max(0.0, min(1.0, float(stress)))
        kind = _stress_kind(t, st, stress is not None)
        depth, exhaling = breath(t, mood)
        bmax = (3 if mood == 'scary' else 2) * (k if k > 1 else 1)
        if pose == 'close':
            bmax = 4 if mood == 'scary' else 3
        sink = int(round(depth * bmax)) if amp < 0.5 else int(round(depth * bmax * 0.5))
        _face_smoke(q, P_, t, k, mood if not exhaling else 'scary')
        open_ = 1 if (amp > 0.8 or (st > 0.6 and mood != 'scary' and kind != 2)) else 0
        hb = heartbeat(t) if mood == 'scary' else 0.0
        heat = max(st, 1.0 if hb >= 1.0 else 0.0, glow)
        fl = 1 if (mood == 'scary' and hb >= 1.0) or st > 0.5 or glow > 0.6 else 0
    if not blink:
        eyes = []
        for j, (ey, ex) in enumerate(P_['eyes']):
            eyes.append((int(ey), int(round(ex + look * k))))
            _slit(q, int(ey), int(round(ex + look * k)), 1 if j == 0 else -1, k, mood, lit or fl, open_, fl, stare)
        if anim and k > 1:
            # the halo: a faint ember-glow at rest, a hot one on a stressed word / the heartbeat / leaning in
            base = 0.22 if mood == 'scary' else 0.12
            _eye_glow(q, m, eyes, k, max(base, 0.85 * heat, 1.0 * stare), rim=max(0.0, glow * 0.9, stare * 0.8, st * 0.3))
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
        # the cloak moves: rows below the waist swing sideways, more toward the hem, rags flutter; with anim the
        # whole hem also leans into the lamp's draught
        wy = P_['waist']
        wind = lamp_wind(t) if anim else 0.0
        for y in range(wy, h):
            d = (y - wy) / max(1, h - wy)
            sh = int(round(math.sin(t * 2 * math.pi / 2.14 + y * 0.09) * 2.2 * d + math.sin(t * 9 + y * 0.7) * 0.6 * d * d
                           + wind * 1.6 * d * d))
            if sh:
                q[y] = np.roll(q[y], sh, 0); m[y] = np.roll(m[y], sh, 0)
    if anim:
        _anim_motion(q, m, P_, t, amp, mood, pose, k, sway, st, kind, sink)
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
         sx=1.0, sy=1.0, scale_f=1.0, tilt=0, anim=True, vx=0.0, ghosts=0, stress=None, stare=0.0, glow=None):
    """anim: the living layer (see sprite_img; plus a heartbeat in the aura and thicker smoke when scary, smoke bent by
    the lamp's draught).  vx: his speed in grid px/s - above ~25 px/s fading afterimages trail behind him.
    ghosts: force n afterimages (0..3) regardless of vx (teleports, fast moves).
    stress: 0..1 stress of the current word (stress_at(words, t)); None = guessed from amp.
    stare: 0..1 the eyes snap wide open.  glow: 0..1 the eyes light the hood and the smoke; None = from scale_f
    (leaning into the lens makes them burn)."""
    if pose in ('stand', 'sit'):
        pose = 'front'
    if glow is None:
        glow = min(1.0, max(0.0, (scale_f - 1.0) / 0.08)) if anim else 0.0
    s = sprite_img(t, amp, blink, look, wave_seed, mood, pose, anim=anim, stress=stress, stare=stare, glow=glow) if look_img \
        else sprite(t, amp, blink, look, wave_seed, mood, pose)
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
        hb = heartbeat(t) if (anim and (mood == 'scary' or glow > 0.5)) else 0.0   # scary: the darkness pulses like a heart
        pad = 12 if hb > 0 else 10
        m = np.zeros((h + 2 * pad, w + 2 * pad), bool)
        m[pad:pad + h, pad:pad + w] = s[..., 3] > 0
        import cv2
        k1, k2 = (7 + int(round(4 * hb)) // 2 * 2, 15 + int(round(8 * hb)) // 2 * 2)
        d1 = cv2.dilate(m.astype(np.uint8), np.ones((k1, k1), np.uint8)).astype(bool)
        d2 = cv2.dilate(m.astype(np.uint8), np.ones((k2, k2), np.uint8)).astype(bool)
        yy, xx = np.mgrid[0:h + 2 * pad, 0:w + 2 * pad]
        bay = _B4[(yy - pad + 10) % 4, (xx - pad + 10) % 4] + 0.47     # same dither phase whatever the pad
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
        pi2[r1] = lut[lut[lut[pi[r1]]]] if hb >= 1.0 else lut[lut[pi[r1]]]
        pi2[r2] = lut[lut[pi[r2]]] if hb >= 1.0 else lut[pi[r2]]
        g[gy0:gy1, gx0:gx1] = BGR[pi2]
        # black smoke rising off the hood and shoulders, drifting and thinning out
        rng = np.random.default_rng(408)
        tq = math.floor(t * 15) / 15
        scary = anim and mood == 'scary'
        wind = lamp_wind(tq) if anim else 0.0
        n_smoke = int(smoke * (1.8 if scary else 1.0) * (1.0 + 1.3 * glow))
        P0 = POSES.get(pose, POSES['front'])
        (e0y, e0x), (e1y, e1x) = P0['eyes']
        ecy = y0 + (e0y + e1y) / 2 * (h / _load_img(pose)[1].shape[0] if (look_img and pose in POSES) else 1.0)
        for k in range(n_smoke):
            sx = cx + (rng.random() - 0.5) * w * 0.85
            life = (2.2 + rng.random() * 1.8) / (1.0 + 0.8 * glow)        # leaning in: the smoke boils faster
            ph = (tq / life + rng.random()) % 1.0
            top = y0 + 6 + abs(sx - cx) * 0.6
            py = top + 18 - ph * 52
            px = sx + math.sin(tq * 1.3 + k) * 3 + (sx - cx) * ph * (0.4 + 1.1 * glow) + wind * ph * ph * 10
            if ph < 0.85 and 0 <= int(py) < g.shape[0] and 0 <= int(px) < g.shape[1]:
                size = (3 if ph < 0.3 else 2 if ph < 0.6 else 1) if (scary or glow > 0.3) else (2 if ph < 0.4 else 1)
                col = BGR[VOID] if ph < 0.5 else BGR[VOID2]
                if glow > 0.3:
                    # the eyes' red light catches the wisps nearest the face; the rest are lit by the monitors
                    near = abs(py - ecy) < 40 * scale and abs(px - cx) < w * 0.45
                    hv = _hash(k * 31 + int(tq * 15), 9)
                    if near and hv < 0.45 * glow:
                        col = BGR[RED2]
                    elif hv < 0.5 * glow:
                        col = BGR[DARK]
                g[int(py):int(py) + size, int(px):int(px) + size] = col
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(g.shape[1], x0 + w), min(g.shape[0], y0 + h)
    if gx1 <= gx0 or gy1 <= gy0:
        return g
    nG = ghosts if ghosts else (0 if not anim or abs(vx) < 25 else (1 if abs(vx) < 50 else 2))
    if nG:
        _afterimages(g, s, x0, y0, -1 if vx >= 0 else 1, nG, t, step=max(4, min(12, int(abs(vx) * 0.12) + 4)))
    sub = s[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    m = sub[..., 3] > 0
    g[gy0:gy1, gx0:gx1][m] = sub[..., :3][m]
    return g


def _afterimages(g, s, x0, y0, back, n, t, step=6):
    """fading copies of him trailing behind (back = -1: to the left): dithered dark silhouettes, the eyes a dim red"""
    h, w = s.shape[:2]
    a = s[..., 3] > 0
    red = a & (s[..., 2].astype(int) - s[..., 1] > 60)
    yy, xx = np.mgrid[0:h, 0:w]
    fi = int(t * 15)
    for j in range(n, 0, -1):                       # the furthest, faintest first
        ox = x0 + back * step * j + (1 if _hash(fi, j) > 0.5 else 0)
        dens = (0.55, 0.32, 0.18)[min(2, j - 1)]
        keep = a & (_B4[yy % 4, xx % 4] + 0.47 < dens)
        col = BGR[D3] if j == 1 else BGR[DARK]
        gx0, gx1 = max(0, ox), min(g.shape[1], ox + w)
        gy0, gy1 = max(0, y0), min(g.shape[0], y0 + h)
        if gx1 <= gx0 or gy1 <= gy0:
            continue
        sl = (slice(gy0 - y0, gy1 - y0), slice(gx0 - ox, gx1 - ox))
        k_ = keep[sl]
        g[gy0:gy1, gx0:gx1][k_] = col
        r_ = red[sl] & a[sl]
        if j == 1:
            g[gy0:gy1, gx0:gx1][r_] = BGR[RED2]


# ------------------------------------------------------------------ ready-made shots (call-site helpers)
def dark_portrait(t, f, eye_y=207, mood='scary', stare=0.0, scale_f=1.0, glow=0.35, blink=False, smoke=20, look=0, stress=None):
    """his close portrait in total darkness: everything two palette steps darker except the red of the eyes (and
    the red light they throw on the hood).  The eye line stays at grid row eye_y whatever scale_f.
    -> (320, 180, 3) BGR grid"""
    g = np.zeros((320, 180, 3), np.uint8); g[:] = BGR[VOID]
    h0 = _load_img('close')[1].shape[0]
    ey = POSES['close']['eyes'][0][0]
    by = int(round(eye_y + (h0 - ey) * scale_f))
    draw(g, 90, by, t, 0, 1, blink, look, 0, f, mood, 'close', aura=False, smoke=smoke, scale_f=scale_f, stare=stare,
         glow=glow, stress=stress)
    pi = _pal_index(g); lut = np.array([_darker(i) for i in range(18)], np.int16)
    red = (pi == RED) | (pi == RED2)
    pi2 = lut[lut[pi]]; pi2[red] = pi[red]
    return BGR[pi2]


def final_card(t, tl, dur, f, eye_y=207):
    """the last card behind «ТЫ ДОЧИТАЛ. / ОНА — НЕТ.»: his eyes in the dark, below the words.  He creeps closer,
    the heart beats in the slits; 0.33 s before the end the eyes go out for two frames, then SNAP wide open - a burning
    almond with a pinprick, the hood lit red from inside - and the cut lands on that stare (the loop's frame 0 opens
    on the same stare: pass stare=max(0, 1 - tl / 0.3) to the hook's close-up).  tl = time in the card, dur = its length"""
    left = dur - tl
    creep = min(1.0, max(0.0, tl / max(0.1, dur - 0.33)))
    sf = 1.0 + 0.05 * creep * creep * (3 - 2 * creep)
    if left <= 0.27:                                          # SNAP
        return dark_portrait(t, f, eye_y, 'scary', stare=1.0, scale_f=sf + 0.07, glow=1.0, smoke=34)
    if left <= 0.34:                                          # the eyes go out
        return dark_portrait(t, f, eye_y, 'scary', blink=True, scale_f=sf, glow=0.0)
    return dark_portrait(t, f, eye_y, 'scary', scale_f=sf, glow=0.25 + 0.35 * creep)


def pip_box(t, amp=0.0, w=46, h=54):
    """the content of his little live window (BGR (h, w, 3)): the close portrait shrunk, breathing and swaying, the
    eyes redrawn crisp at the small size - they glance aside now and then, blink, burn on loud syllables - and the
    broken wave of the mouth while the story's voice runs.  Scanlines included."""
    import cv2
    s_ = sprite_img(t, amp, False, 0, int(t * 15), 'calm', 'close', sway=False)
    sp = s_[30:30 + 150]
    al = (sp[..., 3:4].astype(np.float32) / 255.0) if sp.shape[2] == 4 else 1.0   # transparent pixels -> the void, not grey
    flat = (sp[..., :3].astype(np.float32) * al + np.float32(BGR[VOID]) * (1 - al)).astype(np.uint8)
    small = cv2.resize(flat, (w, int(150 * w / 149)), interpolation=cv2.INTER_AREA)
    small = P.quantize(small, dither=0.0, contrast=1.1)[:h]
    box = np.zeros((h, w, 3), np.uint8); box[:] = BGR[VOID]
    box[:small.shape[0]] = small
    kx = w / 149
    depth, _ = breath(t, 'calm')
    dy = int(round(depth * 3 * kx + 0.2))
    # glances: every ~4.7 s the slits slide to one side for half a second (the letters moved?)
    gp = t % 4.7
    look = 0
    if 1.6 < gp < 2.15:
        look = -1 if _hash(int(t / 4.7), 3) < 0.5 else 1
    blink = (t % 3.3) < 0.09
    beat = speech_beat(t, amp)
    hot = beat > 0.4 or (t % 4.3) < 0.15
    ey = int((88 - 30) * kx) + dy
    if not blink:
        for j, ex in enumerate((int(61 * kx), int(88 * kx))):
            ex += look
            sg = 1 if j == 0 else -1
            box[ey, ex - 1:ex + 2] = BGR[RED2]; box[ey, ex] = BGR[RED]
            box[ey + 1, ex + sg] = BGR[RED2]                     # the inner corner drops
            if hot:
                box[ey, ex - 1:ex + 2] = BGR[RED]; box[ey, ex - 2 * sg] = BGR[RED2]
                if ey - 1 >= 0:
                    box[ey - 1, ex - sg] = BGR[RED2]
    if amp > 0.05:
        my = int((104 - 30) * kx) + dy; r_ = np.random.default_rng(int(t * 15))
        for x in range(int(64 * kx), int(86 * kx), 2):
            hh = int(round(amp * 2 * r_.random()))
            box[my - hh:my + hh + 1, x] = BGR[RED2]
    pi = _pal_index(box)                                          # scanlines inside the palette: one step darker
    lut = np.array([_darker(i) for i in range(len(BGR))], np.int16)
    keep = (pi == RED) | (pi == RED2)
    pi2 = pi.copy(); pi2[::2] = np.where(keep[::2], pi[::2], lut[pi[::2]])
    return BGR[pi2].astype(np.uint8)


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
