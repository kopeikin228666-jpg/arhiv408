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
    v0 = 42.0 if sit else 44.0
    body &= ~((Y >= v0) & (np.abs(X - C) < (Y - v0) * 0.55))
    holes = body & (Y > hem - 8) & (Y < hem) & (_NOISE > 0.985)        # moth holes near the hem
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
    cl = _tone(0.35 * fold + 0.45 * light - 0.1)
    idx[body] = cl[body]
    idx[holes] = VOID
    # sleeves: their own folds, a dark inside of the bell, a dark edge against the cloak
    st = _tone(0.3 * np.sin(X * 1.6 + Y * 0.15) + 0.45 * light - 0.05)
    idx[sleeves] = st[sleeves]
    for sgn in (-1, 1):
        idx[sleeves & _ell(C + sgn * 19.5, 44.4, 5.0, 1.5)] = VOID
        e = sleeves & ~np.roll(sleeves, sgn, 1) & (np.abs(X - C) < 19)
        idx[e] = VOID
    # hood: rounded, lit upper-left, a stitched seam along its front edge
    ht = _tone(-(X - C) / 13 - (Y - 6) / 18 - 0.15)
    idx[hood] = ht[hood]
    seam_line = hood & _ell(C, 13.5, 9.6, 10.5) & ~_ell(C, 13.5, 9.0, 9.9)
    idx[seam_line & ((_xx + _yy) % 3 == 0)] = DARK
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
    # the front edges of the parted cloak
    vedge = body & (Y >= v0) & ((~np.roll(body, -1, 1) & (X < C)) | (~np.roll(body, 1, 1) & (X > C)))
    idx[vedge] = VOID
    # clasp: a red stone on a short chain at the throat
    chain = (np.abs(Y - 26.6 - 0.12 * np.abs(X - C)) < 0.35) & (np.abs(X - C) < 6) & hood
    idx[chain & (_xx % 2 == 0)] = D3
    stone = _ell(C, 27.4, 1.25, 1.25)
    idx[stone] = RED
    idx[stone & ~_ell(C, 27.4, 0.8, 0.8)] = RED2
    idx[int(27.0 * R), int(C * R) - 1] = BONE
    # gloves: black, separate fingers, light on the knuckles
    idx[hands] = VOID2
    for sgn in (-1, 1):
        hx = C + sgn * 19.5
        for f in (-0.9, 0.0, 0.9):
            idx[(np.abs(X - (hx + f)) < 0.25) & (Y > 46.6) & hands] = VOID
        idx[hands & (np.abs(Y - 45.2) < 0.3) & ((_xx % 2) == 0)] = D3
    # trousers: wide, black, folds, thin rim
    lg = legs & ~body
    trt = _tone(0.3 * np.sin(X * 2.1 + Y * 0.6) - (X - C) / 30 - 0.15)
    idx[lg] = trt[lg]
    for (x0, x1, ly1, by0, by1) in geo:
        col = lg & (X >= x0 - 1) & (X < x1 + 1)
        idx[col & (np.abs(X - (x0 + 0.25)) < 0.3)] = D3          # lit edge
        idx[col & (Y > ly1 - 1.2)] = VOID                        # hem falls over the boot
    # boots: heavy black leather, criss-cross laces, an ankle strap with a buckle, a stitched sole, scuffed toe
    bt = boots & ~body
    btt = _tone(-(X - C) / 26 - (Y - 62) / 30 - 0.2, cuts=(-0.3, 0.2, 0.7))
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
    idx[edge & (lit < -3) & ~keep] = D3
    idx[edge & (lit < -10) & ~keep] = TILE
    # ================================================================ face
    face = _ell(C, 13.5, 6.8, 7.8)
    inner = _ell(C, 13.5, 8.2, 9.1) & ~face & hood
    it = _tone(-(X - C) / 10 - (Y - 9) / 8 - 0.55, levels=(VOID, VOID2, DARK, DARK))
    idx[inner] = it[inner]
    idx[face] = VOID
    ex = [C - 3 + look, C + 3 + look]
    ey = 12.5
    if mood == 'scary' and not blink:
        for j, e in enumerate(ex):
            sg = 1 if j == 0 else -1                            # the inner ends sit lower: angry
            slit = (np.abs((Y - ey) - sg * 0.35 * (X - e)) < 0.55) & (np.abs(X - e) < 1.9) & face
            idx[slit] = RED
            glow = (np.abs((Y - ey - 1.4) - sg * 0.35 * (X - e)) < 0.4) & (np.abs(X - e) < 1.4) & face & ((_xx + _yy) % 2 == 0)
            idx[glow] = RED2
            idx[int(round((ey - sg * 0.35 * 1.2) * R)), int(round((e - sg * 1.2) * R))] = BONE
    else:
        for e in ex:
            if blink:
                idx[face & (np.abs(Y - ey - 0.5) < 0.3) & (np.abs(X - e) < 1.2)] = RED
                continue
            halo = face & _ell(e, ey, 2.0, 2.4) & ~_ell(e, ey, 1.1, 1.5) & ((_xx + _yy) % 2 == 0)
            idx[halo] = RED2
            idx[_ell(e, ey, 1.1, 1.5)] = RED
            idx[int((ey - 0.9) * R), int((e - 0.6) * R)] = BONE   # the glint
    # mouth: a digital sound wave, 1-px bars with gaps on a faint baseline
    yb = int(18.0 * R)
    xs = list(range(int((C - 5) * R), int((C + 5) * R) + 1))
    if amp < 0.04:
        idx[yb, xs] = RED2
    else:
        ph = np.random.default_rng(wave_seed).random(64) * 6.28
        n = len(xs)
        for i, x in enumerate(xs):
            idx[yb, x] = RED2
            if i % 2:
                continue
            env = 0.3 + 0.7 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.2 + 0.8 * abs(math.sin(t * 19 + ph[(i // 2) % 64] + i * 0.45)))
            hgt = int(round(v * 5.5))
            idx[yb - hgt:yb + hgt + 1, x] = RED
    out = np.zeros((H, W, 4), np.uint8)
    m = idx >= 0
    out[m, :3] = BGR[idx[m]]
    out[m, 3] = 255
    return out


def draw(g, cx, by, t, amp, scale=1, blink=False, look=0, bob=0, wave_seed=0, mood='calm', pose='stand'):
    s = sprite(t, amp, blink, look, wave_seed, mood, pose)
    if scale != 1:
        s = np.repeat(np.repeat(s, scale, 0), scale, 1)
    h, w = s.shape[:2]
    x0, y0 = int(cx - w // 2), int(by - h + bob)
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
