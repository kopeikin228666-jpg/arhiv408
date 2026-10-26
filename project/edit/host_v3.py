"""The channel host «408»: a chibi pixel figure in a fully black ragged hood.  No face - only darkness, two glowing red eyes,
and a mouth that is a digital sound wave while he speaks.

sprite(t, amp, blink=False, look=0) -> (H, W, 4) uint8 BGRA at sprite resolution (W=44, H=52), palette colours only
draw(g, cx, by, t, amp, scale=2, ...) pastes it on a 180x320 BGR grid (cx = centre x, by = bottom y)
"""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/voynich/look')
import pixelize as P

BGR = np.array(P.PALETTE, np.uint8).reshape(-1, 3)          # the grids are BGR, so is the palette
VOID, VOID2, DARK, D3, TILE, TILE2, MID, LINE, LIGHT, BONE = range(10)
G1, G2, G3, SEPIA, PARCH, PARCH2, RED2, RED = 10, 11, 12, 13, 14, 15, 16, 17
W, H = 52, 74


def _ell(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def sprite(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='stand'):
    """«408» as a little Death: a deep hood, broad shoulders, a long heavy ragged cloak, wide bell sleeves with
    black gloves; wide black trousers and heavy boots show under the hem.  Face = darkness, red eyes,
    a digital sound wave for a mouth.  amp 0..1 voice loudness; mood 'calm' | 'scary'; pose 'stand' | 'sit'"""
    yy, xx = np.mgrid[0:H, 0:W]
    idx = np.full((H, W), -1, np.int16)
    C = 26
    sit = pose == 'sit'
    # ---- hood: deep and a little pointed, the cowl falls wide over the shoulders
    hood = _ell(xx, yy, C, 11.5, 10.0, 10.5)
    hood |= _ell(xx, yy, C + 1.5, 2.5, 5.5, 2.4)
    for y in range(12, 24):
        half = 10.0 + (y - 12) * 0.32
        hood[y, int(round(C - half)):int(round(C + half)) + 1] = True
    for y in range(24, 28):
        half = 9 - (y - 24) * 2.6
        if half > 0:
            hood[y, int(round(C - half)):int(round(C + half)) + 1] = True
    # ---- the cloak: broad shoulders, flares to the floor-length ragged hem
    body = np.zeros((H, W), bool)
    hem = 50 if sit else 58
    for y in range(19, hem):
        if y <= 23:
            half = 10.5 + (y - 19) * 1.0
        else:
            half = 14.5 + (y - 23) / (hem - 23) * 7.5
        body[y, max(0, int(round(C - half))):min(W, int(round(C + half)) + 1)] = True
    lens = np.random.default_rng(408).integers(2, 9, 16)
    for i, x0 in enumerate(range(4, 49, 3)):
        L = lens[i]
        sw = int(round(math.sin(t * 2 * math.pi / 1.07 + i * 1.3)))
        for dy in range(L):
            y = hem + dy
            xa = x0 + (sw if dy >= L // 2 else 0)
            for x in range(xa, xa + (2 if dy < L - 1 else 1)):
                if 0 <= x < W and y < H and body[hem - 1, min(W - 1, max(0, x0))]:
                    body[y, x] = True
    # the front of the cloak parts in an inverted V so the legs show
    v0 = 44 if not sit else 42
    for y in range(v0, H):
        half = (y - v0) * 0.55
        if half >= 0.5:
            body[y, int(round(C - half)):int(round(C + half)) + 1] = False
    # ---- wide bell sleeves hanging at the sides, gloved hands peek out
    sleeves = np.zeros((H, W), bool)
    hands = np.zeros((H, W), bool)
    for j in (0, 1):
        sgn = -1 if j == 0 else 1
        for y in range(22, 44):
            k = (y - 22) / 22
            xc = C + sgn * (15.0 + k * 4.5)
            half = 2.5 + k * 3.5
            sleeves[y, max(0, int(round(xc - half))):min(W, int(round(xc + half)) + 1)] = True
        for x in range(W):                                           # ragged cuff
            if sleeves[43, x] and (x * 7 + j) % 3 == 0:
                sleeves[44:46, x] = True
        hx = int(round(C + sgn * 19.5)) - 1
        hands[44:48, hx:hx + 3] = True
        hands[45:47, hx + (-1 if j == 0 else 3)] = True
    # ---- legs: wide black trousers and heavy black boots under the hem
    LEGS = ((C - 9, C - 2), (C + 2, C + 9))
    legs = np.zeros((H, W), bool); boots = np.zeros((H, W), bool)
    leg_y, boot_y = [], []
    for j, (x0, x1) in enumerate(LEGS):
        out = -1 if j == 0 else 1
        if sit:
            sw = int(round(0.6 + 0.6 * math.sin(t * 2 * math.pi / 1.07 + j * math.pi)))
            ly1 = 61 + sw
            legs[44:ly1, x0:x1] = True
            by0, by1 = ly1 - 1, ly1 + 8
            boots[by0:by1, x0 - 1:x1 + 1] = True
            boots[by0 + 3:by1, (x0 - 2 if out < 0 else x0 - 1):(x1 + 1 if out < 0 else x1 + 2)] = True
        else:
            ly1 = 64
            legs[46:ly1, x0:x1] = True
            legs[58:ly1, (x0 - 1 if out < 0 else x1):(x0 if out < 0 else x1 + 1)] = True
            by0, by1 = 62, 73
            boots[by0:by0 + 5, x0 - 1:x1 + 1] = True
            fx0, fx1 = x0 - (4 if out < 0 else 1), x1 + (4 if out > 0 else 1)
            boots[by0 + 5:by1, fx0:fx1] = True
            tip = fx0 if out < 0 else fx1 - 1
            boots[by0 + 5, tip] = False; boots[by0 + 6, tip] = False; boots[by0 + 5, tip - out] = False
        leg_y.append((48, ly1)); boot_y.append((by0, by1))
    shape = hood | body | sleeves | hands | legs | boots
    idx[shape] = VOID2
    inner = shape.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(np.roll(shape, dy, 0), dx, 1)
    edge = shape & ~inner
    lit = (xx - C) * 0.6 + (yy - 20) * 0.4
    idx[edge] = DARK
    idx[edge & (lit < -2)] = D3
    idx[edge & (lit < -9)] = TILE
    # hood's shadow on the shoulders, the cloak's front opening, folds
    seam = (body | sleeves) & ~hood & np.roll(hood, 1, 0)
    idx[seam] = VOID
    idx[28:v0, C] = VOID
    vedge = body & ~np.roll(body, -1, 1) & (xx < C) & (yy >= v0)
    vedge |= body & ~np.roll(body, 1, 1) & (xx > C) & (yy >= v0)
    idx[vedge] = VOID
    for fx, f0, f1 in ((C - 6, 30, hem), (C + 6, 30, hem), (C - 11, 36, hem), (C + 11, 36, hem), (C - 3, 40, hem), (C + 3, 40, hem)):
        for y in range(f0, f1):
            if inner[y, fx]:
                idx[y, fx] = VOID
    idx[27, C - 1:C + 2] = RED2                                 # a small clasp at the throat
    idx[27, C] = RED
    # sleeves: outlined against the cloak, ragged cuffs
    sl = sleeves & ~hood
    sin_ = sl & ~(np.roll(sl, 1, 1) & np.roll(sl, -1, 1))
    idx[sin_ & (xx > 9) & (xx < W - 9)] = VOID                 # the sleeve's inner edge against the cloak
    for y in range(36, 44):
        for x in range(W):
            if sl[y, x] and not sl[y - 1, x]:
                idx[y, x] = VOID
    idx[43:46][sl[43:46]] = VOID                                 # dark inside of the bell
    # gloves: black with a little light on the knuckles
    idx[hands] = VOID
    for y, x in zip(*np.nonzero(hands & ~np.roll(hands, 1, 0))):
        idx[y, x] = DARK
    # trousers + boots, all black, thin cold rim
    lg = legs & ~body
    idx[lg] = VOID2
    for j, (x0, x1) in enumerate(LEGS):
        col = lg & (xx >= x0 - 1) & (xx <= x1)
        idx[col & (xx == (x0 - 1 if j == 0 and not sit else x0))] = D3
        idx[col & (xx == x0 + 3)] = VOID
        idx[col & (yy == leg_y[j][1] - 1)] = VOID
        if sit:
            idx[51, x0:x1] = VOID                                # the knees
    bt = boots & ~body
    idx[bt] = VOID2
    idx[bt & ~(np.roll(bt, 1, 0) & np.roll(bt, 1, 1))] = DARK
    for j, (x0, x1) in enumerate(LEGS):
        by0, by1 = boot_y[j]
        bj = bt & (xx >= x0 - 5) & (xx <= x1 + 4)
        idx[bj & (yy == by1 - 1)] = VOID
        idx[bj & (yy == by1 - 2)] = DARK
        cx = (x0 + x1) // 2
        for y in range(by0 + 1, by1 - 4, 2):
            if bj[y, cx]:
                idx[y, cx - 1:cx + 1] = DARK
        if sit:
            idx[by1 - 4, x0 + 1:x0 + 3] = D3
        else:
            tx = x0 - 4 if j == 0 else x1 + 2
            idx[by1 - 4, tx + (0 if j == 0 else 1)] = D3
    # ---- face: an oval of pure darkness inside the cowl
    face = _ell(xx, yy, C, 13.5, 6.8, 7.8)
    rim = _ell(xx, yy, C, 13.5, 8.2, 9.1) & ~face & hood
    idx[rim] = VOID2
    idx[rim & (yy < 10) & (xx < C + 1)] = DARK
    idx[face] = VOID
    # ---- eyes
    ex = [C - 3 + look, C + 3 + look]
    ey = 12
    if mood == 'scary' and not blink:
        for j, e in enumerate(ex):
            if j == 0:
                idx[ey, e - 2:e] = RED; idx[ey + 1, e - 1:e + 1] = RED
                idx[ey, e - 2] = BONE
            else:
                idx[ey, e:e + 2] = RED; idx[ey + 1, e - 1:e + 1] = RED
                idx[ey, e + 1] = BONE
            idx[ey + 2, e - 1:e + 1] = RED2
    else:
        for e in ex:
            if blink:
                idx[ey + 1, e - 1:e + 1] = RED
                continue
            idx[ey - 1, e - 1:e + 1] = RED2
            idx[ey + 2, e - 1:e + 1] = RED2
            idx[ey:ey + 2, e - 2] = RED2
            idx[ey:ey + 2, e + 1] = RED2
            idx[ey:ey + 2, e - 1:e + 1] = RED
            idx[ey, e - 1] = BONE
    # ---- mouth: digital sound wave
    mx0, mx1, my = C - 5, C + 5, 17
    if amp < 0.04:
        idx[my, mx0 + 1:mx1] = RED2
    else:
        ph = np.random.default_rng(wave_seed).random(64) * 6.28
        idx[my, mx0:mx1 + 1] = RED2
        n = (mx1 - mx0) // 2 + 1
        for i, x in enumerate(range(mx0, mx1 + 1, 2)):
            env = 0.35 + 0.65 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.25 + 0.75 * abs(math.sin(t * 19 + ph[i] + i * 0.9)))
            hgt = int(round(v * 2.6))
            idx[my - hgt:my + hgt + 1, x] = RED
    out = np.zeros((H, W, 4), np.uint8)
    m = idx >= 0
    out[m, :3] = BGR[idx[m]]
    out[m, 3] = 255
    return out


def draw(g, cx, by, t, amp, scale=2, blink=False, look=0, bob=0, wave_seed=0, mood='calm', pose='stand'):
    s = sprite(t, amp, blink, look, wave_seed, mood, pose)
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
    for amp, bl, lk, md, po in [(0, False, 0, 'calm', 'stand'), (0.7, False, 0, 'calm', 'stand'), (0, False, 0, 'scary', 'stand'),
                                (0, False, 0, 'calm', 'sit'), (0.6, False, -1, 'calm', 'sit')]:
        s_ = sprite(0.3, amp, bl, lk, 3, md, po)
        bg = np.zeros((H + 8, W + 8, 3), np.uint8); bg[:] = BGR[D3]
        bg[(np.indices(bg.shape[:2]).sum(0) // 6) % 2 == 0] = BGR[DARK]
        a_ = s_[..., 3:] / 255
        bg[4:4 + H, 4:4 + W] = (s_[..., :3] * a_ + bg[4:4 + H, 4:4 + W] * (1 - a_)).astype(np.uint8)
        tiles.append(cv2.resize(bg, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite('/home/claude/voynich/team/host_sheet.png', np.hstack(tiles))
