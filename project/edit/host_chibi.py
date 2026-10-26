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
W, H = 44, 64


def _ell(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def sprite(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='stand'):
    """«408»: a chibi in a fully black ragged hood.  amp 0..1 = loudness of the voice (0 = silent).
    mood: 'calm' (round eyes with a glint) | 'scary' (narrow burning slits)"""
    yy, xx = np.mgrid[0:H, 0:W]
    idx = np.full((H, W), -1, np.int16)
    # ---- the hood: big and round (chibi), slightly flattened on top, torn corners under the face
    hood = _ell(xx, yy, 22, 18, 15.5, 13.5) & (yy >= 5)
    hood |= _ell(xx, yy, 22, 9, 11, 5.5)
    # ---- legs: wide black trousers, big heavy black boots
    LEGS = ((11, 19), (25, 33))
    legs = np.zeros((H, W), bool)
    boots = np.zeros((H, W), bool)
    leg_y, boot_y = [], []
    for j, (x0, x1) in enumerate(LEGS):
        out = -1 if j == 0 else 1
        if pose == 'sit':                                   # sitting in a big chair: shins dangle, feet swing
            sw = int(round(0.6 + 0.6 * math.sin(t * 2 * math.pi / 1.07 + j * math.pi)))
            ly1 = 49 + sw
            legs[38:ly1, x0:x1] = True
            by0, by1 = ly1 - 1, ly1 + 8
            boots[by0:by1, x0 - 1:x1 + 1] = True             # big boot seen from the front
            boots[by0 + 3:by1, (x0 - 2 if out < 0 else x0 - 1):(x1 + 1 if out < 0 else x1 + 2)] = True   # wide toe, outward
            for (cy_, cx_) in ((by1 - 1, x0 - 2), (by1 - 1, x1 + 1), (by0 + 3, x0 - 2), (by0 + 3, x1 + 1)):
                if 0 <= cx_ < W: boots[cy_, cx_] = False      # rounded corners
        else:
            ly1 = 53
            legs[38:ly1, x0:x1] = True
            legs[46:ly1, (x0 - 1 if out < 0 else x1):(x0 if out < 0 else x1 + 1)] = True   # wide flare
            by0, by1 = 51, 63
            boots[by0:by0 + 6, x0 - 1:x1 + 1] = True         # tall shaft
            fx0, fx1 = x0 - (4 if out < 0 else 1), x1 + (4 if out > 0 else 1)
            boots[by0 + 6:by1, fx0:fx1] = True               # heavy foot, toe outward
            tip = fx0 if out < 0 else fx1 - 1
            boots[by0 + 6, tip] = False; boots[by0 + 7, tip] = False   # rounded toe
            boots[by0 + 6, tip - out] = False
        leg_y.append((38, ly1)); boot_y.append((by0, by1))
    # ---- the cloak: shorter now (to the knees), flares, ends in long torn rags that flutter
    body = np.zeros((H, W), bool)
    for y in range(26, 41):
        k = (y - 26) / 14
        half = 10.5 + k * 7.5
        body[y, int(round(22 - half)):int(round(22 + half)) + 1] = True
    rag_rng = np.random.default_rng(408)
    lens = rag_rng.integers(2, 9, 12)
    for i, x0 in enumerate(range(4, 41, 3)):               # rag strips, 2-3 px wide, each swings on its own phase
        L = lens[i % 12]
        sw = int(round(math.sin(t * 2 * math.pi / 1.07 + i * 1.3) * 1.0))
        for dy in range(L):
            y = 41 + dy
            w_ = 2 if dy < L - 1 else 1
            xa = x0 + (sw if dy >= L // 2 else 0)
            for x in range(xa, xa + w_):
                if 0 <= x < W and y < H and body[40, min(W - 1, max(0, x0))]:
                    body[y, x] = True
    shape = hood | body | legs | boots
    # fully black: near-black body, pure black folds, only a cold rim of light on the upper-left contour
    idx[shape] = VOID2
    inner = shape.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(np.roll(shape, dy, 0), dx, 1)
    edge = shape & ~inner
    lit = (xx - 22) * 0.6 + (yy - 22) * 0.4
    idx[edge & (lit < -2)] = D3
    idx[edge & (lit < -9)] = TILE
    seam = body & ~hood & np.roll(hood, 1, 0) & (xx > 6) & (xx < 38)
    idx[seam] = VOID                                       # the hood's shadow on the shoulders
    for fx, f0, f1 in ((15, 33, 47), (22, 37, 48), (29, 33, 47), (9, 40, 46), (35, 40, 46)):   # folds
        for y in range(f0, f1):
            if inner[y, fx]:
                idx[y, fx] = VOID
    # a couple of tears in the cloak
    for (x, y) in ((12, 38), (12, 39), (31, 41), (32, 41), (31, 42)):
        idx[y, x] = VOID
    # everything black: trousers and boots near-black, folds pure black, a thin cold rim on the lit edges only
    lg = legs & ~body
    idx[lg] = VOID2
    for j, (x0, x1) in enumerate(LEGS):
        col = lg & (xx >= x0 - 1) & (xx <= x1)
        lx = x0 - 1 if (j == 0 and pose != 'sit') else x0
        idx[col & (xx == lx)] = D3                          # lit edge
        idx[col & (xx == x0 + 3)] = VOID                    # folds of the wide trousers
        idx[col & (xx == x1 - 2) & (yy > 44)] = VOID
        ly0, ly1 = leg_y[j]
        idx[col & (yy == ly1 - 1)] = VOID                   # the hem falls over the boot
    bt = boots & ~body
    idx[bt] = VOID2
    bedge = bt & ~(np.roll(bt, 1, 0) & np.roll(bt, 1, 1))
    idx[bedge] = DARK                                       # upper-left edge of the leather
    for j, (x0, x1) in enumerate(LEGS):
        by0, by1 = boot_y[j]
        bj = bt & (xx >= x0 - 5) & (xx <= x1 + 4)
        idx[bj & (yy == by1 - 1)] = VOID                      # heavy sole
        idx[bj & (yy == by1 - 2)] = DARK                      # sole edge
        cx = (x0 + x1) // 2
        for y in range(by0 + 1, by1 - 4, 2):                # laces, black on black, just a glint
            if bj[y, cx]:
                idx[y, cx - 1:cx + 1] = DARK
        if pose == 'sit':
            idx[by1 - 4, x0:x1] = DARK                        # toe cap
            idx[by1 - 4, x0 + 1:x0 + 3] = D3
        else:
            tx = x0 - 4 if j == 0 else x1 + 2
            idx[by1 - 4, tx:tx + 2] = DARK
            idx[by1 - 4, tx + (0 if j == 0 else 1)] = D3     # toe catches a little light
    # the eyes' red light spills on the lower inner edge of the hood
    spill = _ell(xx, yy, 22, 20.5, 11.6, 9.0) & ~_ell(xx, yy, 22, 20, 10.5, 8.0) & (yy > 22)
    # ---- the face: an opening of pure darkness, ragged on the lower corners
    face = _ell(xx, yy, 22, 20, 10.5, 8.0)
    face |= _ell(xx, yy, 13.2, 25.2, 1.6, 1.4) | _ell(xx, yy, 30.8, 25.2, 1.6, 1.4)
    rim = _ell(xx, yy, 22, 20, 12.0, 9.4) & ~face & hood
    idx[rim] = VOID2
    idx[rim & (yy < 16) & (xx < 23)] = DARK                 # the hood's front edge catches a little light
    idx[face] = VOID
    # ---- eyes
    ex = [17 + look, 27 + look]
    ey = 19
    if mood == 'scary' and not blink:
        for j, e in enumerate(ex):
            idx[ey + 2, e - 1:e + 2] = RED2
            sgn = 1 if j == 0 else -1                        # angry: the inner ends sit lower
            for k, dx in enumerate((-2, -1, 0, 1, 2)):
                yk = ey + (1 if (dx * sgn) > 0 else 0)
                idx[yk, e + dx] = RED
                idx[yk + 1, e + dx] = RED if abs(dx) < 2 else RED2
            idx[ey, e] = BONE
    else:
        for e in ex:
            if blink:
                idx[ey + 1, e - 1:e + 2] = RED
                continue
            idx[ey - 2, e - 1:e + 1] = RED2
            idx[ey + 2, e - 1:e + 1] = RED2
            idx[ey - 1:ey + 2, e - 2] = RED2
            idx[ey - 1:ey + 2, e + 1] = RED2
            idx[ey - 1:ey + 2, e - 1:e + 1] = RED
            idx[ey - 1, e - 1] = BONE                         # the glint
    # ---- mouth: a digital sound wave - thin equaliser bars with gaps on a dim baseline
    mx0, mx1, my = 15, 29, 24
    if amp < 0.04:
        idx[my, mx0 + 2:mx1 - 1] = RED2
    else:
        r = np.random.default_rng(wave_seed)
        ph = r.random(64) * 6.28
        idx[my, mx0:mx1 + 1] = RED2
        n = (mx1 - mx0) // 2 + 1
        for i, x in enumerate(range(mx0, mx1 + 1, 2)):
            env = 0.35 + 0.65 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.25 + 0.75 * abs(math.sin(t * 19 + ph[i] + i * 0.9)))
            hgt = int(round(v * 3.3))
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
