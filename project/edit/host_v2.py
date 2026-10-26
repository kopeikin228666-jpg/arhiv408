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
W, H = 40, 70


def _ell(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def sprite(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='stand'):
    """«408», a little more human: hood ~1/3 of the height, shoulders, arms in long sleeves with black gloves,
    wide black trousers, heavy black boots.  Face = darkness, red eyes, a digital sound wave for a mouth.
    amp 0..1 voice loudness; mood 'calm' | 'scary'; pose 'stand' | 'sit'"""
    yy, xx = np.mgrid[0:H, 0:W]
    idx = np.full((H, W), -1, np.int16)
    C = 20
    sit = pose == 'sit'
    # ---- hood: a rounded cowl over the head that drapes onto the shoulders
    hood = _ell(xx, yy, C, 11, 8.5, 10.0)
    for y in range(12, 23):
        half = 8.5 + (y - 12) * 0.12
        hood[y, int(round(C - half)):int(round(C + half)) + 1] = True
    for y in range(23, 26):                                          # the cowl ends in a soft V on the chest
        half = 7 - (y - 23) * 2.5
        hood[y, int(round(C - half)):int(round(C + half)) + 1] = True
    hood[0:2] = False
    hood |= _ell(xx, yy, C + 1, 2.2, 5, 2.0)                       # the hood's crown, a soft point
    # ---- the cloak over the torso, to mid-thigh, ragged hem
    body = np.zeros((H, W), bool)
    hem = 44 if not sit else 46
    for y in range(20, hem):
        k = (y - 23) / (hem - 23)
        half = 9.5 + k * 3.0 if y > 23 else 8.0 + (y - 20) * 1.6       # shoulders wider than the hood
        body[y, int(round(C - half)):int(round(C + half)) + 1] = True
    lens = np.random.default_rng(408).integers(1, 5, 12)
    for i, x0 in enumerate(range(8, 33, 3)):
        L = lens[i]
        sw = int(round(math.sin(t * 2 * math.pi / 1.07 + i * 1.3)))
        for dy in range(L):
            y = hem + dy
            xa = x0 + (sw if dy >= L // 2 else 0)
            for x in range(xa, xa + (2 if dy < L - 1 else 1)):
                if 0 <= x < W and y < H and body[hem - 1, min(W - 1, max(0, x0))]:
                    body[y, x] = True
    # ---- arms: long sleeves hanging at the sides, gloved hands
    sleeves = np.zeros((H, W), bool)
    hands = np.zeros((H, W), bool)
    for j, sx in enumerate((5, 30)):
        for y in range(22, 40):
            k = (y - 22) / 18
            x0 = int(round(sx + (-1 if j == 0 else 1) * k * 2.0))
            sleeves[y, x0:x0 + 5] = True
        sleeves[21, sx + 1:sx + 5] = True
        hx = int(round(sx + (-1 if j == 0 else 1) * 2.0)) + 1
        hands[39:43, hx:hx + 3] = True
        hands[40:42, hx - (1 if j == 0 else -3):hx - (1 if j == 0 else -3) + 1] = True    # thumb
    # ---- legs: wide black trousers, heavy black boots
    LEGS = ((12, 18), (22, 28))
    legs = np.zeros((H, W), bool); boots = np.zeros((H, W), bool)
    leg_y, boot_y = [], []
    for j, (x0, x1) in enumerate(LEGS):
        out = -1 if j == 0 else 1
        if sit:
            sw = int(round(0.6 + 0.6 * math.sin(t * 2 * math.pi / 1.07 + j * math.pi)))
            legs[44:52, x0 - 1:x1 + 1] = True                  # thighs toward the camera (foreshortened)
            ly1 = 59 + sw
            legs[52:ly1, x0:x1] = True                          # shins dangle
            by0, by1 = ly1 - 1, ly1 + 7
            boots[by0:by1, x0 - 1:x1 + 1] = True
            boots[by0 + 3:by1, (x0 - 2 if out < 0 else x0 - 1):(x1 + 1 if out < 0 else x1 + 2)] = True
        else:
            ly1 = 59
            legs[42:ly1, x0:x1] = True
            legs[52:ly1, (x0 - 1 if out < 0 else x1):(x0 if out < 0 else x1 + 1)] = True
            by0, by1 = 57, 68
            boots[by0:by0 + 5, x0 - 1:x1 + 1] = True
            fx0, fx1 = x0 - (4 if out < 0 else 1), x1 + (4 if out > 0 else 1)
            boots[by0 + 5:by1, fx0:fx1] = True
            tip = fx0 if out < 0 else fx1 - 1
            boots[by0 + 5, tip] = False; boots[by0 + 6, tip] = False; boots[by0 + 5, tip - out] = False
        leg_y.append((44, ly1)); boot_y.append((by0, by1))
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
    idx[26:hem, C] = VOID
    for fx, f0, f1 in ((14, 30, hem), (26, 30, hem), (10, 34, hem)):
        for y in range(f0, f1):
            if inner[y, fx]:
                idx[y, fx] = VOID
    idx[24, C - 1:C + 2] = RED2                                 # a small clasp at the throat
    idx[24, C] = RED
    # sleeves: outlined against the cloak, ragged cuffs
    sl = sleeves & ~hood
    sin_ = sl & ~(np.roll(sl, 1, 1) & np.roll(sl, -1, 1))
    idx[sin_ & (xx > 8) & (xx < 32)] = VOID
    for j, sx in enumerate((6, 30)):
        idx[39, sx + (0 if j == 0 else 2):sx + (3 if j == 0 else 5)] = VOID
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
            idx[50, x0:x0 + 3] = D3
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
    face = _ell(xx, yy, C, 13, 6.3, 7.4)
    rim = _ell(xx, yy, C, 13, 7.6, 8.6) & ~face & hood
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
