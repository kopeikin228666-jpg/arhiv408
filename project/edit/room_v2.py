"""The keeper's room: a dark archive where «408» sits in a big chair in front of a wall of old computers.

frame(t, amp=0.0, mood='calm', blink=False, look=0, screen=None) -> (320, 180, 3) uint8 BGR grid, palette colours only
    amp    voice loudness 0..1 (drives his mouth wave and the waveform monitor)
    screen optional (h, w, 3) grid image shown on the big central monitor (e.g. the case's first shot)
"""
import math, numpy as np
import host as HO
from host import BGR, VOID, VOID2, DARK, D3, TILE, TILE2, MID, LINE, LIGHT, BONE, G1, G2, G3, SEPIA, PARCH, PARCH2, RED2, RED

GW, GH = 180, 320
CAB = ((0, 28), (152, 180))
LED0, LED1 = 30, 150
REELS = (170, 198)
yy, xx = np.mgrid[0:GH, 0:GW]

# layout
MON_BIG = (50, 70, 130, 128)          # x0, y0, x1, y1 of the central CRT (outer bezel)
MONS = [(34, 26, 82, 64, 'text'), (98, 26, 146, 64, 'glyph'), (30, 70, 46, 128, 'bars'), (134, 70, 150, 128, 'radar'),
        (36, 134, 74, 164, 'wave'), (106, 134, 144, 164, 'rec')]
DESK_Y = 172                           # top of the desk surface
CHAIR = (44, 116, 136, 236)
SEAT_Y = 234
HOST_CX, HOST_BY, HOST_S = 90, 298, 1
FLOOR_Y = 236


def _rect(I, x0, y0, x1, y1, c):
    I[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = c


def _crt(I, x0, y0, x1, y1):
    """a chunky CRT: rounded bezel, dark green glass with scanlines; returns the glass box"""
    _rect(I, x0, y0, x1, y1, D3)
    _rect(I, x0 + 1, y0 + 1, x1 - 1, y1 - 1, DARK)
    for (cx, cy) in ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        I[cy, cx] = VOID
    I[y0, x0 + 1:x1 - 1] = TILE                     # top edge catches the lamp
    gx0, gy0, gx1, gy1 = x0 + 3, y0 + 3, x1 - 3, y1 - 4
    _rect(I, gx0, gy0, gx1, gy1, G1)
    I[gy0:gy1:2, gx0:gx1] = VOID2                   # scanlines
    I[y1 - 3, x1 - 6] = RED                         # power led
    return gx0, gy0, gx1, gy1


def _static():
    I = np.full((GH, GW), VOID, np.int16)
    # back wall: dark panels with seams
    _rect(I, 0, 0, GW, FLOOR_Y, VOID2)
    for x in range(0, GW, 30):
        I[0:FLOOR_Y, x] = VOID
    for y in (60, 130):
        I[y, :] = VOID
    # floor
    _rect(I, 0, FLOOR_Y, GW, GH, VOID)
    for y in range(FLOOR_Y + 4, GH, 9):
        I[y, :] = VOID2
    # mainframe cabinets left & right
    for (x0, x1) in CAB:
        _rect(I, x0, 20, x1, FLOOR_Y, DARK)
        _rect(I, x0 + 1, 21, x1 - 1, FLOOR_Y - 1, DARK)
        _rect(I, x0 + 3, LED0 - 4, x1 - 3, LED1, VOID)       # led panel
        I[LED0 - 4, x0 + 3:x1 - 3] = D3
        I[20, x0:x1] = D3
        I[20:FLOOR_Y, x0] = D3
        for y in range(LED0, LED1, 6):                # rows of indicator holes (the leds are drawn per frame)
            I[y, x0 + 4:x1 - 4:3] = VOID
        for cy in REELS:                              # tape reel windows
            _rect(I, x0 + 2, cy - 12, x1 - 2, cy + 12, VOID)
            I[cy - 12, x0 + 2:x1 - 2] = D3
        _rect(I, x0 + 3, 214, x1 - 3, 226, DARK)      # vents
        I[215:226:2, x0 + 3:x1 - 3] = VOID
    # monitors
    glass = {}
    glass['big'] = _crt(I, *MON_BIG)
    for (x0, y0, x1, y1, kind) in MONS:
        glass[kind] = _crt(I, x0, y0, x1, y1)
    # cables hanging from the ceiling and lying on the floor
    for k, (xa, xb, sag) in enumerate(((0, 60, 18), (40, 140, 12), (120, 180, 20), (10, 90, 8))):
        for x in range(xa, xb):
            u = (x - xa) / (xb - xa)
            y = int(2 + sag * 4 * u * (1 - u)) + k
            I[y, x] = VOID if k % 2 else DARK
    for k, (ya, xa, xb) in enumerate(((282, 0, 70), (300, 110, 180), (262, 140, 180))):
        for x in range(xa, xb):
            I[ya + int(2 * math.sin(x / 9 + k)), x] = DARK
    # the desk
    _rect(I, 26, DESK_Y, 154, DESK_Y + 5, D3)
    I[DESK_Y, 26:154] = TILE2
    _rect(I, 28, DESK_Y + 5, 152, FLOOR_Y, VOID2)
    _rect(I, 30, DESK_Y + 5, 36, FLOOR_Y, DARK); _rect(I, 144, DESK_Y + 5, 150, FLOOR_Y, DARK)
    # keyboard (keys light up per frame), stack of case files, the little manuscript, a mug
    _rect(I, 34, DESK_Y - 3, 56, DESK_Y, TILE)
    for (x0, h, c) in ((124, 3, SEPIA), (125, 2, PARCH), (123, 2, SEPIA), (126, 2, PARCH2)):
        pass
    _rect(I, 124, DESK_Y - 9, 144, DESK_Y, SEPIA)          # folders
    for y in range(DESK_Y - 9, DESK_Y, 3):
        I[y, 124:144] = PARCH
    I[DESK_Y - 9, 126:134] = RED2                           # a red «ДЕЛО» tab
    _rect(I, 140, DESK_Y - 13, 150, DESK_Y - 9, PARCH)     # the manuscript on top
    I[DESK_Y - 13, 140:150] = PARCH2; I[DESK_Y - 10, 140:150] = SEPIA
    _rect(I, 60, DESK_Y - 6, 65, DESK_Y, LINE)             # mug
    I[DESK_Y - 4:DESK_Y - 2, 65] = LINE
    I[DESK_Y - 5, 61:64] = VOID
    # the chair: a tall black-leather back that frames him, a little lighter than his cloak so he reads against it
    x0, y0, x1, y1 = CHAIR
    back = (((xx - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2 + ((yy - (y0 + 30)) / 34.0) ** 2 <= 1) | \
           ((xx >= x0) & (xx < x1) & (yy >= y0 + 30) & (yy < y1))
    I[back] = D3
    edge = back & ~(np.roll(back, 1, 0) & np.roll(back, -1, 0) & np.roll(back, 1, 1) & np.roll(back, -1, 1))
    I[edge] = TILE
    I[edge & (xx < (x0 + x1) / 2) & (yy < y0 + 30)] = TILE2
    for yb in range(y0 + 14, y1, 12):                        # tufted leather: rows of buttons
        for xb in range(x0 + 8, x1 - 6, 10):
            if back[yb, xb]:
                I[yb, xb] = DARK; I[yb + 1, xb] = VOID
    _rect(I, x0 - 4, SEAT_Y - 2, x1 + 4, SEAT_Y + 6, DARK)   # seat front
    I[SEAT_Y - 2, x0 - 4:x1 + 4] = TILE
    _rect(I, x0 - 6, SEAT_Y - 26, x0 + 2, SEAT_Y - 2, DARK)   # armrests
    _rect(I, x1 - 2, SEAT_Y - 26, x1 + 6, SEAT_Y - 2, DARK)
    I[SEAT_Y - 26, x0 - 6:x0 + 2] = TILE; I[SEAT_Y - 26, x1 - 2:x1 + 6] = TILE
    _rect(I, 87, SEAT_Y + 6, 93, 266, DARK)                  # the stem
    for (xa, xb) in ((58, 87), (93, 122)):                   # star base + wheels
        _rect(I, xa, 266, xb, 269, DARK)
        I[266, xa:xb] = D3
    for xw in (57, 88, 119):
        _rect(I, xw, 269, xw + 5, 274, D3)
    return I, glass


_BG, GLASS = _static()


def _glyph_text(I, box, t, rows, col, seed):
    gx0, gy0, gx1, gy1 = box
    r = np.random.default_rng(seed)
    line = int(t * 4)
    for k, y in enumerate(range(gy0 + 2, gy1 - 1, 4)):
        rr = np.random.default_rng(seed * 100 + line + k)
        x = gx0 + 2
        while x < gx1 - 3:
            w = int(rr.integers(1, 5))
            if rr.random() < 0.8:
                I[y, x:min(gx1 - 2, x + w)] = col
                if rr.random() < 0.3:
                    I[y - 1, x] = col
            x += w + 1


def _crt_glow(g, box, colour, k=0.10):
    """light from a screen spills onto the room below and around it (simple additive tint, re-quantised later)"""
    gx0, gy0, gx1, gy1 = box


def frame(t, amp=0.0, mood='calm', blink=False, look=0, screen=None, case='001', host_cx=None, host_pose='front', host_flip=False,
          host_on=1.0, bob=0):
    HCX = HOST_CX if host_cx is None else host_cx
    tq = math.floor(t * 15) / 15
    I = _BG.copy()
    r = np.random.default_rng(int(tq * 15))
    # cabinet leds
    for (x0, x1) in CAB:
        for j, y in enumerate(range(LED0, LED1, 6)):
            for i, x in enumerate(range(x0 + 4, x1 - 4, 3)):
                h = (i * 7 + j * 13 + x0) % 11
                on = (int(tq * (2 + h % 4)) + h) % 3 == 0
                I[y, x] = (RED if h % 5 == 0 else G3 if h % 3 == 0 else BONE) if on else VOID
        # tape reels turning
        for cy, sp in zip(REELS, (1.0, -0.7)):
            for cxr in ((x0 + x1) // 2,):
                for rr in (9, 8, 3):
                    for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                        I[int(round(cy + rr * math.sin(a))), int(round(cxr + rr * math.cos(a)))] = LINE if rr == 9 else (MID if rr == 3 else TILE)
                a0 = tq * 4 * sp
                for k in range(3):
                    a = a0 + k * 2 * math.pi / 3
                    for q in range(1, 8):
                        I[int(round(cy + q * math.sin(a))), int(round(cxr + q * math.cos(a)))] = LINE
                I[cy, cxr] = BONE
    # monitors
    big = GLASS['big']
    gx0, gy0, gx1, gy1 = big
    if screen is not None:
        h, w = gy1 - gy0, gx1 - gx0
        sc = screen[:h, :w] if screen.ndim == 2 else None
    _glyph_text(I, big, tq, 0, G3, 7)
    # the case number on the big screen
    cx = (gx0 + gx1) // 2
    _rect(I, gx0 + 2, gy0 + 2, gx1 - 2, gy0 + 9, VOID2)
    # a red eye that opens on the big screen every few seconds
    ph = (t % 6.0)
    if ph > 4.6:
        op = min(1.0, (ph - 4.6) / 0.3) * (1 if ph < 5.7 else max(0, (6.0 - ph) / 0.3))
        ey, ex = (gy0 + gy1) // 2 + 3, cx
        hh = int(round(5 * op))
        _rect(I, gx0 + 4, ey - 7, gx1 - 4, ey + 8, VOID)
        for x in range(ex - 14, ex + 15):
            d = int(round(hh * math.sqrt(max(0, 1 - ((x - ex) / 14) ** 2))))
            if d:
                I[ey - d:ey + d + 1, x] = BONE
                I[ey - d, x] = RED2; I[ey + d, x] = RED2
        if hh >= 2:
            I[ey - 2:ey + 3, ex - 2:ex + 3] = RED
            I[ey - 1:ey + 2, ex - 1:ex + 2] = VOID
    _glyph_text(I, GLASS['text'], tq, 0, G3, 11)
    _glyph_text(I, GLASS['glyph'], tq * 0.5, 0, SEPIA, 23)
    # equaliser bars
    bx0, by0, bx1, by1 = GLASS['bars']
    for i, x in enumerate(range(bx0 + 1, bx1 - 1, 2)):
        hgt = int((by1 - by0 - 4) * (0.2 + 0.8 * abs(math.sin(tq * 3.1 + i * 1.7))))
        I[by1 - 2 - hgt:by1 - 2, x] = G3
    # radar sweep
    rx0, ry0, rx1, ry1 = GLASS['radar']
    rcx, rcy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    a = tq * 2.2
    for q in range(1, 7):
        I[int(round(rcy + q * math.sin(a))), int(round(rcx + q * math.cos(a)))] = G3
    if int(tq * 3) % 4 == 0:
        I[rcy - 4, rcx + 3] = RED
    # his voice on the waveform monitor
    wx0, wy0, wx1, wy1 = GLASS['wave']
    wc = (wy0 + wy1) // 2
    for x in range(wx0 + 1, wx1 - 1):
        v = amp * 5 * math.sin(x * 0.9 + tq * 20) * math.sin(x * 0.23 + tq * 7)
        I[wc + int(round(v)), x] = RED if amp > 0.05 else G2
    # REC
    qx0, qy0, qx1, qy1 = GLASS['rec']
    _rect(I, qx0, qy0, qx1, qy1, VOID)
    if int(t * 2) % 2 == 0:
        _rect(I, qx0 + 3, qy0 + 4, qx0 + 7, qy0 + 8, RED)
    for x in range(qx0 + 9, qx1 - 3, 4):
        I[qy0 + 5:qy0 + 8, x:x + 3] = BONE
    # keyboard keys flicker, the mug steams
    for i, x in enumerate(range(37, 57, 2)):
        I[DESK_Y - 2, x] = G3 if (int(tq * 6) + i * 3) % 7 == 0 else G1
    for k in range(3):
        sy = DESK_Y - 8 - int((tq * 6 + k * 4) % 12)
        I[sy, 28 + int(math.sin(tq * 3 + k) * 1.5)] = LINE
    # dust in the screens' light
    for k in range(18):
        dx = int((k * 37 + tq * (3 + k % 4)) % GW)
        dy = int((k * 53 + tq * (5 + k % 3)) % FLOOR_Y)
        if I[dy, dx] in (VOID, VOID2):
            I[dy, dx] = MID if k % 3 else LINE
    g = BGR[np.clip(I, 0, 17)]
    # the screens light the room: a green haze in front of the monitor wall, stronger close to the glass
    haze = np.clip(1 - np.hypot((xx - 90) / 80, (yy - 100) / 90), 0, 1) * 0.30
    flick = 1 + 0.15 * math.sin(tq * 9) * (r.random() < 0.3)
    gf = g.astype(np.float32)
    gf += (haze * flick)[..., None] * np.array(BGR[G3], np.float32)[None, None] * 0.6
    # vignette
    v = np.clip(1.3 - np.hypot((xx - 90) / 130, (yy - 150) / 210), 0.45, 1)
    gf *= v[..., None]
    import pixelize as P
    g = P.quantize(np.clip(gf, 0, 255).astype(np.uint8), dither=.10, contrast=1.0)
    # an image on the big monitor (already a grid image, BGR): fit, quantise, scanlines
    if screen is not None:
        import cv2
        bx0, by0, bx1, by1 = GLASS['big']
        bw, bh = bx1 - bx0, by1 - by0
        sc = screen
        if sc.shape[0] != bh or sc.shape[1] != bw:
            sh_, sw_ = sc.shape[:2]
            k = max(bw / sw_, bh / sh_)
            sc = cv2.resize(sc, (max(bw, int(sw_ * k + .5)), max(bh, int(sh_ * k + .5))), interpolation=cv2.INTER_AREA)
            oy, ox = (sc.shape[0] - bh) // 2, (sc.shape[1] - bw) // 2
            sc = P.quantize(sc[oy:oy + bh, ox:ox + bw], dither=.08, contrast=1.05)
        sc = sc.copy()
        sc[::2] = (sc[::2].astype(np.int16) * 3 // 5).astype(np.uint8)
        g[by0:by1, bx0:bx1] = sc
    # a pool of cold monitor light on the floor where he stands, so his boots read against it
    pool = (((xx - HCX) / 70.0) ** 2 + ((yy - (HOST_BY - 4)) / 16.0) ** 2) <= 1
    bay = ((xx * 7 + yy * 3) % 4) / 4.0
    fall = 1 - np.sqrt(((xx - HCX) / 70.0) ** 2 + ((yy - (HOST_BY - 4)) / 16.0) ** 2)
    g[pool & (fall + bay * 0.5 > 0.55)] = BGR[D3]
    g[pool & (fall + bay * 0.5 > 0.9)] = BGR[TILE]
    g[pool & (fall + bay * 0.5 <= 0.55) & (fall > 0.05)] = BGR[DARK]
    if host_on <= 0:
        return g
    # him: darkness bleeds out of him onto the room, smoke rises off the hood
    layer = np.zeros_like(g)
    HO.draw(layer, HCX, HOST_BY, tq, amp, HOST_S, blink, look, bob, int(tq * 15), mood, host_pose, aura=False, flip=host_flip)
    m = layer.any(2)
    if host_on >= 1:
        HO.draw(g, HCX, HOST_BY, tq, amp, HOST_S, blink, look, bob, int(tq * 15), mood, host_pose, aura=True, flip=host_flip)
        up = m & ~np.roll(m, 1, 0) & (yy < HOST_BY - 110) & ((xx + yy) % 2 == 0)
        g[up] = BGR[G1]
    else:
        # glitch-teleport: only some of his pixels are here yet, the rest fly in from the sides
        rr = np.random.default_rng(int(tq * 15) + 7)
        keep = m & (rr.random(m.shape) < host_on ** 1.5)
        g[keep] = layer[keep]
        ys, xs = np.nonzero(m & ~keep)
        if len(ys):
            sel = rr.random(len(ys)) < 0.12
            dy = (rr.normal(0, 1, sel.sum()) * 30 * (1 - host_on)).astype(int)
            dx = (rr.normal(0, 1, sel.sum()) * 60 * (1 - host_on)).astype(int)
            ty, tx = np.clip(ys[sel] + dy, 0, GH - 1), np.clip(xs[sel] + dx, 0, GW - 1)
            g[ty, tx] = layer[ys[sel], xs[sel]]
        # torn scanlines through his silhouette
        for _ in range(int(6 * (1 - host_on)) + 1):
            y = int(rr.integers(HOST_BY - 180, HOST_BY))
            if 0 <= y < GH:
                g[y] = np.roll(g[y], int(rr.integers(-20, 20)), 0)
    return g


if __name__ == '__main__':
    import cv2, sys
    sys.path.insert(0, '/home/claude/voynich/look')
    tiles = []
    for t, amp, md in ((0.2, 0.0, 'calm'), (5.0, 0.7, 'calm'), (5.3, 0.9, 'scary')):
        tiles.append(cv2.resize(frame(t, amp, md), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite('/home/claude/voynich/team/room.png', np.hstack(tiles))
