"""pixel scenes for ДЕЛО 002 (float RGB canvases on the 180x225 art grid)"""
import math, numpy as np
from lib408 import *

GLOW_R = (190, 45, 35)
GLOW_G = (60, 120, 90)
MOON = (95, 130, 120)


def aisle(a, cam=None, gc=GLOW_R, seed=1, zmax=16, gk=1.0, led_red=0.10):
    """server aisle receding to a glowing doorway"""
    cam = cam or Cam(90, 92, 105)
    r = np.random.default_rng(seed)
    FY, CY, W = 1.0, -1.35, 0.95
    # end wall + lit doorway
    x0, y0 = cam(-W, CY, zmax); x1, y1 = cam(W, FY, zmax)
    rect(a, x0, y0, x1, y1, (14, 17, 16))
    d0 = cam(-0.35, -0.6, zmax); d1 = cam(0.35, FY, zmax)
    rect(a, d0[0], d0[1], d1[0], d1[1], tuple(min(255, c * 1.25) for c in gc))
    glow(a, cam.cx, (d0[1] + d1[1]) / 2, 14, gc, 0.55 * gk, ry=22, mode='lerp')
    zs = np.arange(zmax, 0.5, -0.5)
    for z1 in zs:
        z0 = max(0.5, z1 - 0.5)
        cam.quad(a, [(-W, FY, z0), (W, FY, z0), (W, FY, z1), (-W, FY, z1)], fogc((26, 32, 30), z0, VOID, 0.05))
        cam.quad(a, [(-W, CY, z0), (W, CY, z0), (W, CY, z1), (-W, CY, z1)], (10, 12, 11))
        if int(z1 * 2) % 2 == 0:
            line(a, cam(-W, FY, z1), cam(W, FY, z1), (16, 20, 19))
    # light spill from the doorway on the floor
    pts = [cam(-0.35, FY, zmax - 0.05), cam(0.35, FY, zmax - 0.05), cam(0.8, FY, zmax * 0.35), cam(-0.8, FY, zmax * 0.35)]
    poly(a, pts, tuple(c * 0.8 for c in gc), 0.22 * gk)
    # ceiling cable trays + lamps
    for X in (-0.5, 0.5):
        line(a, cam(X, CY + 0.05, 0.6), cam(X, CY + 0.05, zmax), (34, 40, 38))
    # racks: cabinets 0.6 deep, far -> near
    for s in (-1, 1):
        cz = zmax
        while cz > 0.55:
            z1, z0 = cz, max(0.5, cz - 0.6)
            k = math.exp(-0.06 * z0)
            base = tuple(c * k for c in (44, 54, 50))
            cam.quad(a, [(s * W, CY + 0.12, z0), (s * W, CY + 0.12, z1), (s * W, FY, z1), (s * W, FY, z0)], base)
            # cabinet seams: a dark gap + a lit edge
            line(a, cam(s * W, CY + 0.12, z0), cam(s * W, FY, z0), VOID)
            e = cam(s * W, CY + 0.12, z0 + 0.03); e2 = cam(s * W, FY, z0 + 0.03)
            line(a, e, e2, tuple(c * k for c in (82, 94, 88)))
            # server units
            for v in np.arange(CY + 0.2, FY - 0.05, 0.13):
                line(a, cam(s * W, v, z0 + 0.06), cam(s * W, v, z1 - 0.02), tuple(c * k for c in (30, 37, 35)))
            if z0 < 10:
                for v in np.arange(CY + 0.26, FY - 0.1, 0.13):
                    for u in (0.15, 0.22, 0.29):
                        if r.random() < 0.55:
                            p = cam(s * W, v, z0 + u * 0.6 / 0.6 * (z1 - z0) / 0.6 * 0.6)
                            cc = RED if r.random() < led_red else (G3 if r.random() < 0.7 else BONE)
                            px(a, p[0], p[1], tuple(c * (0.5 + 0.5 * k) for c in cc))
            # toe kick
            cam.quad(a, [(s * W, FY - 0.06, z0), (s * W, FY - 0.06, z1), (s * W, FY, z1), (s * W, FY, z0)], VOID2)
            cz -= 0.6
    # coloured light from the end on the near cabinets' edges
    glow(a, cam.cx, cam.cy, 30, gc, 0.18 * gk, ry=40, mode='lerp')
    return cam


def crt(a, x0, y0, x1, y1, scr=(30, 60, 45), depth=5, glowk=0.6, gc=None):
    """a CRT monitor (front view); returns the glass rect"""
    gc = gc or scr
    glow(a, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.75, gc, glowk * 0.35, ry=(y1 - y0) * 0.9)
    # body depth (bottom/right shade)
    rect(a, x0 + depth, y0 + depth, x1 + depth, y1 + depth, VOID)
    rect(a, x0, y0, x1, y1, (44, 52, 49))
    rect(a, x0, y0, x1, y0 + 1, (70, 80, 74))
    rect(a, x0, y0, x0 + 1, y1, (60, 68, 64))
    rect(a, x0 + 1, y1 - 2, x1, y1, (26, 32, 30))
    gx0, gy0, gx1, gy1 = x0 + 6, y0 + 5, x1 - 6, y1 - 9
    rect(a, gx0 - 1, gy0 - 1, gx1 + 1, gy1 + 1, VOID)
    vgrad(a, gy0, gy1, tuple(c * 1.0 for c in scr), tuple(c * 0.6 for c in scr), gx0, gx1)
    glow(a, (gx0 + gx1) / 2, (gy0 + gy1) / 2, (gx1 - gx0) * 0.4, scr, 0.35, ry=(gy1 - gy0) * 0.45)
    for y in range(int(gy0), int(gy1), 2):
        a[y, int(gx0):int(gx1)] *= 0.86
    # rounded glass corners
    for (cx, cy) in ((gx0, gy0), (gx1 - 1, gy0), (gx0, gy1 - 1), (gx1 - 1, gy1 - 1)):
        px(a, cx, cy, VOID)
    # power led + knobs
    px(a, x1 - 8, y1 - 5, RED); rect(a, x1 - 16, y1 - 6, x1 - 13, y1 - 4, (30, 36, 34))
    return (gx0, gy0, gx1, gy1)


def wallet_icon(a, x, y, s=1, c=BONE, c2=MID):
    """chunky pixel wallet with a coin (no real logo)"""
    W, H = 26 * s, 17 * s
    rect(a, x, y + 3 * s, x + W, y + H, c2)
    rect(a, x + s, y + 4 * s, x + W - s, y + H - s, (40, 46, 44))
    rect(a, x, y + 3 * s, x + W, y + 4 * s, c)
    rect(a, x, y + H - s, x + W, y + H, c)
    rect(a, x, y + 3 * s, x + s, y + H, c); rect(a, x + W - s, y + 3 * s, x + W, y + H, c)
    # flap
    rect(a, x + 15 * s, y + 7 * s, x + W + s, y + 13 * s, c)
    rect(a, x + 16 * s, y + 8 * s, x + W, y + 12 * s, (40, 46, 44))
    rect(a, x + 18 * s, y + 9 * s, x + 20 * s, y + 11 * s, RED)
    # coin peeking out
    cx, cy, rr = x + 7 * s, y + 2 * s, 4 * s
    for yy in range(-rr, rr + 1):
        for xx in range(-rr, rr + 1):
            if xx * xx + yy * yy <= rr * rr and cy + yy < y + 4 * s:
                px(a, cx + xx, cy + yy, PARCH2 if xx * xx + yy * yy < (rr - s) ** 2 else PARCH)


def desk(a, y, x0=0, x1=AW, c=(34, 40, 38)):
    rect(a, x0, y, x1, AH, VOID2)
    rect(a, x0, y, x1, y + 3, c)
    rect(a, x0, y, x1, y + 1, (56, 64, 60))


def ward(a, seed=2, zmax=10):
    """dark hospital ward: two rows of empty beds, moonlit windows on the end wall"""
    cam = Cam(90, 64, 100)
    W, FY, CY, BT = 1.6, 1.7, -0.8, 0.95      # walls, floor, ceiling, bed top (camera sits above the beds)
    x0, y0 = cam(-W, CY, zmax); x1, y1 = cam(W, FY, zmax)
    rect(a, x0, y0, x1, y1, (22, 30, 28))
    wins = (-0.85, 0.0, 0.85)
    for wx in wins:
        p0 = cam(wx - 0.28, -0.5, zmax); p1 = cam(wx + 0.28, 0.75, zmax)
        rect(a, p0[0], p0[1], p1[0], p1[1], (92, 124, 112))
        line(a, ((p0[0] + p1[0]) / 2, p0[1]), ((p0[0] + p1[0]) / 2, p1[1]), (30, 40, 38))
        line(a, (p0[0], (p0[1] + p1[1]) / 2), (p1[0], (p0[1] + p1[1]) / 2), (30, 40, 38))
        glow(a, (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 8, MOON, 0.5)
    zs = np.arange(zmax, 0.5, -0.5)
    for z1 in zs:
        z0 = max(0.5, z1 - 0.5)
        cam.quad(a, [(-W, FY, z0), (W, FY, z0), (W, FY, z1), (-W, FY, z1)], fogc((30, 38, 36), z0, (16, 22, 20), 0.08))
        if int(z1 * 2) % 2 == 0:
            line(a, cam(-W, FY, z1), cam(W, FY, z1), (22, 28, 26))
        cam.quad(a, [(-W, CY, z0), (W, CY, z0), (W, CY, z1), (-W, CY, z1)], (12, 15, 14))
        for s in (-1, 1):
            cam.quad(a, [(s * W, CY, z0), (s * W, CY, z1), (s * W, FY, z1), (s * W, FY, z0)], fogc((26, 33, 31), z0, (14, 18, 17), 0.07))
    for wx in wins:
        pts = [cam(wx - 0.28, FY, zmax - 0.05), cam(wx + 0.28, FY, zmax - 0.05), cam(wx * 2.2 + 0.45, FY, 1.3), cam(wx * 2.2 - 0.45, FY, 1.3)]
        poly(a, pts, MOON, 0.13)
    for z in np.arange(1.5, zmax, 1.6):
        p = cam(0, CY + 0.02, z); rect(a, p[0] - 10 / z * 2, p[1], p[0] + 10 / z * 2, p[1] + max(1, 3 / z), (40, 48, 46))
    for zc in np.arange(zmax - 1.0, 1.0, -1.5):
        for s in (-1, 1):
            xw, xi = s * (W - 0.05), s * 0.55
            za, zb = zc - 0.42, zc + 0.42
            k = math.exp(-0.09 * zc)
            fr = tuple(c * k for c in (78, 88, 84))
            for xx in (xi, xw):
                for zz in (za, zb):
                    line(a, cam(xx, BT, zz), cam(xx, FY, zz), tuple(c * k for c in (40, 48, 45)))
            cam.quad(a, [(xi, BT, za), (xi, BT, zb), (xi, BT + 0.14, zb), (xi, BT + 0.14, za)], tuple(c * k for c in (70, 82, 78)))
            cam.quad(a, [(xw, BT, za), (xi, BT, za), (xi, BT, zb), (xw, BT, zb)], tuple(c * k for c in (108, 122, 114)))
            # sheet fold + shadow line
            line(a, cam(xi + s * 0.35, BT, za), cam(xi + s * 0.35, BT, zb), tuple(c * k for c in (80, 92, 86)))
            cam.quad(a, [(xw, BT, za), (xi, BT, za), (xi, BT + 0.14, za), (xw, BT + 0.14, za)], tuple(c * k for c in (64, 74, 70)))
            cam.quad(a, [(xw - s * 0.05, BT - 0.07, za + 0.08), (xw - s * 0.4, BT - 0.07, za + 0.08), (xw - s * 0.4, BT - 0.07, zb - 0.08),
                         (xw - s * 0.05, BT - 0.07, zb - 0.08)], tuple(c * k for c in (196, 204, 194)))
            # headboard: a dark panel with bars against the wall
            cam.quad(a, [(xw, BT - 0.55, za), (xw, BT - 0.55, zb), (xw, BT, zb), (xw, BT, za)], tuple(c * k for c in (24, 30, 28)))
            for zz in np.linspace(za, zb, 6):
                line(a, cam(xw, BT - 0.5, zz), cam(xw, BT, zz), fr)
            line(a, cam(xw, BT - 0.55, za), cam(xw, BT - 0.55, zb), fr)
            # IV stand between the beds
            if zc < zmax - 2:
                ix, iz = s * (W - 0.25), zb + 0.33
                line(a, cam(ix, FY, iz), cam(ix, BT - 1.25, iz), tuple(c * k for c in (90, 100, 96)))
                bx, by = cam(ix, BT - 1.2, iz)
                rect(a, bx - 1, by, bx + 2, by + max(2, 4 / zc), tuple(c * k for c in (120, 140, 130)))
    return cam


def cell(a, seed=3):
    """sealed room, terminal on the floor, cable runs to the light under the door"""
    cam = Cam(90, 92, 95)
    W, Z0, Z1 = 1.25, 0.8, 4.2
    # walls
    for (P4, c) in (
        ([(-W, -1.4, Z1), (W, -1.4, Z1), (W, 1, Z1), (-W, 1, Z1)], (34, 40, 38)),
        ([(-W, -1.4, Z0), (-W, -1.4, Z1), (-W, 1, Z1), (-W, 1, Z0)], (26, 31, 30)),
        ([(W, -1.4, Z0), (W, -1.4, Z1), (W, 1, Z1), (W, 1, Z0)], (24, 29, 28)),
        ([(-W, 1, Z0), (W, 1, Z0), (W, 1, Z1), (-W, 1, Z1)], (30, 36, 34)),
        ([(-W, -1.4, Z0), (W, -1.4, Z0), (W, -1.4, Z1), (-W, -1.4, Z1)], (14, 18, 17)),
    ):
        cam.quad(a, P4, c)
    # concrete blocks on the back wall
    for v in np.arange(-1.4, 1.0, 0.3):
        line(a, cam(-W, v, Z1), cam(W, v, Z1), (26, 31, 30))
    r = np.random.default_rng(seed)
    # door on the back wall (right of centre)
    d0 = cam(0.15, -0.75, Z1); d1 = cam(0.85, 1.0, Z1)
    rect(a, d0[0] - 2, d0[1] - 2, d1[0] + 2, d1[1], (18, 22, 21))
    rect(a, d0[0], d0[1], d1[0], d1[1], (52, 60, 57))
    rect(a, d0[0], d0[1], d0[0] + 1, d1[1], (70, 80, 75))
    # rivets
    for yy in np.linspace(d0[1] + 3, d1[1] - 4, 6):
        px(a, d0[0] + 3, yy, (90, 100, 95)); px(a, d1[0] - 3, yy, (90, 100, 95))
    # barred window in the door
    wx0, wy0 = d0[0] + 7, d0[1] + 6; wx1, wy1 = d1[0] - 7, d0[1] + 16
    rect(a, wx0, wy0, wx1, wy1, (150, 165, 150))
    glow(a, (wx0 + wx1) / 2, (wy0 + wy1) / 2, 10, (120, 140, 125), 0.35)
    for bx in np.arange(wx0 + 2, wx1, 3):
        rect(a, bx, wy0, bx + 1, wy1, (20, 24, 23))
    # handle / lock
    rect(a, d0[0] + 3, (d0[1] + d1[1]) / 2, d0[0] + 6, (d0[1] + d1[1]) / 2 + 2, (100, 105, 100))
    # light under the door
    rect(a, d0[0], d1[1] - 1, d1[0], d1[1] + 1, (230, 225, 205))
    glow(a, (d0[0] + d1[0]) / 2, d1[1] + 1, 18, (200, 200, 180), 0.55, ry=4)
    poly(a, [(d0[0], d1[1]), (d1[0], d1[1]), (d1[0] + 12, d1[1] + 14), (d0[0] - 14, d1[1] + 14)], (180, 180, 160), 0.18)
    # terminal on a crate, left foreground
    rect(a, 18, 160, 74, 205, (30, 26, 20)); rect(a, 18, 160, 74, 162, (60, 52, 38))
    for yy in (172, 186):
        rect(a, 18, yy, 74, yy + 1, (22, 18, 14))
    g = crt(a, 20, 118, 72, 160, scr=(40, 90, 60), depth=3, glowk=0.7)
    # the cable: from the crate, across the floor, under the door
    pts = []
    ex, ey = (d0[0] + d1[0]) / 2 + 2, d1[1] + 0.5
    for i in range(41):
        t = i / 40
        x = 74 + (ex - 74) * t + 14 * math.sin(t * math.pi * 1.6) * (1 - t)
        y = 196 + (ey - 196) * t ** 0.7 + 7 * math.sin(t * math.pi * 2.2) * (1 - t)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        line(a, pts[i], pts[i + 1], (12, 14, 13), 2 if i < 24 else 1)
        line(a, (pts[i][0], pts[i][1] - 1), (pts[i + 1][0], pts[i + 1][1] - 1), (60, 66, 60) if i < 24 else (45, 50, 46))
    glow(a, ex, ey, 5, RED2, 0.6, mode='lerp')
    px(a, ex, ey, RED); px(a, ex + 1, ey, RED)
    # dim overhead bulb
    bx, by = cam(-0.4, -1.38, 2.0)
    line(a, (bx, by - 5), (bx, by), (30, 34, 32))
    glow(a, bx, by + 2, 26, (70, 80, 70), 0.25)
    return g, (d0, d1)


def drawers(a, n=19, hot=(1, 5, 11, 16), cols=5, seed=4, x0=16, y0=62, cw=30, ch=25):
    """archive wall of drawers; returns the label rects (art coords) in reading order"""
    rect(a, 0, 0, AW, AH, (16, 20, 19))
    labels = []
    rows = (n + cols) // cols
    rect(a, x0 - 6, y0 - 6, x0 + cols * cw + 6, y0 + rows * ch + 6, (12, 14, 13))
    k = 0
    for rr in range(rows):
        for cc in range(cols):
            X, Y = x0 + cc * cw, y0 + rr * ch
            if k >= n:
                rect(a, X + 2, Y + 2, X + cw - 1, Y + ch - 1, (6, 8, 7))   # the empty slot
                k += 1; continue
            is_hot = k in hot
            rect(a, X + 1, Y + 1, X + cw - 1, Y + ch - 1, (52, 60, 56) if not is_hot else (80, 40, 34))
            rect(a, X + 1, Y + 1, X + cw - 1, Y + 2, (74, 84, 78) if not is_hot else (130, 60, 50))
            rect(a, X + 1, Y + ch - 2, X + cw - 1, Y + ch - 1, (32, 38, 36))
            # label frame + handle
            rect(a, X + 6, Y + 5, X + cw - 6, Y + 11, (20, 24, 22) if not is_hot else (30, 10, 8))
            rect(a, X + cw / 2 - 4, Y + ch - 7, X + cw / 2 + 4, Y + ch - 5, (100, 106, 100) if not is_hot else (200, 120, 100))
            if is_hot:
                glow(a, X + cw / 2, Y + ch / 2, 16, RED2, 0.45, mode='lerp')
            labels.append((X + 6, Y + 5, X + cw - 6, Y + 11, is_hot))
            k += 1
    # floor + lamp
    rect(a, 0, y0 + rows * ch + 6, AW, AH, (10, 12, 11))
    glow(a, 90, 30, 70, (60, 70, 64), 0.35, ry=50)
    return labels


def rack_open(a, seed=5):
    """one rack, the door swung open, a red unit, rows of data pouring out into a heap"""
    r = np.random.default_rng(seed)
    rect(a, 0, 0, AW, AH, (12, 15, 14))
    glow(a, 90, 70, 80, (40, 70, 55), 0.35, ry=70)
    # neighbouring racks (dark)
    for X0 in (-30, 142):
        rect(a, X0, 18, X0 + 66, 200, (22, 27, 25))
        for y in range(24, 196, 6):
            rect(a, X0 + 4, y, X0 + 62, y + 4, (28, 34, 32))
            if r.random() < 0.6:
                px(a, X0 + 8 + r.integers(0, 50), y + 2, G3 if r.random() < 0.7 else RED2)
    # the rack
    X0, X1, Y0, Y1 = 48, 132, 14, 196
    rect(a, X0 - 3, Y0 - 3, X1 + 3, Y1 + 2, (8, 10, 9))
    rect(a, X0, Y0, X1, Y1, (36, 44, 41))
    hot_y = 92
    for y in range(Y0 + 4, Y1 - 4, 7):
        rect(a, X0 + 4, y, X1 - 4, y + 5, (48, 58, 54))
        rect(a, X0 + 4, y, X1 - 4, y + 1, (66, 76, 70))
        for i in range(6):
            if r.random() < 0.55:
                px(a, X0 + 8 + i * 3, y + 3, G3 if r.random() < 0.8 else BONE)
        rect(a, X1 - 22, y + 2, X1 - 8, y + 3, (28, 34, 32))
    # the pulled-out red unit
    rect(a, X0 - 6, hot_y, X1 + 6, hot_y + 8, (60, 22, 18))
    rect(a, X0 - 6, hot_y, X1 + 6, hot_y + 1, (200, 70, 55))
    for i in range(10):
        px(a, X0 - 2 + i * 3, hot_y + 4, RED)
    glow(a, 90, hot_y + 4, 44, RED2, 0.55, ry=18, mode='lerp')
    # open glass door swung to the left (parallelogram)
    poly(a, [(X0 - 3, Y0 - 3), (X0 - 34, Y0 + 8), (X0 - 34, Y1 - 6), (X0 - 3, Y1 + 2)], (60, 80, 74), 0.35)
    line(a, (X0 - 3, Y0 - 3), (X0 - 34, Y0 + 8), (110, 125, 118)); line(a, (X0 - 34, Y0 + 8), (X0 - 34, Y1 - 6), (110, 125, 118))
    line(a, (X0 - 34, Y1 - 6), (X0 - 3, Y1 + 2), (80, 90, 86))
    # data rows pouring from the unit to the floor heap (right side)
    for i in range(170):
        t = r.random()
        x = X1 + 4 + t * 36 + r.normal(0, 2 + 6 * t)
        y = hot_y + 6 + (t ** 1.6) * 110 + r.normal(0, 2)
        ln = int(2 + r.integers(0, 4))
        c = BONE if r.random() < 0.35 else (LIGHT if r.random() < 0.5 else RED)
        rect(a, x, y, x + ln, y + 1, c)
    # heap of rows on the floor
    rect(a, 0, 200, AW, AH, (10, 12, 11))
    for i in range(260):
        x = 150 + r.normal(0, 12); y = 214 - abs(r.normal(0, 6))
        rect(a, x, y, x + r.integers(2, 5), y + 1, BONE if r.random() < 0.3 else (MID if r.random() < 0.6 else RED2))
    return (X0, Y0, X1, Y1)
