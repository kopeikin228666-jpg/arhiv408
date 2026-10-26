#!/usr/bin/env python3
"""Generate test footage for voyfx (no source footage exists yet).

  synth.mp4      ffmpeg testsrc2 + moving gradient + drawn shapes + text, 1080x1920, 5 s
  parchment.mp4  dark procedural parchment with Voynich-like script and a botanical drawing,
                 slow Ken Burns zoom via ffmpeg zoompan, 5 s
  figure.mp4     candle-lit "photographic" scene: dark room, head-and-shoulders silhouette
                 rim-lit, moving candle glow, handheld drift — used for masks / datamosh B, 5 s
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from voyfx.io import FrameWriter  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "test_src")
os.makedirs(OUT, exist_ok=True)
W, H, FPS, DUR = 1080, 1920, 30, 5
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def fbm(h, w, octaves=6, base=4, seed=0, persistence=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.standard_normal((max(2, h * n // w), n)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-6)


def synth():
    """testsrc2 + moving gradient + moving shapes (overlay with time expressions) + text."""
    p = os.path.join(OUT, "synth.mp4")
    fc = (
        f"[0:v]scale=1080:1920:flags=neighbor,format=rgba,colorchannelmixer=aa=0.35[ts];"
        f"[1:v]format=rgba[gr];"
        f"[gr][ts]overlay=0:0[b0];"
        f"[2:v]format=rgba,rotate=a='t*1.3':c=none:ow=420:oh=420[sq];"
        f"[b0][sq]overlay=x='330+330*sin(t*1.7)':y='1100+260*cos(t*1.3)'[b1];"
        f"[3:v]format=rgba[bar];"
        f"[b1][bar]overlay=x='-200+t*300':y='420+160*sin(t*2.3)'[b2];"
        f"[4:v]format=rgba,geq=r='230':g='200':b='120':a='255*lte(hypot(X-160,Y-160),150)'[disc];"
        f"[b2][disc]overlay=x='700-500*t/5':y='300+t*260'[b3];"
        f"[b3]drawtext=fontfile={FONT}:text='VOYNICH MS 408':fontsize=104:fontcolor=white:"
        f"x='(w-text_w)/2+120*sin(t*1.6)':y='160+40*cos(t*2)',"
        f"drawtext=fontfile={FONT}:text='%{{n}}':fontsize=150:fontcolor=0xFFE08A:x=60:y=h-260,"
        f"format=yuv420p[v]"
    )
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=270x480:rate={FPS}:duration={DUR}",
           "-f", "lavfi", "-i", f"gradients=size=1080x1920:rate={FPS}:duration={DUR}:c0=0x0f3a43:c1=0xd0a050:c2=0x2a0f37:c3=0x113322:nb_colors=4:speed=0.03:type=radial",
           "-f", "lavfi", "-i", f"color=c=0x1B8B8B:size=300x300:rate={FPS}:duration={DUR}",
           "-f", "lavfi", "-i", f"color=c=0xC03030:size=200x520:rate={FPS}:duration={DUR}",
           "-f", "lavfi", "-i", f"color=c=black:size=320x320:rate={FPS}:duration={DUR}",
           "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", p]
    subprocess.run(cmd, check=True)
    print("wrote", p)


def voynich_glyph_line(img, x0, y0, length, rng, scale=1.0, color=(38, 46, 52), thick=3):
    """Draw a line of pseudo-Voynichese: loops, gallows, 'o', '9', 'c' shapes."""
    x = x0
    while x < x0 + length:
        g = rng.integers(0, 7)
        s = scale * rng.uniform(0.85, 1.15)
        if g == 0:   # o
            cv2.ellipse(img, (int(x + 9 * s), int(y0)), (int(9 * s), int(10 * s)), 0, 0, 360, color, thick, cv2.LINE_AA)
            x += 22 * s
        elif g == 1:  # 9
            cv2.ellipse(img, (int(x + 9 * s), int(y0 - 3 * s)), (int(8 * s), int(8 * s)), 0, 0, 360, color, thick, cv2.LINE_AA)
            cv2.line(img, (int(x + 17 * s), int(y0)), (int(x + 12 * s), int(y0 + 22 * s)), color, thick, cv2.LINE_AA)
            x += 24 * s
        elif g == 2:  # gallows (tall loop letter)
            cv2.line(img, (int(x + 4 * s), int(y0 + 10 * s)), (int(x + 4 * s), int(y0 - 34 * s)), color, thick, cv2.LINE_AA)
            cv2.line(img, (int(x + 16 * s), int(y0 + 10 * s)), (int(x + 16 * s), int(y0 - 30 * s)), color, thick, cv2.LINE_AA)
            cv2.ellipse(img, (int(x + 20 * s), int(y0 - 28 * s)), (int(12 * s), int(7 * s)), -10, 180, 400, color, thick, cv2.LINE_AA)
            x += 36 * s
        elif g == 3:  # c / e
            cv2.ellipse(img, (int(x + 8 * s), int(y0)), (int(8 * s), int(9 * s)), 0, 50, 310, color, thick, cv2.LINE_AA)
            x += 18 * s
        elif g == 4:  # i strokes
            for k in range(rng.integers(1, 3)):
                cv2.line(img, (int(x + 3 * s), int(y0 - 9 * s)), (int(x + 5 * s), int(y0 + 9 * s)), color, thick, cv2.LINE_AA)
                x += 9 * s
            x += 6 * s
        elif g == 5:  # 8 / daiin tail
            cv2.ellipse(img, (int(x + 8 * s), int(y0 - 8 * s)), (int(6 * s), int(7 * s)), 0, 0, 360, color, thick, cv2.LINE_AA)
            cv2.ellipse(img, (int(x + 8 * s), int(y0 + 6 * s)), (int(8 * s), int(8 * s)), 0, 0, 360, color, thick, cv2.LINE_AA)
            x += 22 * s
        else:  # word space
            x += 26 * s


def parchment_still(w=2160, h=3840, seed=3):
    rng = np.random.default_rng(seed)
    base = fbm(h // 4, w // 4, 7, 3, seed)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = fbm(h // 2, w // 2, 5, 40, seed + 1)
    fine = cv2.resize(fine, (w, h), interpolation=cv2.INTER_LINEAR)
    stains = fbm(h // 8, w // 8, 4, 2, seed + 2)
    stains = cv2.resize(stains, (w, h), interpolation=cv2.INTER_CUBIC)
    # warm vellum colour (BGR)
    vellum = np.array([120, 168, 196], np.float32) / 255.0
    dark = np.array([28, 44, 60], np.float32) / 255.0
    t = 0.55 + 0.35 * base[..., None] + 0.1 * fine[..., None]
    img = dark + (vellum - dark) * t
    st = np.clip((stains - 0.55) * 3.5, 0, 1)[..., None]
    img = img * (1 - 0.45 * st) + np.array([30, 60, 90], np.float32) / 255 * 0.45 * st
    # fibres
    fib = np.zeros((h, w), np.float32)
    for _ in range(1400):
        x, y = rng.integers(0, w), rng.integers(0, h)
        a = rng.uniform(0, math.pi)
        L = rng.uniform(10, 60)
        cv2.line(fib, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), float(rng.uniform(0.2, 1)), 1, cv2.LINE_AA)
    img -= 0.05 * cv2.GaussianBlur(fib, (0, 0), 0.8)[..., None]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    ink = np.zeros_like(img)
    ink_layer = np.full((h, w, 3), 255, np.uint8)
    # botanical drawing: stem, leaves, roots, bloom
    cx = int(w * 0.55)
    col = (30, 40, 48)
    pts = np.array([[cx + int(40 * math.sin(y / 180.0)), y] for y in range(int(h * 0.30), int(h * 0.78), 8)], np.int32)
    cv2.polylines(ink_layer, [pts], False, col, 7, cv2.LINE_AA)
    for k in range(9):
        y = int(h * 0.34 + k * h * 0.045)
        side = 1 if k % 2 else -1
        x = cx + int(40 * math.sin(y / 180.0))
        axes = (int(rng.uniform(120, 220)), int(rng.uniform(40, 70)))
        ang = side * rng.uniform(15, 35)
        center = (x + side * axes[0], y - 30)
        cv2.ellipse(ink_layer, center, axes, ang, 0, 360, col, 5, cv2.LINE_AA)
        cv2.ellipse(img, center, axes, ang, 0, 360, (70, 120, 95), -1, cv2.LINE_AA)  # faded green wash
        cv2.line(ink_layer, (x, y), center, col, 3, cv2.LINE_AA)
    bloom_c = (cx + int(40 * math.sin(h * 0.3 / 180.0)), int(h * 0.28))
    for k in range(7):
        a = k * 2 * math.pi / 7
        c = (int(bloom_c[0] + 110 * math.cos(a)), int(bloom_c[1] + 110 * math.sin(a)))
        cv2.ellipse(img, c, (95, 45), math.degrees(a), 0, 360, (60, 70, 150), -1, cv2.LINE_AA)  # faded red/blue
        cv2.ellipse(ink_layer, c, (95, 45), math.degrees(a), 0, 360, col, 4, cv2.LINE_AA)
    cv2.circle(img, bloom_c, 55, (120, 90, 50), -1, cv2.LINE_AA)
    cv2.circle(ink_layer, bloom_c, 55, col, 5, cv2.LINE_AA)
    for k in range(14):  # roots
        a = rng.uniform(0.2, math.pi - 0.2)
        p0 = (pts[-1][0], pts[-1][1])
        L = rng.uniform(150, 420)
        p1 = (int(p0[0] + L * math.cos(a)), int(p0[1] + L * math.sin(a) * 0.6))
        cv2.line(ink_layer, p0, p1, col, 3, cv2.LINE_AA)
    # script block
    for r in range(12):
        voynich_glyph_line(ink_layer, int(w * 0.10), int(h * 0.06 + r * 62), int(w * 0.78), rng, 1.6, col, 4)
    for r in range(9):
        voynich_glyph_line(ink_layer, int(w * 0.08), int(h * 0.83 + r * 62), int(w * 0.80), rng, 1.6, col, 4)
    inkf = 1 - ink_layer.astype(np.float32) / 255.0
    inkf = cv2.GaussianBlur(inkf, (0, 0), 1.1)
    bleed = cv2.GaussianBlur(inkf, (0, 0), 5) * 0.25
    img = img.astype(np.float32) * (1 - 0.85 * np.clip(inkf + bleed, 0, 1))
    # candle light falloff: bright upper-left, dark edges
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.42) / w) ** 2 + ((yy - h * 0.40) / h) ** 2 * 0.8)
    light = np.clip(1.15 - d * 1.9, 0.08, 1.0) ** 1.6
    img = img * light[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def parchment():
    still = os.path.join(OUT, "parchment_4k.png")
    cv2.imwrite(still, parchment_still())
    p = os.path.join(OUT, "parchment.mp4")
    n = FPS * DUR
    # Ken Burns: slow push-in with slight drift toward the bloom
    vf = (f"zoompan=z='1.0+0.18*on/{n}':x='iw/2-(iw/zoom/2)+40*on/{n}':y='ih*0.42-(ih/zoom/2)':"
          f"d={n}:s=1080x1920:fps={FPS},format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", still, "-vf", vf, "-frames:v", str(n),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", p], check=True)
    print("wrote", p)


def figure():
    """A rim-lit head/shoulders silhouette in a dark room with a moving candle glow."""
    p = os.path.join(OUT, "figure.mp4")
    rng = np.random.default_rng(11)
    wall = fbm(H // 4, W // 4, 6, 3, 21)
    wall = cv2.resize(wall, (W, H), interpolation=cv2.INTER_CUBIC)
    wr = FrameWriter(p, W, H, FPS, crf=12)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(FPS * DUR):
        t = i / FPS
        ox = 30 * math.sin(t * 0.9) + 6 * math.sin(t * 5.1)   # handheld drift
        oy = 18 * math.sin(t * 0.7 + 1) + 4 * math.sin(t * 6.3)
        hx = W * 0.5 + 140 * math.sin(t * 0.8) + ox             # head sways/turns
        hy = H * 0.42 + oy
        lx = W * 0.5 + 330 * math.cos(t * 1.1)                  # candle moves
        ly = H * 0.62 + 90 * math.sin(t * 1.6)
        flick = 0.85 + 0.15 * math.sin(t * 23) * math.sin(t * 7.3)
        d = np.sqrt((xx - lx) ** 2 + ((yy - ly) * 0.8) ** 2) / W
        glow = np.exp(-d * 3.2) * flick
        bg = (0.10 + 0.25 * wall)[..., None] * glow[..., None] * np.array([0.55, 0.75, 1.0])
        # silhouette: head ellipse + neck + shoulders
        sil = np.zeros((H, W), np.uint8)
        cv2.ellipse(sil, (int(hx), int(hy)), (175, 235), 4 * math.sin(t), 0, 360, 255, -1, cv2.LINE_AA)
        cv2.rectangle(sil, (int(hx - 85), int(hy + 150)), (int(hx + 85), int(hy + 380)), 255, -1)
        cv2.ellipse(sil, (int(W * 0.5 + ox * 0.6), int(H * 0.86 + oy)), (560, 330), 0, 180, 360, 255, -1, cv2.LINE_AA)
        cv2.rectangle(sil, (int(W * 0.5 + ox * 0.6 - 560), int(H * 0.86 + oy)), (int(W * 0.5 + ox * 0.6 + 560), H), 255, -1)
        silf = cv2.GaussianBlur(sil.astype(np.float32) / 255, (0, 0), 2.0)
        # rim light from candle side + face fill
        er = cv2.GaussianBlur(sil, (0, 0), 14).astype(np.float32) / 255
        edge = np.clip(silf - er, 0, 1) * 2.5
        side = np.clip((lx - xx) / -W * 0 + 1, 0, 1)
        skin = np.array([0.42, 0.55, 0.72])
        fill = np.exp(-np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / W * 4.0) * flick
        face = silf[..., None] * (0.05 + 0.75 * fill[..., None]) * skin
        rim = edge[..., None] * glow[..., None] * np.array([0.7, 0.85, 1.0]) * 1.4
        # eyes/brows hints for structure
        img = bg * (1 - silf[..., None]) + face + rim
        e = np.zeros((H, W), np.float32)
        for sx in (-62, 62):
            cv2.ellipse(e, (int(hx + sx), int(hy - 20)), (34, 14), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.line(e, (int(hx), int(hy + 10)), (int(hx + 6), int(hy + 70)), 0.6, 6, cv2.LINE_AA)
        e = cv2.GaussianBlur(e, (0, 0), 4)
        img = img * (1 - 0.6 * e[..., None])
        # candle flame
        cf = np.exp(-(((xx - lx) / 18) ** 2 + ((yy - ly + 40) / 46) ** 2)) * 1.6
        img += cf[..., None] * np.array([0.6, 0.9, 1.0])
        img = img ** (1 / 1.15)
        img += rng.normal(0, 0.012, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
        wr.write(np.clip(img * 255, 0, 255).astype(np.uint8))
    wr.close()
    print("wrote", p)


if __name__ == "__main__":
    which = sys.argv[1:] or ["synth", "parchment", "figure"]
    for w in which:
        globals()[w]()
