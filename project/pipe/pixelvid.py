"""Video -> approved pixel look (see look/pixel_reader.png).

usage: python3 pixelvid.py IN.mp4 OUT.mp4 [--hold 2] [--gamma 0.8] [--ui reader|none]
       [--caption "ОНА|СМОТРИТ"] [--dissolve 0.0..1.0] [--start S] [--dur D]

Everything is computed on a 180x320 grid and upscaled x6 with nearest neighbour.
Dither threshold is fixed in screen space, so static areas never shimmer.
"""
import sys, argparse, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/claude/voynich/look')
from pixelize import quantize, PALETTE

GW, GH, S = 180, 320, 6
OW, OH = GW * S, GH * S
COL = {k: tuple(int(v) for v in PALETTE[i][::-1]) for k, i in
       dict(void=0, void2=1, dark=2, tile=4, tile2=5, line=7, bone=9, red=17).items()}  # RGB
FMONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
FCOND = '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf'


def probe(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'stream=width,height,r_frame_rate', '-of', 'csv=p=0', path]).decode().strip()
    w, h, r = out.split(',')[:3]
    a, b = r.split('/')
    return int(w), int(h), float(a) / float(b)


def reader(path, start, dur):
    w, h, fps = probe(path)
    cmd = ['ffmpeg', '-v', 'error']
    if start: cmd += ['-ss', str(start)]
    cmd += ['-i', path]
    if dur: cmd += ['-t', str(dur)]
    cmd += ['-vf', 'fps=30', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def writer(path, audio_from=None, start=0, dur=None):
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', '30', '-i', '-']
    if audio_from:
        if start: cmd += ['-ss', str(start)]
        cmd += ['-i', audio_from]
        if dur: cmd += ['-t', str(dur)]
        cmd += ['-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-b:a', '192k', '-shortest']
    cmd += ['-c:v', 'libx264', '-crf', '14', '-preset', 'medium', '-pix_fmt', 'yuv420p', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def to_grid(frame, gamma):
    h, w = frame.shape[:2]
    tgt = GW / GH
    if w / h > tgt:
        nw = int(h * tgt); x0 = (w - nw) // 2; frame = frame[:, x0:x0 + nw]
    else:
        nh = int(w / tgt); y0 = (h - nh) // 2; frame = frame[y0:y0 + nh]
    small = cv2.resize(frame, (GW, GH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    small = small ** gamma
    return (small * 255).astype(np.uint8)


class UI:
    def __init__(self, caption=None, label='SUBJECT_01'):
        self.f9 = ImageFont.truetype(FMONO, 9)
        self.fbig = ImageFont.truetype(FCOND, 15)
        self.caption = caption
        self.label = label
        self.box = None

    def draw(self, grid_rgb, t, bbox=None, book=None):
        pil = Image.fromarray(grid_rgb)
        d = ImageDraw.Draw(pil); d.fontmode = '1'
        if bbox is not None and getattr(self, 'gy', None) is None:
            self.gy = int(bbox[1]) - 18
        # guides
        d.line([GW // 2, 18, GW // 2, GH - 60], fill=COL['dark'])
        gy = self.gy if getattr(self, 'gy', None) else 100
        d.line([6, gy, GW - 6, gy], fill=COL['dark'])
        for px in (6, GW // 2, GW - 7):
            d.rectangle([px - 1, gy - 1, px + 1, gy + 1], fill=COL['red'])
        d.text((8, 22), 'SURFACE_01', font=self.f9, fill=COL['line'])
        if int(t * 7) % 11:
            d.text((8, 30), 'SIGNAL --', font=self.f9, fill=COL['tile2'])
        d.text((GW - 50, 22), f'{412 + int(t * 30):04d}', font=self.f9, fill=COL['tile2'])
        # tracking box, smoothed, with a slight lag
        if bbox is not None:
            bb = np.array(bbox, np.float32)
            self.box = bb if self.box is None else self.box * 0.75 + bb * 0.25
            x0, y0, x1, y1 = [int(round(v)) for v in self.box]
            d.rectangle([x0, y0, x1, y1], outline=COL['bone'])
            for (cx, cy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                d.rectangle([cx - 1, cy - 1, cx + 1, cy + 1], fill=COL['bone'])
            lw = int(d.textlength(self.label, font=self.f9)) + 4
            d.rectangle([x0, y0 - 10, x0 + lw, y0 - 1], fill=COL['bone'])
            d.text((x0 + 2, y0 - 10), self.label, font=self.f9, fill=COL['void'])
            conf = 0.97
            d.text((x0 + lw + 3, y0 - 10), f'.{int(conf * 100):02d}', font=self.f9, fill=COL['line'])
        if book is not None:
            bb = np.array(book, np.float32)
            self.bk = bb if getattr(self, 'bk', None) is None else self.bk * 0.7 + bb * 0.3
            x0, y0, x1, y1 = [int(round(v)) for v in self.bk]
            d.rectangle([x0, y0, x1, y1], outline=COL['red'])
            d.text((x0, y1 + 2), 'MS408', font=self.f9, fill=COL['red'])
            d.text((x0, y1 + 11), f'READ {min(99, int(t * 3))}%', font=self.f9, fill=COL['red'])
        # caption: "WORD|KEYWORD" -> keyword on a red plate
        if self.caption:
            a, b = (self.caption.split('|') + [''])[:2]
            a = a + ' ' if a else ''
            w1 = d.textlength(a, font=self.fbig); w2 = d.textlength(b, font=self.fbig)
            cx = int((GW - w1 - w2 - 4) / 2); cy = GH - 44
            for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1),(1,1)):
                d.text((cx+ddx, cy+ddy), a, font=self.fbig, fill=COL['void'])
            d.text((cx, cy), a, font=self.fbig, fill=COL['bone'])
            if b:
                d.rectangle([cx + w1 - 1, cy + 1, cx + w1 + w2 + 2, cy + 17], fill=COL['red'])
                d.text((cx + w1, cy), b, font=self.fbig, fill=COL['bone'])
        return np.array(pil)


class Segmenter:
    """GrabCut at 270x480, initialised from a rect, then from the previous mask."""
    def __init__(self, rect_frac):
        self.rect_frac = rect_frac
        self.prev = None

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        tgt = GW / GH
        if w / h > tgt:
            nw = int(h * tgt); x0 = (w - nw) // 2; frame_bgr = frame_bgr[:, x0:x0 + nw]
        else:
            nh = int(w / tgt); y0 = (h - nh) // 2; frame_bgr = frame_bgr[y0:y0 + nh]
        self.full = frame_bgr
        img = cv2.resize(frame_bgr, (270, 480), interpolation=cv2.INTER_AREA)
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        if self.prev is None:
            fx0, fy0, fx1, fy1 = self.rect_frac
            rect = (int(fx0 * 270), int(fy0 * 480), int((fx1 - fx0) * 270), int((fy1 - fy0) * 480))
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.grabCut(img, m, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
        else:
            m = np.where(self.prev, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            core = cv2.erode(self.prev.astype(np.uint8), np.ones((9, 9), np.uint8))
            far = cv2.dilate(self.prev.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
            m[core > 0] = cv2.GC_FGD; m[far] = cv2.GC_BGD
            cv2.grabCut(img, m, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n > 1:
            i = 1 + int(np.argmax(st[1:, 4])); fg = (lab == i).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.prev = fg > 0
        self.m270 = fg
        small = cv2.resize(fg.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA) > 0.5
        ys, xs = np.where(small)
        if len(xs) < 20:
            return None, None
        return small, (xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 2)


class MaskSeg:
    """Uses precomputed per-source-frame masks (540x968) from shotmask.py."""
    def __init__(self, path, src_fps):
        self.M = np.load(path)
        self.src_fps = src_fps
        u = self.M.max(0)
        u270 = cv2.resize(u.astype(np.float32), (270, 480), interpolation=cv2.INTER_AREA) > 0.3
        ys, xs = np.where(u270)
        self.fitbox = (max(0, xs.min() - 6), max(0, ys.min() - 6), min(270, xs.max() + 6), min(480, ys.max() + 6))
        self.prev = None

    def __call__(self, frame_bgr, t):
        h, w = frame_bgr.shape[:2]
        tgt = GW / GH
        if w / h > tgt:
            nw = int(h * tgt); x0 = (w - nw) // 2; frame_bgr = frame_bgr[:, x0:x0 + nw]
        else:
            nh = int(w / tgt); y0 = (h - nh) // 2; frame_bgr = frame_bgr[y0:y0 + nh]
        self.full = frame_bgr
        k = min(len(self.M) - 1, int(round(t * self.src_fps)))
        self.m270 = (cv2.resize(self.M[k].astype(np.float32), (270, 480), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        small = cv2.resize(self.m270.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA) > 0.5
        ys, xs = np.where(small)
        return small, (xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 2)


def fog_canvas(seed=11, floor_y=None):
    rng = np.random.default_rng(seed)
    yy_, xx_ = np.mgrid[0:GH, 0:GW]
    fogv = 0.10 + 0.22 * np.exp(-((yy_ - 200) / 95.0) ** 2) * (1 - 0.5 * np.abs(xx_ - GW / 2) / (GW / 2))
    fogimg = np.dstack([fogv * 235, fogv * 250, fogv * 240]).clip(0, 255).astype(np.uint8)
    can = quantize(fogimg, dither=0.16, contrast=1.0)
    if floor_y is not None:
        T = 9
        tile = PALETTE[4].astype(np.uint8); tile2 = PALETTE[5].astype(np.uint8); dark = PALETTE[2].astype(np.uint8)
        for gy in range(-2, 6):
            for gx in range(0, GW // T + 1):
                x, y = gx * T, floor_y + gy * T
                dist = abs(gy - 0.5) * 1.0 + abs(x + T / 2 - GW / 2) / T * 0.45
                if rng.random() > np.clip(1.3 - dist * 0.24, 0, 1):
                    continue
                drop = int(rng.integers(4, 40)) if rng.random() < 0.2 else 0
                yy = y + drop
                if yy < 0 or yy + T - 1 > GH:
                    continue
                can[yy:yy + T - 1, x:x + T - 1] = tile if (gx + gy) % 2 else tile2
                sp = rng.random((T - 1, T - 1)) < 0.18
                can[yy:yy + T - 1, x:x + T - 1][sp] = dark
    return can


def figure_mask(grid_bgr, bg_model):
    """foreground vs. running median background (static camera)."""
    diff = np.abs(grid_bgr.astype(np.int16) - bg_model.astype(np.int16)).sum(2)
    m = (diff > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, None
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i, 4] < 30:
        return None, None
    x, y, w, h = st[i, :4]
    return (lab == i), (x - 3, y - 3, x + w + 2, y + h + 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--hold', type=int, default=2)
    ap.add_argument('--gamma', type=float, default=0.8)
    ap.add_argument('--contrast', type=float, default=1.25)
    ap.add_argument('--dither', type=float, default=0.12)
    ap.add_argument('--ui', default='reader')
    ap.add_argument('--label', default='SUBJECT_01')
    ap.add_argument('--caption', default=None)
    ap.add_argument('--dissolve', type=float, default=0.0)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--dur', type=float, default=None)
    ap.add_argument('--audio', action='store_true')
    ap.add_argument('--seg', default=None, help='x0,y0,x1,y1 fractions: GrabCut start rect -> figure is cut out')
    ap.add_argument('--bg', default='source', help='source | fog (approved look: fog + floating tiles)')
    ap.add_argument('--book', action='store_true', help='red MS408 box on the brightest parchment area')
    ap.add_argument('--fit', type=int, default=0, help='approved layout: figure scaled to this many cells wide, centred, feet at y=250')
    ap.add_argument('--bottom', type=int, default=250)
    ap.add_argument('--masks', default=None, help='npy from shotmask.py (overrides --seg)')
    a = ap.parse_args()

    rng = np.random.default_rng(3)
    ui = UI(a.caption, a.label) if a.ui != 'none' else None
    w = writer(a.out, a.inp if a.audio else None, a.start, a.dur)
    frames = list(reader(a.inp, a.start, a.dur))
    grids = [to_grid(f, a.gamma) for f in frames]
    bg = np.median(np.stack(grids[::max(1, len(grids) // 30)]), axis=0).astype(np.uint8)
    particles = []  # [y, x, vy, vx, color, life]
    seg = Segmenter([float(v) for v in a.seg.split(',')]) if a.seg else None
    if a.masks:
        seg = MaskSeg(a.masks, probe(a.inp)[2])
    canvas = None
    fitbox = None
    held = None
    for i, g in enumerate(grids):
        t = i / 30
        if held is None or i % a.hold == 0:
            q = quantize(g, dither=a.dither, contrast=a.contrast)
            if seg is not None:
                mask, bbox = seg(frames[i], a.start + t) if isinstance(seg, MaskSeg) else seg(frames[i])
                if a.fit and mask is not None:
                    full, m270 = seg.full, seg.m270
                    if fitbox is None and isinstance(seg, MaskSeg):
                        fitbox = seg.fitbox
                    if fitbox is None:
                        ys_, xs_ = np.where(m270)
                        mx = 6
                        fitbox = (max(0, xs_.min() - mx), max(0, ys_.min() - mx), min(270, xs_.max() + mx), min(480, ys_.max() + mx))
                    bx0, by0, bx1, by1 = fitbox
                    sx = full.shape[1] / 270; sy = full.shape[0] / 480
                    crop = full[int(by0 * sy):int(by1 * sy), int(bx0 * sx):int(bx1 * sx)]
                    fw_ = a.fit; fh_ = int(round((by1 - by0) * fw_ / (bx1 - bx0)))
                    sm = cv2.resize(crop, (fw_, fh_), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                    sm = (255 * sm ** a.gamma).astype(np.uint8)
                    fq = quantize(sm, dither=a.dither, contrast=a.contrast)
                    mk = cv2.resize(m270[by0:by1, bx0:bx1].astype(np.float32), (fw_, fh_), interpolation=cv2.INTER_AREA) > 0.5
                    ox_ = (GW - fw_) // 2 + 2; oy_ = a.bottom - fh_
                    q = np.zeros_like(q); mask = np.zeros((GH, GW), bool)
                    yy0, yy1 = max(0, oy_), min(GH, oy_ + fh_)
                    q[yy0:yy1, ox_:ox_ + fw_] = fq[yy0 - oy_:yy1 - oy_]
                    mask[yy0:yy1, ox_:ox_ + fw_] = mk[yy0 - oy_:yy1 - oy_]
                    ys2, xs2 = np.where(mask)
                    bbox = (xs2.min() - 3, ys2.min() - 3, xs2.max() + 3, ys2.max() + 2) if len(xs2) else None
                    g = np.zeros_like(g); g[yy0:yy1, ox_:ox_ + fw_] = sm[yy0 - oy_:yy1 - oy_]
            else:
                mask, bbox = figure_mask(g, bg)
            if a.bg == 'fog' and mask is not None:
                if canvas is None:
                    canvas = fog_canvas(floor_y=int(bbox[3]) - 6)
                comp = canvas.copy(); comp[mask] = q[mask]; q = comp
            if a.dissolve > 0 and mask is not None:
                ys, xs = np.where(mask)
                x_lo, x_hi = xs.min(), xs.max()
                cut = x_lo + (x_hi - x_lo) * (1 - a.dissolve)
                drng = np.random.default_rng(1000 + i // (a.hold * 3))   # pattern changes slowly
                for y2 in range(ys.min() // 2 * 2, ys.max() + 1, 2):
                    for x2 in range(int(cut) // 2 * 2, x_hi + 1, 2):
                        if not mask[y2:y2 + 2, x2:x2 + 2].any():
                            continue
                        e = (x2 - cut) / max(1.0, x_hi - cut)
                        r_ = drng.random()
                        if r_ < e * 0.9:
                            c = q[y2:y2 + 2, x2:x2 + 2].reshape(-1, 3).mean(0)
                            if r_ < e * 0.45:
                                q[y2:y2 + 2, x2:x2 + 2] = canvas[y2:y2 + 2, x2:x2 + 2] if canvas is not None else PALETTE[1]
                                if rng.random() < 0.25:
                                    particles.append([float(y2), float(x2), -rng.uniform(0.2, 0.9), rng.uniform(0.05, 0.5), c.copy(), rng.integers(25, 70)])
                            else:
                                q[y2:y2 + 2, x2:x2 + 2] = c
            book = None
            if a.book and mask is not None:
                gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
                thr = np.percentile(gray[mask], 92)
                br = ((gray >= thr) & mask).astype(np.uint8)
                br = cv2.morphologyEx(br, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(br)
                if n_ > 1:
                    k = 1 + int(np.argmax(st_[1:, 4])); x_, y_, w_, h_ = st_[k, :4]
                    book = (x_ - 2, y_ - 3, x_ + w_ + 2, y_ + h_ + 3)
            held = (q, bbox, book)
        q, bbox, book = held
        frame = q.copy()
        alive = []
        for p in particles:
            p[0] += p[2]; p[1] += p[3]; p[5] -= 1
            y, x = int(p[0]), int(p[1])
            if p[5] > 0 and 0 <= y < GH - 2 and 0 <= x < GW - 2:
                frame[y:y + 2, x:x + 2] = p[4]
                alive.append(p)
        particles = alive
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ui:
            rgb = ui.draw(rgb, t, bbox, book)
        big = cv2.resize(rgb, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        big[::S] *= 0.82
        big += np.random.default_rng(i).normal(0, 4, big.shape[:2])[..., None]
        w.stdin.write(np.clip(big, 0, 255).astype(np.uint8).tobytes())
    w.stdin.close(); w.wait()
    print('frames', len(grids))


if __name__ == '__main__':
    main()
