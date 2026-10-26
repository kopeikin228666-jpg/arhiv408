"""ДЕЛО 003: «Мэри Селест» v2 - format C «ДЕЛО ЗА 15 СЕКУНД», now with the host: 408 sits in his live window
(make_ep1e.pip / host.pip_box, '● 408' tag, mouth wave + hot eyes driven by his voice amplitude) and SPEAKS:
«Я — четыреста восемь. Открываю дело.» under the hook, «Никто… не смог.» on «ПОЧЕМУ?». Voice = Callum with the
Дело 001 "from the darkness" chain (make_ep1e.VOICE); the bed is ducked under him. The loop is 17 s now; every
period below divides 17 s (DUR / n) and the pip is fed a time that is continuous across the seam.
v1 (make_delo003.py, 15 s, no host) is untouched.
A seamless loop: one monitor frame (pixel brigantine rolling in a grey sea, empty deck), one fact, own ambience.
Same look as Дело 001: 180x320 grid x6 -> 1080x1920, 30 fps (scene animated at 15 fps), 18-colour palette,
scanlines + grain (render/make_ep1e `finish`), slam text, hook in the first frame, level() to -14 LUFS.
Everything is periodic in 17 s (all motion periods divide 17 s, noise is circular), so frame 510 == frame 0.
usage: python3 make_delo003.py            -> renders everything into this folder
"""
import os, sys, math, subprocess
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
sys.path.insert(0, '/home/claude/voynich/team/host')
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import render as R
import host as HO
import pixelize as P
from ep_level import level, integrated, true_peak_db

R.CAP_BOTTOM = 96; R.CAP_MAXW = 118; R.CAP_CX = 84          # same safe-zone constants as make_ep1e
OUT = '/home/claude/voynich/edit/delo003'
GW, GH, OW, OH, S = 180, 320, 1080, 1920, 6
FPS, SR = 30, 48000
DUR = 17.0; NF = int(DUR * FPS); NS = int(DUR * SR)
ICON_X = 146                                                 # right Reels icon column starts here (grid x) for y > 150
SAFE_BOTTOM = GH - R.CAP_BOTTOM                              # 224: nothing of ours below this
TAU = 2 * math.pi
P5, P3, P25, P15 = DUR / 4, DUR / 6, DUR / 7, DUR / 11            # v1's 5 / 3 / 2.5 / 1.5 s periods, re-fit to 17 s

C = R.P                                                      # BGR palette colours by name
RGB = R.RGB
SEA_PAL = P.PALETTE[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]    # greys + swamp greens only (no parchment, no red)

# ------------------------------------------------------------------ timeline (s) - text never past the safe zone
HOOK_IN = 16.10                                              # hook slams back in; it is fully settled at the loop seam
HOOK_OUT = 4.00                                              # the hook holds while 408 introduces himself
HOOK = ['КОРАБЛЬ НАШЛИ', 'ЦЕЛЫМ.', 'ДЕСЯТИ ЧЕЛОВЕК', 'НА НЁМ НЕ БЫЛО.']
HOOK_KEY = 2
CARDS = [  # (t0, t1, big line, red?, sub line)
    (4.05, 6.75, '4 ДЕКАБРЯ 1872', False, 'АТЛАНТИКА'),
    (6.90, 9.45, 'ГРУЗ НА МЕСТЕ', False, '1701 БОЧКА СПИРТА'),
    (9.60, 11.30, 'ШЛЮПКИ НЕТ.', False, None),
    (11.40, 13.50, 'ЛЮДЕЙ НЕТ.', True, 'НИ ОДНОГО ИЗ 10'),
    (13.65, 15.98, 'ПОЧЕМУ?', False, None),
]
HIT = 11.40                                                  # the single hit: «ЛЮДЕЙ НЕТ.»
SLAMS = [HOOK_IN] + [c[0] for c in CARDS if c[0] != HIT]


# ------------------------------------------------------------------ fonts
def cond(size):
    return ImageFont.truetype(R.FCOND, size)


_MEAS = ImageDraw.Draw(Image.new('RGB', (8, 8)))


def fit_font(texts, maxw, sizes=(30, 28, 26, 24, 22, 21, 20, 19, 18, 17, 16, 15)):
    for s_ in sizes:
        f = cond(s_)
        if max(_MEAS.textlength(t, font=f) for t in texts) <= maxw:
            return f
    return cond(sizes[-1])


TEXT_MAXW = 160                                              # grid px: x 10..170 (all text sits above row 150, where the icon column starts)
TEXT_CX = 90
F_HOOK = fit_font(HOOK, TEXT_MAXW)
HOOK_FONTS = [fit_font([h], TEXT_MAXW, sizes=(24, 22, 21, 20, 19, 18, 17, 16)) for h in HOOK]   # each line as big as it fits
F_SUB = fit_font([c[4] for c in CARDS if c[4]], TEXT_MAXW - 8, sizes=(18, 17, 16, 15))


# ------------------------------------------------------------------ circular noise / periodic helpers
def circ_noise2d(h, w, seed, scale):
    """smooth tileable noise (FFT low-pass of white noise) in [0, 1]"""
    r = np.random.default_rng(seed).normal(0, 1, (h, w))
    F = np.fft.fft2(r)
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    F *= np.exp(-((fx * scale) ** 2 + (fy * scale * 0.5) ** 2) * 30)
    n = np.real(np.fft.ifft2(F)); n = (n - n.min()) / (n.max() - n.min())
    return n.astype(np.float32)


CLOUD = circ_noise2d(GH, GW, 7, 9.0)                         # wide, flat clouds
CLOUD2 = circ_noise2d(GH, GW, 11, 4.0)


def roll_frac(a, sx):
    """np.roll by a fractional x shift (linear), tileable"""
    i = int(math.floor(sx)); fr = sx - i
    return (1 - fr) * np.roll(a, i, 1) + fr * np.roll(a, i + 1, 1)


# ------------------------------------------------------------------ the ship (pixel art, drawn once)
SW, SH_ = 140, 150
WL = 132                                                     # waterline row inside the sprite
PIV = (70, WL)                                               # roll pivot


def ship_sprite():
    im = Image.new('RGBA', (SW, SH_), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    col = lambda k, a=255: RGB[k] + (a,)
    # rigging first (behind masts and sails): stays and shrouds, thin and dark
    for a_, b_ in [((86, 14), (134, 104)), ((86, 14), (46, 10)), ((46, 10), (10, 112)), ((86, 30), (100, 118)),
                   ((86, 30), (72, 118)), ((46, 16), (58, 118)), ((46, 16), (34, 118)), ((86, 14), (118, 112))]:
        d.line([a_, b_], fill=col('d3'), width=1)
    # masts
    d.line([(86, 8), (86, 120)], fill=col('void2'), width=2)
    d.line([(46, 4), (46, 120)], fill=col('void2'), width=2)
    # fore yards
    for y, hw in ((24, 15), (40, 19), (58, 23), (80, 26)):
        d.line([(86 - hw, y), (86 + hw, y)], fill=col('void2'), width=1)
    # lower fore topsail: set, grey canvas (40..58)
    d.polygon([(68, 41), (104, 41), (108, 57), (64, 57)], fill=col('light'))
    d.line([(68, 41), (64, 57)], fill=col('line')); d.line([(104, 41), (108, 57)], fill=col('line'))
    for x in range(70, 104, 6):                                # seams
        d.line([(x, 42), (x - 1 if x < 86 else x + 1, 56)], fill=col('line'))
    # foresail furled on the lowest yard: a lumpy roll
    for x in range(61, 112, 3):
        d.rectangle([x, 80, x + 2, 82 + (x * 7) % 3], fill=col('mid'))
    # mainmast: gaff + boom, the mainsail furled on the boom
    d.line([(46, 16), (14, 34)], fill=col('void2'), width=1)
    d.line([(46, 100), (6, 104)], fill=col('void2'), width=1)
    for x in range(8, 46, 3):
        d.rectangle([x, 97 + (x * 5) % 2, x + 2, 101], fill=col('mid'))
    # jib: set, triangle from the bowsprit to the foremast head
    d.polygon([(88, 20), (126, 100), (92, 100)], fill=col('bone'))
    d.line([(88, 20), (92, 100)], fill=col('line')); d.line([(92, 100), (126, 100)], fill=col('line'))
    d.polygon([(88, 30), (106, 104), (90, 104)], fill=col('light'))   # fore staysail
    # bowsprit
    d.line([(118, 113), (139, 101)], fill=col('void2'), width=2)
    # hull
    hull = [(6, 108), (14, 110), (60, 114), (112, 112), (124, 106), (126, 110), (118, 124), (108, 140), (22, 142), (10, 128)]
    d.polygon(hull, fill=col('void2'))
    d.line([(6, 108), (14, 110), (60, 114), (112, 112), (124, 106)], fill=col('tile'), width=1)       # rail
    d.line([(12, 118), (60, 121), (114, 119)], fill=col('d3'), width=1)                               # wale
    d.line([(16, 126), (60, 128), (112, 126)], fill=col('dark'), width=1)
    for x in range(18, 112, 7):                                                                         # stanchions
        d.point((x, 111 + (x > 60)), fill=col('tile'))
    # deckhouse aft of the mainmast, a dark doorway, the empty wheel at the stern
    d.rectangle([24, 104, 40, 112], fill=col('d3')); d.rectangle([24, 104, 40, 104], fill=col('tile'))
    d.rectangle([30, 106, 33, 112], fill=col('void'))
    d.ellipse([9, 101, 15, 107], outline=col('tile')); d.line([(12, 107), (12, 111)], fill=col('tile'))
    # main hatch with the EMPTY chocks where the yawl was stowed
    d.rectangle([58, 110, 74, 113], fill=col('dark'))
    d.rectangle([59, 107, 60, 110], fill=col('tile')); d.rectangle([72, 107, 73, 110], fill=col('tile'))
    # anchor at the bow
    d.line([(114, 114), (114, 120)], fill=col('tile')); d.line([(112, 120), (116, 120)], fill=col('tile'))
    return np.array(im)


SHIP = ship_sprite()
SHIP_X0, SHIP_WL = 14, 210                                   # sprite x on the grid; waterline row on the grid
HORIZON = 168


def ship_pose(t):
    roll = 2.4 * math.sin(TAU * t / P5) + 0.7 * math.sin(TAU * t / P3 + 1.0)     # degrees, periods divide 15 s
    bob = 1.6 * math.sin(TAU * t / P5 + 0.9) + 0.5 * math.sin(TAU * t / P25)
    return roll, bob


def ship_frame(t):
    """the sprite with its live parts (pennant, a loose sheet), rolled about the waterline -> RGBA (SH_, SW)"""
    im = Image.fromarray(SHIP.copy()); d = ImageDraw.Draw(im)
    # pennant at the mainmast head, fluttering (period 1.5 s)
    pts = [(46, 5)]
    for k in range(1, 9):
        pts.append((46 - k * 2, 5 + round(1.4 * math.sin(TAU * t / P15 - k * 0.8)) + k * 0.15))
    for a_, b_ in zip(pts, pts[1:]):
        d.line([a_, b_], fill=RGB['red2'] + (255,))
    # a loose sheet hanging from the upper fore yard (the upper sail is gone), swinging (period 2.5 s)
    sw = 3 * math.sin(TAU * t / P25)
    d.line([(76, 24), (76 + sw, 38)], fill=RGB['line'] + (255,))
    d.line([(97, 24), (97 + sw * 0.6, 33)], fill=RGB['line'] + (255,))
    # the topsail breathes with the wind (period 3 s): its foot bellies out a pixel or two
    b = round(1.5 + 1.5 * math.sin(TAU * t / P3))
    d.line([(64, 57), (86, 57 + b), (108, 57)], fill=RGB['light'] + (255,))
    a = np.array(im)
    roll, _ = ship_pose(t)
    M = cv2.getRotationMatrix2D(PIV, roll, 1.0)
    return cv2.warpAffine(a, M, (SW, SH_), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))


# ------------------------------------------------------------------ sky + sea (quantised to the palette)
YY, XX = np.mgrid[0:GH, 0:GW].astype(np.float32)
ROWPH = np.repeat(np.random.default_rng(5).uniform(0, TAU, GH // 2 + 1), 2)[:GH, None].astype(np.float32)   # each 2-row band its own phase: horizontal swell lines


def sea_sky(t, dark=0.0):
    img = np.zeros((GH, GW), np.float32)
    # sky: overcast, darker at the top, a pale band of fog over the horizon, clouds drifting (one tile per 15 s)
    sky = 0.16 + 0.30 * np.clip(YY / HORIZON, 0, 1) ** 2.2
    cl = roll_frac(CLOUD, GW * t / DUR) * 0.6 + roll_frac(CLOUD2, -GW * t / DUR) * 0.4
    sky = sky + (cl - 0.5) * 0.22 * (0.4 + 0.6 * np.clip(YY / HORIZON, 0, 1))
    fog = np.exp(-((YY - HORIZON) / 10.0) ** 2) * 0.12
    # sea: dark, lighter rows near the horizon, wave crests moving left/right (all phases periodic in 15 s)
    dep = np.clip((YY - HORIZON) / (GH - HORIZON), 0, 1)
    kx = 0.10 / (0.18 + dep)                                    # perspective: far waves are finer
    w = (np.sin(XX * kx + TAU * 6 * t / DUR + ROWPH) * 0.55 + np.sin(XX * kx * 0.47 - TAU * 3 * t / DUR + ROWPH * 2.3) * 0.45
         + np.sin(XX * 0.031 + TAU * 3 * t / DUR + YY * 0.11) * 0.5)
    sea = 0.30 - 0.16 * dep + w * (0.035 + 0.06 * dep)
    crest = np.clip(w - 1.05, 0, 1) * (0.25 + 0.5 * dep)        # foam on the tallest crests
    sea = sea + crest
    img = np.where(YY < HORIZON, sky + fog, sea + fog * 0.6)
    img = np.clip(img * (1 - dark), 0, 1)
    rgb = np.stack([img * 0.93, img * 1.0, img * 0.96], -1) * 255   # BGR, a hint of green-grey
    return P.quantize(rgb.astype(np.uint8), palette=SEA_PAL, dither=0.16, contrast=1.05)


def waterline(t, x):
    return 1.2 * np.sin(x * 0.21 + TAU * 6 * t / DUR) + 0.8 * np.sin(x * 0.09 - TAU * 3 * t / DUR)


def scene(t, f):
    """BGR grid of the deck at time t (t is the 15 fps-held time); f = output frame number"""
    hit_k = (t - HIT) % DUR
    dark = 0.0
    if hit_k < 2.2:                                             # the world goes dim while his eyes are in the sky
        dark = 0.45 * min(1, hit_k / 0.08) * (1 - max(0, (hit_k - 1.6) / 0.6))
    g = sea_sky(t, dark)
    bg = g.copy()
    # ship
    _, bob = ship_pose(t)
    sp = ship_frame(t)
    y0 = int(round(SHIP_WL - WL + bob)); x0 = SHIP_X0
    h, w = sp.shape[:2]
    sub = g[y0:y0 + h, x0:x0 + w]
    m = sp[..., 3] > 0
    sub[m] = sp[..., 2::-1][m]                                  # RGBA -> BGR
    if dark > 0:                                                # dim the ship with the world (palette steps, exact)
        lut = np.array([HO._darker(i) for i in range(18)], np.int16)
        if dark > 0.2:
            pi = HO._pal_index(sub); pi2 = lut[pi]; sub[m] = HO.BGR[pi2][m]
    # the sea in front of the hull: water rises over the bottom of the hull, foam where it touches
    xs = np.arange(GW)
    wl = SHIP_WL + 2 + waterline(t, xs)
    in_front = (YY >= wl[None, :]) & (XX >= x0 + 6) & (XX <= x0 + 126)
    g[in_front] = bg[in_front]
    for x in range(x0 + 8, x0 + 124):
        y = int(round(wl[x]))
        if 0 <= y < GH and ((x * 13 + int(t * 15) * 7) % 5) < 3:
            g[y, x] = C['light'] if (x + int(t * 15)) % 4 else C['bone']
    # his eyes in the overcast after the hit (the case is his): red slits + the hood's edge, sky only
    if hit_k < 2.2:
        a = min(1, hit_k / 0.25) * (1 - max(0, (hit_k - 1.5) / 0.7))
        por = HO.dark_portrait(t, f, eye_y=132, mood='scary', stare=max(0, 1 - hit_k / 0.4), glow=0.3 + 0.6 * a, smoke=10)
        pi = HO._pal_index(por)
        por = np.roll(por, -38, 1); pi = np.roll(pi, -38, 1)          # left of the foremast, in open sky
        red = (pi == 16) | (pi == 17)
        rim = (pi >= 2) & ~red
        thr = np.tile(P.bayer(4) + 0.5, (GH // 4, GW // 4))
        sky = (YY < HORIZON - 4) & (g == bg).all(2)          # never over the ship
        g[red & sky & (thr < a)] = por[red & sky & (thr < a)]
        g[rim & sky & (thr < a * 0.5)] = C['void2']
    return g



# ------------------------------------------------------------------ 408: his voice and his live window
A = '/home/claude/voynich/audio'
# make_ep1e.VOICE - the Дело 001 "from the darkness" chain, verbatim
VOICE = ('aresample=48000,highpass=f=70,lowpass=f=11000,equalizer=f=150:t=q:w=1:g=3,equalizer=f=3000:t=q:w=1.5:g=2,'
         'acompressor=threshold=-24dB:ratio=4:attack=5:release=120:makeup=5,aecho=0.8:0.4:60|140:0.12|0.07')
LINES = [  # (file, src a, src b, at = loop time of src a, text) - see audio/phrases/phrases.md
    ('vo_vidiq_phrases.mp3', 0.000, 1.956, 0.30, 'Я — четыреста восемь.'),
    ('vo_vidiq_phrases.mp3', 2.483, 3.844, 2.62, 'Открываю дело.'),
    ('vo_narrator.mp3', 11.27, 13.03, 14.00, 'Никто… не смог.'),
]
VO_RMS_DB = -21.0                                            # every line's active RMS before mixing (lines matched)
PIP_ON = (13.55, 4.15)                                       # his window: from just before «ПОЧЕМУ?» through the seam to the cards
PIP_X, PIP_Y, PIP_W, PIP_H = 6, 148, 48, 56                  # left, over the stern; clear of text, tag, icon column


def _load(path, a, b, af=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', path] + (['-af', af] if af else []) + \
          ['-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.check_output(cmd), np.float32).reshape(-1, 2).astype(np.float64)


def voice_track():
    """(NS, 2) 408's lines placed on the loop (wrapping), each through VOICE and matched in level; + per-frame amp"""
    vo = np.zeros((NS, 2))
    for (fn, a, b, at, tx) in LINES:
        a0 = max(0.0, a - 0.06)
        x = _load(f'{A}/{fn}', a0, b + 0.3, VOICE)
        n = len(x); fi, fo = int(0.012 * SR), int(0.06 * SR)
        x[:fi] *= np.linspace(0, 1, fi)[:, None]; x[n - fo:] *= np.linspace(1, 0, fo)[:, None]
        m = x.mean(1); F_ = SR // 100
        fr = np.array([np.sqrt((m[i:i + F_] ** 2).mean()) for i in range(0, n - F_, F_)])
        act = fr[20 * np.log10(fr + 1e-12) > 20 * np.log10(fr.max()) - 30]
        g = 10 ** ((VO_RMS_DB - 20 * np.log10(np.sqrt((act ** 2).mean()))) / 20)
        place(vo, x * g, at - (a - a0))
        print(f'  voice: {tx:24s} {fn} {a:.3f}-{b:.3f} -> loop {at:.2f}-{at + b - a:.2f}  gain {20 * np.log10(g):+.1f} dB')
    return vo


def amp_track(vo):
    """make_ep1e's AMP: per output frame, dB of the voice RMS mapped (-44..-20 dBFS) -> 0..1, quiet frames 0"""
    vm = vo.mean(1); hop = SR // FPS
    env = np.array([np.sqrt((vm[i * hop:(i + 1) * hop] ** 2).mean() + 1e-12) for i in range(NF)])
    a = np.clip((20 * np.log10(env + 1e-9) + 44) / 24, 0, 1); a[a < 0.12] = 0
    return a



def pip_on(t):
    a, b = PIP_ON
    return t >= a or t < b


def pip_time(t):
    """a host time continuous across the loop seam (the window is closed at DUR/2, where it jumps)"""
    return 100.0 + (t if t < DUR / 2 else t - DUR)


def pip_layer(rgb, t, f):
    """make_ep1e.pip on the text layer (RGB): his close portrait in a window, '● 408' tag blinking, mouth wave and hot
    eyes driven by AMP; 2 held frames of static as the window opens / closes"""
    if not pip_on(t):
        return rgb
    x0, y0, w, h = PIP_X, PIP_Y, PIP_W, PIP_H
    k_in = (t - PIP_ON[0]) % DUR; k_out = (PIP_ON[1] - t) % DUR
    if k_in < 0.14 or k_out <= 0.14:                         # opening / closing: static in the frame
        v = np.random.default_rng(int(t * 15) + 7).integers(0, 4, (h, w))
        cols = np.array([P.PALETTE[0], P.PALETTE[3], P.PALETTE[6], P.PALETTE[8]], np.uint8).reshape(-1, 3)
        box = cols[v]
    else:
        box = HO.pip_box(pip_time(t), float(AMP[f % NF]), w, h)
    rgb[y0 - 1:y0 + h + 1, x0 - 1:x0 + w + 1] = HO.BGR[HO.D3][::-1]
    rgb[y0:y0 + h, x0:x0 + w] = box[..., ::-1]
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    d.rectangle([x0 - 1, y0 - 10, x0 + 30, y0 - 2], fill=RGB['red'] if int(t * 2) % 2 == 0 else RGB['red2'])
    d.text((x0 + 1, y0 - 10), '● 408', font=R.F7, fill=RGB['bone'])
    return np.array(pil)

# ------------------------------------------------------------------ text (RGB grid on a magenta key)
KEY = (255, 0, 255)


def _txt_img(text, font, col, plate_col=None, pad=3):
    w = int(_MEAS.textlength(text, font=font)) + 2 * pad; h = font.size + 2 * pad + 3
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); d.fontmode = '1'
    if plate_col:
        d.rectangle([0, 0, w - 1, h - 1], fill=plate_col + (255,))
    else:
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (2, 2)):
            d.text((pad + ox, pad + oy - 1), text, font=font, fill=RGB['void'] + (255,))
    d.text((pad, pad - 1), text, font=font, fill=col + (255,))
    return np.array(im)


def _paste(rgb, im, cx, cy, scale=1.0, dx=0, dy=0):
    if scale != 1.0:
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    h, w = im.shape[:2]
    x0, y0 = int(cx - w / 2 + dx), int(cy - h / 2 + dy)
    gx0, gy0, gx1, gy1 = max(0, x0), max(0, y0), min(GW, x0 + w), min(GH, y0 + h)
    if gx1 > gx0 and gy1 > gy0:
        sub = im[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
        m = sub[..., 3] > 0
        rgb[gy0:gy1, gx0:gx1][m] = sub[..., :3][m]
    return rgb


def slam_line(rgb, k, life, text, y, font, red=False, delay=0.0):
    """make_ep1e.slam: the line slams in from big, shakes, settles; glitches out over its last 0.12 s"""
    k = k - delay
    if k < 0 or k > life - delay:
        return rgb
    sc = 1.0 + 0.9 * math.exp(-k / 0.05)
    shake = 3 * math.exp(-k / 0.08)
    r_ = np.random.default_rng(int(k * 1000) + y)
    dx, dy = (r_.random(2) - 0.5) * 2 * shake
    if k > 0.6:                                                 # settled: exact position (the seam must be bit-identical)
        sc, dx, dy = 1.0, 0.0, 0.0
    if k > life - delay - 0.12:
        dx += (r_.random() - 0.5) * 30
    im = _txt_img(text, font, RGB['bone'], RGB['red'] if red else None)
    return _paste(rgb, im, TEXT_CX, y, sc, dx, dy)


def hook_layer(rgb, k):
    """the hook: four lines, the key line bone on a red plate (like make_ep1e.hook_text), slamming in line by line
    from HOOK_IN; fully settled long before the seam, so frame 0 (the cover) shows it complete"""
    life = DUR - HOOK_IN + HOOK_OUT
    y = 34
    for n, s_ in enumerate(HOOK):
        fn = HOOK_FONTS[n]
        y += fn.size // 2 + 4
        rgb = slam_line(rgb, k, life, s_, y, fn, red=(n == HOOK_KEY), delay=0.07 * n)
        y += fn.size // 2 + 4
    return rgb


def hud(rgb, t, f):
    """the monitor: case header, blinking REC, corner brackets, the honest «РЕКОНСТРУКЦИЯ» tag - and, on the beats,
    a tracking box over the empty chocks / the empty deck"""
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    d.text((8, 20), 'АРХИВ 408 · ДЕЛО 003', font=R.F9, fill=RGB['line'])
    if int(t * 2) % 2 == 0:                                     # blink 1 Hz (periodic)
        d.rectangle([141, 22, 145, 26], fill=RGB['red'])
    d.text((148, 20), 'REC', font=R.F9, fill=RGB['red'] if int(t * 2) % 2 == 0 else RGB['red2'])
    for (x, y, sx, sy) in ((4, 16, 1, 1), (175, 16, -1, 1), (4, 222, 1, -1)):   # no bracket in the icon column
        d.line([(x, y), (x + 8 * sx, y)], fill=RGB['tile']); d.line([(x, y), (x, y + 8 * sy)], fill=RGB['tile'])
    d.rectangle([6, 209, 6 + d.textlength('РЕКОНСТРУКЦИЯ', font=R.F7) + 4, 218], fill=RGB['void2'])
    d.text((8, 210), 'РЕКОНСТРУКЦИЯ', font=R.F7, fill=RGB['line'])
    _, bob = ship_pose(t)
    oy = int(round(bob))
    def box(x0, y0, x1, y1, label, blink_from):
        if (t - blink_from) < 0.25 and int((t - blink_from) * 16) % 2:
            return
        d.rectangle([x0, y0 + oy, x1, y1 + oy], outline=RGB['bone'])
        w = d.textlength(label, font=R.F7)
        d.rectangle([x0, y0 + oy - 10, x0 + w + 3, y0 + oy - 1], fill=RGB['red'])
        d.text((x0 + 2, y0 + oy - 10), label, font=R.F7, fill=RGB['bone'])
    c3, c4 = CARDS[2], CARDS[3]
    if c3[0] <= t < c3[1]:
        box(SHIP_X0 + 54, 192, SHIP_X0 + 78, 202, 'ШЛЮПКА: —', c3[0])
    if c4[0] <= t < c4[1]:
        box(SHIP_X0 + 4, 176, SHIP_X0 + 124, 202, 'НА БОРТУ: 0 / 10', c4[0])
    return np.array(pil)


def text_layer(t, f, ta=None):
    """-> (RGB grid on the magenta key, bbox of the big text or None)"""
    rgb = np.zeros((GH, GW, 3), np.uint8); rgb[:] = KEY
    hk = (t - HOOK_IN) % DUR
    if hk < DUR - HOOK_IN + HOOK_OUT:
        rgb = hook_layer(rgb, hk)
    for (t0, t1, big, red, sub) in CARDS:
        if t0 <= t < t1:
            fb = fit_font([big], TEXT_MAXW)
            rgb = slam_line(rgb, t - t0, t1 - t0, big, 70, fb, red=red)
            if sub:
                rgb = slam_line(rgb, t - t0, t1 - t0, sub, 70 + fb.size // 2 + 18, F_SUB, red=not red, delay=0.18)
    m = ~((rgb[..., 0] == 255) & (rgb[..., 1] == 0) & (rgb[..., 2] == 255))
    bb = None
    if m.any():
        ys, xs = np.nonzero(m); bb = (ys.min(), ys.max(), xs.min(), xs.max())
    rgb = hud(rgb, t if ta is None else ta, f)
    rgb = pip_layer(rgb, t if ta is None else ta, f)
    return rgb, bb


# ------------------------------------------------------------------ frame assembly (= make_ep1e.finish, without drift)
noise_bank = [np.random.default_rng(k).normal(0, 4.0, (OH, OW, 1)).astype(np.float32) for k in range(6)]  # 6 | 450
scan = np.ones((OH, 1, 1), np.float32); scan[::S] = 0.82


def zoom_of(t):
    return 1.0 + 0.035 * (0.5 - 0.5 * math.cos(TAU * t / DUR))   # slow push in and out, periodic


def frame(f, return_grid=False):
    t = f / FPS
    ta = (((f + 1) // 2) * 2 / FPS) % DUR                       # scene held at 15 fps; pairs (449,0),(1,2).. so frame 449 == frame 0
    g = scene(ta, f)
    hk = (ta - HIT) % DUR
    if hk < 2 / FPS:                                            # tear on the hit (2 frames)
        g = R.tear(g, np.random.default_rng(f), 5, 26)
    grid_rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    big = cv2.resize(grid_rgb, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    z = zoom_of(ta)
    if z > 1.0005:
        w, h = OW / z, OH / z; x0 = (OW - w) * 0.5; y0 = (OH - h) * 0.55
        M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
        big = cv2.warpAffine(big, M, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=(10, 12, 11))
    big *= scan; big += noise_bank[((f + 1) // 2) % 3]           # grain also held per pair (225 pairs, 3 | 225)
    big = np.clip(big, 0, 255).astype(np.uint8)
    lay, bb = text_layer(t, f, ta)
    m = ~((lay[..., 0] == 255) & (lay[..., 1] == 0) & (lay[..., 2] == 255))
    if bb is not None:                                          # scrim behind the words (sails / sky never cut them)
        y0, y1, x0, x1 = bb
        y0, y1, x0, x1 = max(0, y0 - 3), min(GH, y1 + 4), max(0, x0 - 5), min(GW, x1 + 6)
        big[y0 * S:y1 * S, x0 * S:x1 * S] = (big[y0 * S:y1 * S, x0 * S:x1 * S].astype(np.float32) * 0.5).astype(np.uint8)
    lb = cv2.resize(lay, (OW, OH), interpolation=cv2.INTER_NEAREST)
    mb = cv2.resize(m.astype(np.uint8), (OW, OH), interpolation=cv2.INTER_NEAREST).astype(bool)
    big[mb] = (lb[mb].astype(np.float32) * scan[np.nonzero(mb)[0]][:, 0]).astype(np.uint8)
    if return_grid:
        comp = grid_rgb.copy(); comp[m] = lay[m]
        return big, comp, m
    return big


# ------------------------------------------------------------------ sound (all circular: the loop is seamless)
def circ_band(seed, lo, hi, tilt=0.0, n=NS):
    """white noise band-limited in the frequency domain (circular => periodic in n samples), unit RMS"""
    X = np.fft.rfft(np.random.default_rng(seed).normal(0, 1, n))
    fr = np.fft.rfftfreq(n, 1 / SR)
    msk = 1 / (1 + (lo / np.maximum(fr, 1e-3)) ** 4) / (1 + (fr / hi) ** 4)
    if tilt:
        msk *= (np.maximum(fr, 20) / 100.0) ** (-tilt)
    y = np.fft.irfft(X * msk, n)
    return y / (y.std() + 1e-12)


def place(buf, x, at):
    """add x (n, 2) at time `at`, wrapping around the loop"""
    i0 = int(at * SR) % NS
    for k in range(0, len(x)):
        pass
    idx = (np.arange(len(x)) + i0) % NS
    np.add.at(buf, idx, x)


def creak(seed, amp, dur=0.55, f0=380.0):
    """wooden stick-slip: an accelerating train of tiny impulses through three hull resonances"""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed); n = int(dur * SR); tt = np.arange(n) / SR
    rate = 18 + 70 * (tt / dur) ** 1.4 * (1 - 0.6 * (tt / dur) ** 3)
    ph = np.cumsum(rate / SR); imp = np.zeros(n)
    hits = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    imp[hits] = rng.uniform(0.5, 1.0, len(hits))
    y = np.zeros(n)
    for fr_, q, g in ((f0, 30, 1.0), (f0 * 1.63, 26, 0.6), (f0 * 2.71, 22, 0.35)):
        r = math.exp(-math.pi * fr_ / q / SR)
        y += g * lfilter([1 - r], [1, -2 * r * math.cos(TAU * fr_ / SR), r * r], imp)
    env = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 0.6
    y = y * env; y = y / (np.abs(y).max() + 1e-9) * amp
    pan = rng.uniform(-0.4, 0.4)
    return np.stack([y * (1 - pan), y * (1 + pan)], 1)


def crushed_kick(amp=0.6):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    f = 110 * np.exp(-tt * 9) + 38
    x = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-tt * 6)
    x = np.round(x * 8) / 8; x = np.repeat(x[::4], 4)[:n]
    return np.stack([x * amp] * 2, 1)


def card_hit(amp=1.0):
    """make_ep1e.card_hit: crushed kick + long sub drop + low body thump"""
    from scipy.signal import butter, sosfilt
    n = int(1.6 * SR); tt = np.arange(n) / SR
    y = np.zeros((n, 2)); k = crushed_kick(0.5 * amp); y[:len(k)] += k
    sw = np.sin(TAU * np.cumsum(62 * np.exp(-tt * 2.4) + 27) / SR) * np.exp(-tt * 2.0) * np.minimum(1, tt / 0.003) * 0.34 * amp
    body = sosfilt(butter(2, 220, 'low', fs=SR, output='sos'), np.random.default_rng(93).normal(0, 1, n)) * np.exp(-tt * 9) * 0.5 * amp
    ring = (np.sin(TAU * 740 * tt) + 0.7 * np.sin(TAU * 781 * tt)) * np.exp(-tt * 3.0) * 0.05 * amp   # a dark metallic tail
    y += np.stack([sw + body + ring, sw + body + ring * 0.8], 1)
    return y


def tick(amp=0.05, seed=0):
    n = int(0.05 * SR); x = np.random.default_rng(seed).normal(0, 1, n)
    x = np.repeat(x[::6], 6)[:n]; x = np.round(x * 6) / 6 * np.exp(-np.linspace(0, 4, n)) * amp
    return np.stack([x, x], 1)


def audio():
    tt = np.arange(NS) / SR
    mix = np.zeros((NS, 2))
    # sea: low wash rising and falling with the swell (period 5 s, like the ship's roll), stereo-decorrelated
    sw = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(TAU * tt / P5 - 0.6)) ** 2
    for ch, sd in ((0, 1), (1, 2)):
        mix[:, ch] += circ_band(sd, 60, 900, tilt=0.6) * 0.10 * sw
        mix[:, ch] += circ_band(sd + 10, 1500, 6000) * 0.018 * (0.5 + 0.5 * np.sin(TAU * tt / P25 + ch)) ** 3   # spray hiss
    # wind: two bands breathing against each other (periods 15 and 7.5 s), a faint whistle in the rigging
    wa = 0.5 + 0.5 * np.sin(TAU * tt / DUR); wb = 0.5 + 0.5 * np.sin(TAU * tt / (DUR / 2) + 2)
    for ch, sd in ((0, 3), (1, 4)):
        mix[:, ch] += circ_band(sd, 350, 750) * 0.030 * wa + circ_band(sd + 20, 800, 1400) * 0.016 * wb
    whistle_f = 1180 + 60 * np.sin(TAU * tt / DUR)
    mix += (np.sin(TAU * np.cumsum(whistle_f) / SR) * 0.004 * wb)[:, None]
    # drone: integer cycles per 15 s (seamless), tension builds to the hit, drops after, returns to the start level
    def denv(t):
        k = (t - HIT) % DUR                                     # 0 at the hit
        pre = np.clip(1 - k / DUR, 0, 1)                        # (not used directly)
        e = np.where(k < 1.8, 0.25 + 0.15 * k / 1.8, 0)         # after the hit: low
        rise = np.clip((k - 1.8) / (DUR - 1.8), 0, 1)           # then a slow rise all the way back to the next hit
        e = np.where(k >= 1.8, 0.40 + 0.75 * rise ** 1.6, e)
        return e
    de = denv(tt)
    for fc, a in ((742 / DUR, 0.10), (1113 / DUR, 0.05), (1485 / DUR, 0.022)):     # = v1's pitches, whole cycles per loop
        mix += (np.sin(TAU * fc * tt) * a * de)[:, None]
    mix[:, 0] += np.sin(TAU * (743 / DUR) * tt) * 0.04 * de         # slow beating, left only
    mix += (circ_band(9, 30, 140) * 0.05 * de)[:, None]
    # duck everything for 0.18 s before the hit (it lands out of a breath), then the hit
    k = (tt - HIT) % DUR
    duck = np.ones(NS)
    pre = (k > DUR - 0.22)
    duck[pre] = 1 - 0.75 * np.clip((k[pre] - (DUR - 0.22)) / 0.08, 0, 1)
    post = k < 0.25
    duck[post] = np.minimum(duck[post], 0.25 + 0.75 * k[post] / 0.25)
    mix *= duck[:, None]
    # creaks at the ends of the roll (the hull and the masts working), varied
    roll_ext = [P5 / 4 + P5 / 2 * i for i in range(8)]
    for i, te in enumerate(roll_ext):
        if abs(((te - HIT) % DUR)) < 0.6 or abs(((HIT - te) % DUR)) < 0.3:
            continue
        place(mix, creak(100 + i, 0.07 + 0.03 * (i % 2), 0.45 + 0.1 * (i % 3), 330 + 45 * (i % 3)), te - 0.25)
    place(mix, creak(201, 0.05, 0.9, 260), 7.6)                 # a long groan of the mast
    # text ticks and the single hit
    for i, ts in enumerate(SLAMS):
        place(mix, tick(0.06, i), ts)
    place(mix, card_hit(1.0), HIT)
    # his window opens / closes with a crackle of static
    for at in (PIP_ON[0], PIP_ON[1] - 0.14):
        n = int(0.12 * SR); x = np.random.default_rng(int(at * 100)).normal(0, 1, n)
        x = np.repeat(x[::5], 5)[:n] * np.exp(-np.linspace(0, 3, n)) * 0.05
        place(mix, np.stack([x, x], 1), at)
    # duck the whole bed under his voice (~ -8 dB, 0.35 s attack/release, circular), then add him
    v = np.abs(VO.mean(1)); k1 = int(0.03 * SR); k2 = int(0.35 * SR)
    circ = lambda x, k: np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.r_[np.ones(k) / k, np.zeros(len(x) - k)])))
    v = np.roll(circ(v, k1), -k1 // 2)
    talk = np.roll(circ((v > 0.004).astype(float), k2), -k2 // 2)
    duck = 1 - 0.6 * np.clip(talk * 1.5, 0, 1)
    return mix * duck[:, None] + VO


VO = voice_track()
AMP = amp_track(VO)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    # sound
    mix = audio()
    lev = level(mix, [{'kind': 'outro', 't0': 0.0, 't1': DUR}], SR, voice=VO)
    wav = f'{OUT}/delo003_v2_mix.wav'
    import wave
    pcm = (np.clip(lev, -1, 1) * 32767).astype('<i2')
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())
    print('audio: I', round(integrated(lev, SR), 2), 'LUFS, TP', round(true_peak_db(lev), 2), 'dBTP')
    # video
    vid = f'{OUT}/_video_v2.mp4'
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', vid]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    qc = {'text_bad': [], 'off_palette': 0, 'grids': {}}
    pal = set(map(tuple, P.PALETTE.astype(np.uint8)[:, ::-1].tolist()))    # RGB
    for f in range(NF):
        big, comp, m = frame(f, return_grid=True)
        p.stdin.write(big.tobytes())
        ys, xs = np.nonzero(m)
        if len(ys):
            bad = (ys >= SAFE_BOTTOM) | ((ys > 150) & (xs >= ICON_X)) | (ys < 14)
            if bad.any():
                qc['text_bad'].append(f)
        u = np.unique(comp.reshape(-1, 3), axis=0)
        qc['off_palette'] += sum(1 for c in map(tuple, u.tolist()) if c not in pal)
        if f in (0, NF - 1):
            qc['grids'][f] = comp
        if f % 60 == 0:
            print('frame', f, flush=True)
    p.stdin.close(); p.wait()
    print('QC text outside safe zone on frames:', qc['text_bad'][:20], 'count', len(qc['text_bad']))
    print('QC off-palette colours (grid, all frames):', qc['off_palette'])
    np.save(f'{OUT}/_qc_grids_v2.npy', np.stack([qc['grids'][0], qc['grids'][NF - 1]]))
    # master: H.264 + AAC
    master = f'{OUT}/delo003_v2_master.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-t', f'{DUR}', '-movflags', '+faststart', master], check=True)
    # IG: two-pass to a size budget
    ig = f'{OUT}/delo003_v2_IG.mp4'
    vb = 3900                                                    # 17 s: (3900 + 160) k * 17 s ~ 8.6 MB
    for ps in (1, 2):
        c_ = ['ffmpeg', '-v', 'error', '-y', '-i', master, '-c:v', 'libx264', '-preset', 'slow', '-b:v', f'{vb}k',
              '-maxrate', f'{vb * 2}k', '-bufsize', f'{vb * 2}k', '-pix_fmt', 'yuv420p', '-r', '30', '-g', '60',
              '-profile:v', 'high', '-pass', str(ps), '-passlogfile', f'{OUT}/_2pass_v2']
        if ps == 1:
            c_ += ['-an', '-f', 'mp4', '/dev/null']
        else:
            c_ += ['-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-movflags', '+faststart', ig]
        subprocess.run(c_, check=True)
    for fn in os.listdir(OUT):
        if fn.startswith('_2pass_v2'):
            os.remove(f'{OUT}/{fn}')
    # cover + QC stills
    cov = frame(0)
    Image.fromarray(cov).save(f'{OUT}/cover_v2.png')
    stills = []
    for i, ts in enumerate((1.2, 3.2, 7.0, 11.9, 14.5)):
        im = frame(int(round(ts * FPS)))
        Image.fromarray(im).save(f'{OUT}/qc_v2_{i + 1}_{ts:04.1f}s.png'); stills.append(im)
    sheet = np.hstack([cv2.resize(x, (540, 960), interpolation=cv2.INTER_AREA) for x in [cov] + stills])
    Image.fromarray(sheet).save(f'{OUT}/qc_sheet_v2.png')
    print('done')


if __name__ == '__main__':
    main()
