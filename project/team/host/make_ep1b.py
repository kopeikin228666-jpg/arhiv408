"""Episode 1 (v2): «408» tells the case of the Voynich manuscript as a real host - he moves, points, teleports, the camera
cuts between a wide lab shot and close-ups, story shots play full screen between his lines, and then he dives into the
big screen and the main reel plays with his little live window in the corner.  1080x1920, 30 fps, <= 90 s.
"""
import os, sys, math, subprocess, argparse
os.environ['VOY_VER'] = 'B'
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import numpy as np, cv2
from PIL import Image, ImageDraw
import render as R
import room as RM
import host as HO
import pixelize as P
from align import scribe_words, scribe_line, align_line, true_attacks, display_chunks
from timeline import LINES as RLINES

E = '/home/claude/voynich/edit'; A = '/home/claude/voynich/audio'; KF = '/home/claude/voynich/kf'
OUTD = '/home/claude/voynich/team/host/ep1b'; os.makedirs(OUTD, exist_ok=True)
FPS, SR = 30, 48000
GW, GH, OW, OH, S = 180, 320, 1080, 1920, 6
PHRASE = f'{A}/vo_408.mp3' if os.path.exists(f'{A}/vo_408.mp3') else None

# ------------------------------------------------------------------ the edit
# kinds: 'cold' (close-up in the dark), 'lab' (wide), 'close' (close-up), 'reel' (v6 master frames), 'dive' (into the big screen)
# a line = (file, src_a, src_b, text); 'pose' changes = [(t_rel, pose, flip)], host x, screen keyframe, camera (z0, z1)
SEG = []


def seg(kind, dur, **kw):
    SEG.append(dict(kind=kind, dur=dur, **kw))


seg('cold', 2.0 + (4.0 if PHRASE else 0.0), line=(('vo_408.mp3', 0.0, 3.84, 'Я — четыреста{четыреста} восемь… Открываю *дело.'), 0.9) if PHRASE else None,
    trans_out=('tear', 6))
seg('lab', None, line=(('vo_intro.mp3', 0.0, 5.49, 'Двести сорок *страниц… и ни одного… *прочитанного слова.'), 0.75),
    hx=62, poses=[(0.0, 'front', False), ('страниц', 'point', False), ('слова', 'front', False)], screen='s02_parchment',
    cam=(1.0, 1.1), teleport=0.6, tail=0.35, trans_out=('zoom', 8))
seg('reel', None, src=(7.7, 11.0), trans_out=('flash', 4))
seg('close', None, line=(('vo_written.mp3', 0.06, 3.3, 'Она *написана… *неизвестно *кем.'), 0.25), cam=(1.0, 1.12), tail=0.2,
    trans_out=('glyph', 8))
seg('reel', None, src=(36.55, 38.55), trans_out=('slice', 6))
seg('lab', None, line=(('vo_narrator.mp3', 4.74, 10.76, 'В 1912-м{тысяча девятьсот двенадцатом} её *купили. Потом пытались *читать.'), 0.55),
    hx=118, poses=[(0.0, 'point', True), ('ПОТОМ', 'front', False)], screen='s09_tag', screen2=('ПОТОМ', 's15_reading'),
    cam=(1.1, 1.0), teleport=0.45, tail=0.3, trans_out=('iris', 8))
seg('reel', None, src=(26.33, 30.0), trans_out=('blocks', 6))
seg('close', None, line=(('vo_narrator.mp3', 0.0, 3.81, 'Эта *книга… не хочет, чтобы её *читали.'), 0.2), cam=(1.0, 1.3), tail=0.45,
    scary_from='ЧИТАЛИ.', trans_out=None)
seg('dive', 1.5)
seg('reel', None, src=(47.467, 69.0), pip=True, trans_out=('tear', 3))
seg('reel', None, src=(100.73, 130.0), pip=True, trans_out=('pixelrain', 10))
seg('outro', 3.4)

# ------------------------------------------------------------------ words of his lines
def words_of(f, a, b, text, at):
    if scribe_words(f):
        toks, res = scribe_line(f, a, b, text)
    else:
        toks, res = align_line(f'{A}/{f}', a, b, text)
    res = true_attacks(f, a, toks, res, trusted_ends=bool(scribe_words(f)))
    ids = display_chunks(toks)
    return [dict(w=d, key=k, t0=at + r0, t1=at + r1, chunk=c) for (d, sp, k, e), (r0, r1), c in zip(toks, res, ids)]


if PHRASE and not os.path.exists(f'{A}/scribe_vo_408.txt'):
    open(f'{A}/scribe_vo_408.txt', 'w').write('Я 0 0.32\nчетыреста 0.8 1.36\nвосемь… 1.44 2.48\nОткрываю 2.56 3.12\nдело. 3.2 3.84\n')

T = 0.0
for sg in SEG:
    sg['t0'] = T
    if sg.get('line'):
        (f, a, b, text), lead = sg['line']
        sg['at'] = T + lead
        sg['words'] = words_of(f, a, b, text, sg['at'])
        if sg['dur'] is None:
            sg['dur'] = lead + (b - a) + sg.get('tail', 0.3)
    if sg['kind'] == 'reel':
        sg['dur'] = sg['src'][1] - sg['src'][0]
    T += sg['dur']
    sg['t1'] = T
T_END = T
for sg in SEG:
    print(f"{sg['kind']:6s} {sg['t0']:6.2f} {sg['t1']:6.2f}", sg.get('src', ''), sg.get('line', [''])[0][3] if sg.get('line') else '')
print('TOTAL', round(T_END, 2))
ALLW = [w for sg in SEG for w in sg.get('words', [])]


def seg_at(t):
    for i, sg in enumerate(SEG):
        if sg['t0'] <= t < sg['t1']:
            return i, sg
    return len(SEG) - 1, SEG[-1]


def word_time(sg, word):
    for w in sg.get('words', []):
        if w['w'].upper().startswith(word.upper()):
            return w['t0']
    return sg['t0']


def caption_at(sg, t):
    ws = sg.get('words')
    if not ws or t < ws[0]['t0'] - 0.2 or t > min(ws[-1]['t1'] + 0.7, sg['t1'] - 0.02):
        return None
    heads, tails = {}, {}
    for w in ws:
        heads.setdefault(w['chunk'], w['t0']); tails[w['chunk']] = w['t1']
    def shows(c):
        h = heads[c] - 0.2
        if c - 1 in tails:
            h = max(h, min(heads[c] - 0.03, tails[c - 1] + 0.08))
        return h
    ch = max([c for c in heads if shows(c) <= t] or [min(heads)])
    return [(w['w'], w['key'], w['t0'] - 0.03 <= t, 0 <= t - w['t0'] + 0.03 < 2 / FPS) for w in ws if w['chunk'] == ch], 'narr'


# ------------------------------------------------------------------ audio helpers
def load(path, a=None, b=None, af=None, ch=2):
    cmd = ['ffmpeg', '-v', 'error'] + (['-ss', str(a)] if a is not None else []) + (['-to', str(b)] if b is not None else []) + ['-i', path]
    if af:
        cmd += ['-af', af]
    cmd += ['-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-']
    x = np.frombuffer(subprocess.check_output(cmd), np.float32).copy()
    return x.reshape(-1, ch) if ch > 1 else x


VOICE = ('aresample=48000,highpass=f=70,lowpass=f=10500,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,'
         'acrusher=bits=11:mix=0.12:mode=log,aecho=0.8:0.55:55|130|260:0.16|0.10|0.06')
N = int((T_END + 1) * SR)


def fade(x, fi=0.012, fo=0.06):
    x = x.copy(); n = len(x); a_, b_ = int(fi * SR), int(fo * SR)
    x[:a_] *= np.linspace(0, 1, a_)[:, None]; x[n - b_:] *= np.linspace(1, 0, b_)[:, None]
    return x


def place(buf, x, at):
    i = int(round(at * SR))
    if i < 0:
        x = x[-i:]; i = 0
    n = min(len(x), len(buf) - i)
    if n > 0:
        buf[i:i + n] += x[:n]


vo = np.zeros((N, 2), np.float32)
for sg in SEG:
    if sg.get('line'):
        (f, a, b, text), lead = sg['line']
        a0 = max(0, a - 0.06)
        x = load(f'{A}/{f}', a0, b + 0.3, VOICE)
        place(vo, fade(x), sg['at'] - (a - a0))
vo_mono = vo.mean(1)
hop = SR // FPS
nfr = int(T_END * FPS) + 2
env = np.array([np.sqrt((vo_mono[i * hop:(i + 1) * hop] ** 2).mean() + 1e-12) for i in range(nfr)])
AMP = np.clip((20 * np.log10(env + 1e-9) + 44) / 24, 0, 1); AMP[AMP < 0.12] = 0
# the reel's own narrator stem (16 kHz mono, film time) drives his little window in the corner
VO_REEL = np.load(f'{E}/vo_B.npy')
NARR_SPANS = [(st - 0.1, st + (b - a) + 0.3) for (f, a, b, st, tx, v) in RLINES if v in ('narr', 'final', 'intro')]


def reel_amp(ft):
    i = int(ft * 16000); seg_ = VO_REEL[max(0, i - 270):i + 270]
    if not len(seg_):
        return 0.0
    db = 20 * np.log10(np.sqrt((seg_ ** 2).mean()) + 1e-9)
    return float(np.clip((db + 44) / 24, 0, 1)) if db > -44 else 0.0


# ------------------------------------------------------------------ picture helpers
noise_bank = [np.random.default_rng(k).normal(0, 4.0, (OH, OW, 1)).astype(np.float32) for k in range(8)]
scan = np.ones((OH, 1, 1), np.float32); scan[::S] = 0.82
_KF = {}


def kf(name):
    if name not in _KF:
        _KF[name] = cv2.imread(f'{KF}/{name}.jpg')
    return _KF[name]


def static_img(seed, w=80, h=60):
    v = np.random.default_rng(seed).integers(0, 4, (h, w))
    cols = np.array([P.PALETTE[0], P.PALETTE[3], P.PALETTE[6], P.PALETTE[8]], np.uint8).reshape(-1, 3)
    return cols[v]


def plate(rgb, t, t0, closing=False):
    k = t - t0
    if k < 0:
        return rgb
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    def typed(s, start, cps=28):
        return s[:int(max(0, k - start) * cps)]
    s1 = typed('АРХИВ 408 · ДЕЛО №001' + (' · ОТКРЫТО' if closing else ''), 0.0)
    if s1:
        d.rectangle([6, 6, 6 + d.textlength(s1, font=R.F9) + 4, 17], fill=R.RGB['void2'])
        d.text((8, 7), s1, font=R.F9, fill=R.RGB['bone'])
    if not closing:
        s2 = typed('РУКОПИСЬ ВОЙНИЧА', 0.5, 22)
        if s2:
            w = d.textlength(s2, font=R.FCAP)
            d.rectangle([6, 20, 6 + w + 6, 38], fill=R.RGB['red']); d.text((9, 21), s2, font=R.FCAP, fill=R.RGB['bone'])
        s3 = typed('MS 408 · ЙЕЛЬСКИЙ УНИВЕРСИТЕТ · США', 1.4, 34)
        if s3:
            d.rectangle([6, 41, 6 + d.textlength(s3, font=R.F7) + 4, 50], fill=R.RGB['void2'])
            d.text((8, 42), s3, font=R.F7, fill=R.RGB['line'])
    return np.array(pil)


def question(rgb, t, t0):
    k = t - t0
    if k < 0:
        return rgb
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    for i, (s, col) in enumerate((('ЧТО, ПО-ТВОЕМУ,', 'bone'), ('В НЕЙ НАПИСАНО?', 'red'))):
        s = s[:int(max(0, k - i * 0.5) * 24)]
        if not s:
            continue
        w = d.textlength(s, font=R.FCAP); x = int((GW - w) / 2); y = 262 + i * 20
        if col == 'red':
            d.rectangle([x - 3, y + 1, x + w + 3, y + 17], fill=R.RGB['red']); d.text((x, y), s, font=R.FCAP, fill=R.RGB['bone'])
        else:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                d.text((x + ox, y + oy), s, font=R.FCAP, fill=R.RGB['void'])
            d.text((x, y), s, font=R.FCAP, fill=R.RGB['bone'])
    return np.array(pil)


def blink_at(t):
    return (t % 3.9) < 0.1


def pose_at(sg, t):
    cur = ('front', False)
    for (when, pose, flip) in sg.get('poses', [(0, 'front', False)]):
        tw = sg['t0'] + when if isinstance(when, (int, float)) else word_time(sg, when) - 0.05
        if t >= tw:
            cur = (pose, flip, tw)
    return cur


def lab_grid(sg, t, f):
    """wide shot of the lab, BGR grid"""
    amp = float(AMP[min(len(AMP) - 1, int(t * FPS))])
    pose, flip, tw = pose_at(sg, t) if len(pose_at(sg, t)) == 3 else (*pose_at(sg, t), sg['t0'])
    tl = t - sg['t0']
    on = 1.0
    if sg.get('teleport') and tl < sg['teleport']:
        on = max(0.0, tl / sg['teleport'])
    scr = kf(sg['screen'])
    if sg.get('screen2') and t >= word_time(sg, sg['screen2'][0]) - 0.3:
        scr = kf(sg['screen2'][1])
    if tl < 0.25:
        scr = static_img(f)
    bob = int(round(math.sin(t * 2 * math.pi / 2.14)))
    g = RM.frame(t, amp, 'calm', blink_at(t) and amp == 0, 0, screen=scr, host_cx=sg['hx'], host_pose=pose, host_flip=flip,
                 host_on=on, bob=bob)
    # the gesture lands with a tiny glitch on the frame the pose changes
    if 0 <= t - tw < 2 / FPS and t - sg['t0'] > 0.1:
        g = R.rgbsplit(g, 1)
    return g


_LABBG = {}


def close_grid(sg, t, f, mood='calm', amp=None):
    """close-up: the lab behind him, pushed in and darker; him big in front"""
    if amp is None:
        amp = float(AMP[min(len(AMP) - 1, int(t * FPS))])
    bg = RM.frame(t, 0, 'calm', screen=kf('s03_eye') if sg['kind'] == 'outro' else None, host_on=0)
    crop = bg[40:200, 45:135]
    bg = cv2.resize(crop, (GW, GH), interpolation=cv2.INTER_NEAREST)
    pi = HO._pal_index(bg)
    lut = np.array([HO._darker(i) for i in range(18)], np.int16)
    bg = HO.BGR[lut[lut[pi]]]
    bob = int(round(math.sin(t * 2 * math.pi / 2.14)))
    HO.draw(bg, 90, 160 - 88 + 273 + 2, t, amp, 1, blink_at(t) and amp == 0, 0, bob, int(t * 15), mood, 'close', aura=True, smoke=30)
    return bg


def finish(rgb_grid, f, zoom=1.0, cx=0.5, cy=0.45):
    big = cv2.resize(rgb_grid, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big *= scan; big += noise_bank[f % 8]
    big = np.clip(big, 0, 255).astype(np.uint8)
    if zoom > 1.001:
        w, h = OW / zoom, OH / zoom
        x0 = (OW - w) * cx; y0 = (OH - h) * cy
        M = np.float32([[zoom, 0, -x0 * zoom], [0, zoom, -y0 * zoom]])
        big = cv2.warpAffine(big, M, (OW, OH), flags=cv2.INTER_NEAREST)
    return big


def reel_grids(a, b):
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{a:.3f}', '-i', f'{E}/video_v6_B.mp4', '-t', f'{b - a:.3f}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE); n = OW * OH * 3; out = []
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
        out.append(fr[3::S, 3::S].copy())                  # back to the 180x320 grid (pixel centres)
    p.wait()
    return out


def pip(g, t, ft):
    """his little live window in the corner while the story plays: he is the one telling it"""
    if not any(a <= ft <= b for a, b in NARR_SPANS):
        return g
    amp = reel_amp(ft)
    x0, y0, w, h = 128, 60, 46, 54
    s = HO.sprite_img(t, 0, False, 0, 0, 'calm', 'close', sway=False)
    sp = s[30:30 + 150]
    small = cv2.resize(sp[..., :3], (w, int(150 * w / 149)), interpolation=cv2.INTER_AREA)
    small = P.quantize(small, dither=0.0, contrast=1.1)[:h]
    box = np.zeros((h, w, 3), np.uint8); box[:] = HO.BGR[HO.VOID]
    box[:small.shape[0]] = small
    # eyes + mouth redrawn at this size
    ey = int((88 - 30) * w / 149); exs = (int(61 * w / 149), int(88 * w / 149))
    for j, ex in enumerate(exs):
        box[ey, ex - 1:ex + 2] = HO.BGR[HO.RED2]; box[ey, ex] = HO.BGR[HO.RED]
    if amp > 0.05:
        my = int((104 - 30) * w / 149); r = np.random.default_rng(int(t * 15))
        for x in range(int(64 * w / 149), int(86 * w / 149), 2):
            hh = int(round(amp * 2 * r.random()))
            box[my - hh:my + hh + 1, x] = HO.BGR[HO.RED2]
    box[::2] = (box[::2].astype(np.int16) * 3 // 4).astype(np.uint8)
    g = g.copy()
    g[y0 - 1:y0 + h + 1, x0 - 1:x0 + w + 1] = HO.BGR[HO.D3]
    g[y0:y0 + h, x0:x0 + w] = box
    pil = Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil); d.fontmode = '1'
    d.rectangle([x0 - 1, y0 - 10, x0 + 30, y0 - 2], fill=R.RGB['red'] if int(t * 2) % 2 == 0 else R.RGB['red2'])
    d.text((x0 + 1, y0 - 10), '● 408', font=R.F7, fill=R.RGB['bone'])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ one frame of any segment (BGR grid + camera)
REELS = {}


def seg_frame(i, t, f):
    """-> (BGR grid, zoom, cx, cy) for segment i at time t (t may lie a little outside the segment for transitions)"""
    sg = SEG[i]
    tl = t - sg['t0']
    k = 'lab' if sg['kind'] == 'lab' else sg['kind']
    if k == 'reel':
        fr = REELS[i]
        j = int(np.clip(round(tl * FPS), 0, len(fr) - 1))
        g = fr[j]
        if sg.get('pip'):
            g = pip(g, t, sg['src'][0] + tl)
        return g, 1.0, .5, .45
    if k == 'lab':
        g = lab_grid(sg, t, f)
        z0, z1 = sg.get('cam', (1.0, 1.0))
        u = np.clip(tl / max(sg['dur'], 1e-3), 0, 1)
        z = z0 + (z1 - z0) * (u * u * (3 - 2 * u))
        # a punch-in on the key words
        for w in sg.get('words', []):
            if w['key'] and 0 <= t - w['t0'] < 0.3:
                z += 0.035 * math.exp(-(t - w['t0']) / 0.1)
        rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        rgb = plate(rgb, t, sg['t0'] + 0.3) if i == 1 else rgb
        rgb = R.draw_caption(rgb, caption_at(sg, t), t)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), z, sg['hx'] / GW * 0.6 + 0.2, .42
    if k in ('close', 'cold', 'outro'):
        mood = 'calm'
        if k == 'cold':
            mood = 'scary' if 0.85 <= tl < 1.25 else 'calm'
        if k == 'outro':
            mood = 'scary'
        if sg.get('scary_from') and t >= word_time(sg, sg['scary_from']) - 0.05:
            mood = 'scary'
        amp = None
        g = close_grid(sg, t, f, mood, amp)
        if k == 'cold' and tl < 0.5:                       # the eyes have not lit yet: only darkness and smoke
            g = close_grid(sg, t, f, mood, 0.0)
            red = (g[..., 2] > 120) & (g[..., 1] < 90)
            g[red] = HO.BGR[HO.VOID]
        z0, z1 = sg.get('cam', (1.0, 1.08))
        u = np.clip(tl / max(sg['dur'], 1e-3), 0, 1)
        z = z0 + (z1 - z0) * u
        rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        if k == 'outro':
            rgb = plate(rgb, t, sg['t0'] + 0.2, closing=True)
            rgb = question(rgb, t, sg['t0'] + 0.6)
        else:
            rgb = R.draw_caption(rgb, caption_at(sg, t), t)
        if k == 'cold' and tl < 0.35:
            g2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            st = static_img(f, GW, GH); m = np.random.default_rng(f).random((GH, GW)) < (1 - tl / 0.35) * 0.9
            g2[m] = st[m]; rgb = cv2.cvtColor(g2, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), z, .5, .4
    if k == 'dive':
        # wide again: he turns to the screen and points at it, the screen bursts into static, we dive through the glass
        prev = SEG[i - 1]
        g = RM.frame(t, 0, 'scary', False, 0, screen=static_img(f) if tl < 0.6 else REELS[i + 1][0], host_cx=50,
                     host_pose='point', host_flip=False)
        return g, 1.0, .5, .45
    return np.zeros((GH, GW, 3), np.uint8), 1.0, .5, .45


def dive_frame(i, t, f):
    sg = SEG[i]; tl = t - sg['t0']
    g, _, _, _ = seg_frame(i, t, f)
    rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    big = finish(rgb, f)
    k = np.clip((tl - 0.55) / (sg['dur'] - 0.55), 0, 1)
    if k <= 0:
        return big
    e = k * k * (3 - 2 * k)
    BX0, BY0, BX1, BY1 = RM.GLASS['big']
    gx0, gy0, gx1, gy1 = BX0 * S, BY0 * S, BX1 * S, BY1 * S
    gw, gh = gx1 - gx0, gy1 - gy0
    nxt = cv2.resize(REELS[i + 1][0], (OW, OH), interpolation=cv2.INTER_NEAREST)
    nxt = cv2.cvtColor(nxt, cv2.COLOR_BGR2RGB)
    sh_ = int(OW * gh / gw)
    big[gy0:gy1, gx0:gx1] = cv2.resize(nxt[(OH - sh_) // 2:(OH - sh_) // 2 + sh_], (gw, gh), interpolation=cv2.INTER_AREA)
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    w = OW + (gw * 0.92 - OW) * e; h = w * OH / OW
    x0 = OW / 2 + (cx - OW / 2) * e - w / 2; y0 = OH / 2 + (cy - OH / 2) * e - h / 2
    M = np.float32([[OW / w, 0, -x0 * OW / w], [0, OH / h, -y0 * OH / h]])
    fr = cv2.warpAffine(big, M, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=(10, 12, 11))
    if k > 0.8:
        fr = np.clip(fr.astype(np.int16) + int(255 * (k - 0.8) / 0.2 * 0.6), 0, 255).astype(np.uint8)
    return fr


def render_video(out, a=0.0, b=None):
    b = T_END if b is None else b
    for i, sg in enumerate(SEG):
        if sg['kind'] == 'reel' and sg['t1'] > a - 1 and sg['t0'] < b + 1 or (sg['kind'] == 'reel' and i > 0 and SEG[i - 1]['kind'] == 'dive' and SEG[i - 1]['t1'] > a - 2 and SEG[i - 1]['t0'] < b):
            REELS[i] = reel_grids(*sg['src'])
            print('reel', i, len(REELS[i]), flush=True)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(int(round(a * FPS)), int(round(b * FPS))):
        t = f / FPS
        i, sg = seg_at(t)
        rng = np.random.default_rng(f)
        frame = None
        # transitions centred on the cut between i-1 -> i (or i -> i+1)
        for (j0, j1) in ((i - 1, i), (i, i + 1)):
            if j0 < 0 or j1 >= len(SEG) or not SEG[j0].get('trans_out'):
                continue
            kind, n = SEG[j0]['trans_out']
            fc = int(round(SEG[j1]['t0'] * FPS)); pre = n // 2
            if not (fc - pre <= f < fc - pre + n):
                continue
            prog = (f - (fc - pre) + 1) / (n + 1)
            ga, za, cxa, cya = seg_frame(j0, min(t, SEG[j0]['t1'] - 1 / FPS), f)
            gb, zb, cxb, cyb = seg_frame(j1, max(t, SEG[j1]['t0']), f)
            g = R.run_trans(kind, ga, gb, prog, j1, rng)
            z = za if prog < 0.5 else zb
            frame = finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, z, cxa if prog < .5 else cxb, cya if prog < .5 else cyb)
            break
        if frame is None:
            if sg['kind'] == 'dive':
                frame = dive_frame(i, t, f)
            else:
                g, z, cx, cy = seg_frame(i, t, f)
                if sg['kind'] == 'reel' and t - sg['t0'] < 2 / FPS and SEG[i - 1]['kind'] == 'dive':
                    g = R.tear(g, rng, 4, 30)
                rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
                if sg['kind'] == 'outro' and t > T_END - 0.5:
                    rgb = (rgb.astype(np.float32) * max(0, (T_END - t) / 0.5)).astype(np.uint8)
                frame = finish(rgb, f, z, cx, cy)
        p.stdin.write(frame.tobytes())
        if f % 60 == 0:
            print('t', round(t, 1), sg['kind'], flush=True)
            cv2.imwrite(f'{OUTD}/f_{f:05d}.jpg', cv2.cvtColor(cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    p.stdin.close(); p.wait()


# ------------------------------------------------------------------ audio
def synth_static(dur, amp=0.25, seed=0):
    n = int(dur * SR); x = np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)
    x = np.repeat(x[::6], 6)[:n]; x = np.round(x * 6) / 6
    return np.stack([x * np.exp(-np.linspace(0, 3, n)) * amp] * 2, 1).astype(np.float32)


def crushed_kick(amp=0.6):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    f = 110 * np.exp(-tt * 9) + 38
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 6)
    x = np.repeat(np.round(x * 8) / 8, 1)
    x = np.repeat(x[::4], 4)[:n]
    return np.stack([x * amp] * 2, 1).astype(np.float32)


def swell(dur=1.2, amp=0.25, seed=3):
    from scipy.signal import butter, sosfilt
    n = int(dur * SR); x = np.random.default_rng(seed).normal(0, 1, n)
    x = sosfilt(butter(2, [200, 1800], 'bandpass', fs=SR, output='sos'), x)
    x = x / (np.abs(x).max() + 1e-9) * np.linspace(0, 1, n) ** 2.5
    return np.stack([x * amp] * 2, 1).astype(np.float32)


def whoosh(dur=0.35, amp=0.2, seed=5):
    from scipy.signal import butter, sosfilt
    n = int(dur * SR); x = np.random.default_rng(seed).normal(0, 1, n)
    x = sosfilt(butter(2, [400, 4000], 'bandpass', fs=SR, output='sos'), x)
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return np.stack([x * e * amp / (np.abs(x).max() + 1e-9)] * 2, 1).astype(np.float32)


def render_audio(out):
    mix = np.zeros((N, 2), np.float32)
    # the lab's bed: a quiet stretch of Mara running under all his segments (a constant room tone), ducked under his voice
    bed = load(f'{A}/track2.mp3', 6.0, 6.0 + T_END + 1)
    ref = load(f'{E}/mix_B.wav', 47.5, 50.5); raw = load(f'{A}/track2.mp3', 54.2, 57.2)
    bed *= np.sqrt((ref ** 2).mean()) / (np.sqrt((raw ** 2).mean()) + 1e-9) * 0.55
    bed = bed[:N]
    if len(bed) < N:
        bed = np.concatenate([bed, np.zeros((N - len(bed), 2), np.float32)])
    v = np.convolve(np.abs(vo_mono), np.ones(int(0.03 * SR)) / int(0.03 * SR), 'same')
    talk = np.convolve((v > 0.01).astype(np.float32), np.ones(int(0.35 * SR)) / int(0.35 * SR), 'same')
    duck = 1 - 0.6 * np.clip(talk * 1.5, 0, 1)
    gate = np.zeros(N, np.float32)                          # the bed only plays under his own segments
    for sg in SEG:
        if sg['kind'] != 'reel':
            gate[int(sg['t0'] * SR):int(sg['t1'] * SR)] = 1
    gate = np.convolve(gate, np.ones(int(0.12 * SR)) / int(0.12 * SR), 'same')
    mix += bed * (duck * gate)[:, None]
    mix += vo * 1.35
    # the story shots carry their own finished sound
    for sg in SEG:
        if sg['kind'] == 'reel':
            a, b = sg['src']
            place(mix, fade(load(f'{E}/mix_B.wav', a, b), 0.015, 0.04), sg['t0'])
    # sound for every cut and move
    place(mix, synth_static(0.4, 0.28, 1), 0.0)
    place(mix, crushed_kick(0.55), SEG[0]['t0'] + 0.85)          # the eyes ignite
    for i, sg in enumerate(SEG[1:], 1):
        kind = SEG[i - 1].get('trans_out')
        if sg['kind'] == 'lab' and sg.get('teleport'):
            place(mix, synth_static(sg['teleport'], 0.2, i), sg['t0'])
        if kind:
            place(mix, whoosh(0.35, 0.16, i), sg['t0'] - 0.25)
            place(mix, crushed_kick(0.35), sg['t0'])
        for (when, pose, flip) in sg.get('poses', [])[1:]:
            place(mix, whoosh(0.2, 0.1, i + 50), word_time(sg, when) - 0.12)
        if sg['kind'] == 'dive':
            place(mix, synth_static(0.6, 0.22, 77), sg['t0'])
            place(mix, swell(sg['dur'] - 0.4, 0.22), sg['t0'] + 0.4)
            place(mix, crushed_kick(0.6), sg['t1'])
        if sg['kind'] == 'outro':
            place(mix, synth_static(0.4, 0.25, 11), sg['t0'])
            place(mix, crushed_kick(0.6), sg['t0'] + 0.02)
            n = int(sg['dur'] * SR); tt = np.arange(n) / SR
            dr = (np.sin(2 * np.pi * 41 * tt) * 0.6 + np.sin(2 * np.pi * 61.5 * tt) * 0.3) * np.minimum(1, tt / 0.4) * np.maximum(0, 1 - tt / sg['dur']) * 0.18
            place(mix, np.stack([dr, dr], 1).astype(np.float32), sg['t0'])
    mix = mix[:int(T_END * SR)]
    mix = np.tanh(mix / max(np.abs(mix).max(), 1e-6) * 1.3) * 0.92
    q = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-c:a', 'pcm_s16le', out], stdin=subprocess.PIPE)
    q.stdin.write(mix.astype(np.float32).tobytes()); q.stdin.close(); q.wait()
    print('audio ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='a', type=float, default=0.0)
    ap.add_argument('--to', dest='b', type=float, default=None)
    ap.add_argument('--audio-only', action='store_true'); ap.add_argument('--video-only', action='store_true')
    o = ap.parse_args()
    if not o.video_only:
        render_audio(f'{OUTD}/ep.wav')
    if not o.audio_only:
        render_video(f'{OUTD}/ep_video.mp4', o.a, o.b)
