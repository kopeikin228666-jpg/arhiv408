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
R.CAP_BOTTOM = 96                       # every caption sits above the TikTok/Reels description and buttons
R.CAP_MAXW = 118; R.CAP_CX = 84         # captions span x ~150-858 px: clear of the right icon column (x > 900, y 900-1700)


def _fits_narrow(words):
    """True when draw_caption lays these words out in <= 2 lines at the normal size within R.CAP_MAXW (mirrors its wrap)"""
    def nlines(font, lim):
        n, cur = 1, 0.0
        for w in words:
            wl = R._MEAS.textlength(w.upper() + ' ', font=font)
            if cur and cur + wl > lim:
                n += 1; cur = wl
            else:
                cur += wl
        return n
    if any(R._MEAS.textlength(w.upper() + ' ', font=R.FCAP) > R.CAP_MAXW + 6 for w in words):
        return nlines(R.FCAP_S, R.CAP_MAXW + 4) <= 2           # one over-wide word: draw_caption uses the small size
    return nlines(R.FCAP, R.CAP_MAXW) <= 2


_WEAK = {'И', 'В', 'НЕ', 'НА', 'С', 'К', 'О', 'У', 'А', 'ПО', 'ИЗ', 'ЗА', 'ЧТО', 'КАК', 'ИХ'}


def _split_narrow(idx, ws):
    """R._split for the narrow column + a chunk never ends on a preposition / conjunction («ЖЕНЩИНЫ В» -> «В ЗЕЛЁНОЙ»)"""
    words = [ws[i]['w'] for i in idx]
    if _fits_narrow(words) or len(idx) < 2:
        return [idx]
    total = sum(len(w) for w in words)
    best, bestk = None, None
    for k in range(1, len(idx)):
        left = sum(len(w) for w in words[:k])
        score = abs(left - total / 2) - (12 if words[k - 1].endswith(',') else 0)
        if min(k, len(idx) - k) < 2:
            score += 8
        if words[k - 1].upper() in _WEAK:
            score += 20
        if not (_fits_narrow(words[:k]) and _fits_narrow(words[k:])):
            score += 100
        if best is None or score < best:
            best, bestk = score, k
    return _split_narrow(idx[:bestk], ws) + _split_narrow(idx[bestk:], ws)


R._fits = _fits_narrow                  # the narrower caption column: re-split the reel's caption chunks so none needs 3 lines
R._split = _split_narrow
for _ws in R._LINES:
    R._chunks(_ws)


def rechunk(ws):
    """split his own caption chunks the same way (never mid-thought, never an orphan word), renumbered in order"""
    groups, c = {}, 0
    for i, w in enumerate(ws):
        groups.setdefault(w['chunk'], []).append(i)
    for cid in sorted(groups):
        for part in R._split(groups[cid], ws):
            for i in part:
                ws[i]['chunk'] = c
            c += 1
    return ws
REEL_SRC = f'/home/claude/voynich/edit/video_v6_B_clean.mp4'   # the reel without its burned-in captions
import room as RM
import host as HO
import pixelize as P
from align import scribe_words, scribe_line, align_line, true_attacks, display_chunks
from timeline import LINES as RLINES

E = '/home/claude/voynich/edit'; A = '/home/claude/voynich/audio'; KF = '/home/claude/voynich/kf'
PART = int(os.environ.get('PART', '1'))
OUTD = f'/home/claude/voynich/team/host/ep1e_p{PART}'; os.makedirs(OUTD, exist_ok=True)
FPS, SR = 30, 48000
GW, GH, OW, OH, S = 180, 320, 1080, 1920, 6
PHRASE = f'{A}/vo_408.mp3' if os.path.exists(f'{A}/vo_408.mp3') else None

# ------------------------------------------------------------------ the edit
# kinds: 'cold' (close-up in the dark), 'lab' (wide), 'close' (close-up), 'reel' (v6 master frames), 'dive' (into the big screen)
# a line = (file, src_a, src_b, text); 'pose' changes = [(t_rel, pose, flip)], host x, screen keyframe, camera (z0, z1)
SEG = []


def seg(kind, dur, **kw):
    SEG.append(dict(kind=kind, dur=dur, **kw))


if PART == 1:
    seg('hook', 6.4, line=(('vo_narrator.mp3', 0.0, 3.81, 'Эта *книга… не хочет, чтобы её *читали.'), 0.35),
        flashes=[(0.80, 0.33, 9.2), (2.08, 0.30, 52.6), (2.76, 0.28, 12.9)],
        montage=[(4.25, 0.26, 73.2), (4.51, 0.26, 111.6), (4.77, 0.26, 65.6)],
        clip=None, title=5.05, title_text=('ДЕЛО №001', 'РУКОПИСЬ ВОЙНИЧА'), trans_out=('tear', 6))
    seg('lab', None, line=(('vo_intro.mp3', 0.0, 5.49, 'Двести сорок *страниц… и ни одного… *прочитанного слова.'), 0.5),
        hx=62, poses=[(0.0, 'front', False), ('страниц', 'point', False), ('ни', 'front', False)], screen='s02_parchment',
        cam=(1.03, 1.12), teleport=0.5, tail=0.2, lean=('ни', 0.9), trans_out=('zoom', 8),
        kin=[('Двести', 'count', 240, 'СТРАНИЦ'), ('ни', 'big', '0', 'ПРОЧИТАННЫХ СЛОВ')])
    seg('reel', None, src=(8.9, 10.6), trans_out=('flash', 4))
    seg('close', None, line=(('vo_written.mp3', 0.06, 3.3, 'Она *написана… *неизвестно *кем.'), 0.2), cam=(1.03, 1.16), tail=0.15,
        looks=[('неизвестно', -1), ('кем', 0)], kin=[('неизвестно', 'big', 'АВТОР: ???', None)], trans_out=('glyph', 8))
    seg('reel', None, src=(36.55, 38.55), trans_out=('slice', 6))
    seg('lab', None, line=(('vo_narrator.mp3', 4.74, 10.76, 'В 1912-м{тысяча девятьсот двенадцатом} её *купили. Потом пытались *читать.'), 0.45),
        hx=118, walk=(36, 118, 0.0, 1.9), poses=[(0.0, 'front', False), ('ПОТОМ', 'point', True)], turn='ПОТОМ',
        screen='s09_tag', screen2=('ПОТОМ', 's15_reading'), cam=(1.12, 1.03), tail=0.25, trans_out=('iris', 8),
        kin=[('1912', 'big', '1912', 'КУПЛЕНА')])
    seg('reel', None, src=(26.9, 29.9), trans_out=('shutter', 6))
    seg('reel', None, src=(42.55, 46.6), pip=True, trans_out=('blocks', 6))
    seg('lean', 1.4)
    seg('card', 1.9, lines=('ЧАСТЬ 2', 'ЗАВТРА'))
elif PART == 3:
    # one video: the hook, his story, the dive, the horror, the last line over the real book, a hard cut that loops
    seg('hook', 5.9, line=(('vo_narrator.mp3', 0.0, 3.81, 'Эта *книга… не хочет, чтобы её *читали.'), 0.35),
        flashes=[(0.80, 0.33, 9.2), (2.08, 0.30, 52.6), (2.76, 0.28, 12.9)],
        montage=[(4.25, 0.26, 73.2), (4.51, 0.26, 111.6), (4.77, 0.26, 65.6)],
        clip=None, title=5.05, title_text=('ДЕЛО 001', 'РУКОПИСЬ ВОЙНИЧА'), trans_out=('tear', 6),
        hooktext=(('ПОЧЕМУ ЕЁ', 'НЕЛЬЗЯ', 'ПРОЧИТАТЬ?'), 1))   # frame 0 asks the question (muted autoplay, thumbnail)
    seg('lab', None, line=(('vo_intro.mp3', 0.0, 2.13, 'Двести сорок *страниц…'), 0.15), hx=62,
        poses=[(0.0, 'front', False), ('страниц', 'point', False)], screen='s02_parchment', cam=(1.03, 1.10), teleport=0.4,
        tail=0.15, kin=[('Двести', 'count', 240, 'СТРАНИЦ')], inserts=[('страниц', 6.9, 0.30)], trans_out=('zoom', 6))
    seg('close', None, line=(('vo_intro.mp3', 2.40, 5.49, 'и ни одного… *прочитанного слова.'), 0.05), cam=(1.06, 1.18), tail=0.15,
        kin=[('ни', 'big', '0', 'ПРОЧИТАННЫХ СЛОВ')], inserts=[('прочитанного', 11.6, 0.28)], trans_out=('flash', 4))
    seg('reel', None, src=(8.9, 9.85), trans_out=('glyph', 6))
    seg('close', None, line=(('vo_written.mp3', 0.06, 3.3, 'Она *написана… *неизвестно *кем.'), 0.2), cam=(1.0, 1.25), tail=0.15,
        looks=[('написана', 1), ('неизвестно', -1), ('кем', 0)], kin=[('неизвестно', 'big', 'АВТОР: ???', None)], kin_y=60,
        inserts=[('написана', 60.0, 0.35), ('кем', 39.6, 0.28)], trans_out=('slice', 6))
    seg('reel', None, src=(36.55, 38.55), trans_out=('iris', 6))
    # «её» -> he turns and points at the tag on the big screen; «купили» -> the real lot tag «УИЛФРИД ВОЙНИЧ · АНТИКВАР»
    seg('lab', None, line=(('vo_narrator.mp3', 4.74, 8.07, 'В 1912-м{тысяча девятьсот двенадцатом} её *купили.'), 0.3), hx=118,
        walk=(36, 118, 0.0, 1.6), poses=[(0.0, 'front', False), ('её', 'point', True)], turn='её', screen='s09_tag',
        cam=(1.12, 1.03), tail=0.1, kin=[('1912', 'big', '1912', 'КУПИЛ ВОЙНИЧ')],
        inserts=[('купили', 20.80, 0.48)], trans_out=('shutter', 6))
    seg('reel', None, src=(22.55, 25.15), trans_out=('slice', 4))
    # «не смог» used to sit on the black tail of the swirl: cut to the empty reading desk (the reader is gone) + «0 %»
    seg('reel', None, src=(27.40, 29.55), over=[(1.14, 31.16, 1.01, 1.0, 1.10)],
        slams=[(1.20, '0 %', 'РАСШИФРОВАНО', 128, 0.95)])
    # «Буквы двигаются» -> the letters crawl: every grid row slides on a sine
    seg('reel', None, src=(42.55, 46.6), pip=True, crawl=(43.45, 44.54), trans_out=('blocks', 6))
    seg('lean', 0.7, slams=[(0.0, 'ЗАГЛЯНЕМ ВНУТРЬ.', None, 52, 0.7)])
    seg('dive', 1.2)
    seg('reel', None, src=(47.467, 50.05), pip=True, trans_out=('slice', 4))
    seg('reel', None, src=(51.55, 54.25), pip=True, trans_out=('glyph', 4))
    # «то, чего нельзя видеть» -> the eye with the plant growing in its iris
    seg('reel', None, src=(56.15, 62.55), pip=True, over=[(5.54, 12.68, 0.32, 1.04, 1.12)], trans_out=('tear', 3))
    seg('reel', None, src=(110.30, 113.45), pip=True, trans_out=('tear', 2))
    # the real book in its real case (a frozen frame): a push-in + punches on the words, a tear when «КНИГОЙ.» lands,
    # the museum's own tiny label replaced by a readable one
    seg('reel', None, src=(121.05, 124.40), pip=True, pip_y=94, push=0.16, cy=0.40, mask=(196, 224, 14, 166),
        idplate=0.10, tear_on=123.71)
    seg('card', 1.9, lines=('ТЫ ДОЧИТАЛ.', 'ОНА — НЕТ.'), hit=0.45)
else:
    seg('hook', 4.2, line=None, flashes=[], montage=[], clip=(0.45, 104.9, 107.6), title=3.2,
        title_text=('ДЕЛО №001', 'ЧАСТЬ 2'))
    seg('dive', 1.5)
    seg('reel', None, src=(47.467, 64.3), pip=True, trans_out=('tear', 3))
    seg('reel', None, src=(108.77, 113.7), pip=True)
    seg('card', 2.4, lines=('ПИШИ,', 'ЧТО ТЫ ВИДИШЬ'))

# ------------------------------------------------------------------ words of his lines
def words_of(f, a, b, text, at):
    if scribe_words(f):
        toks, res = scribe_line(f, a, b, text)
    else:
        toks, res = align_line(f'{A}/{f}', a, b, text)
    res = true_attacks(f, a, toks, res, trusted_ends=bool(scribe_words(f)))
    ids = display_chunks(toks)
    return rechunk([dict(w=d, key=k, t0=at + r0, t1=at + r1, chunk=c) for (d, sp, k, e), (r0, r1), c in zip(toks, res, ids)])


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


VOICE = ('aresample=48000,highpass=f=70,lowpass=f=11000,equalizer=f=150:t=q:w=1:g=3,equalizer=f=3000:t=q:w=1.5:g=2,'
         'acompressor=threshold=-24dB:ratio=4:attack=5:release=120:makeup=5,aecho=0.8:0.4:60|140:0.12|0.07')
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


def plate(rgb, t, t0, closing=False, rows23=True):
    k = t - t0
    if k < 0:
        return rgb
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    def typed(s, start, cps=28):
        return s[:int(max(0, k - start) * cps)]
    y0 = 34                                                   # below the app's top bar (~200 px of 1920 = 33 rows)
    s1 = typed('АРХИВ 408 · ДЕЛО 001' + (' · ОТКРЫТО' if closing else ''), 0.0)
    if s1:
        d.rectangle([6, y0, 6 + d.textlength(s1, font=R.F9) + 4, y0 + 11], fill=R.RGB['void2'])
        d.text((8, y0 + 1), s1, font=R.F9, fill=R.RGB['bone'])
    if not closing and rows23:                                # rows 2-3 step aside while a fact slams in

        s2 = typed('РУКОПИСЬ ВОЙНИЧА', 0.3, 40)
        if s2:
            w = d.textlength(s2, font=R.FCAP)
            d.rectangle([6, y0 + 14, 6 + w + 6, y0 + 32], fill=R.RGB['red']); d.text((9, y0 + 15), s2, font=R.FCAP, fill=R.RGB['bone'])
        s3 = typed('MS 408 · ЙЕЛЬ, США', 0.7, 40)
        if s3:
            w3 = d.textlength(s3, font=R.FCAP)
            d.rectangle([6, y0 + 35, 6 + w3 + 6, y0 + 53], fill=R.RGB['void2'])
            d.text((9, y0 + 36), s3, font=R.FCAP, fill=R.RGB['bone'])
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
    return t > 0.5 and (t % 3.9) < 0.1                    # never on the first frames: the eyes are the hook


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def pose_at(sg, t):
    cur = ('front', False, sg['t0'])
    for (when, pose, flip) in sg.get('poses', [(0, 'front', False)]):
        tw = sg['t0'] + when if isinstance(when, (int, float)) else word_time(sg, when) - 0.05
        if t >= tw:
            cur = (pose, flip, tw)
    return cur


# ---------------- kinetic type: facts slam onto the screen
def _txt_img(text, font, col, plate_col=None, pad=3):
    d0 = ImageDraw.Draw(Image.new('RGB', (4, 4)))
    w = int(d0.textlength(text, font=font)) + 2 * pad; h = font.size + 2 * pad + 2
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); d.fontmode = '1'
    if plate_col:
        d.rectangle([0, 0, w - 1, h - 1], fill=plate_col + (255,))
    else:
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (2, 2)):
            d.text((pad + ox, pad + oy - 1), text, font=font, fill=R.RGB['void'] + (255,))
    d.text((pad, pad - 1), text, font=font, fill=col + (255,))
    return np.array(im)


def _paste_rgba(rgb, im, cx, cy, scale=1.0, dx=0, dy=0):
    if scale != 1.0:
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    h, w = im.shape[:2]
    x0, y0 = int(cx - w / 2 + dx), int(cy - h / 2 + dy)
    gx0, gy0, gx1, gy1 = max(0, x0), max(0, y0), min(GW, x0 + w), min(GH, y0 + h)
    if gx1 <= gx0 or gy1 <= gy0:
        return rgb
    sub = im[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    m = sub[..., 3] > 0
    rgb[gy0:gy1, gx0:gx1][m] = sub[..., :3][m]
    return rgb


def slam(rgb, t, t0, text, small=None, y=92, big_font=None, red=False, life=1.7, f=0, maxw=GW - 8, cx=GW / 2):
    """a fact slams in from big and shakes; maxw = widest it may be once settled (grid px; the font steps down)"""
    k = t - t0
    if k < 0 or k > life:
        return rgb
    rgb = rgb.copy()
    big_font = big_font or R.FBIG
    sc = 1.0 + 0.9 * math.exp(-k / 0.05)                      # slams in from big
    shake = 3 * math.exp(-k / 0.08)
    r_ = np.random.default_rng(int(t * 30))
    dx, dy = (r_.random(2) - 0.5) * 2 * shake
    if k > life - 0.15:                                       # glitches out
        dx += (r_.random() - 0.5) * 30
    col = R.RGB['bone']
    im = _txt_img(text, big_font, col, R.RGB['red'] if red else None)
    if im.shape[1] > maxw:                                    # never wider than maxw: step the font down
        from PIL import ImageFont
        for size in (26, 22, 19, 16, 13):
            if size >= big_font.size:
                continue
            im = _txt_img(text, ImageFont.truetype(R.FCOND, size), col, R.RGB['red'] if red else None)
            if im.shape[1] <= maxw:
                break
    def keep_clear(c, w, yc, h):                              # rows >= 150 (900 px+): right edge left of the icon column
        return min(c, ICON_X - w / 2) if yc + h / 2 > 150 else c
    rgb = _paste_rgba(rgb, im, keep_clear(cx, im.shape[1], y, im.shape[0]), y, sc, dx, dy)
    if small and k > 0.12:
        n = int((k - 0.12) * 40)
        im_s = _txt_img(small, R.FCAP, R.RGB['bone'], R.RGB['red'])       # the full line fixes the position
        im2 = _txt_img(small[:n], R.FCAP, R.RGB['bone'], R.RGB['red'])
        c2 = keep_clear(cx, im_s.shape[1], y + 26, im_s.shape[0])
        rgb = _paste_rgba(rgb, im2, c2, y + 26, 1.0)
    return rgb


ICON_X = 146                                                  # grid x of the right icon column's inner edge (~880 px)


KIN_LIFE = {'count': 2.2}


def kin_alive(sg, t):
    """True while one of this segment's facts is on screen"""
    return any(0 <= t - (word_time(sg, w) - 0.04) <= KIN_LIFE.get(kd, 1.8) for (w, kd, _, _) in sg.get('kin', []))


def kin_draw(rgb, sg, t, f):
    for (word, kind, val, small) in sg.get('kin', []):
        t0 = word_time(sg, word) - 0.04
        if kind == 'count':
            k = t - t0
            if 0 <= k <= KIN_LIFE['count']:
                v = int(val * ease(k / 0.6))
                rgb = slam(rgb, t, t0, f'{v:03d}', small, y=110, life=KIN_LIFE['count'], f=f)
        else:
            y = sg.get('kin_y') or (110 if (sg['kind'] == 'lab' or SEG.index(sg) == 2) else 60)
            rgb = slam(rgb, t, t0, str(val), small, y=y, red=(kind == 'red'), life=1.8, f=f)
    return rgb


def hook_text(rgb, sg, t):
    """frame 0 already asks the question in words (muted autoplay, thumbnail): «ПОЧЕМУ ЕЁ / НЕЛЬЗЯ / ПРОЧИТАТЬ?»,
    fully drawn from the very first frame (no typing), centred left of the icon column, far above the captions.
    The key line is bone on a red plate, the others bone with a void outline. Stays until the title."""
    ht = sg.get('hooktext')
    if not ht:
        return rgb
    lines, key = ht
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    for n_, (s, y) in enumerate(zip(lines, (40, 59, 78))):
        w = d.textlength(s, font=R.FCAP); x = int(round(84 - w / 2))
        if n_ == key:
            d.rectangle([x - 3, y + 1, x + w + 3, y + 17], fill=R.RGB['red'])
        else:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
                d.text((x + ox, y + oy), s, font=R.FCAP, fill=R.RGB['void'])
        d.text((x, y), s, font=R.FCAP, fill=R.RGB['bone'])
    return np.array(pil)


# ---------------- the lab (wide)
def lab_grid(sg, t, f):
    amp = float(AMP[min(len(AMP) - 1, int(t * FPS))])
    pose, flip, tw = pose_at(sg, t)
    tl = t - sg['t0']
    on = 1.0
    if sg.get('teleport') and tl < sg['teleport']:
        on = max(0.0, tl / sg['teleport'])
    scr = kf(sg['screen'])
    if sg.get('screen2') and t >= word_time(sg, sg['screen2'][0]) - 0.3:
        scr = kf(sg['screen2'][1])
    if tl < 0.25:
        scr = static_img(f)
    vx = 0.0
    hx = sg['hx']; kw = {}; bob = int(round(math.sin(t * 2 * math.pi / 2.14)))
    if sg.get('walk'):                                       # he walks across the lab: steps bob, the body sways
        x0, x1, ta, tb = sg['walk']
        u = (tl - ta) / (tb - ta)
        hx = x0 + (x1 - x0) * ease(u)
        if 0 <= u <= 1:
            ph = (tl - ta) * 2 * math.pi * 1.7
            bob = -int(round(abs(math.sin(ph)) * 2))
            kw['tilt'] = int(round(math.sin(ph) * 1.5)) + 2
            vx = (x1 - x0) * 6 * u * (1 - u) / (tb - ta)
    if sg.get('turn'):                                       # the turn: squash to a sliver, flip, open up again
        tt = word_time(sg, sg['turn']) - 0.05
        d = t - tt
        if -0.12 <= d < 0:
            kw['sx'] = max(0.25, 1 + d / 0.12 * 0.75); pose, flip = 'front', False
        elif 0 <= d < 0.12:
            kw['sx'] = 0.25 + 0.75 * d / 0.12
    if sg.get('lean'):                                       # he leans toward us on a word
        lt = word_time(sg, sg['lean'][0])
        d = t - lt
        if 0 <= d < sg['lean'][1]:
            e = math.sin(math.pi * d / sg['lean'][1])
            kw['scale_f'] = 1 + 0.07 * e; kw['tilt'] = int(round(3 * e))
    if 0 <= t - tw < 0.1 and tl > 0.1:                        # anticipation squash on a gesture
        kw['sy'] = 0.96
    papers = None
    if sg.get('teleport') and tl < 1.4:
        papers = (min(1.0, tl / 1.4), hx, 236)
    flick = -1.0 if (int(t * 30) % 47 == 0) else 0.0
    kw['stress'] = HO.stress_at(sg.get('words'), t)           # his eyes / hood react exactly on the stressed words
    g = RM.frame(t, amp, 'calm', blink_at(t) and amp == 0, 0, screen=scr, host_cx=int(round(hx)), host_pose=pose,
                 host_flip=flip, host_on=on, bob=bob, host_kw=kw, papers=papers, flicker=flick, host_vx=vx)
    if 0 <= t - tw < 2 / FPS and tl > 0.1:
        g = R.rgbsplit(g, 1)
    sg['_hx'] = hx
    return g


def close_grid(sg, t, f, mood='calm', amp=None, kw=None, look=0, eyes_on=True, bg_screen=None):
    """close-up: the lab behind him, pushed in and darker; him big in front"""
    if amp is None:
        amp = float(AMP[min(len(AMP) - 1, int(t * FPS))])
    bg = RM.frame(t, 0, 'calm', screen=bg_screen, host_on=0)
    bg = cv2.resize(bg[40:200, 45:135], (GW, GH), interpolation=cv2.INTER_NEAREST)
    pi = HO._pal_index(bg)
    lut = np.array([HO._darker(i) for i in range(18)], np.int16)
    bg = HO.BGR[lut[lut[pi]]]
    bob = int(round(math.sin(t * 2 * math.pi / 2.14)))
    kw = dict(kw or {})
    kw.setdefault('stress', HO.stress_at(sg.get('words'), t))   # eyes / hood react exactly on the stressed words
    by = 160 - 88 + 273 + 2
    if kw.get('scale_f', 1) != 1:
        by += int(273 * (kw['scale_f'] - 1) * 0.45)
    HO.draw(bg, 90, by, t, amp if eyes_on else 0, 1, (blink_at(t) and amp == 0) or not eyes_on, look, bob, int(t * 15), mood,
            'close', aura=True, smoke=30, **kw)
    return bg


def finish(rgb_grid, f, zoom=1.0, cx=0.5, cy=0.45, rot=0.0, drift=True):
    big = cv2.resize(rgb_grid, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big *= scan; big += noise_bank[f % 8]
    big = np.clip(big, 0, 255).astype(np.uint8)
    if drift and zoom > 1.01:                                  # handheld: the camera is never perfectly still
        cx += 0.010 * math.sin(f * 0.071) + 0.004 * math.sin(f * 0.23)
        cy += 0.008 * math.sin(f * 0.053 + 1.0) + 0.003 * math.sin(f * 0.31)
        cx, cy = min(1, max(0, cx)), min(1, max(0, cy))
    if rot:
        zoom = max(zoom, 1.0 + abs(rot) * 0.035)
    if zoom > 1.001 or rot:
        w, h = OW / zoom, OH / zoom
        x0 = (OW - w) * cx; y0 = (OH - h) * cy
        M = np.float32([[zoom, 0, -x0 * zoom], [0, zoom, -y0 * zoom]])
        if rot:
            Rm = cv2.getRotationMatrix2D((OW / 2, OH / 2), rot, 1.0)
            M = (np.vstack([Rm, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)
        big = cv2.warpAffine(big, M, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=(10, 12, 11))
    return big


def reel_grids(a, b):
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{a:.3f}', '-i', REEL_SRC, '-t', f'{b - a:.3f}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE); n = OW * OH * 3; out = []
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
        out.append(fr[3::S, 3::S].copy())
    p.wait()
    return out


def pip(g, t, ft, y0=60):
    """his little live window in the corner while the story plays: he is the one telling it"""
    if not any(a <= ft <= b for a, b in NARR_SPANS):
        return g
    amp = reel_amp(ft)
    x0, w, h = 128, 46, 54
    box = HO.pip_box(t, amp, w, h)                            # breathing, glances, blinks, hot eyes on loud syllables
    g = g.copy()
    g[y0 - 1:y0 + h + 1, x0 - 1:x0 + w + 1] = HO.BGR[HO.D3]
    g[y0:y0 + h, x0:x0 + w] = box
    pil = Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil); d.fontmode = '1'
    d.rectangle([x0 - 1, y0 - 10, x0 + 30, y0 - 2], fill=R.RGB['red'] if int(t * 2) % 2 == 0 else R.RGB['red2'])
    d.text((x0 + 1, y0 - 10), '● 408', font=R.F7, fill=R.RGB['bone'])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ one frame of any segment (BGR grid + camera)
REELS = {}
CLIPS = {}
REEL_KEYS = [w['t0'] for ws in R._LINES for w in ws if w['key']]      # film times of the story's stressed words


def clip_grids(film, dur):
    key = (film, dur)
    if key not in CLIPS:
        CLIPS[key] = reel_grids(film, film + dur + 0.05)
    return CLIPS[key]


def seg_frame(i, t, f):
    """-> (BGR grid, zoom, cx, cy, rot)"""
    sg = SEG[i]
    tl = t - sg['t0']
    k = sg['kind']
    if k == 'reel':
        ft = sg['src'][0] + tl
        cx, cy = .5, sg.get('cy', .45)
        z = 1.0 + sg.get('push', 0.06) * ease(tl / sg['dur'])  # every story shot creeps in; the frozen museum frame more
        g = None
        for (st, film, du, oz0, oz1) in sg.get('over', []):    # picture-only cutaway: the reel's sound and captions go on
            if st <= tl < st + du:
                fr = clip_grids(film, du)
                if fr:
                    g = fr[min(len(fr) - 1, int((tl - st) * FPS))].copy()
                    if tl - st < 1.5 / FPS:
                        g = R.tear(g, np.random.default_rng(f), 4, 26)
                    z, cx, cy = oz0 + (oz1 - oz0) * (tl - st) / du, .5, .45
                break
        if g is None:
            fr = REELS[i]
            j = int(np.clip(round(tl * FPS), 0, len(fr) - 1))
            g = fr[j]
            if sg.get('mask'):                                   # hide the reel's own unreadable small print
                y0_, y1_, x0_, x1_ = sg['mask']
                g = g.copy(); sub = g[y0_:y1_, x0_:x1_]
                floor = np.percentile(sub.reshape(-1, 3), 35, axis=0).astype(np.uint8)
                g[y0_:y1_, x0_:x1_] = np.minimum(sub, floor[None, None, :])
        if sg.get('crawl') and sg['crawl'][0] <= ft < sg['crawl'][1]:   # «Буквы двигаются»: the letters crawl
            g = g.copy()
            for y in range(GH):
                s_ = int(2 * math.sin(y * 0.15 + t * 9))
                if s_:
                    g[y] = np.roll(g[y], s_, axis=0)
        if sg.get('tear_on') is not None and 0 <= ft - sg['tear_on'] < 2 / FPS:   # the last word rips the picture
            g = R.rgbsplit(R.tear(g, np.random.default_rng(f + 11), 5, 30), 2)
        for kt in REEL_KEYS:                                     # small camera punch on every stressed word of the story
            if 0 <= ft - kt < 0.35:
                z += 0.035 * math.exp(-(ft - kt) / 0.11)
        return g, z, cx, cy, 0.0
    for (w_, film, du) in sg.get('inserts', []):
        st = word_time(sg, w_) - 0.02
        if st <= t < st + du:
            fr = clip_grids(film, du)
            if not fr:
                break
            g = fr[min(len(fr) - 1, int((t - st) * FPS))].copy()
            if t - st < 1.5 / FPS:
                g = R.tear(g, np.random.default_rng(f), 4, 26)
            return g, 1.04 + 0.05 * (t - st) / du, .5, .45, 0.0
    if k == 'lab':
        g = lab_grid(sg, t, f)
        z0, z1 = sg.get('cam', (1.0, 1.0))
        z = z0 + (z1 - z0) * ease(tl / max(sg['dur'], 1e-3))
        for w in sg.get('words', []):
            if w['key'] and 0 <= t - w['t0'] < 0.3:
                z += 0.04 * math.exp(-(t - w['t0']) / 0.1)
        hx = sg.get('_hx', sg['hx'])
        return g, z, min(0.8, max(0.2, hx / GW)), .42, 0.0
    if k == 'close':
        look = 0
        for (w_, lv) in sg.get('looks', []):
            if t >= word_time(sg, w_):
                look = lv
        g = close_grid(sg, t, f, 'calm', None, look=look)
        z0, z1 = sg.get('cam', (1.0, 1.08))
        z = z0 + (z1 - z0) * ease(tl / sg['dur'])
        for w in sg.get('words', []):
            if w['key'] and 0 <= t - w['t0'] < 0.3:
                z += 0.035 * math.exp(-(t - w['t0']) / 0.1)
        return g, z, .5, .4, 0.0
    if k == 'hook':
        cap = caption_at(sg, t)
        # flash inserts and the fast montage: the scariest frames of the story, cut on his words
        for (st, du, film) in sg['flashes'] + sg['montage']:
            if st <= tl < st + du:
                fr = clip_grids(film, du)
                g = fr[min(len(fr) - 1, int((tl - st) * FPS))].copy()
                if tl - st < 1.5 / FPS:
                    g = R.tear(g, np.random.default_rng(f), 4, 26)
                return g, 1.06 + 0.05 * (tl - st) / du, .5, .45, 0.0
        cs, ca, cb = sg['clip'] if sg.get('clip') else (99, 0, 0)
        if cs <= tl < cs + (cb - ca):                           # «Ты слишком долго смотришь…»
            fr = clip_grids(ca, cb - ca)
            g = fr[min(len(fr) - 1, int((tl - cs) * FPS))]
            return g, 1.0 + 0.06 * (tl - cs) / (cb - ca), .5, .45, 0.0
        if tl >= sg['title']:                                    # the title slams in the dark, his eyes behind it
            g = HO.dark_portrait(t, f, eye_y=207)               # his eyes below the title, never behind it
            rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
            k2 = tl - sg['title']
            t1_, t2_ = sg.get('title_text', ('ДЕЛО 001', 'РУКОПИСЬ ВОЙНИЧА'))
            ts0 = sg['t0'] + sg['title']
            rgb = slam(rgb, t, ts0, t1_, None, y=100, life=9, f=f, maxw=132)
            if k2 > 0.25:
                rgb = slam(rgb, t, ts0 + 0.25, t2_, None, y=126, big_font=R.FCAP, red=True, life=9, f=f, maxw=132)
            if k2 > 0.45:                                        # the real book: readable, not 9-px small print
                rgb = slam(rgb, t, ts0 + 0.45, 'MS 408 · ЙЕЛЬ, США', None, y=146, big_font=R.FCAP, life=9, f=f, maxw=132)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 1.0 + 0.04 * k2, .5, .42, 0.0
        # him, close: dark until the eyes ignite, scary on «читали», a dutch tilt
        eyes_on = True                                          # frame 0 already has the red slits: it stops the scroll
        wt = word_time(sg, 'ЧИТАЛИ') if sg.get('words') else 1e9
        mood = 'scary' if (wt - 0.05 <= t or not sg.get('words')) else 'calm'
        if tl < 0.25:
            mood = 'scary'
        # frame 0 opens on the wide burning stare the card ends on; it narrows into the slits in 0.3 s
        g = close_grid(sg, t, f, mood, None if eyes_on else 0.0, eyes_on=eyes_on, kw=dict(stare=max(0.0, 1 - tl / 0.3)))
        if tl < 0.1:
            st_ = static_img(f, GW, GH); m = np.random.default_rng(f).random((GH, GW)) < (1 - tl / 0.1) * 0.2
            g[m] = st_[m]
        rot = 0.0; z = 1.30 - 0.18 * ease(tl / 1.0)
        if t >= wt - 0.05:
            d = t - wt
            z += 0.18 * (1 - math.exp(-d / 0.08))               # snap zoom into the eyes
            rot = -3.0 * min(1, d / 0.15)
        return g, z, .5, .33, rot
    if k == 'lean':
        # he leans right into the lens, the slits glance left and right as if the letters were moving, then burn
        u = tl / sg['dur']
        look = -1 if 0.15 < u < 0.4 else (1 if 0.4 <= u < 0.62 else 0)
        mood = 'scary' if u > 0.66 else 'calm'
        g = close_grid(sg, t, f, mood, 0.0, kw=dict(scale_f=1.0 + 0.16 * ease(u)), look=look)
        return g, 1.04 + 0.12 * ease(u), .5, .36, -2.5 * ease((u - 0.6) / 0.4)
    if k == 'dive':
        g = RM.frame(t, 0, 'scary', False, 0, screen=static_img(f) if tl < 0.6 else REELS[i + 1][0], host_cx=50,
                     host_pose='point', host_flip=False)
        return g, 1.0, .5, .45, 0.0
    if k == 'card':
        # hard cut to black: his eyes in the dark, the last words slam in (text_layer). It is framed exactly like frame 0
        # of the hook (same close-up, same eyes, z 1.30 / cy .33 at the end) so the loop back to the start is one image.
        # HO.final_card: he creeps closer, the eyes go out, then SNAP wide open - the cut lands on that stare, and
        # frame 0 of the hook opens on the same stare at the same place (eye row 162 = close_grid's, camera 1.30/.5/.33)
        hit = sg.get('hit', 0.45)                                 # «ОНА — НЕТ.» lands
        g = HO.final_card(t, tl, sg['dur'], f, eye_y=162)
        if tl < 2 / FPS:
            g = R.tear(g, np.random.default_rng(f), 5, 40)
        d = tl - hit
        if 0 <= d < 2 / FPS:
            g = R.rgbsplit(R.tear(g, np.random.default_rng(f + 3), 4, 22), 2)
        z = 1.30
        if 0 <= d:
            z += 0.06 * math.exp(-d / 0.09)                       # punch on the hit, settled long before the loop
        return g, z, .5, .33, 0.0
    if k == 'outro':
        g = close_grid(sg, t, f, 'scary', 0.0, bg_screen=kf('s03_eye'))
        rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        rgb = plate(rgb, t, sg['t0'] + 0.2, closing=True)
        rgb = question(rgb, t, sg['t0'] + 0.6)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 1.03 + 0.06 * ease(tl / sg['dur']), .5, .4, 0.0
    return np.zeros((GH, GW, 3), np.uint8), 1.0, .5, .45, 0.0


KEY = (255, 0, 255)


def text_layer(t, f):
    """captions, facts and the case plate for time t, on a magenta key (RGB grid) - laid over the frame after the
    camera move, so the camera never crops the words"""
    global CAP_BOX
    CAP_BOX = None
    i, sg = seg_at(t)
    k = sg['kind']
    tl = t - sg['t0']
    def cap(rgb, c):                                          # caption + remember its box for the scrim
        global CAP_BOX
        before = rgb.copy()
        rgb = R.draw_caption(rgb, c, t)
        ys, xs = np.nonzero((rgb != before).any(2))
        if len(ys):                                           # snap to the line grid: the box never twitches
            yb = GH - R.CAP_BOTTOM
            n1 = max(0, (yb - int(ys.min()) + 5) // 19)
            CAP_BOX = (yb - 19 * n1 - 1, yb + 20, int(xs.min()), int(xs.max()) + 1)
        return rgb
    if k == 'reel':
        rgb = np.zeros((GH, GW, 3), np.uint8); rgb[:] = KEY
        if sg.get('pip'):                                     # his live window rides above the camera: never cropped
            rgb = cv2.cvtColor(pip(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), t, sg['src'][0] + tl, sg.get('pip_y', 60)),
                               cv2.COLOR_BGR2RGB)
        for (st, big, small, y, life) in sg.get('slams', []):
            rgb = slam(rgb, t, sg['t0'] + st, big, small, y=y, life=life, f=f)
        if sg.get('idplate') is not None:
            rgb = id_plate(rgb, tl - sg['idplate'])
        return cap(rgb, R.caption_at(sg['src'][0] + tl))
    if k == 'card':                                           # above the camera, so the zoom never crops the words
        rgb = np.zeros((GH, GW, 3), np.uint8); rgb[:] = KEY
        l1, l2 = sg['lines']
        rgb = slam(rgb, t, sg['t0'] + 0.06, l1, None, y=100, life=99, f=f, maxw=132)
        rgb = slam(rgb, t, sg['t0'] + sg.get('hit', 0.45), l2, None, y=126, red=True, life=99, f=f, maxw=132)
        return rgb
    if k == 'lean':
        if not sg.get('slams'):
            return None
        rgb = np.zeros((GH, GW, 3), np.uint8); rgb[:] = KEY
        for (st, big, small, y, life) in sg['slams']:
            rgb = slam(rgb, t, sg['t0'] + st, big, small, y=y, life=life, f=f)
        return rgb
    if k not in ('lab', 'close', 'hook'):
        return None
    if k == 'hook' and (sg.get('title') is not None and t - sg['t0'] >= sg['title']):
        return None
    if k == 'hook' and sg.get('clip'):
        cs, ca, cb = sg['clip']
        if cs <= t - sg['t0'] < cs + (cb - ca):
            return None
    rgb = np.zeros((GH, GW, 3), np.uint8); rgb[:] = KEY
    if i in (1, 2):                                           # the case plate stays up over his first two lines
        rgb = plate(rgb, t, SEG[1]['t0'] + 0.3, rows23=not kin_alive(sg, t))
    rgb = kin_draw(rgb, sg, t, f)
    if k == 'hook':
        rgb = hook_text(rgb, sg, t)
    return cap(rgb, caption_at(sg, t))


CAP_BOX = None


def id_plate(rgb, k):
    """over the real book in its case: «MS 408» + «ЙЕЛЬ, США», typed in, above the case and left of his window"""
    if k < 0:
        return rgb
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    s1 = 'MS 408'[:int(k * 24) + 1]
    w = d.textlength(s1, font=R.FBIG); x = 8
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (2, 2)):
        d.text((x + ox, 36 + oy), s1, font=R.FBIG, fill=R.RGB['void'])
    d.text((x, 36), s1, font=R.FBIG, fill=R.RGB['bone'])
    s2 = 'ЙЕЛЬ, США'[:max(0, int((k - 0.25) * 30))]
    if s2:
        w2 = d.textlength(s2, font=R.FCAP)
        d.rectangle([x - 2, 72, x + w2 + 3, 90], fill=R.RGB['red']); d.text((x + 1, 72), s2, font=R.FCAP, fill=R.RGB['bone'])
    return np.array(pil)


def with_text(frame, t, f):
    lay = text_layer(t, f)
    if lay is None:
        return frame
    m = ~((lay[..., 0] == 255) & (lay[..., 1] == 0) & (lay[..., 2] == 255))
    if not m.any():
        return frame
    big = cv2.resize(lay, (OW, OH), interpolation=cv2.INTER_NEAREST)
    mb = cv2.resize(m.astype(np.uint8), (OW, OH), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = frame.copy()
    if CAP_BOX is not None:                                   # scrim: the reel's HUD / white parchment never cut the words
        y0, y1, x0, x1 = CAP_BOX
        y0, x0 = max(0, y0 - 2), max(0, x0 - 4); y1, x1 = min(GH, y1 + 1), min(GW, x1 + 4)
        k_ = 0.35 if seg_at(t)[1]['kind'] == 'reel' else 0.6
        sub = out[y0 * S:y1 * S, x0 * S:x1 * S].astype(np.float32) * k_
        out[y0 * S:y1 * S, x0 * S:x1 * S] = sub.astype(np.uint8)
    out[mb] = (big[mb].astype(np.float32) * scan[np.nonzero(mb)[0]][:, 0] ).astype(np.uint8)
    return out


def dive_frame(i, t, f):
    sg = SEG[i]; tl = t - sg['t0']
    g, _, _, _, _ = seg_frame(i, t, f)
    big = finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, drift=False)
    k = np.clip((tl - 0.55) / (sg['dur'] - 0.55), 0, 1)
    if k <= 0:
        return big
    e = k * k * (3 - 2 * k)
    BX0, BY0, BX1, BY1 = RM.GLASS['big']
    gx0, gy0, gx1, gy1 = BX0 * S, BY0 * S, BX1 * S, BY1 * S
    gw, gh = gx1 - gx0, gy1 - gy0
    nxt = cv2.cvtColor(cv2.resize(REELS[i + 1][0], (OW, OH), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
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
        if sg['kind'] == 'reel':
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
        for (j0, j1) in ((i - 1, i), (i, i + 1)):
            if j0 < 0 or j1 >= len(SEG) or not SEG[j0].get('trans_out'):
                continue
            kind, n = SEG[j0]['trans_out']
            fc = int(round(SEG[j1]['t0'] * FPS)); pre = n // 2
            if not (fc - pre <= f < fc - pre + n):
                continue
            prog = (f - (fc - pre) + 1) / (n + 1)
            ga, za, cxa, cya, ra = seg_frame(j0, min(t, SEG[j0]['t1'] - 1 / FPS), f)
            gb, zb, cxb, cyb, rb = seg_frame(j1, max(t, SEG[j1]['t0']), f)
            g = R.run_trans(kind, ga, gb, prog, j1, rng)
            frame = finish(cv2.cvtColor(g, cv2.COLOR_BGR2RGB), f, za if prog < .5 else zb, cxa if prog < .5 else cxb,
                           cya if prog < .5 else cyb, ra if prog < .5 else rb)
            break
        if frame is None:
            if sg['kind'] == 'dive':
                frame = dive_frame(i, t, f)
            else:
                g, z, cx, cy, rot = seg_frame(i, t, f)
                if sg['kind'] == 'reel' and t - sg['t0'] < 2 / FPS and SEG[i - 1]['kind'] == 'dive':
                    g = R.tear(g, rng, 4, 30)
                rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
                if sg['kind'] == 'outro' and t > T_END - 0.5:
                    rgb = (rgb.astype(np.float32) * max(0, (T_END - t) / 0.5)).astype(np.uint8)
                frame = finish(rgb, f, z, cx, cy, rot, drift=sg['kind'] != 'reel')
        frame = with_text(frame, t, f)
        p.stdin.write(frame.tobytes())
        if f % 30 == 0:
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
    ref = load(f'{E}/mix_B_C.wav', 47.5, 50.5); raw = load(f'{A}/track2.mp3', 54.2, 57.2)
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
        if sg['kind'] == 'hook' and sg.get('clip'):
            cs, ca, cb = sg['clip']
            gate[int((sg['t0'] + cs) * SR):int((sg['t0'] + cs + cb - ca) * SR)] = 0
        if sg['kind'] == 'card':
            gate[int(sg['t0'] * SR):int(sg['t1'] * SR)] = 0
    gate = np.convolve(gate, np.ones(int(0.12 * SR)) / int(0.12 * SR), 'same')
    mix += bed * (duck * gate)[:, None]
    mix += vo * 1.35
    # the story shots carry their own finished sound
    for sg in SEG:
        if sg['kind'] == 'reel':
            a, b = sg['src']
            place(mix, fade(load(f'{E}/mix_B_C.wav', a, b), 0.015, 0.04), sg['t0'])
    # sound for every cut and move
    hk = SEG[0]
    place(mix, synth_static(0.12, 0.28, 1), 0.0)
    place(mix, crushed_kick(0.6), hk['t0'] + 0.1)   # the eyes ignite
    for n_, (st, du, film) in enumerate(hk['flashes'] + hk['montage']):
        place(mix, crushed_kick(0.16), hk['t0'] + st)
        place(mix, synth_static(0.12, 0.10, 30 + n_), hk['t0'] + st)
        x = load(f'{E}/mix_B_C.wav', film, film + du)            # a breath of each shot's own sound
        place(mix, fade(x, 0.01, 0.05) * 0.2, hk['t0'] + st)
    if hk.get('clip'):
        cs, ca, cb = hk['clip']
        place(mix, fade(load(f'{E}/mix_B_C.wav', ca, cb), 0.01, 0.08), hk['t0'] + cs)
    place(mix, swell(1.1, 0.26, 8), hk['t0'] + hk['title'] - 1.1)
    place(mix, crushed_kick(0.55), hk['t0'] + hk['title'])
    n = int(1.4 * SR); tt = np.arange(n) / SR
    sub = np.sin(2 * np.pi * np.cumsum(55 * np.exp(-tt * 1.5) + 28) / SR) * np.exp(-tt * 2.2) * 0.30
    place(mix, np.stack([sub, sub], 1).astype(np.float32), hk['t0'] + hk['title'])
    place(mix, crushed_kick(0.4), hk['t0'] + hk['title'] + 0.35)
    for sg in SEG:
        for (word, kind, val, small) in sg.get('kin', []):        # every fact lands with a hit
            place(mix, crushed_kick(0.3), word_time(sg, word) - 0.04)
            place(mix, synth_static(0.08, 0.1, 40), word_time(sg, word) - 0.04)
        if sg.get('walk'):                                        # heavy boots on the floor
            x0, x1, ta, tb = sg['walk']
            k_ = 0
            while ta + k_ / 3.4 < tb:
                st = sg['t0'] + ta + k_ / 3.4
                n = int(0.18 * SR); tt = np.arange(n) / SR
                thud = (np.sin(2 * np.pi * (70 + 30 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 28) +
                        np.random.default_rng(k_).normal(0, 1, n) * np.exp(-tt * 60) * 0.25) * 0.32
                pan = 0.5 + 0.4 * ((x0 + (x1 - x0) * (k_ / 3.4) / (tb - ta)) / GW - 0.5)
                place(mix, np.stack([thud * (1 - pan) * 2, thud * pan * 2], 1).astype(np.float32), st)
                k_ += 1
        for (w_, film, du) in sg.get('inserts', []):
            place(mix, crushed_kick(0.14), word_time(sg, w_) - 0.02)
            place(mix, synth_static(0.08, 0.08, 70), word_time(sg, w_) - 0.02)
        if sg.get('turn'):
            place(mix, whoosh(0.3, 0.07, 61), word_time(sg, sg['turn']) - 0.2)
        if sg['kind'] == 'card':                                  # the hard cut: a hit, then a low hum in the dark
            place(mix, crushed_kick(0.55), sg['t0'])
            place(mix, synth_static(0.2, 0.22, 91), sg['t0'])
            n = int(sg['dur'] * SR); tt = np.arange(n) / SR
            dr = (np.sin(2 * np.pi * 41 * tt) * 0.6 + np.sin(2 * np.pi * 61.5 * tt) * 0.3) * np.minimum(1, tt / 0.1) * np.maximum(0, 1 - tt / sg['dur']) * 0.2
            place(mix, np.stack([dr, dr], 1).astype(np.float32), sg['t0'])
            place(mix, crushed_kick(0.4), sg['t0'] + 0.38)
        if sg['kind'] == 'lean':
            wh = load(f'{A}/vo_whisper.mp3', af='areverse,aecho=0.8:0.9:120|260:0.5|0.35,lowpass=f=3000')
            n = min(len(wh), int(sg['dur'] * SR))
            wh = wh[:n] * np.linspace(0.2, 1.0, n)[:, None] * 0.4
            place(mix, wh, sg['t0'])
            place(mix, swell(sg['dur'], 0.12, 12), sg['t0'])
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
            place(mix, synth_static(0.6, 0.14, 77), sg['t0'])
            place(mix, swell(sg['dur'] - 0.4, 0.14), sg['t0'] + 0.4)
            place(mix, crushed_kick(0.30), sg['t1'] - 0.05)
        if sg['kind'] == 'outro':
            place(mix, synth_static(0.4, 0.25, 11), sg['t0'])
            place(mix, crushed_kick(0.6), sg['t0'] + 0.02)
            n = int(sg['dur'] * SR); tt = np.arange(n) / SR
            dr = (np.sin(2 * np.pi * 41 * tt) * 0.6 + np.sin(2 * np.pi * 61.5 * tt) * 0.3) * np.minimum(1, tt / 0.4) * np.maximum(0, 1 - tt / sg['dur']) * 0.18
            place(mix, np.stack([dr, dr], 1).astype(np.float32), sg['t0'])
    mix = mix[:int(T_END * SR)]
    # master: equal dialogue loudness in every segment, sfx held near the dialogue, -14 LUFS, <= -1 dBTP (ep_level.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ep_level import level
    mix = level(mix, SEG, SR, voice=(vo * 1.35)[:len(mix)])
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
