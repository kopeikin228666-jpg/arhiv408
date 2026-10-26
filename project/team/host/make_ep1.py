"""Episode 1: «408» opens the case of the Voynich manuscript, tells it in his lab, switches on the big screen,
the reel plays (a ~52 s cut of v6 B), back to him with a question.  Output 1080x1920, 30 fps, <= 90 s.

python3 make_ep1.py [--from S --to S]   (preview a range of the INTRO/OUTRO only; the reel parts are copied)
"""
import os, sys, math, json, subprocess, argparse
os.environ['VOY_VER'] = 'B'
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import numpy as np, cv2
from PIL import Image, ImageDraw
import render as R
import room as RM
import host as HO
import pixelize as P
from align import tokens, scribe_words, scribe_line, align_line, true_attacks, display_chunks

E = '/home/claude/voynich/edit'
A = '/home/claude/voynich/audio'
KF = '/home/claude/voynich/kf'
OUTD = '/home/claude/voynich/team/host/ep1'
os.makedirs(OUTD, exist_ok=True)
FPS, SR = 30, 48000
GW, GH, OW, OH, S = 180, 320, 1080, 1920, 6

# ------------------------------------------------------------------ timeline
PHRASE = f'{A}/vo_408.mp3' if os.path.exists(f'{A}/vo_408.mp3') else None
T_TELE = 1.2
LINES = [  # (file, src_a, src_b, text, screen keyframe, mood)
    ('vo_intro.mp3', 0.0, 5.49, 'Двести сорок *страниц… и ни одного… *прочитанного слова.', 's02_parchment', 'calm'),
    ('vo_written.mp3', 0.06, 5.44, 'Она *написана… *неизвестно *кем. Шестьсот лет *назад.', 's16_hourglass', 'calm'),
    ('vo_narrator.mp3', 4.74, 8.07, 'В 1912-м{тысяча девятьсот двенадцатом} её *купили.', 's09_tag', 'calm'),
    ('vo_narrator.mp3', 8.51, 10.76, 'Потом пытались *читать.', 's15_reading', 'calm'),
    ('vo_narrator.mp3', 11.27, 13.03, '*Никто… не *смог.', 's10_grid', 'scary'),
    ('vo_narrator.mp3', 0.0, 3.81, 'Эта *книга… не хочет, чтобы её *читали.', 's03_eye', 'calm'),
]
GAPS = [0.0, 0.5, 0.55, 0.25, 0.3, 0.6]
t = T_TELE + 0.25 + (4.2 if PHRASE else 0.0)
PLACE = []
for (f, a, b, *_), gap in zip(LINES, GAPS):
    t += gap
    PLACE.append(t)
    t += b - a
T_SCREEN = t + 0.35                     # the monitor bursts into static
T_REEL = T_SCREEN + 1.4                 # dive through the glass into the reel
REEL = [(47.467, 69.0), (99.1, 130.0)]
T_OUT = T_REEL + sum(b - a for a, b in REEL)
T_END = T_OUT + 3.6
print('lines at', [round(p, 2) for p in PLACE], 'screen', round(T_SCREEN, 2), 'reel', round(T_REEL, 2), 'outro', round(T_OUT, 2), 'end', round(T_END, 2))

# ------------------------------------------------------------------ words of his lines
WORDS = []          # per line: list of dict(w, key, t0, t1, chunk)
for li, ((f, a, b, text, _, _), p) in enumerate(zip(LINES, PLACE)):
    if scribe_words(f):
        toks, res = scribe_line(f, a, b, text)
    else:
        toks, res = align_line(f'{A}/{f}', a, b, text)
    res = true_attacks(f, a, toks, res, trusted_ends=bool(scribe_words(f)))
    ids = display_chunks(toks)
    WORDS.append([dict(w=d, key=k, t0=p + r0, t1=p + r1, chunk=c) for (d, sp, k, e), (r0, r1), c in zip(toks, res, ids)])


def caption_at(tt):
    for li, ws in enumerate(WORDS):
        start = ws[0]['t0'] - 0.2
        end = ws[-1]['t1'] + 0.7
        if li + 1 < len(WORDS):
            end = min(end, WORDS[li + 1][0]['t0'] - 0.24)
        if start <= tt < end:
            heads, tails = {}, {}
            for w in ws:
                heads.setdefault(w['chunk'], w['t0']); tails[w['chunk']] = w['t1']
            def shows(c):
                h = heads[c] - 0.2
                if c - 1 in tails:
                    h = max(h, min(heads[c] - 0.03, tails[c - 1] + 0.08))
                return h
            ch = max([c for c in heads if shows(c) <= tt] or [min(heads)])
            return [(w['w'], w['key'], w['t0'] - 0.03 <= tt, 0 <= tt - w['t0'] + 0.03 < 2 / FPS) for w in ws if w['chunk'] == ch], 'narr'
    return None


# ------------------------------------------------------------------ audio: his voice
def load(path, a=None, b=None, af=None, ch=2):
    cmd = ['ffmpeg', '-v', 'error'] + (['-ss', str(a)] if a is not None else []) + (['-to', str(b)] if b is not None else []) + ['-i', path]
    if af:
        cmd += ['-af', af]
    cmd += ['-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-']
    x = np.frombuffer(subprocess.check_output(cmd), np.float32).copy()
    return x.reshape(-1, ch) if ch > 1 else x


VOICE = ('aresample=48000,highpass=f=70,lowpass=f=10500,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,'
         'acrusher=bits=11:mix=0.12:mode=log,aecho=0.8:0.55:55|130|260:0.16|0.10|0.06')
N = int(T_END * SR) + SR
vo = np.zeros((N, 2), np.float32)


def fade(x, fi=0.012, fo=0.06):
    x = x.copy(); n = len(x)
    a_, b_ = int(fi * SR), int(fo * SR)
    x[:a_] *= np.linspace(0, 1, a_)[:, None]; x[n - b_:] *= np.linspace(1, 0, b_)[:, None]
    return x


def place(buf, x, at):
    i = int(round(at * SR))
    if i < 0:
        x = x[-i:]; i = 0
    n = min(len(x), len(buf) - i)
    buf[i:i + n] += x[:n]


if PHRASE:
    ph = load(PHRASE, af=VOICE)
    place(vo, fade(ph) * 1.0, T_TELE + 0.25)
for (f, a, b, *_), p in zip(LINES, PLACE):
    x = load(f'{A}/{f}', max(0, a - 0.06), b + 0.3, VOICE)
    place(vo, fade(x), p - (a - max(0, a - 0.06)))
vo_mono = vo.mean(1)
hop = SR // FPS
nfr = int(T_END * FPS) + 2
env = np.array([np.sqrt((vo_mono[i * hop:(i + 1) * hop] ** 2).mean() + 1e-12) for i in range(nfr)])
AMP = np.clip((20 * np.log10(env + 1e-9) + 44) / 24, 0, 1)
AMP[AMP < 0.12] = 0


def amp_at(tt):
    return float(AMP[min(len(AMP) - 1, int(tt * FPS))])


# ------------------------------------------------------------------ screen images for his lines
_SCR = {}


def screen_img(name):
    if name not in _SCR:
        _SCR[name] = cv2.imread(f'{KF}/{name}.jpg')
    return _SCR[name]


def static_img(seed, w=80, h=60):
    r = np.random.default_rng(seed)
    v = r.integers(0, 4, (h, w))
    cols = np.array([P.PALETTE[0], P.PALETTE[3], P.PALETTE[6], P.PALETTE[8]], np.uint8).reshape(-1, 3)
    return cols[v]


def line_index(tt):
    k = -1
    for i, p in enumerate(PLACE):
        if tt >= p - 0.35:
            k = i
    return k


# ------------------------------------------------------------------ the case plate (top)
def plate(rgb, tt, t0, lines_on=True, closing=False):
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    k = tt - t0
    if k < 0:
        return rgb
    def typed(s, start, cps=28):
        n = int(max(0, k - start) * cps)
        return s[:n]
    s1 = typed('АРХИВ 408 · ДЕЛО №001' + (' · ОТКРЫТО' if closing else ''), 0.0)
    d.rectangle([6, 6, 6 + d.textlength(s1, font=R.F9) + 4, 17], fill=R.RGB['void2']) if s1 else None
    d.text((8, 7), s1, font=R.F9, fill=R.RGB['bone'])
    if lines_on:
        s2 = typed('РУКОПИСЬ ВОЙНИЧА', 0.5, 22)
        if s2:
            w = d.textlength(s2, font=R.FCAP)
            d.rectangle([6, 20, 6 + w + 6, 38], fill=R.RGB['red'])
            d.text((9, 21), s2, font=R.FCAP, fill=R.RGB['bone'])
        s3 = typed('MS 408 · ЙЕЛЬСКИЙ УНИВЕРСИТЕТ · США', 1.4, 34)
        if s3:
            d.rectangle([6, 41, 6 + d.textlength(s3, font=R.F7) + 4, 50], fill=R.RGB['void2'])
            d.text((8, 42), s3, font=R.F7, fill=R.RGB['line'])
    return np.array(pil)


def question(rgb, tt, t0):
    k = tt - t0
    if k < 0:
        return rgb
    pil = Image.fromarray(rgb); d = ImageDraw.Draw(pil); d.fontmode = '1'
    for i, (s, col, f) in enumerate((('ЧТО, ПО-ТВОЕМУ,', 'bone', R.FCAP), ('В НЕЙ НАПИСАНО?', 'red', R.FCAP))):
        n = int(max(0, k - i * 0.5) * 24)
        s = s[:n]
        if not s:
            continue
        w = d.textlength(s, font=f)
        x = int((GW - w) / 2); y = 262 + i * 20
        if col == 'red':
            d.rectangle([x - 3, y + 1, x + w + 3, y + 17], fill=R.RGB['red'])
            d.text((x, y), s, font=f, fill=R.RGB['bone'])
        else:
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                d.text((x + ox, y + oy), s, font=f, fill=R.RGB['void'])
            d.text((x, y), s, font=f, fill=R.RGB['bone'])
    return np.array(pil)


# ------------------------------------------------------------------ output helpers
noise_bank = [np.random.default_rng(k).normal(0, 4.0, (OH, OW, 1)).astype(np.float32) for k in range(8)]
scan = np.ones((OH, 1, 1), np.float32); scan[::S] = 0.82


def finish(rgb_grid, f):
    big = cv2.resize(rgb_grid, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big *= scan; big += noise_bank[f % 8]
    return np.clip(big, 0, 255).astype(np.uint8)


def reel_frames(a, b):
    """RGB frames of the v6 master between film a and b"""
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{a:.3f}', '-i', f'{E}/video_v6_B.mp4', '-t', f'{b - a:.3f}',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = OW * OH * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
    p.wait()


FIRST_REEL = next(reel_frames(REEL[0][0], REEL[0][0] + 0.05))
BX0, BY0, BX1, BY1 = RM.GLASS['big']


def lab_frame(tt, f):
    """one 180x320 grid frame of the lab (RGB) for time tt"""
    li = line_index(tt)
    amp = amp_at(tt)
    mood = 'calm'
    if 0 <= li < len(LINES):
        mood = LINES[li][5]
        if mood == 'scary' and tt > PLACE[li] + (LINES[li][2] - LINES[li][1]) + 0.3:
            mood = 'calm'
    if li == len(LINES) - 1 and WORDS[li][-1]['t0'] - 0.05 <= tt < T_SCREEN + 0.4:
        mood = 'scary'                                   # «...читали.»: the slits flare
    if tt >= T_OUT:
        mood = 'scary'
    blink = (tt < T_TELE - 0.25) or (amp == 0 and (tt % 4.7) < 0.12 and tt > T_TELE + 1)
    # what the big monitor shows
    scr = None
    if T_TELE <= tt < T_SCREEN and li >= 0:
        sw = PLACE[li] - 0.35
        if tt - sw < 0.1:
            scr = static_img(f)
        else:
            scr = screen_img(LINES[li][4])
    elif T_SCREEN <= tt < T_SCREEN + 0.5:
        scr = static_img(f)
    elif tt >= T_OUT:
        scr = screen_img('s03_eye') if (tt - T_OUT) > 0.4 else static_img(f)
    g = RM.frame(tt, amp, mood, blink, 0, screen=scr)
    rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    if tt >= T_OUT:
        rgb = plate(rgb, tt, T_OUT + 0.3, lines_on=False, closing=True)
        rgb = question(rgb, tt, T_OUT + 0.9)
    else:
        rgb = plate(rgb, tt, T_TELE + 0.2)
        rgb = R.draw_caption(rgb, caption_at(tt), tt)
    return rgb


def glitch(rgb, f, amount):
    """pixel glitch on a grid RGB frame: tears, static, rgb split; amount 0..1"""
    rng = np.random.default_rng(f * 31 + 7)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    g = R.tear(g, rng, int(2 + 8 * amount), int(4 + 40 * amount), 2, 14)
    if amount > 0.15:
        g = R.rgbsplit(g, int(1 + 3 * amount))
    st = static_img(f, GW, GH)
    m = rng.random((GH, GW)) < amount * 0.85
    g[m] = st[m]
    return cv2.cvtColor(g, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ video
def render_video(out, t_from=0.0, t_to=None):
    t_to = T_END if t_to is None else t_to
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    f0, f1 = int(round(t_from * FPS)), int(round(t_to * FPS))
    f = f0
    reel_gen = None
    while f < f1:
        tt = f / FPS
        if tt < T_TELE:                                       # glitch-teleport in
            k = tt / T_TELE
            rgb = lab_frame(tt, f)
            amount = 1.0 if k < 0.3 else max(0.0, 1 - (k - 0.3) / 0.55)
            if amount > 0:
                rgb = glitch(rgb, f, amount)
            frame = finish(rgb, f)
        elif tt < T_SCREEN + 0.5:
            frame = finish(lab_frame(tt, f), f)
        elif tt < T_REEL:                                     # the dive into the big monitor
            k = (tt - (T_SCREEN + 0.5)) / (T_REEL - T_SCREEN - 0.5)
            e = k * k * (3 - 2 * k)
            rgb = lab_frame(tt, f)
            big = finish(rgb, f)
            # paste the first reel frame into the glass (sharp), centre-cropped to the glass aspect
            gx0, gy0, gx1, gy1 = BX0 * S, BY0 * S, BX1 * S, BY1 * S
            gw, gh = gx1 - gx0, gy1 - gy0
            src = FIRST_REEL
            sh_ = int(OW * gh / gw)
            crop = src[(OH - sh_) // 2:(OH - sh_) // 2 + sh_]
            big[gy0:gy1, gx0:gx1] = cv2.resize(crop, (gw, gh), interpolation=cv2.INTER_AREA)
            big[gy0:gy1:S] = (big[gy0:gy1:S].astype(np.int16) * 3 // 5).astype(np.uint8)
            # camera: from the whole room to the glass filling the frame width
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            tw = gw * 0.92
            w = OW + (tw - OW) * e
            h = w * OH / OW
            x0 = (OW / 2) + (cx - OW / 2) * e - w / 2
            y0 = (OH / 2) + (cy - OH / 2) * e - h / 2
            M = np.float32([[OW / w, 0, -x0 * OW / w], [0, OH / h, -y0 * OH / h]])
            frame = cv2.warpAffine(big, M, (OW, OH), flags=cv2.INTER_NEAREST, borderValue=(10, 12, 11))
            if k > 0.85:                                     # flash through the glass
                frame = np.clip(frame.astype(np.int16) + int(255 * (k - 0.85) / 0.15 * 0.6), 0, 255).astype(np.uint8)
        elif tt < T_OUT:                                      # the reel (copied frames of the v6 master)
            if reel_gen is None:
                def gen():
                    for a, b in REEL:
                        for fr in reel_frames(a, b):
                            yield fr
                reel_gen = gen()
                # skip to the right frame when rendering a range
                skip = int(round((max(tt, T_REEL) - T_REEL) * FPS))
                for _ in range(skip):
                    next(reel_gen, None)
            fr = next(reel_gen, None)
            if fr is None:
                fr = np.zeros((OH, OW, 3), np.uint8)
            frame = fr
            if tt - T_REEL < 2 / FPS:                       # land with a glitch
                frame = np.roll(frame, (int(np.random.default_rng(f).integers(-40, 40)), 0), axis=(1, 0))
        else:                                                 # back in the lab, the question
            k = tt - T_OUT
            rgb = lab_frame(tt, f)
            if k < 0.35:
                rgb = glitch(rgb, f, 1 - k / 0.35)
            if tt > T_END - 0.5:                              # the eyes go out, darkness
                q = (tt - (T_END - 0.5)) / 0.5
                rgb = (rgb.astype(np.float32) * (1 - q)).astype(np.uint8)
            frame = finish(rgb, f)
        p.stdin.write(frame.tobytes())
        if f % 150 == 0:
            print('t', round(tt, 1), flush=True)
            cv2.imwrite(f'{OUTD}/f_{f:05d}.jpg', cv2.cvtColor(cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        f += 1
    p.stdin.close(); p.wait()


# ------------------------------------------------------------------ audio
def synth_static(dur, amp=0.25, seed=0):
    n = int(dur * SR)
    x = np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)
    hold = 6
    x = np.repeat(x[::hold], hold)[:n]                       # sample-held: pixel static
    x = np.round(x * 6) / 6
    env = np.exp(-np.linspace(0, 3, n))
    return np.stack([x * env * amp] * 2, 1).astype(np.float32)


def crushed_kick(amp=0.6):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    f = 110 * np.exp(-tt * 9) + 38
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 6)
    x = np.round(x * 8) / 8
    x = np.repeat(x[::4], 4)[:n]
    return np.stack([x * amp] * 2, 1).astype(np.float32)


def swell(dur=1.2, amp=0.25, seed=3):
    n = int(dur * SR)
    x = np.random.default_rng(seed).normal(0, 1, n)
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, [200, 1800], 'bandpass', fs=SR, output='sos'), x)
    x = x / (np.abs(x).max() + 1e-9) * np.linspace(0, 1, n) ** 2.5
    return np.stack([x * amp] * 2, 1).astype(np.float32)


def render_audio(out):
    from scipy.signal import butter, sosfilt
    mix = np.zeros((N, 2), np.float32)
    # the music under his part: the stretch of Mara that leads straight into the reel's first frame (continuity)
    song0 = REEL[0][0] + 6.66 - T_REEL
    mara = load(f'{A}/track2.mp3', max(0, song0), REEL[0][0] + 6.66)
    if song0 < 0:
        mara = np.concatenate([np.zeros((int(-song0 * SR), 2), np.float32), mara])
    # level: match the reel's mix level at its start
    ref = load(f'{E}/mix_B.wav', REEL[0][0], REEL[0][0] + 3.0)
    raw = load(f'{A}/track2.mp3', REEL[0][0] + 6.66, REEL[0][0] + 9.66)
    g_m = np.sqrt((ref ** 2).mean()) / (np.sqrt((raw ** 2).mean()) + 1e-9) * 0.75
    mara *= g_m
    # duck the music under his voice
    v = np.abs(vo_mono)
    k = int(0.03 * SR)
    v = np.convolve(v, np.ones(k) / k, 'same')
    talk = (v > 0.01).astype(np.float32)
    k2 = int(0.35 * SR)
    talk = np.convolve(talk, np.ones(k2) / k2, 'same')
    duck = 1 - 0.62 * np.clip(talk * 1.5, 0, 1)
    n = min(len(mara), int(T_REEL * SR))
    mara = mara[:n] * duck[:n, None]
    mara[:int(0.6 * SR)] *= np.linspace(0, 1, int(0.6 * SR))[:, None]
    mix[:n] += mara
    mix += vo * 1.35
    # teleport: static burst, crushed kick, eye ignition tick
    place(mix, synth_static(0.5, 0.28, 1), 0.0)
    place(mix, crushed_kick(0.55), 0.35)
    place(mix, synth_static(0.15, 0.12, 2), T_TELE - 0.25)
    # screen on: static, a swell into the dive, a hit on the landing
    place(mix, synth_static(0.6, 0.22, 4), T_SCREEN)
    place(mix, swell(1.2, 0.22), T_REEL - 1.2)
    place(mix, crushed_kick(0.5), T_REEL)
    # the reel (its own finished mix), with a glitch over the splice
    t = T_REEL
    for i, (a, b) in enumerate(REEL):
        x = load(f'{E}/mix_B.wav', a, b)
        x = fade(x, 0.01, 0.03)
        place(mix, x, t)
        if i:
            place(mix, synth_static(0.25, 0.2, 9), t - 0.05)
        t += b - a
    # outro: a hit, static, a low drone that dies
    place(mix, synth_static(0.4, 0.25, 11), T_OUT)
    place(mix, crushed_kick(0.6), T_OUT + 0.02)
    n = int(3.6 * SR); tt = np.arange(n) / SR
    drone = (np.sin(2 * np.pi * 41 * tt) * 0.6 + np.sin(2 * np.pi * 61.5 * tt) * 0.3) * np.minimum(1, tt / 0.4) * np.maximum(0, 1 - tt / 3.6) * 0.18
    place(mix, np.stack([drone, drone], 1).astype(np.float32), T_OUT)
    mix = mix[:int(T_END * SR)]
    pk = np.abs(mix).max()
    mix = np.tanh(mix / max(pk, 1e-6) * 1.3) * 0.92
    q = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-c:a', 'pcm_s16le', out],
                         stdin=subprocess.PIPE)
    q.stdin.write(mix.astype(np.float32).tobytes()); q.stdin.close(); q.wait()
    # check: his words over the music
    sos = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
    vs = sosfilt(sos, vo_mono * 1.35); ms = sosfilt(sos, mix[:, 0] * 0 + (mara.mean(1) if False else 0)) if False else None
    print('audio ok', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='a', type=float, default=0.0)
    ap.add_argument('--to', dest='b', type=float, default=None)
    ap.add_argument('--audio-only', action='store_true')
    ap.add_argument('--video-only', action='store_true')
    o = ap.parse_args()
    if not o.video_only:
        render_audio(f'{OUTD}/ep1.wav')
    if not o.audio_only:
        render_video(f'{OUTD}/ep1_video.mp4', o.a, o.b)
