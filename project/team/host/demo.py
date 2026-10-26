import os, sys, math, subprocess, numpy as np, cv2
os.environ['VOY_VER'] = 'B'
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import render as R, host as HO, pixelize as P
FPS, GW, GH, OW, OH, S = 30, 180, 320, 1080, 1920, 6
SR = 48000
DUR = 6.6
VO_AT = 1.25
A = '/home/claude/voynich/audio'
def load(path, a=None, b=None, af=None):
    cmd = ['ffmpeg', '-v', 'error'] + (['-ss', str(a)] if a is not None else []) + (['-to', str(b)] if b is not None else []) + ['-i', path]
    if af: cmd += ['-af', af]
    cmd += ['-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.check_output(cmd), np.float32).copy()
vo = load(f'{A}/vo_narrator.mp3', 0.0, 3.86, 'highpass=f=70,acompressor=threshold=-22dB:ratio=4:attack=5:release=120:makeup=4,aecho=0.8:0.6:45|90:0.18|0.10')
vo[-int(0.05*SR):] *= np.linspace(1, 0, int(0.05*SR))
# loudness per video frame -> mouth amplitude
def amp_at(t):
    i0 = int((t - VO_AT) * SR); i1 = i0 + int(SR / FPS)
    if i1 <= 0 or i0 >= len(vo): return 0.0
    seg = vo[max(0, i0):min(len(vo), i1)]
    db = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
    return float(np.clip((db + 42) / 26, 0, 1))
# background: the archive, darkened
bg = cv2.imread('/home/claude/voynich/kf/s08_archive.jpg')
h, w = bg.shape[:2]; sc = max(GW / w, GH / h)
bg = cv2.resize(bg, (int(w * sc + 1), int(h * sc + 1)), interpolation=cv2.INTER_AREA)
y0 = (bg.shape[0] - GH) // 2; x0 = (bg.shape[1] - GW) // 2
bg = bg[y0:y0 + GH, x0:x0 + GW].astype(np.float32)
yy, xx = np.mgrid[0:GH, 0:GW]
vig = np.clip(1.15 - np.hypot((xx - 90) / 110, (yy - 190) / 190), 0.15, 1)[..., None]
bgq = P.quantize(np.clip(bg * 0.55 * vig, 0, 255).astype(np.uint8), dither=.12, contrast=1.1)
SCALE = 3
CX, BY = 90, 268
words = [('ЭТА', False, 0.16), ('КНИГА…', True, 0.42), ('НЕ', False, 1.5), ('ХОЧЕТ,', False, 1.74), ('ЧТОБЫ', False, 2.42), ('ЕЁ', False, 2.72), ('ЧИТАЛИ.', True, 3.14)]
chunks = [words[:2], words[2:]]
noise_bank = [np.random.default_rng(k).normal(0, 4.0, (OH, OW, 1)).astype(np.float32) for k in range(8)]
scan = np.ones((OH, 1, 1), np.float32); scan[::S] = 0.82
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '/home/claude/voynich/team/host/v.mp4'], stdin=subprocess.PIPE)
rngm = np.random.default_rng(7)
order = rngm.random((52 * SCALE, 44 * SCALE))
for f in range(int(DUR * FPS)):
    t = f / FPS
    tq = math.floor(t * 15) / 15                       # 15 fps sprite hold
    g = bgq.copy()
    # dust in the dark
    r = np.random.default_rng(int(t * 15))
    for k in range(14):
        dx = (k * 37 + tq * (6 + k % 5)) % GW; dy = (k * 91 - tq * (9 + k % 3) * 2) % GH
        g[int(dy), int(dx)] = HO.BGR[HO.LINE if k % 3 else HO.MID]
    amp = amp_at(tq)
    blink = (2.55 <= tq < 2.62 and False) or (5.6 <= tq < 5.7) or (0.62 <= tq < 0.69)
    look = -1 if 5.1 <= tq < 5.55 else 0
    bob = int(round(math.sin(tq * 2 * math.pi / 2.14) * 1.0))
    layer = np.zeros_like(g); HO.draw(layer, CX, BY, tq, amp, SCALE, blink, look, bob, wave_seed=int(tq * 15))
    m = layer.any(2)
    if t < 0.55:
        # only the eyes in the void: they light up
        eyes = (layer[..., 2] > 150) & (layer[..., 1] < 80)
        on = t > 0.25 and not (0.36 < t < 0.42)
        g[:] = (g * 0.25).astype(np.uint8)
        g = P.quantize(g, dither=.0, contrast=1.0)
        if on: g[eyes] = layer[eyes]
    elif t < 1.1:
        # the body assembles out of pixels around the eyes
        k = (t - 0.55) / 0.55
        ys, xs = np.nonzero(m)
        sy0 = BY - 52 * SCALE + bob; sx0 = CX - 44 * SCALE // 2
        oy, ox = np.clip(ys - sy0, 0, 52 * SCALE - 1), np.clip(xs - sx0, 0, 44 * SCALE - 1)
        show = order[oy, ox] < k ** 0.8
        eyes = (layer[..., 2] > 150) & (layer[..., 1] < 80)
        g[ys[show], xs[show]] = layer[ys[show], xs[show]]
        g[eyes] = layer[eyes]
        # loose pixels flying in
        for j in range(int(60 * (1 - k))):
            jj = rngm.integers(len(ys)); d = (1 - k) * 40
            ty, tx = int(ys[jj] + rngm.normal(0, d)), int(xs[jj] + rngm.normal(0, d))
            if 0 <= ty < GH and 0 <= tx < GW: g[ty, tx] = layer[ys[jj], xs[jj]]
    else:
        g[m] = layer[m]
        if 1.1 <= t < 1.14:
            g = R.rgbsplit(g, 1)
    rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    rgb = R.draw_ui(rgb, t, dict(surface='KEEPER_408', boxes=[dict(r=(CX - 50, BY - 150, CX + 50, BY - 70), label='ХРАНИТЕЛЬ', col='red', sub='АРХИВ 408')]) if t >= 1.1 else dict(surface='KEEPER_408'))
    # caption
    tv = t - VO_AT
    cap = None
    if -0.2 <= tv < 4.6:
        ch = chunks[0] if tv < 1.4 else chunks[1]
        cap = ([(w, k, w0 - 0.03 <= tv, 0 <= tv - w0 + 0.03 < 2 / FPS) for w, k, w0 in ch], 'narr')
    rgb = R.draw_caption(rgb, cap, t)
    big = cv2.resize(rgb, (OW, OH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big *= scan; big += noise_bank[f % 8]
    p.stdin.write(np.clip(big, 0, 255).astype(np.uint8).tobytes())
    if f in (12, 25, 45, 70, 100, 150, 160):
        cv2.imwrite(f'/home/claude/voynich/team/host/f{f:03d}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
p.stdin.close(); p.wait()
# audio: a low drone from Mara + the voice + a crushed blip when the eyes light up
N = int(DUR * SR)
mara = load(f'{A}/track2.mp3', 40.0, 40.0 + DUR, 'lowpass=f=2500')[:N]
mix = np.zeros(N, np.float32); mix[:len(mara)] += mara * 0.35
i = int(VO_AT * SR); mix[i:i + len(vo)] += vo[:N - i] * 1.3
tb = np.arange(int(0.25 * SR)) / SR
blip = np.sign(np.sin(2 * np.pi * (90 - 200 * tb) * tb)) * np.exp(-tb / 0.06) * 0.25
blip = np.round(blip * 8) / 8
mix[int(0.25 * SR):int(0.25 * SR) + len(blip)] += blip
mix[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))
mix = np.tanh(mix * 1.2) * 0.9
st = np.stack([mix, mix], 1).astype(np.float32)
q = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '/home/claude/voynich/team/host/a.wav'], stdin=subprocess.PIPE)
q.stdin.write(st.tobytes()); q.stdin.close(); q.wait()
subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', '/home/claude/voynich/team/host/v.mp4', '-i', '/home/claude/voynich/team/host/a.wav',
                       '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart',
                       '/mnt/user-data/outputs/host_demo.mp4'])
print('ok')
