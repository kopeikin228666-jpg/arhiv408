"""Loudness levelling / mastering for the 408 episode (owned by the sound engineer).

    level(mix, segs, SR, voice=None) -> np.float32 (N, 2)

mix   : the summed pre-master mix (N, 2), float, any level (it may exceed 0 dBFS)
segs  : the SEG list of make_ep1d.py (dicts with kind, t0, t1; reel segments also carry src=(film_a, film_b))
voice : optional (N, 2) - the part of `mix` that is 408's own voice (exactly as it was summed into mix).
        With it the music/sfx under his words get a speech-band dynamic EQ and are never allowed to mask him.

Chain
 1. split mix into voice + rest (rest = mix - voice)
 2. speech-band ducking of the rest under 408's words (complementary 250-4000 Hz split, only the mid band moves,
    so kicks keep their low thump); target >= SNR_DB in 300-3400 Hz on every 10 ms frame where he speaks
 3. per-segment gain so that the DIALOGUE (speech-gated BS.1770 loudness) is the same in every segment - 408's
    lines and the reel narrator alike; segments without words get a target relative to the dialogue
    (music-only reel -2, lean -1, dive +1.5, black card -5 LU); gain steps are raised-cosine ramps of RAMP s
 4. sfx rider: 1 s / 3 s windowed loudness of the non-voice part is held to <= dialogue + 3 LU
 5. integrated loudness -> TARGET_I (-14 LUFS), then a 4x-oversampled look-ahead true-peak limiter to CEIL dBTP
Loudness is measured with ITU-R BS.1770-4 K-weighting (scipy biquads) + absolute/relative gating.
"""
import os
import numpy as np
from scipy.signal import lfilter, butter, sosfiltfilt, resample_poly
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d

TARGET_I = -14.0       # LUFS integrated
CEIL = -1.3            # dBTP limiter ceiling (leaves margin for the 16-bit / AAC stage under -1 dBTP)
SNR_DB = 10.0          # wanted voice-over-rest margin in 300-3400 Hz under 408's words (spec: >= 8)
RAMP = 0.05            # s, half-width of the raised-cosine gain ramp at a segment boundary (100 ms crossfade)
SFX_OVER = 3.0         # LU allowed above the dialogue for hits / swells (1 s window)
REL = {'reel': -2.0, 'lean': -2.5, 'dive': 1.5, 'card': -5.0, 'outro': -5.0}   # LU vs dialogue, segments without words
# (round 2: lean -1 -> -2.5: he leans in, the room holds its breath, then the dive opens up +1.5 - a little more range)
WHISPER_REL = -3.0     # the whispered «Она уже… внутри…» sits under normal speech (round 2: -1.5 -> -3, over a thinner bed)
VO_REEL = '/home/claude/voynich/edit/vo_B.npy'   # the reel's narrator stem (16 kHz mono, film time)
LAST = {}              # internals of the last level() call (for verification scripts)


# ------------------------------------------------------------------ BS.1770
def _kcoef(sr):
    if sr == 48000:
        return ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585],
                [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])
    # generic design (same analog prototypes as the 48 kHz constants)
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    K = np.tan(np.pi * fc / sr); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    Q2, fc2 = 0.5003270373253953, 38.13547087613982
    K = np.tan(np.pi * fc2 / sr)
    a2 = [1.0, 2 * (K * K - 1) / (1 + K / Q2 + K * K), (1 - K / Q2 + K * K) / (1 + K / Q2 + K * K)]
    return b1, a1, [1.0, -2.0, 1.0], a2


def kpower(x, sr):
    """K-weighted power, summed over channels, per sample."""
    b1, a1, b2, a2 = _kcoef(sr)
    x = np.atleast_2d(np.asarray(x, np.float64).T).T
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    return (y ** 2).sum(1)


def _lufs(p):
    return -0.691 + 10 * np.log10(np.maximum(p, 1e-20))


def window_loudness(pk, sr, win, hop):
    """loudness of every window of `win` s (centred), at `hop` s steps; returns (centres_s, LUFS)."""
    c = np.cumsum(np.concatenate([[0.0], pk])); n = len(pk); w = int(win * sr); h = int(hop * sr)
    cen = np.arange(0, n, h); a = np.clip(cen - w // 2, 0, n); b = np.clip(cen + w // 2, 0, n)
    p = (c[b] - c[a]) / np.maximum(b - a, 1)
    return cen / sr, _lufs(p)


def gated(pk, sr, mask=None):
    """BS.1770 gated loudness of the 400 ms / 75 % blocks (optionally only blocks where mask is mostly true)."""
    w, h = int(0.4 * sr), int(0.1 * sr)
    c = np.cumsum(np.concatenate([[0.0], pk]))
    st = np.arange(0, max(1, len(pk) - w + 1), h)
    p = (c[np.minimum(st + w, len(pk))] - c[st]) / w
    keep = np.ones(len(p), bool)
    if mask is not None:
        cm = np.cumsum(np.concatenate([[0.0], mask.astype(np.float64)]))
        keep = (cm[np.minimum(st + w, len(pk))] - cm[st]) / w > 0.5
    p = p[keep & (_lufs(p) > -70)]
    if not len(p):
        return -np.inf
    rel = _lufs(p.mean()) - 10
    p = p[_lufs(p) > rel]
    return _lufs(p.mean()) if len(p) else -np.inf


def integrated(x, sr):
    return gated(kpower(x, sr), sr)


def true_peak_db(x, os_=4):
    y = resample_poly(np.asarray(x, np.float64), os_, 1, axis=0)
    return 20 * np.log10(np.abs(y).max() + 1e-12)


# ------------------------------------------------------------------ helpers
def _frames_db(x, F):
    m = len(x) // F
    e = (x[:m * F].reshape(m, F) ** 2).mean(1)
    return 10 * np.log10(e + 1e-14)


def _smooth_min(g, n_min, n_avg):
    """gain curve: take the minimum over n_min (so a dip covers its whole window), then a box average."""
    g = minimum_filter1d(g, max(1, n_min), mode='nearest')
    return uniform_filter1d(g, max(1, n_avg), mode='nearest')


def _upsample(gf, F, n):
    """frame-rate curve -> per-sample (linear interpolation between frame centres)."""
    xs = (np.arange(len(gf)) + 0.5) * F
    return np.interp(np.arange(n), xs, gf).astype(np.float32)


def _activity_from_stem(v, sr, F):
    """speech activity per frame from a voice stem: above (loud level - 30 dB), hangover 120 ms, holes < 150 ms filled."""
    d = _frames_db(v, F)
    if not np.isfinite(d).any() or d.max() < -90:
        return np.zeros(len(d), bool)
    loud = np.percentile(d[d > -90], 95) if (d > -90).sum() > 10 else d.max()
    a = d > loud - 30
    fr = sr / F
    a = maximum_filter1d(a.astype(np.uint8), int(0.12 * fr) * 2 + 1).astype(bool)
    a = minimum_filter1d(maximum_filter1d(a.astype(np.uint8), int(0.15 * fr)), int(0.15 * fr)).astype(bool)
    return a


def _reel_activity(segs, N, sr, F):
    """reel narrator activity in episode time (frames), from the reel's own narrator stem."""
    act = np.zeros(N // F, bool); whisper = np.zeros(N // F, bool)
    if not os.path.exists(VO_REEL):
        return act, whisper, False
    vr = np.load(VO_REEL).astype(np.float64); R = 16000
    d_all = _frames_db(vr, R // 100)
    loud = np.percentile(d_all[d_all > -90], 95)
    try:
        import sys
        sys.path.insert(0, '/home/claude/voynich/edit')
        from timeline import LINES
        wspans = [(st, st + (b - a)) for (f, a, b, st, tx, v) in LINES if v == 'whisper']
    except Exception:
        wspans = []
    for sg in segs:
        if sg['kind'] != 'reel' or not sg.get('src'):
            continue
        fa, fb = sg['src']
        d = d_all[int(fa * 100):int(fb * 100)]                 # 10 ms frames of film time
        a = d > loud - 30
        a = maximum_filter1d(a.astype(np.uint8), 25).astype(bool)
        t = sg['t0'] + (np.arange(len(a)) + 0.5) / 100
        i = (t * sr / F).astype(int); ok = (i >= 0) & (i < len(act))
        act[i[ok]] |= a[ok]
        ft = fa + (np.arange(len(a)) + 0.5) / 100
        for (wa, wb) in wspans:
            m = ok & (ft >= wa - 0.3) & (ft <= wb + 0.5) & a
            whisper[i[m]] = True
    # fill the frame grid (10 ms film frames may be coarser than F)
    act = maximum_filter1d(act.astype(np.uint8), 3).astype(bool)
    return act, whisper, True


def _ramp_curve(points, n, sr, ramp=RAMP):
    """piecewise-constant dB curve [(t0, t1, dB)] -> per-sample linear gain, raised-cosine ramp centred on each boundary."""
    g = np.zeros(n, np.float64)
    pts = sorted(points)
    for k, (t0, t1, db) in enumerate(pts):
        g[int(t0 * sr):int(t1 * sr)] = db
    if pts:
        g[:int(pts[0][0] * sr)] = pts[0][2]; g[int(pts[-1][1] * sr):] = pts[-1][2]
    r = int(ramp * sr); h = np.hanning(r * 2 + 1)[:r * 2 + 1]
    for k in range(1, len(pts)):
        b = int(pts[k][0] * sr); a_, b_ = max(0, b - r), min(n, b + r)
        d0, d1 = pts[k - 1][2], pts[k][2]
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, b_ - a_))
        g[a_:b_] = d0 + (d1 - d0) * w
    return 10 ** (g / 20)


def limiter(x, sr, ceil_db=CEIL, look=0.006, release=0.12, os_=4):
    """look-ahead true-peak limiter: gain from the 4x-oversampled peak, min over +-look, smoothed, slow release."""
    c = 10 ** (ceil_db / 20)
    up = resample_poly(x.astype(np.float64), os_, 1, axis=0)
    pk = np.abs(up).max(1).reshape(-1, os_).max(1)[:len(x)]
    if len(pk) < len(x):
        pk = np.concatenate([pk, np.zeros(len(x) - len(pk))])
    req = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = int(look * sr)
    g = minimum_filter1d(req, 2 * L + 1, mode='nearest')
    # release: one-pole recovery at 1 ms resolution (instant attack is already handled by the look-ahead min)
    B = sr // 1000; m = len(g) // B + 1
    gb = np.concatenate([g, np.ones(m * B - len(g))]).reshape(m, B).min(1)
    a = np.exp(-1.0 / (release * 1000)); out = np.empty(m); s = 1.0
    for i in range(m):
        v = gb[i]
        s = v if v < s else v + (s - v) * a
        out[i] = s
    gs = np.repeat(out, B)[:len(g)]
    gs = np.minimum(gs, g)
    gs = uniform_filter1d(gs, L, mode='nearest')
    gs = np.minimum(gs, minimum_filter1d(req, L + 1, mode='nearest'))   # never above a peak's requirement
    return (x * gs[:, None]).astype(np.float32), gs


# ------------------------------------------------------------------ main
def level(mix, segs, SR, voice=None, verbose=True):
    mix = np.asarray(mix, np.float64)
    N = len(mix); F = SR // 100                                  # 10 ms analysis frames
    nf = N // F
    if voice is not None:
        voice = np.asarray(voice, np.float64)[:N]
        if len(voice) < N:
            voice = np.concatenate([voice, np.zeros((N - len(voice), 2))])
    else:
        voice = np.zeros_like(mix)
    rest = mix - voice
    have_vo = np.abs(voice).max() > 1e-6

    # --- 1. speech-band dynamic EQ of the rest under 408's words
    sos_m = butter(2, [250, 4000], 'bandpass', fs=SR, output='sos')
    sos_s = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
    act408 = np.zeros(nf, bool)
    if have_vo:
        vm = voice.mean(1)
        act408 = _activity_from_stem(vm, SR, F)
        rm = sosfiltfilt(sos_m, rest, axis=0)                   # zero-phase -> rest = (rest - rm) + rm exactly
        sm = lambda d: 10 * np.log10(uniform_filter1d(10 ** (d / 10), 3, mode='nearest'))   # 30 ms power average
        vb = sm(_frames_db(sosfiltfilt(sos_s, vm), F))
        rb = sm(_frames_db(sosfiltfilt(sos_s, rest.mean(1)), F))
        # judge the voice by its syllable envelope (150 ms power average), not by single quiet consonant frames
        venv = 10 * np.log10(uniform_filter1d(10 ** (vb / 10), 15, mode='nearest'))
        p95 = np.percentile(venv[act408], 95) if act408.any() else venv.max()
        speech = act408 & (venv > p95 - 25)
        need = np.where(speech, np.clip((venv - SNR_DB) - rb, -18.0, 0.0), 0.0)   # dB of cut needed per frame
        gdb = _smooth_min(need, 15, 10)                          # hold each cut 150 ms, 100 ms smoothing (no flutter)
        gm = _upsample(10 ** (gdb / 20), F, N)
        if verbose:
            print(f'  level: speech-band EQ under 408 cuts >3 dB on {(gdb < -3).mean() * 100:.1f}% of frames, max {-gdb.min():.1f} dB')
        rest = rest + rm * (gm[:, None] - 1.0)
    mix1 = voice + rest

    # --- 2. dialogue activity everywhere (408 stem + reel narrator stem)
    actR, whis, have_reel = _reel_activity(segs, N, SR, F)
    act = act408 | actR
    act_s = np.repeat(act, F); act_s = np.concatenate([act_s, np.zeros(N - len(act_s), bool)])

    # --- 3. per-segment gains: dialogue segments -> common dialogue loudness D, others -> D + REL[kind]
    pk = kpower(mix1, SR)
    dl = []
    for sg in segs:
        a, b = int(sg['t0'] * SR), int(min(sg['t1'], N / SR) * SR)
        L = gated(pk[a:b], SR, act_s[a:b]) if act_s[a:b].mean() > 0.12 else -np.inf
        sg_w = whis[a // F:b // F].mean() > 0.1
        dl.append((L, sg_w))
    Ls = [L for L, w in dl if np.isfinite(L)]
    D = float(np.median(Ls)) if Ls else integrated(mix1, SR)    # common dialogue reference (pre-gain units)
    tg = []                                                     # (target, measured-on-dialogue?) per segment
    gdb = []
    for sg, (L, w) in zip(segs, dl):
        a, b = int(sg['t0'] * SR), int(min(sg['t1'], N / SR) * SR)
        if np.isfinite(L):
            tg.append((D + (WHISPER_REL if w else 0.0), True)); g0 = tg[-1][0] - L
        else:
            Lseg = gated(pk[a:b], SR)
            tg.append((D + REL.get(sg['kind'], -2.0), False)); g0 = (tg[-1][0] - Lseg) if np.isfinite(Lseg) else 0.0
        gdb.append(float(np.clip(g0, -12, 12)))

    def seg_measure(x):
        pkx = kpower(x, SR); res = []
        for sg, (t_, on_dlg) in zip(segs, tg):
            a, b = int(sg['t0'] * SR), int(min(sg['t1'], N / SR) * SR)
            res.append(gated(pkx[a:b], SR, act_s[a:b] if on_dlg else None))
        return res

    # --- 4. sfx rider on the rest: windowed loudness <= dialogue + 4.5 (0.4 s), + SFX_OVER (1 s), + SFX_OVER-0.5 (3 s);
    #        then re-measure every segment and correct its gain (the rider may have touched the dialogue too)
    win_specs = ((0.4, SFX_OVER + 1.5), (1.0, SFX_OVER), (3.0, SFX_OVER - 0.5))   # momentary / 1 s / short-term
    hop = 0.01
    for npass in range(3):
        pts = [(sg['t0'], sg['t1'], g) for sg, g in zip(segs, gdb)]
        gseg = _ramp_curve(pts, N, SR)
        voice2 = voice * gseg[:, None]; rest2 = rest * gseg[:, None]
        rgain = gseg.astype(np.float64)                         # total linear gain on the rest (for verification)
        for it in range(3):
            tot = voice2 + rest2
            pkt = kpower(tot, SR); pkv = kpower(voice2, SR) if have_vo else np.zeros(N)
            cut = np.zeros(nf)
            for win, over in win_specs:
                _, Lt = window_loudness(pkt, SR, win, hop)
                _, Lv = window_loudness(pkv, SR, win, hop)
                Lt = Lt[:nf]; Lv = Lv[:nf]
                ceil = np.maximum(D + over + 1.0 * actR[:nf], Lv + 1.0)      # the reel narrator may peak 1 LU more
                ex = np.maximum(np.minimum(0.0, ceil - Lt), -12.0)
                c = -maximum_filter1d(-ex, int(min(win, 1.0) / hop) | 1)  # spread over (<= 1 s of) its window
                cut = np.minimum(cut, c)
            if cut.min() > -0.2:
                break
            cutdb = uniform_filter1d(cut, int(0.15 / hop), mode='nearest')
            cg = _upsample(10 ** (cutdb / 20), F, N)
            rest2 = rest2 * cg[:, None]; rgain = rgain * cg
        out = voice2 + rest2
        meas = seg_measure(out)
        err = [(t_ - m) if np.isfinite(m) else 0.0 for (t_, _), m in zip(tg, meas)]
        if verbose:
            print(f'  level pass {npass}: worst segment error {max(abs(e) for e in err):.2f} LU')
        if max(abs(e) for e in err) < 0.3 or npass == 2:
            break
        gdb = [float(np.clip(g + e, -14, 14)) for g, e in zip(gdb, err)]
    if verbose:
        for sg, (t_, on_dlg), g, m in zip(segs, tg, gdb, meas):
            print(f"  level {sg['kind']:6s} {sg['t0']:6.2f}-{sg['t1']:6.2f}  gain {g:+5.1f} dB  "
                  f"{'dialogue' if on_dlg else 'segment '} {m - D:+5.1f} LU vs dialogue (target {t_ - D:+.1f})")
    rider_db = 20 * np.log10(np.maximum(np.abs(rest2[:, 0]), 1e-12) / np.maximum(np.abs(rest[:, 0] * gseg), 1e-12))

    # --- 5. integrated loudness -> target, true-peak limiter, re-trim
    ks = 10 ** ((TARGET_I - integrated(out, SR)) / 20)
    for it in range(4):
        lim, gl = limiter(out * ks, SR)
        I1 = integrated(lim, SR)
        if abs(I1 - TARGET_I) < 0.1:
            break
        ks *= 10 ** ((TARGET_I - I1) / 20)
    tp = true_peak_db(lim)
    if tp > CEIL + 0.05:                                       # safety: plain trim if overs survived
        lim *= 10 ** ((CEIL - tp) / 20); ks *= 10 ** ((CEIL - tp) / 20)
    if verbose:
        print(f'  level: dialogue ref {D:.1f} (pre), integrated {integrated(lim, SR):.2f} LUFS, '
              f'true peak {true_peak_db(lim):.2f} dBTP, limiter max GR {20 * np.log10(gl.min()):.1f} dB, '
              f'GR>1dB on {100 * (gl < 0.891).mean():.1f}% of samples')
    LAST.clear()
    LAST.update(D=D, gseg=gseg, rider_db=rider_db, scale=ks, lim_gain=gl, act=act, segs_db=list(gdb),
                rest_gain=(rgain * ks * gl).astype(np.float32),
                voice_out=(voice2 * ks * gl[:, None]).astype(np.float32))
    return np.clip(lim, -1, 1).astype(np.float32)
