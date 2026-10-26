"""Word-level alignment without ASR: the text of every line is known, so words are spread over the
voiced frames of the line by syllable weight, then each boundary snaps to the nearest energy dip.
Pauses inside a line ("…", ".") are matched to the largest silent gaps of that line.
Output: edit/words.json  [{line, chunk, w, key, t0, t1}]  in FILM time.
"""
import re, json, subprocess, numpy as np
from timeline import LINES, A

SR = 16000
HOP = 0.01
VOW = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')


def load(path, a, b):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(a), '-to', str(b), '-i', path,
                                   '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'])
    return np.frombuffer(raw, np.float32)


def envelope(x):
    n = int(HOP * SR)
    k = len(x) // n
    e = np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1))
    return np.convolve(e, np.ones(3) / 3, 'same')


def tokens(text):
    """-> list of (display, spoken, key, ends_chunk)"""
    out = []
    for m in re.finditer(r'(\*?)([^\s{]+)(?:\{([^}]*)\})?(\S*)', text):
        key, disp, spoken = m.group(1) == '*', m.group(2) + m.group(4), m.group(3)
        sp = spoken if spoken else disp
        end = bool(re.search(r'(…|\.|!|\?)$', disp))
        out.append((disp.upper(), sp, key, end))
    return out


def weight(sp):
    syl = sum(c in VOW for c in sp)
    digits = sum(c.isdigit() for c in sp)
    if syl == 0 and digits == 0:
        return 0.35
    return syl + digits * 2.5 + 0.5


def band_env(path, a, b, n):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(a), '-to', str(b), '-i', path,
                                   '-af', 'highpass=f=300,lowpass=f=2500', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'])
    x = np.frombuffer(raw, np.float32)
    e = envelope(x)
    e = np.convolve(e, np.ones(5) / 5, 'same')
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def islands(e, thr):
    v = e > thr
    # close gaps shorter than 90 ms, drop blips shorter than 40 ms
    idx = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(int), [0]])))
    segs = [[a, b] for a, b in zip(idx[::2], idx[1::2])]
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 9:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= 4]


def align_line(path, a, b, text, snap=5):
    x = load(path, a, b + 0.05); e = envelope(x)
    isl = islands(e, e.max() * 0.07); toks = tokens(text)
    chunks, cur = [], []
    for i, t in enumerate(toks):
        cur.append(i)
        if t[3] and i != len(toks) - 1: chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    if len(chunks) > 1 and len(isl) >= len(chunks):
        gaps = [(isl[j + 1][0] - isl[j][1], j) for j in range(len(isl) - 1)]
        cut = sorted(j for _, j in sorted(gaps, reverse=True)[:len(chunks) - 1])
        groups, s = [], 0
        for j in cut: groups.append(isl[s:j + 1]); s = j + 1
        groups.append(isl[s:])
    else:
        groups = [isl]; chunks = [sum(chunks, [])]
    res = [None] * len(toks)
    for ch, grp in zip(chunks, groups):
        frames = np.concatenate([np.arange(s, e_) for s, e_ in grp])
        w = np.array([weight(toks[i][1]) for i in ch])
        cum = np.concatenate([[0], np.cumsum(w)]) / w.sum()
        bounds = [frames[min(len(frames) - 1, int(c * (len(frames) - 1)))] for c in cum]
        bounds[-1] = frames[-1] + 1
        for k in range(1, len(bounds) - 1):
            c = bounds[k]; lo, hi = max(bounds[k - 1] + 3, c - snap), min(bounds[-1] - 3, c + snap)
            if hi > lo: bounds[k] = lo + int(np.argmin(e[lo:hi]))
        for k, i in enumerate(ch): res[i] = (bounds[k] * HOP, bounds[k + 1] * HOP)
    return toks, res

def display_chunks(toks, maxc=30):
    """split each punctuation chunk further so a caption never exceeds ~2 lines"""
    ids, cid, n = [], 0, 0
    for i, t in enumerate(toks):
        L = len(t[0]) + 1
        if n and n + L > maxc:
            cid += 1; n = 0
        ids.append(cid); n += L
        if t[3]:
            cid += 1; n = 0
    return ids


import os


def scribe_words(f):
    p = f'{A}/scribe_{f[:-4]}.txt'
    if not os.path.exists(p):
        return None
    return [(l.split()[0], float(l.split()[1]), float(l.split()[2])) for l in open(p, encoding='utf-8') if l.strip()]


_ENV = {}


def _src_env(f):
    if f not in _ENV:
        x = load(f'{A}/{f}', 0, 10 ** 4)
        e = envelope(x)
        _ENV[f] = 20 * np.log10(e + 1e-6)
    return _ENV[f]


def scribe_line(f, a, b, text):
    """real word timestamps from ElevenLabs Scribe, each start pulled back to the true attack of the word
    (Scribe marks words ~60 ms late on average, phrase openings up to 150 ms late)"""
    toks = tokens(text)
    ws = [w for w in scribe_words(f) if a - 0.12 <= w[1] < b + 0.05]
    assert len(ws) == len(toks), (text, [w[0] for w in ws])
    le = _src_env(f)
    out = []
    for w in ws:
        i = int(round(w[1] / HOP))
        lo, hi = max(1, i - 17), i + 4
        j = lo + int(np.argmax(np.diff(le[lo - 1:hi])))
        t0 = min(w[1], j * HOP)
        if out:
            t0 = max(t0, out[-1][0] + a + 0.06)      # keep the words in order
        out.append((t0 - a, w[2] - a))
    return toks, out


_RAW = {}
FRIC = set('сшщчцжзфх')


def _env_db(x, sr, hop=0.0025, win=0.010):
    h, w = int(hop * sr), int(win * sr)
    n = max(0, (len(x) - w) // h + 1)
    idx = np.arange(n)[:, None] * h + np.arange(w)[None]
    return 20 * np.log10(np.sqrt((x[idx].astype(np.float64) ** 2).mean(1)) + 1e-9)


def _raw(f):
    if f not in _RAW:
        from scipy.signal import stft
        x = load(f'{A}/{f}', 0, 10 ** 4)
        fr, tt, Z = stft(x, SR, nperseg=320, noverlap=280, boundary=None, padded=False)
        P = np.abs(Z) ** 2
        hfr = 10 * np.log10(P[fr >= 3500].sum(0) + 1e-12) - 10 * np.log10(P.sum(0) + 1e-12)
        _RAW[f] = (_env_db(x, SR), tt, hfr)
    return _RAW[f]


def true_attacks(f, a, toks, res, trusted_ends=True):
    """the real start of every word in the source: onset after a pause, else the hiss of с/ш/ч..., else the prior.
    Also the audible end (level under -45 dBFS for 60 ms).  Same model the independent QA uses."""
    env, FTr, HFR = _raw(f)
    out, prev_att, prev_end = [], -9, -9
    for (disp, sp, key, end_), (t0, t1) in zip(toks, res):
        prior, prior_end = a + t0, a + t1
        lo_lim = max(prior - 0.25, prev_att + 0.08, prev_end - 0.02)
        att = None
        k0, k1 = int(lo_lim / 0.0025), int((prior + 0.12) / 0.0025)
        seg = env[max(0, k0):k1]; run = 0; best = None
        for k in range(len(seg)):
            if seg[k] < -55:
                run += 1
            else:
                if run * 0.0025 >= 0.05 or (k0 + k - run <= 0 and run > 0):
                    up = np.flatnonzero(seg[k:] > -45)
                    if len(up) and up[0] * 0.0025 < 0.08:
                        best = max(0, k0) + k + up[0]
                run = 0
        if k0 <= 0 and len(seg) and seg[0] > -45:
            best = 0
        if best is not None and abs(best * 0.0025 + 0.005 - prior) < 0.25:
            att = best * 0.0025 + 0.005
        if att is None and disp.lstrip('*').lower()[:1] in FRIC:
            lo, hi = np.searchsorted(FTr, lo_lim), np.searchsorted(FTr, prior + 0.06)
            cand = [jj for jj in range(lo, hi) if HFR[jj] > -10]
            if cand:
                jj = cand[0]
                while jj > lo and HFR[jj - 1] > -10:
                    jj -= 1
                att = float(FTr[jj])
        if att is None:
            att = prior
        e_ = None
        for k in range(int((prior_end - 0.15) / 0.0025), int((prior_end + 0.8) / 0.0025)):
            if k + 24 <= len(env) and (env[k:k + 24] < -45).all():
                e_ = k * 0.0025 + 0.005
                break
        if e_ is None:
            e_ = min(prior_end + 0.1, len(env) * 0.0025)
        prev_att, prev_end = att, (prior_end if trusted_ends else -9)
        out.append((att - a, max(e_, prior_end) - a))
    # keep order, no overlaps of starts
    for k in range(1, len(out)):
        if out[k][0] < out[k - 1][0] + 0.06:
            out[k] = (out[k - 1][0] + 0.06, out[k][1])
    return out


def main():
    words = []
    for li, (f, a, b, start, text, voice) in enumerate(LINES):
        if scribe_words(f):
            toks, res = scribe_line(f, a, b, text)
        else:
            toks, res = align_line(f'{A}/{f}', a, b, text)
        res = true_attacks(f, a, toks, res, trusted_ends=bool(scribe_words(f)))
        ids = display_chunks(toks)
        for (disp, sp, key, end), (t0, t1), c in zip(toks, res, ids):
            words.append(dict(line=li, chunk=c, w=disp, key=key, t0=round(start + t0, 3), t1=round(start + t1, 3), voice=voice))
        print(f'{start:6.2f} ' + ' '.join(f'{w["w"]}@{w["t0"]:.2f}' for w in words if w['line'] == li))
    json.dump(words, open('/home/claude/voynich/edit/words.json', 'w'), ensure_ascii=False, indent=0)


if __name__ == '__main__':
    main()
