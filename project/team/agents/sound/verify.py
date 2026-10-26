"""verification of a levelled mix: dialogue loudness per segment, per-word speech-band SNR, limiter activity."""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/voynich/team/host'); sys.path.insert(0, '/home/claude/voynich/edit')
import ep_level as EL
from scipy.signal import butter, sosfiltfilt, resample_poly
SR = 48000; F = 480

def report(out, voice_out, segs, words408, lim_gain=None, act=None):
    N = len(out)
    pk = EL.kpower(out, SR)
    if act is not None:
        a_s = np.repeat(act, F); a_s = np.concatenate([a_s, np.zeros(max(0, N - len(a_s)), bool)])[:N]
        print('dialogue loudness per segment (speech-gated BS.1770):')
        for sg in segs:
            a, b = int(sg['t0'] * SR), int(min(sg['t1'], N / SR) * SR)
            if a_s[a:b].mean() > 0.12:
                print(f"   {sg['kind']:6s} {sg['t0']:5.1f}-{sg['t1']:5.1f}  {EL.gated(pk[a:b], SR, a_s[a:b]):6.1f} LUFS")
    sos = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
    V = sosfiltfilt(sos, voice_out.mean(1)); Rr = sosfiltfilt(sos, (out - voice_out).mean(1))
    vf, rf = EL._frames_db(V, F), EL._frames_db(Rr, F)
    def word_snr(t0, t1, vf, rf):
        i0, i1 = int(t0 * 100), max(int(t0 * 100) + 3, int(t1 * 100))
        v, r = vf[i0:i1], rf[i0:i1]
        m = v > v.max() - 12                           # the word's voiced frames
        return 10 * np.log10((10 ** (v[m] / 10)).mean()) - 10 * np.log10((10 ** (r[m] / 10)).mean()), (v[m] - r[m]).min()
    print('408 words: energy SNR / worst 10 ms frame (300-3400 Hz):')
    bad = []
    row = []
    for w in words408:
        s, wm = word_snr(w['t0'], min(w['t1'], w['t0'] + 1.0), vf, rf)
        row.append(f"{w['w'][:10]:10s}{s:5.1f}/{wm:5.1f}")
        if s < 8: bad.append((w['w'], round(w['t0'], 2), round(s, 1)))
    for j in range(0, len(row), 4): print('   ' + ' | '.join(row[j:j + 4]))
    print('   408 words < 8 dB:', bad if bad else 'none')
    # reel narrator: its stem (16 kHz mono film time) mapped into episode time, fitted to the mix per segment
    vr = np.load(EL.VO_REEL); W = json.load(open('/home/claude/voynich/edit/words.json'))
    sos16 = butter(4, [300, 3400], 'bandpass', fs=16000, output='sos')
    ob = sosfiltfilt(sos, out.mean(1))
    rbad = []; rrow = []
    for sg in segs:
        if sg['kind'] != 'reel': continue
        fa, fb = sg['src']; a = int(sg['t0'] * SR)
        x = vr[int(fa * 16000):int(fb * 16000)].astype(np.float64)
        if np.abs(x).max() < 1e-4: continue
        xb = resample_poly(sosfiltfilt(sos16, x), 3, 1)
        ob_ = ob[a:a + len(xb)]; xb = xb[:len(ob_)]
        c = (ob_ * xb).sum() / (xb * xb).sum()
        vv = np.zeros(N); vv[a:a + len(xb)] = c * xb
        rr = ob - vv
        vf2, rf2 = EL._frames_db(vv, F), EL._frames_db(rr, F)
        for w in W:
            if fa <= w['t0'] < fb - 0.1:
                t0 = sg['t0'] + w['t0'] - fa; t1 = sg['t0'] + min(w['t1'], w['t0'] + 0.8, fb) - fa
                s, wm = word_snr(t0, t1, vf2, rf2)
                rrow.append(f"{w['w'][:10]:10s}{s:5.1f}")
                if s < 8: rbad.append((w['w'], round(t0, 2), round(s, 1)))
    print('reel narrator words (approx., stem fit - pessimistic):')
    for j in range(0, len(rrow), 5): print('   ' + ' | '.join(rrow[j:j + 5]))
    print('   reel words < 8 dB:', rbad if rbad else 'none')
    if lim_gain is not None:
        g = 20 * np.log10(lim_gain)
        idx = np.where(g < -1.0)[0]
        if len(idx):
            t = idx / SR; spans = []; s0 = t[0]; pv = t[0]
            for x in t[1:]:
                if x - pv > 0.05: spans.append((s0, pv)); s0 = x
                pv = x
            spans.append((s0, pv))
            print('limiter > 1 dB GR spans:', ', '.join(f'{a:.2f}-{b:.2f}({g[int(a*SR):int(b*SR)+1].min():.1f})' for a, b in spans))
        print('limiter: max GR %.1f dB, time with GR>1 dB %.2f s, >3 dB %.2f s' % (-g.min(), (g < -1).sum() / SR, (g < -3).sum() / SR))

if __name__ == '__main__':
    d = np.load(sys.argv[1]); J = json.load(open('/home/claude/voynich/team/agents/sound/segs.json'))
    report(d['out'], d['voice_out'], J['segs'], J['words'], d['lim_gain'] if 'lim_gain' in d else None, d['act'] if 'act' in d else None)
