import subprocess, sys, json, numpy as np
from scipy.signal import find_peaks, stft
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 22050

def load(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'])
    return np.frombuffer(raw, np.float32)

def analyze(path):
    y = load(path)
    hop = 512
    f, t, Z = stft(y, SR, nperseg=2048, noverlap=2048 - hop)
    mag = np.log1p(np.abs(Z) * 50)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(0)
    flux = np.concatenate([[0], flux])
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    low = mag[f < 150].mean(0)                     # kick / sub energy
    rms = np.sqrt(np.convolve(y ** 2, np.ones(hop) / hop, 'same'))[::hop]
    rms = np.pad(rms, (0, max(0, len(t) - len(rms))), mode='edge')[:len(t)]
    fps = SR / hop
    # tempo by autocorrelation of the onset envelope (60..180 bpm)
    env = flux - flux.mean()
    ac = np.correlate(env, env, 'full')[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm_l = 60 * fps / np.maximum(lags, 1)
    ok = (bpm_l > 60) & (bpm_l < 180)
    lag = lags[ok][np.argmax(ac[ok])]
    bpm = 60 * fps / lag
    peaks, props = find_peaks(flux, height=1.5, distance=int(fps * 0.12))
    strong, _ = find_peaks(flux, height=3.0, distance=int(fps * 0.25))
    return dict(dur=len(y) / SR, bpm=float(bpm), t=t, flux=flux, low=low, rms=rms,
                onsets=(t[peaks]).tolist(), strong=(t[strong]).tolist())

if __name__ == '__main__':
    out = {}
    fig, axs = plt.subplots(2, 1, figsize=(18, 7))
    for ax, (name, p) in zip(axs, [('track1', 'track1.mp3'), ('track2', 'track2.mp3')]):
        r = analyze(p)
        out[name] = dict(dur=r['dur'], bpm=r['bpm'], onsets=r['onsets'], strong=r['strong'])
        ax.plot(r['t'], r['rms'] / r['rms'].max(), lw=0.6, label='rms')
        ax.plot(r['t'], (r['low'] - r['low'].min()) / (np.ptp(r['low']) + 1e-9), lw=0.4, alpha=0.6, label='low')
        for s in r['strong']:
            ax.axvline(s, color='r', lw=0.4, alpha=0.5)
        ax.set_xticks(np.arange(0, r['dur'], 5)); ax.grid(alpha=0.3)
        ax.set_title(f"{name}  dur {r['dur']:.1f}s  ~{r['bpm']:.1f} bpm  strong onsets {len(r['strong'])}")
        ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig('energy.png', dpi=80)
    json.dump(out, open('analysis.json', 'w'))
    for k, v in out.items():
        print(k, round(v['dur'], 1), 'bpm', round(v['bpm'], 1), 'strong', len(v['strong']), [round(x, 2) for x in v['strong'][:40]])
