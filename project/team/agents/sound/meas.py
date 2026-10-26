"""ffmpeg ebur128 report: integrated, LRA, true peak, short-term every 0.5 s, per-segment ST ranges."""
import subprocess, re, sys, json, numpy as np
def ebur(path):
    p = subprocess.run(['ffmpeg', '-nostats', '-v', 'verbose', '-i', path, '-af', 'ebur128=peak=true:framelog=verbose', '-f', 'null', '-'],
                       capture_output=True, text=True)
    t, M, S = [], [], []
    for ln in p.stderr.splitlines():
        m = re.search(r't:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.]+|-inf)\s+S:\s*(-?[\d.]+|-inf)', ln)
        if m:
            t.append(float(m.group(1))); M.append(float(m.group(2))); S.append(float(m.group(3)))
    tail = p.stderr[p.stderr.rfind('Summary:'):]
    I = float(re.search(r'I:\s*(-?[\d.]+)', tail).group(1)); LRA = float(re.search(r'LRA:\s*([\d.]+)', tail).group(1))
    TP = float(re.search(r'Peak:\s*(-?[\d.]+|-inf)', tail).group(1))
    return np.array(t), np.array(M), np.array(S), I, LRA, TP
if __name__ == '__main__':
    path = sys.argv[1]
    t, M, S, I, LRA, TP = ebur(path)
    print(f'{path}\n  integrated {I:.1f} LUFS   LRA {LRA:.1f} LU   true peak {TP:.1f} dBTP')
    sel = (t >= 3.0)
    print('  short-term (3 s) range over t>=3: %.1f .. %.1f' % (S[sel].min(), S[sel].max()))
    sel2 = (t >= 3.0) & (t <= 58.3)
    print('  short-term range excluding card: %.1f .. %.1f  (spread %.1f)' % (S[sel2].min(), S[sel2].max(), S[sel2].max()-S[sel2].min()))
    row = []
    for k in np.arange(0.5, t.max() + 0.01, 0.5):
        i = np.argmin(abs(t - k)); row.append(f'{k:4.1f}:{S[i]:6.1f}')
    for j in range(0, len(row), 8):
        print('   ' + '  '.join(row[j:j + 8]))
    if len(sys.argv) > 2:
        segs = json.load(open(sys.argv[2]))['segs']
        for sg in segs:
            s = (t > sg['t0'] + 0.2) & (t <= sg['t1'])
            print(f"  {sg['kind']:6s} {sg['t0']:5.1f}-{sg['t1']:5.1f}  M max {M[s].max():6.1f}  S min {S[s].min():6.1f} max {S[s].max():6.1f}")
