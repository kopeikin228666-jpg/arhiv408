"""Music bed for the 2:00 reel.
film 0:00-0:19.5  track1 from T1_OFF (drop lands at 0:00.5)
film 0:19.5       tape-stop over 8 frames, then silence
film 0:20.0       track2 from T2_OFF (a boom lands exactly on 0:20)
"""
import subprocess, json, numpy as np

SR = 48000
FILM = 120.0
T1_OFF = 4.7
T2_OFF = 94.27
CUT = 19.5
TS = 8 / 30
GAP_END = 20.0

def load(path, start, dur):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(dur), '-i', path,
                                   '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'])
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()

out = np.zeros((int(FILM * SR), 2), np.float32)
a = load('track1.mp3', T1_OFF, CUT + 1.0)
n1 = int(CUT * SR)
out[:n1] = a[:n1]
# tape stop: playback speed ramps 1 -> 0 over TS seconds
nts = int(TS * SR)
speed = np.linspace(1, 0, nts) ** 1.4
pos = n1 + np.cumsum(speed)
idx = np.clip(pos.astype(int), 0, len(a) - 1)
seg = a[idx] * np.linspace(1, 0.2, nts)[:, None]
out[n1:n1 + nts] = seg
# short fade-in at the very start, de-click at the cut
fi = int(0.05 * SR); out[:fi] *= np.linspace(0, 1, fi)[:, None]

b = load('track2.mp3', T2_OFF, FILM - GAP_END)
s2 = int(GAP_END * SR)
out[s2:s2 + len(b)] = b[:len(out) - s2]
# gentle limiter
peak = np.abs(out).max()
out *= 0.89 / max(peak, 1e-6)

p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                      '-c:a', 'pcm_s16le', 'music_bed.wav'], stdin=subprocess.PIPE)
p.stdin.write(out.astype(np.float32).tobytes()); p.stdin.close(); p.wait()

an = json.load(open('analysis.json'))
t1 = [round(x - T1_OFF, 3) for x in an['track1']['onsets'] if T1_OFF <= x < T1_OFF + CUT]
t2 = [round(x - T2_OFF + GAP_END, 3) for x in an['track2']['strong'] if T2_OFF - 0.1 <= x < T2_OFF + FILM - GAP_END]
json.dump({'act1_hits': t1, 'act23_booms': t2, 'tape_stop': CUT, 'track2_start': GAP_END}, open('cue_sheet.json', 'w'), indent=1)
print('act1 hits', len(t1), t1[:60])
print('act2/3 booms', t2)
