import sys, json, time, subprocess, numpy as np
sys.path.insert(0, '/home/claude/voynich/team/host')
import importlib, ep_level; importlib.reload(ep_level)
d = np.load('raw.npz'); J = json.load(open('segs.json')); SR = 48000
segs = J['segs']
t = time.time()
out = ep_level.level(d['mix'], segs, SR, voice=d['vo'])
print('time', round(time.time() - t, 1))
p = '/home/claude/voynich/team/agents/sound/test.wav'
q = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-c:a', 'pcm_s16le', p], stdin=subprocess.PIPE)
q.stdin.write(out.tobytes()); q.stdin.close(); q.wait()
L = ep_level.LAST
np.savez('/home/claude/voynich/team/agents/sound/test_out.npz', out=out, voice_out=L['voice_out'], lim_gain=L['lim_gain'], act=L['act'])
