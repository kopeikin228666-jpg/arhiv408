"""fast loop: run sound_design on the saved stems (bed_d before plug-ins, vo, fx, raw bed) like mix2 does"""
import os, sys, time, json, numpy as np
os.environ['VOY_VER'] = 'B'
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/team/sound')
os.chdir('/home/claude/voynich/edit')
import sync, render as RN, importlib
D = '/home/claude/voynich/team/sound/'
SR = 48000
bed_d = np.load(D + 'st_bed.npy'); vo = np.load(D + 'st_vo.npy'); fx = np.load(D + 'st_fx.npy'); bed = np.load(D + 'st_bedraw.npy')
N = len(vo)
shots = RN.build()
W = json.load(open('/home/claude/voynich/edit/words.json'))
import sound_design as SD
t = time.time()
lay = SD.design('B', shots, W, sync, SR, N)
g = SD.duck('B', shots, W, sync, SR, N)
print('design+duck s', round(time.time() - t, 1), 'layer peak', float(np.abs(lay).max()))
fx2 = fx + lay
bed2 = bed_d * g[:, None]
mix = bed2 * .9 + vo * 1.5 + fx2
mix[int(129.4 * SR):] *= np.linspace(1, 0, N - int(129.4 * SR))[:, None]
pk = np.abs(mix).max()
base = bed_d * .9 + vo * 1.5 + fx
pk0 = np.abs(base).max()
drv = np.abs(mix).max(1) / pk * 1.4
drv0 = np.abs(base).max(1) / pk0 * 1.4
print(f'pre-limiter peak {pk:.4f} (base {pk0:.4f}) at {np.argmax(np.abs(mix).max(1))/SR:.2f}s; '
      f'samples with drive>1: {(drv>1).mean()*100:.4f}% (base {(drv0>1).mean()*100:.4f}%), >1.2: {(drv>1.2).mean()*100:.4f}%')
np.save(D + 'tl_layer.npy', lay); np.save(D + 'tl_gain.npy', g)
np.save(D + 'tl_mix.npy', (np.tanh(mix / pk * 1.4) * .92).astype(np.float32))
for k in ('vacuums', 'words_worst', 'words_min_db', 'words_median_db', 'peak_guard_min_gain', 'peak_guard_time_below_0.9_s', 'layer_peak', 'bed_gain_min_on_words'):
    print(k, SD.REPORT.get(k))
