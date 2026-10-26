"""run mix2.py B with the sound_design plug-in disabled and dump the stems (48k stereo float32)"""
import os, sys, runpy, numpy as np
os.environ['SD_OFF'] = '1'
os.chdir('/home/claude/voynich/edit')
sys.argv = ['mix2.py', 'B']
sys.path.insert(0, '/home/claude/voynich/edit')
g = runpy.run_path('/home/claude/voynich/edit/mix2.py', run_name='__main__')
D = '/home/claude/voynich/team/sound/'
np.save(D + 'st_bed.npy', g['bed_d'].astype(np.float32))
np.save(D + 'st_vo.npy', g['vo'].astype(np.float32))
np.save(D + 'st_fx.npy', g['fx'].astype(np.float32))
np.save(D + 'st_bedraw.npy', g['bed'].astype(np.float32))
print('pk', float(np.abs(g['bed_d'] * .9 + g['vo'] * 1.5 + g['fx']).max()))
