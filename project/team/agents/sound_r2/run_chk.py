"""render the episode audio exactly like `make_ep1e.py --audio-only` (to OUT, default ep1e_p3/ep.wav) and keep ep_level's
exact 408 voice for the checker.  usage: cd /home/claude/voynich/edit && PART=3 python3 run_chk.py [OUT.wav]"""
import os, sys, numpy as np
os.environ.setdefault('PART', '3')
out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/voynich/team/host/ep1e_p3/ep.wav'
sys.argv = [sys.argv[0]]
sys.path.insert(0, '/home/claude/voynich/team/host')
import make_ep1e as M
for k_ in ('BOOST_DB', 'BOOST_AUTO'):
    if os.environ.get(k_):
        setattr(M, k_, float(os.environ[k_]))
M.render_audio(out)
import ep_level as EL
np.savez('/home/claude/voynich/team/agents/sound_r2/last.v408.npz', voice_out=EL.LAST['voice_out'])
print('saved last.v408.npz')
