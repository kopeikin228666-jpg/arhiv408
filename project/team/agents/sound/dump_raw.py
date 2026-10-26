# dump the pre-master mix + 408 voice stem + SEG list from make_ep1d (PART=3) without touching its outputs
import os, sys, inspect, json, numpy as np
os.environ['PART'] = '3'
sys.path.insert(0, '/home/claude/voynich/team/host')
import make_ep1d as M
src = inspect.getsource(M.render_audio)
i = src.index('    mix = mix[:int(T_END * SR)]')
src = src[:i] + "    mix = mix[:int(T_END * SR)]\n    return mix\n"
src = src.replace('def render_audio(out):', 'def render_audio_raw():')
exec(src, M.__dict__)
mix = M.render_audio_raw()
segs = [{k: (v if isinstance(v, (int, float, str)) else (list(v) if isinstance(v, tuple) and k == 'src' else None))
         for k, v in sg.items() if k in ('kind', 't0', 't1', 'dur', 'src', 'pip')} for sg in M.SEG]
words = [dict(w=w['w'], t0=w['t0'], t1=w['t1']) for w in M.ALLW]
np.savez('/home/claude/voynich/team/agents/sound/raw.npz', mix=mix, vo=(M.vo * 1.35)[:len(mix)], vo_raw=M.vo[:len(mix)])
json.dump(dict(segs=segs, words=words, T_END=M.T_END), open('/home/claude/voynich/team/agents/sound/segs.json', 'w'), ensure_ascii=False, indent=0)
print('ok', mix.shape, np.abs(mix).max())
