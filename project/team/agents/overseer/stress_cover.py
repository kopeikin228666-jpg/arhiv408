"""overseer: for every stressed (*) word of his lines, is 408 actually on screen during its stress peak
(t0 .. t0+0.2), or does an insert / flash / title / reel cover it?  no rendering."""
import os, sys, importlib.util
os.environ['PART'] = '3'; sys.path.insert(0, '/home/claude/voynich/edit')
spec = importlib.util.spec_from_file_location('ep', os.environ.get('EP', '/home/claude/voynich/team/host/make_ep1e.py'))
ep = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(ep)
tot = vis = 0
for i, sg in enumerate(ep.SEG):
    for w in sg.get('words', []):
        if not w['key']: continue
        tot += 1; cov = []
        for k in range(7):                         # 7 frames from the attack
            t = w['t0'] + k / ep.FPS
            j, s2 = ep.seg_at(t); tl = t - s2['t0']; why = s2['kind']
            if s2['kind'] in ('lab', 'close', 'lean', 'hook'):
                for (w_, film, du) in s2.get('inserts', []):
                    st = ep.word_time(s2, w_) + getattr(ep, 'INSERT_DELAY', -0.02)
                    if st <= t < st + du: why = 'INSERT'
                if s2['kind'] == 'hook':
                    for (st, du, film) in s2['flashes'] + s2['montage']:
                        if st <= tl < st + du: why = 'FLASH'
                    if tl >= s2['title']: why = 'title'
                    cs, ca, cb = s2['clip'] if s2.get('clip') else (99, 0, 0)
                    if cs <= tl < cs + (cb - ca): why = 'CLIP'
            cov.append(why)
        seen = sum(c in ('lab', 'close', 'lean', 'hook', 'title') for c in cov)
        vis += seen >= 4
        print(f'seg{i:2d} {sg["kind"]:5s} {w["t0"]:6.2f} {w["w"]:16s} 408 visible {seen}/7  {",".join(dict.fromkeys(cov))}')
print(f'stressed words with the reaction on screen (>=4 of 7 frames): {vis}/{tot}')
