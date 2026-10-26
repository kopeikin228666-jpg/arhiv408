"""overseer: every spoken word -> is it on screen (caption, visible flag) at its own attack, and is it cut by a segment edge?
his words: caption_at(sg, t0); reel words: film time mapped through each reel segment's src."""
import os, sys, importlib.util
os.environ['PART'] = '3'; sys.path.insert(0, '/home/claude/voynich/edit')
spec = importlib.util.spec_from_file_location('ep', os.environ.get('EP', '/home/claude/voynich/team/host/make_ep1e.py'))
ep = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(ep)
R = ep.R; bad = 0
def shown(t, word):
    lay_cap = None
    i, sg = ep.seg_at(t)
    if sg['kind'] == 'reel':
        c = R.caption_at(sg['src'][0] + t - sg['t0'])
    elif sg['kind'] in ('lab', 'close', 'hook'):
        c = ep.caption_at(sg, t)
        if sg['kind'] == 'hook' and t - sg['t0'] >= sg['title']: c = None
    else:
        c = None
    if not c: return False
    return any(w[0] == word and w[2] for w in c[0])
for sg in ep.SEG:
    for w in sg.get('words', []):
        ok = shown(w['t0'] + 0.001, w['w']) and shown(min(w['t1'], sg['t1'] - 0.001) - 0.001, w['w'])
        cut = w['t1'] > sg['t1'] + 0.01
        if not ok or cut:
            bad += 1; print('HIS ', sg['kind'], round(w['t0'], 2), round(w['t1'], 2), w['w'], 'NOT SHOWN' if not ok else '', 'CUT by seg end %.2f' % sg['t1'] if cut else '')
for ws in R._LINES:
    for w in ws:
        segs = [sg for sg in ep.SEG if sg['kind'] == 'reel' and sg['src'][0] - 0.02 <= w['t0'] < sg['src'][1]]
        if not segs:
            print('REEL word not in any reel segment (never heard?):', round(w['t0'], 2), w['w']); bad += 1; continue
        sg = segs[0]; te = sg['t0'] + w['t0'] - sg['src'][0]
        ok = shown(te + 0.001, w['w'])
        cut = w['t1'] > sg['src'][1] + 0.01
        if not ok or cut:
            bad += 1; print('REEL', round(w['t0'], 2), '->', round(te, 2), w['w'], 'NOT SHOWN' if not ok else '', f"CUT: word ends {w['t1']:.2f} > src end {sg['src'][1]:.2f}" if cut else '')
print('problems:', bad)
