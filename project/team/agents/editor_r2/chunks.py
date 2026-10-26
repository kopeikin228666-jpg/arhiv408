import os, sys, importlib.util
os.environ['PART']='3'
spec=importlib.util.spec_from_file_location('ep','/home/claude/voynich/team/host/make_ep1e.py'); ep=importlib.util.module_from_spec(spec)
sys.argv=['x']; spec.loader.exec_module(ep)
R=ep.R
def show(ws, tag):
    g={}
    for w in ws: g.setdefault(w['chunk'],[]).append(w['w'])
    for c,v in sorted(g.items()):
        print(tag, c, ' '.join(v), '| fits' if ep._fits_narrow(v) else '| 3+ LINES')
for sg in ep.SEG:
    if sg.get('words'): show(sg['words'], 'HIS')
used=[sg['src'] for sg in ep.SEG if sg['kind']=='reel']
for ws in R._LINES:
    if any(a-1<ws[0]['t0']<b for a,b in used): show(ws, 'REEL %.1f'%ws[0]['t0'])
