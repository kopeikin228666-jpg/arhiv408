import os, sys, importlib.util
os.environ['PART']='3'
spec=importlib.util.spec_from_file_location('ep','/home/claude/voynich/team/host/make_ep1e.py'); ep=importlib.util.module_from_spec(spec)
sys.argv=['x']; spec.loader.exec_module(ep)
for i,sg in enumerate(ep.SEG):
    print(i, sg['kind'], round(sg['t0'],2), round(sg['t1'],2), round(sg['dur'],2), sg.get('src',''))
    for w in sg.get('words',[]): print('     ', w['w'], round(w['t0'],2), round(w['t1'],2))
