# raw 180x320 grids of chosen times (no camera, no text) side by side, x2
import os, sys, importlib.util, numpy as np, cv2, io, contextlib
os.environ['PART']='3'
sys.path.insert(0,'/home/claude/voynich/edit')
spec=importlib.util.spec_from_file_location('ep',os.environ.get('EP','/home/claude/voynich/team/host/make_ep1e.py')); ep=importlib.util.module_from_spec(spec)
sys.argv=['x']
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(ep)
ts=[float(x) for x in os.environ['TS'].split(',')]
mode=os.environ.get('MODE','grid')
out=[]
for t in ts:
    f=int(round(t*ep.FPS)); i,sg=ep.seg_at(t)
    g,z,cx,cy,rot=ep.seg_frame(i,t,f)
    if mode=='full':
        fr=ep.finish(cv2.cvtColor(g,cv2.COLOR_BGR2RGB),f,z,cx,cy,rot,drift=sg['kind']!='reel')
        fr=ep.with_text(fr,t,f)
        sm=cv2.cvtColor(cv2.resize(fr,(360,640),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
    else:
        sm=cv2.resize(g,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    cv2.putText(sm,f'{t:.2f}',(4,14),0,0.45,(0,255,0),1)
    out.append(sm)
cv2.imwrite(os.environ['OUT'],np.hstack(out))
print('ok')
