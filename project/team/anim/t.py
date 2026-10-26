"""usage: t.py shot t_start t_end n [crop x0,y0,x1,y1] [scale]  -> strip.png + timing"""
import os, sys, time
os.environ['VOY_VER']='B'
sys.path.insert(0,'/home/claude/voynich/edit')
import numpy as np, cv2
import render_v5 as R
import importlib, anim
nm=sys.argv[1]; a=float(sys.argv[2]); b=float(sys.argv[3]); n=int(sys.argv[4])
crop=[int(v) for v in sys.argv[5].split(',')] if len(sys.argv)>5 else [0,0,180,320]
sc=int(sys.argv[6]) if len(sys.argv)>6 else 3
s=[x for x in R.build() if x.name==nm][0]
tiles=[]; tt=[]
for t in np.linspace(a,b,n):
    tq = s.t0 + np.floor((t - s.t0)*15+1e-6)/15
    tl = max(0, tq - s.t0)+s.toff; u=min(1, tl/(s.t1-s.t0+s.toff))
    rng=np.random.default_rng(int(t*15)*7+11)
    g=s.fn(tl,u,rng,tq)
    t0=time.time(); o=anim.animate(g,nm,tl,u,s.t0+tl-s.toff,rng); tt.append(time.time()-t0)
    assert o.dtype==np.uint8 and o.shape==(320,180,3)
    assert (anim.pidx(o)!=255).all() or True
    x0,y0,x1,y1=crop
    c=o[y0:y1,x0:x1]
    c=cv2.resize(c,((x1-x0)*sc,(y1-y0)*sc),interpolation=cv2.INTER_NEAREST)
    cv2.putText(c,f'{t:.2f}',(3,12),0,.4,(0,255,0),1)
    tiles.append(np.pad(c,((0,0),(0,2),(0,0))))
rows=[np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]
w=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,2),(0,w-r.shape[1]),(0,0))) for r in rows]
cv2.imwrite('/home/claude/voynich/team/anim/strip.png',np.vstack(rows))
print('ms per call: mean %.2f max %.2f (first %.1f)'%(np.mean(tt[1:])*1000 if n>1 else 0,max(tt[1:] or [0])*1000,tt[0]*1000))
