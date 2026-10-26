import os, sys
os.environ['VOY_VER']='B'
sys.path.insert(0,'/home/claude/voynich/edit')
import numpy as np, cv2
import render_v5 as R
names = sys.argv[1].split(',')
us = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [0,.5,1]
sh = R.build()
for nm in names:
    s = [x for x in sh if x.name==nm][0]
    tiles=[]
    for u in us:
        tl = u*(s.t1-s.t0)*0.999
        g = s.fn(tl+s.toff, u, np.random.default_rng(0), s.t0+tl)
        np.save(f'/home/claude/voynich/team/anim/raw/{nm}_{u:.2f}.npy', g)
        b = cv2.resize(g,(540,960),interpolation=cv2.INTER_NEAREST)
        for x in range(0,180,10):
            b[:, x*3] = (0,0,255) if x%50==0 else ((0,140,255) if x%20==0 else b[:,x*3])
        for y in range(0,320,10):
            b[y*3,:] = (0,0,255) if y%50==0 else ((0,140,255) if y%20==0 else b[y*3,:])
        cv2.putText(b,f'{nm} u{u}',(5,20),0,.6,(0,255,0),1)
        tiles.append(b)
    cv2.imwrite(f'/home/claude/voynich/team/anim/raw/{nm}.png', np.hstack(tiles))
