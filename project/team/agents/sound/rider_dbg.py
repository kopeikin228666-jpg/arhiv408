import numpy as np, sys
sys.path.insert(0,'/home/claude/voynich/team/host'); import ep_level as EL, json
d=np.load('raw.npz'); J=json.load(open('segs.json'))
out=EL.level(d['mix'],J['segs'],48000,voice=d['vo'],verbose=False)
r=EL.LAST['rider_db']; r=np.where(np.isfinite(r),r,0)
for t in np.arange(0,60.7,0.25):
    a=int(t*48000); m=np.median(r[a:a+12000])
    if m<-0.5: print(f'{t:5.2f} rider {m:5.1f}')
