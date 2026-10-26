import os, sys, importlib.util, numpy as np, cv2
os.environ['PART']='3'
sys.path.insert(0,'/home/claude/voynich/edit')
spec=importlib.util.spec_from_file_location('ep',os.environ.get('EP','/home/claude/voynich/team/host/make_ep1e.py')); ep=importlib.util.module_from_spec(spec)
sys.argv=['x']; spec.loader.exec_module(ep)
out_path=os.environ.get('OUT','/home/claude/voynich/team/agents/anim_r2/sheet.jpg')
ts=[float(x) for x in os.environ.get('TS','0.0,0.3,2.9,7.5,10.0,15.0,20.0,31.5,31.9,52.3').split(',')]
out=[]
for t in ts:
    f=int(round(t*ep.FPS)); i,sg=ep.seg_at(t)
    g,z,cx,cy,rot=ep.seg_frame(i,t,f)
    fr=ep.finish(cv2.cvtColor(g,cv2.COLOR_BGR2RGB),f,z,cx,cy,rot,drift=sg['kind']!='reel')
    fr=ep.with_text(fr,t,f)
    sm=cv2.resize(fr,(270,480),interpolation=cv2.INTER_AREA)
    cv2.putText(sm,f'{t:.2f} {sg["kind"]}',(4,14),0,0.45,(0,255,0),1)
    out.append(sm)
while len(out)%5: out.append(np.zeros_like(out[0]))
rows=[np.hstack(out[k:k+5]) for k in range(0,len(out),5)]
cv2.imwrite(out_path,cv2.cvtColor(np.vstack(rows),cv2.COLOR_RGB2BGR))
# pip test: his live window on a dark grid, a few times
if os.environ.get('PIP','1')=='1':
    sp=ep.NARR_SPANS[0]; tiles=[]
    for k in range(5):
        ft=sp[0]+0.3+k*0.37; t=40.0+k*0.37
        g=np.zeros((320,180,3),np.uint8); g[:]=ep.HO.BGR[ep.HO.D3]
        g=ep.pip(g,t,ft)
        tiles.append(cv2.resize(g[45:120,120:180],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(out_path.replace('.jpg','_pip.png'),np.hstack(tiles))
print('ok')
