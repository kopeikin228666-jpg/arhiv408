import cv2, numpy as np, glob
fs=sorted(glob.glob('sm/*.jpg'))
prev=None; rows=[]
for i,f in enumerate(fs):
    g=cv2.imread(f,0).astype(np.float32)
    h=cv2.calcHist([g.astype(np.uint8)],[0],None,[32],[0,256]).ravel(); h/=h.sum()
    if prev is None: d=0; hd=0
    else:
        d=np.mean(np.abs(g-prev[0])); hd=0.5*np.abs(h-prev[1]).sum()
    rows.append((i/30, g.mean(), d, hd)); prev=(g,h)
np.save('stats.npy',np.array(rows))
