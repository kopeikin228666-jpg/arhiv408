import sys, cv2, numpy as np
# sheet.py out t0 t1 step cols width
out=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3]); st=float(sys.argv[4]); cols=int(sys.argv[5]); W=int(sys.argv[6]) if len(sys.argv)>6 else 135
H=int(W*16/9); ims=[]
t=t0
while t<t1-1e-6:
    n=int(round(t*30))+1
    im=cv2.imread(f'sm/{n:05d}.jpg')
    if im is None: break
    im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)
    s=W/1080
    cv2.line(im,(0,int(H-380*s)),(W,int(H-380*s)),(255,0,255),1)
    cv2.line(im,(0,int(200*s)),(W,int(200*s)),(255,0,255),1)
    cv2.rectangle(im,(int(900*s),int(900*s)),(W-1,int(1700*s)),(255,0,255),1)
    cv2.rectangle(im,(0,0),(46,14),(0,0,0),-1)
    cv2.putText(im,f'{t:.2f}',(2,11),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1)
    ims.append(im); t+=st
rows=[]
for i in range(0,len(ims),cols):
    r=ims[i:i+cols]
    while len(r)<cols: r.append(np.zeros_like(ims[0]))
    rows.append(np.hstack(r))
cv2.imwrite(out,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,90])
