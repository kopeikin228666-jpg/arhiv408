import numpy as np, subprocess
from scipy.signal import butter, sosfiltfilt, resample_poly, correlate
SR=48000
x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i','final.wav','-ac','1','-f','f32le','-']),np.float32)*0.5
vb=np.load('/home/claude/voynich/edit/vo_B.npy'); vb=resample_poly(vb,3,1).astype(np.float32)
sos=butter(4,[300,4000],'bandpass',fs=SR,output='sos')
xb=sosfiltfilt(sos,x); vbb=sosfiltfilt(sos,vb)
segs=[((47.467,50.05),33.43,'plants'),((51.55,54.25),36.0,'women'),((56.15,62.55),38.7,'bodies/artist'),((110.30,113.45),45.1,'whisper'),((121.05,124.40),48.3,'museum'),((22.55,25.15),22.6,'potom'),((27.40,29.55),25.1,'nikto'),((42.55,46.6),27.3,'bukvy')]
for (a,b),guess,name in segs:
    v=vbb[int(a*SR):int(b*SR)]
    if np.mean(v**2)<1e-9: print(name,'no stem voice'); continue
    lo=int((guess-0.4)*SR); hi=int((guess+0.4+(b-a))*SR)
    seg=xb[lo:hi]
    c=correlate(seg,v,mode='valid',method='fft'); k=np.argmax(np.abs(c)); t0=(lo+k)/SR
    e=xb[lo+k:lo+k+len(v)]
    g=np.dot(e,v)/np.dot(v,v)
    W=int(0.1*SR); out=[]
    for i in range(0,len(v)-W,W):
        l0=max(0,i-3*W); l1=min(len(v),i+4*W); g=np.dot(e[l0:l1],v[l0:l1])/np.dot(v[l0:l1],v[l0:l1])
        vv=g*v[i:i+W]; pv=np.mean(vv**2)
        if np.mean(v[i:i+W]**2)<0.05*np.mean(v**2): continue
        bg=e[i:i+W]-vv
        out.append((t0+i/SR, 10*np.log10(pv/(np.mean(bg**2)+1e-12))))
    print(f"{name:14s} start {t0:6.2f} gain {g:.2f} ", ' '.join(f"{t:.1f}:{s:.0f}" for t,s in out))
