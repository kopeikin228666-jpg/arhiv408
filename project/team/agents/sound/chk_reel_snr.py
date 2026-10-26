import numpy as np, subprocess, sys
sys.path.insert(0,'/home/claude/voynich/edit'); from timeline import LINES
E='/home/claude/voynich/edit'
x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',f'{E}/mix_B_C.wav','-ac','1','-ar','16000','-f','f32le','-']),np.float32)
v=np.load(f'{E}/vo_B.npy'); n=min(len(x),len(v)); x=x[:n]; v=v[:n]
from scipy.signal import butter,sosfiltfilt
sos=butter(4,[300,3400],'bandpass',fs=16000,output='sos')
X=sosfiltfilt(sos,x); V=sosfiltfilt(sos,v)
# best lag+gain
for lag in range(-3,4):
    vv=np.roll(V,lag); c=(X*vv).sum()/(vv*vv).sum(); r=X-c*vv
    print(lag,c,10*np.log10((r**2).sum()/(X**2).sum()))
c=(X*V).sum()/(V*V).sum(); R=X-c*V
F=320
def fr(z): m=len(z)//F; return 10*np.log10((z[:m*F].reshape(m,F)**2).mean(1)+1e-12)
VF,RF=fr(c*V),fr(R)
for (f,a,b,st,tx,kind) in LINES:
    i0,i1=int(st*50),int((st+b-a)*50)
    act=VF[i0:i1]>VF[i0:i1].max()-20
    snr=(VF[i0:i1]-RF[i0:i1])[act]
    print(f'{st:6.2f}-{st+b-a:6.2f} {kind:7s} med SNR {np.median(snr):5.1f}  p10 {np.percentile(snr,10):5.1f}  {tx[:30]}')
