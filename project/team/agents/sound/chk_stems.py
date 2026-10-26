import numpy as np, subprocess
E='/home/claude/voynich/edit'
x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',f'{E}/mix_B_C.wav','-ac','1','-ar','16000','-f','f32le','-']),np.float32)
v,b,o=[np.load(f'{E}/{n}_B.npy') for n in ('vo','bed','other')]
n=min(len(x),len(v)); x=x[:n]
A=np.stack([v[:n],b[:n],o[:n]],1)
c,res,_,_=np.linalg.lstsq(A,x,rcond=None)
r=x-A@c
print('coef',c,'resid rel dB',10*np.log10((r**2).sum()/(x**2).sum()))
from scipy.signal import butter,sosfiltfilt
sos=butter(4,[300,3400],'bandpass',fs=16000,output='sos')
V=sosfiltfilt(sos,c[0]*v[:n]); X=sosfiltfilt(sos,x); R=X-V
for a,bb in [(8.9,10.6),(36.55,38.55),(26.9,29.9),(42.55,46.6),(47.467,64.3),(108.77,113.7)]:
    s=slice(int(a*16000),int(bb*16000))
    print(a,bb,'voice band dB %.1f rest band dB %.1f'%(10*np.log10((V[s]**2).mean()+1e-12),10*np.log10((R[s]**2).mean()+1e-12)))
