"""same word-window SNR as chk.py, but on the reel mix itself (film time): python3 chk_film.py [mix.wav]"""
import sys, json, subprocess, numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
SR=48000; F=480
wav = sys.argv[1] if len(sys.argv)>1 else '/home/claude/voynich/edit/mix_B_C.wav'
x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-ss','40','-to','130','-i',wav,'-ac','1','-f','f32le','-']),np.float32).astype(np.float64)
O=40.0
st=resample_poly(np.load('/home/claude/voynich/edit/vo_B.npy').astype(np.float64),3,1)[int(O*SR):int(O*SR)+len(x)]
sos=butter(4,[300,4000],'bandpass',fs=SR,output='sos'); xb=sosfiltfilt(sos,x); rb=sosfiltfilt(sos,st)
def fdb(z):
    m=len(z)//F; return 10*np.log10((z[:m*F].reshape(m,F)**2).mean(1)+1e-14)
def snr(t0,t1):
    a,b=int((t0-O)*SR),int((t1-O)*SR); r,e=rb[a:b],xb[a:b]; g=np.dot(e,r)/np.dot(r,r)
    vf,of=fdb(g*r),fdb(e-g*r); m=vf>vf.max()-12
    return 10*np.log10((10**(vf[m]/10)).sum())-10*np.log10((10**(of[m]/10)).sum())
for n,a,b in [('Растения',47.59,47.90),('не-',48.72,49.0),('в',52.86,53.08),('Их',56.41,56.68),('не(те)',58.13,58.41),('видеть',62.0,62.39)]:
    print(f'{n:10s} {snr(a,b):5.1f}')
