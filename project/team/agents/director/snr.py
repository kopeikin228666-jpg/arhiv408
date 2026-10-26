import numpy as np, subprocess
from scipy.signal import butter, sosfiltfilt, resample_poly
SR=48000; A='/home/claude/voynich/audio'; E='/home/claude/voynich/edit'
def load(path,a=None,b=None,af=None):
    cmd=['ffmpeg','-v','error']+(['-ss',str(a)] if a is not None else [])+(['-to',str(b)] if b is not None else [])+['-i',path]
    if af: cmd+=['-af',af]
    cmd+=['-ac','1','-ar',str(SR),'-f','f32le','-']
    return np.frombuffer(subprocess.check_output(cmd),np.float32).copy()
VOICE=('aresample=48000,highpass=f=70,lowpass=f=11000,equalizer=f=150:t=q:w=1:g=3,equalizer=f=3000:t=q:w=1.5:g=2,'
       'acompressor=threshold=-24dB:ratio=4:attack=5:release=120:makeup=5,aecho=0.8:0.4:60|140:0.12|0.07')
ep=load('/home/claude/voynich/team/host/ep1d_p3/ep.wav'); N=len(ep)
vo=np.zeros(N,np.float32)
def place(buf,x,at):
    i=int(round(at*SR)); n=min(len(x),len(buf)-i); buf[i:i+n]+=x[:n]
# host lines (file,a,b,at)
for f,a,b,at in [('vo_narrator.mp3',0.0,3.81,0.35),('vo_intro.mp3',0.0,5.49,6.9),('vo_written.mp3',0.06,3.3,14.49),('vo_narrator.mp3',4.74,10.76,20.33)]:
    a0=max(0,a-0.06); x=load(f'{A}/{f}',a0,b+0.3,VOICE); place(vo,x,at-(a-a0))
# reel narrator stem
vb=np.load(f'{E}/vo_B.npy'); vb48=resample_poly(vb,3,1).astype(np.float32)
for t0,a,b in [(12.59,8.9,10.6),(17.88,36.55,38.55),(26.6,26.9,29.9),(29.6,42.55,46.6),(36.55,47.467,64.3),(53.383,108.77,113.7)]:
    place(vo,vb48[int(a*SR):int(b*SR)],t0)
sos=butter(4,[300,4000],'bandpass',fs=SR,output='sos')
epb=sosfiltfilt(sos,ep); vob=sosfiltfilt(sos,vo)
W=int(0.25*SR); res=[]
# fit gain per 2 s windows by least squares, residual = background
for k in range(0,N-W,W):
    s=slice(k,k+W); v=vob[s]; e=epb[s]
    pv=np.mean(v**2)
    if pv<1e-7: continue
    lo=max(0,k-4*W); hi=min(N,k+5*W)
    g=np.dot(epb[lo:hi],vob[lo:hi])/(np.dot(vob[lo:hi],vob[lo:hi])+1e-12)
    bg=e-g*v
    snr=10*np.log10(np.mean((g*v)**2)/(np.mean(bg**2)+1e-12))
    vdb=10*np.log10(np.mean((g*v)**2)+1e-12)
    res.append((k/SR,snr,vdb))
for t,s,v in res:
    flag=' <8dB' if s<8 else ''
    print(f'{t:6.2f} voice {v:6.1f} dB  speech/bg {s:5.1f} dB{flag}')
