"""director_r2/snr3.py, unchanged maths, but the wav is an argument and the 'guess' starts come from the current SEG.
usage: cd /home/claude/voynich/edit && PART=3 python3 snr3_ep.py ep.wav"""
import sys, os, io, contextlib, subprocess, numpy as np
wav = sys.argv[1]; sys.argv = [sys.argv[0]]
sys.path.insert(0, '/home/claude/voynich/team/host')
with contextlib.redirect_stdout(io.StringIO()):
    import make_ep1e as M
from scipy.signal import butter, sosfiltfilt, resample_poly, correlate
SR=48000
x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',wav,'-ac','1','-f','f32le','-']),np.float32)*0.5
vb=np.load(os.environ.get('STEM','/home/claude/voynich/edit/vo_B.npy')); vb=resample_poly(vb,3,1).astype(np.float32)
sos=butter(4,[300,4000],'bandpass',fs=SR,output='sos')
xb=sosfiltfilt(sos,x); vbb=sosfiltfilt(sos,vb)
LIST=[(47.59,47.90),(48.72,49.0),(52.86,53.08),(56.41,56.68),(58.13,58.41),(62.0,62.39)]
for sg in M.SEG:
    if sg['kind']!='reel': continue
    a,b=sg['src']; guess=sg['t0']
    v=vbb[int(a*SR):int(b*SR)]
    if np.mean(v**2)<1e-9: print(a,'no stem voice'); continue
    lo=int((guess-0.4)*SR); hi=int((guess+0.4+(b-a))*SR)
    seg=xb[lo:hi]
    c=correlate(seg,v,mode='valid',method='fft'); k=np.argmax(np.abs(c)); t0=(lo+k)/SR
    e=xb[lo+k:lo+k+len(v)]
    W=int(0.1*SR); out=[]
    for i in range(0,len(v)-W,W):
        l0=max(0,i-3*W); l1=min(len(v),i+4*W); g=np.dot(e[l0:l1],v[l0:l1])/np.dot(v[l0:l1],v[l0:l1])
        vv=g*v[i:i+W]; pv=np.mean(vv**2)
        if np.mean(v[i:i+W]**2)<0.05*np.mean(v**2): continue
        bg=e[i:i+W]-vv
        ft=a+i/SR; tag='*' if any(p<=ft+0.1 and ft<q for p,q in LIST) else ''
        out.append((ft, 10*np.log10(pv/(np.mean(bg**2)+1e-12)),tag))
    print(f"film {a:7.2f} start {t0:6.2f} (seg {guess:6.2f})", ' '.join(f"{t:.1f}:{s:.0f}{tg}" for t,s,tg in out))
