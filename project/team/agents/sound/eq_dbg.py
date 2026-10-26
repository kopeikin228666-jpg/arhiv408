import numpy as np, sys, json
sys.path.insert(0,'/home/claude/voynich/team/host'); import ep_level as EL
from scipy.signal import butter, sosfiltfilt
d=np.load('raw.npz'); SR=48000; F=480
# replicate the EQ curve
voice=d['vo'].astype(np.float64); rest=d['mix']-voice; vm=voice.mean(1)
sos_s=butter(4,[300,3400],'bandpass',fs=SR,output='sos')
act=EL._activity_from_stem(vm,SR,F)
sm=lambda x: 10*np.log10(EL.uniform_filter1d(10**(x/10),3,mode='nearest'))
vb=sm(EL._frames_db(sosfiltfilt(sos_s,vm),F)); rb=sm(EL._frames_db(sosfiltfilt(sos_s,rest.mean(1)),F))
need=np.minimum(0,(vb-EL.SNR_DB)-rb); sp=act&(vb>vb[act].max()-35); need=np.where(sp,np.maximum(need,-24),0)
g=EL._smooth_min(need,15,10)
print('>3dB %.1f%%  >6dB %.1f%%'%((g<-3).mean()*100,(g<-6).mean()*100))
for t in np.arange(0,60.7,0.5):
    seg=g[int(t*100):int(t*100)+50]
    if seg.min()<-3: print(f'{t:5.1f} min {seg.min():6.1f} mean {seg.mean():6.1f}')
p95=np.percentile(vb[act],95)
m=need<-12
print('p95',p95,'frames need<-12:',m.sum(),' vb rel p95 quantiles',np.percentile(vb[m]-p95,[10,50,90]),' rb rel',np.percentile(rb[m]-p95,[10,50,90]))
m2=sp&(vb>p95-15)
print('loud speech frames',m2.sum(),'need quantiles',np.percentile(need[m2],[5,25,50]))
