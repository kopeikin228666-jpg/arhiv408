import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SR=48000
def rmsdb(x,hop=2400):
    x=x.mean(1) if x.ndim==2 else x
    k=len(x)//hop; return 20*np.log10(np.sqrt((x[:k*hop].reshape(k,hop)**2).mean(1))+1e-6)
def plot(stems, out, t0=0, t1=130, marks=()):
    fig,ax=plt.subplots(figsize=(26,6))
    for nm,x in stems.items():
        d=rmsdb(x); t=np.arange(len(d))*0.05
        m=(t>=t0)&(t<=t1); ax.plot(t[m],d[m],label=nm,lw=.8)
    for m in marks: ax.axvline(m,color='k',lw=.4,alpha=.4)
    ax.set_ylim(-70,5); ax.legend(); ax.grid(alpha=.3); ax.set_xticks(np.arange(t0,t1+1,2 if t1-t0>40 else .5))
    fig.tight_layout(); fig.savefig(out,dpi=70)
if __name__=='__main__':
    import meas
    bed,vo,fx=meas.load()
    import os; os.environ['VOY_VER']='B'; sys.path.insert(0,'/home/claude/voynich/edit'); import sync
    plot({'bed':bed*.9,'vo':vo*1.5,'fx':fx}, 'plots/base_levels.png', marks=sync.strong(0,130))
