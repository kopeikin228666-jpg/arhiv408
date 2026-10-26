# face crops of the close sprite under several states, x4
import os, sys, numpy as np, cv2
sys.path.insert(0,'/home/claude/voynich/edit')
import host as HO
states=eval(os.environ.get('ST',"[dict(t=0.3),dict(t=0.3,amp=0.9),dict(t=0.3,mood='scary'),dict(t=0.0,mood='scary')]"))
tiles=[]
for st in states:
    pose=st.pop('pose','close'); t=st.pop('t',0.0)
    g=np.zeros((320,180,3),np.uint8); g[:]=HO.BGR[HO.DARK]
    HO.draw(g,90,347,t,st.pop('amp',0.0),1,st.pop('blink',False),st.pop('look',0),0,int(t*15),st.pop('mood','calm'),pose,aura=True,smoke=30,**st)
    tiles.append(cv2.resize(g[40:240,10:170],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
cv2.imwrite(os.environ.get('OUT','/home/claude/voynich/team/agents/anim_r2/faces.png'),np.hstack(tiles))
