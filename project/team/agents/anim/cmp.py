import sys, numpy as np, importlib
which = sys.argv[1]
sys.path.insert(0, '/home/claude/voynich/look')
sys.path.insert(0, '/home/claude/voynich/team/agents/anim/old' if which == 'old' else '/home/claude/voynich/edit')
import room as RM, host as HO
scr = np.zeros((60, 80, 3), np.uint8); scr[:] = (40, 90, 120)
kw = {} if which == 'old' else dict(anim=False)
out = []
for t, amp, mood, pose, on, flip in ((0.3, 0, 'calm', 'front', 1, False), (1.2, 1.0, 'calm', 'point', 1, True), (2.0, 0.9, 'scary', 'point', 1, False), (0.4, 0, 'calm', 'front', 0.4, False)):
    out.append(RM.frame(t, amp, mood, screen=scr, host_cx=62, host_pose=pose, host_on=on, host_flip=flip, host_kw=dict(tilt=2), **kw))
g = np.zeros((320, 180, 3), np.uint8)
HO.draw(g, 90, 347, 1.3, 0.8, 1, False, 0, 0, 5, 'scary', 'close', aura=True, smoke=30, **kw); out.append(g)
np.save(f'/home/claude/voynich/team/agents/anim/cmp_{which}.npy', np.stack(out))
