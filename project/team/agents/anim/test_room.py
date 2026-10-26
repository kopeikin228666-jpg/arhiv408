import sys, time, math, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import host as HO, room as RM
tag = sys.argv[1]; OUT = '/home/claude/voynich/team/agents/anim'
def up(g, k=3): return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
scr = np.zeros((60, 80, 3), np.uint8); scr[:] = (40, 90, 120); scr[20:40, 20:60] = (200, 220, 230)
fr = [up(RM.frame(3 + i / 15, 1.0, 'calm', screen=scr, host_cx=62, host_pose='front')[0:170], 2) for i in range(4)]
fr += [up(RM.frame(3 + i / 15, 0.0, 'scary', screen=None, host_cx=62, host_pose='point')[0:170], 2) for i in range(2)]
cv2.imwrite(f'{OUT}/{tag}_monitors.png', np.hstack(fr))
fr = [up(RM.frame(0.5 + i / 15, 0.0, 'calm', host_cx=62, host_pose='front', host_on=on)[90:300, 0:150], 2) for i, on in enumerate((0.15, 0.35, 0.55, 0.75, 0.92))]
fr.append(up(RM.frame(1.0, 0.0, 'calm', host_cx=90, host_pose='front', host_vx=60)[90:300, 0:150], 2))
cv2.imwrite(f'{OUT}/{tag}_teleport.png', np.hstack(fr))
# timing anim on / off
for an in (False, True):
    t0 = time.time()
    for i in range(16):
        RM.frame(i / 15, 1.0 * (i % 2), 'scary' if i % 4 == 0 else 'calm', screen=scr, host_cx=62, host_pose='point' if i % 2 else 'front', anim=an)
    print('anim', an, 'ms/frame', round((time.time() - t0) / 16 * 1000, 1))
