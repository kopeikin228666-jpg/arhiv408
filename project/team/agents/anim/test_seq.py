import sys, time, math, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import host as HO, room as RM
tag = sys.argv[1]
OUT = '/home/claude/voynich/team/agents/anim'
def up(g, k=3): return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
amps = [0, 0.7, 1.0, 1.0, 0.9, 1.0, 0.4, 0.0]
# close-up face sequence
def close(t, amp, mood):
    bg = np.zeros((320, 180, 3), np.uint8); bg[:] = HO.BGR[HO.VOID2]
    HO.draw(bg, 90, 160 - 88 + 273 + 2, t, amp, 1, False, 0, 0, int(t * 15), mood, 'close', aura=True, smoke=30)
    return bg
rows = []
for mood in ('calm', 'scary'):
    rows.append(np.hstack([up(close(1.0 + i / 15 * 2, amps[i], mood)[66:196, 20:160], 3) for i in range(8)]))
cv2.imwrite(f'{OUT}/{tag}_closeseq.png', np.vstack(rows))
# lab point sequence (crop around him)
rows = []
for pose, flip in (('point', False), ('front', False)):
    fr = []
    for i in range(8):
        t = 2.0 + i / 15 * 2
        g = RM.frame(t, amps[i], 'calm', False, 0, host_cx=62, host_pose=pose, host_flip=flip)
        fr.append(up(g[110:240, 10:140], 3))
    rows.append(np.hstack(fr))
cv2.imwrite(f'{OUT}/{tag}_labseq.png', np.vstack(rows))
