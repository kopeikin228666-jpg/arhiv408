import sys, time, math, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import host as HO, room as RM
tag = sys.argv[1]
OUT = '/home/claude/voynich/team/agents/anim'
def up(g, k=3): return cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
# --- lab strips
cases = [('front', 0.0, 'calm', 1.0, False, 0.10), ('front', 0.8, 'calm', 1.0, False, 0.37), ('point', 0.9, 'calm', 1.0, False, 1.23),
         ('point', 0.2, 'calm', 1.0, True, 2.07), ('point', 0.0, 'scary', 1.0, False, 3.5), ('front', 0.0, 'calm', 0.45, False, 0.2)]
tiles = []; ts = []
for pose, amp, mood, on, flip, t in cases:
    t0 = time.time()
    g = RM.frame(t, amp, mood, False, 0, host_cx=62, host_pose=pose, host_flip=flip, host_on=on)
    ts.append(time.time() - t0)
    tiles.append(up(g[100:320, 0:180]))
cv2.imwrite(f'{OUT}/{tag}_lab.png', np.hstack(tiles))
# timing over 20 frames
t0 = time.time()
for i in range(20):
    RM.frame(i / 30, 0.5 * (i % 3) / 2, 'calm', False, 0, host_cx=62, host_pose='point' if i % 2 else 'front')
print('lab frame ms', (time.time() - t0) / 20 * 1000, [round(x * 1000) for x in ts])
# --- close-ups
def close(t, amp, mood, look=0, blink=False):
    bg = np.zeros((320, 180, 3), np.uint8); bg[:] = HO.BGR[HO.VOID2]
    bg[(np.indices((320, 180)).sum(0) // 10) % 2 == 0] = HO.BGR[HO.DARK]
    HO.draw(bg, 90, 160 - 88 + 273 + 2, t, amp, 1, blink, look, 0, int(t * 15), mood, 'close', aura=True, smoke=30)
    return bg
tiles = []
t0 = time.time()
for t, amp, mood in ((0.1, 0.0, 'calm'), (0.43, 0.3, 'calm'), (0.77, 0.9, 'calm'), (1.2, 0.6, 'scary'), (1.5, 0.0, 'scary')):
    tiles.append(up(close(t, amp, mood)[20:250], 2))
print('close ms', (time.time() - t0) / 5 * 1000)
cv2.imwrite(f'{OUT}/{tag}_close.png', np.hstack(tiles))
