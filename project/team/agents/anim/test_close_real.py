import sys, math, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import host as HO, room as RM
tag = sys.argv[1]; OUT = '/home/claude/voynich/team/agents/anim'
def close_grid(t, mood, amp, look=0, scale_f=1.0):
    bg = RM.frame(t, 0, 'calm', host_on=0)
    bg = cv2.resize(bg[40:200, 45:135], (180, 320), interpolation=cv2.INTER_NEAREST)
    pi = HO._pal_index(bg); lut = np.array([HO._darker(i) for i in range(18)], np.int16)
    bg = HO.BGR[lut[lut[pi]]]
    bob = int(round(math.sin(t * 2 * math.pi / 2.14)))
    by = 160 - 88 + 273 + 2
    HO.draw(bg, 90, by, t, amp, 1, False, look, bob, int(t * 15), mood, 'close', aura=True, smoke=30)
    return bg
fr = [close_grid(4.6 + i / 30 * 2, 'scary', 0.0) for i in range(8)]   # 2 frames per step at 30 fps
print([HO.heartbeat(4.6 + i / 30 * 2) for i in range(8)])
cv2.imwrite(f'{OUT}/{tag}_heartbeat.png', cv2.resize(np.hstack([f[30:260] for f in fr]), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
