import sys, math, subprocess, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/edit'); sys.path.insert(0, '/home/claude/voynich/look')
import host as HO, room as RM
OUT = '/home/claude/voynich/team/agents/anim/preview_anim.mp4'
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '540x960', '-r', '30', '-i', '-',
                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', OUT], stdin=subprocess.PIPE)
def amp_at(t):                                   # fake speech: syllables, pauses
    if (t % 2.2) > 1.6: return 0.0
    return min(1.0, 0.55 + 0.5 * abs(math.sin(t * 7.3)))
for f in range(90):                              # 3 s lab: teleport in, then speaking while pointing, scary at the end
    t = f / 30; on = min(1.0, t / 0.5)
    g = RM.frame(t, amp_at(t) if t > 0.6 else 0, 'scary' if t > 2.4 else 'calm', host_cx=62, host_pose='point' if t > 0.8 else 'front', host_on=on)
    p.stdin.write(cv2.resize(g, (540, 960), interpolation=cv2.INTER_NEAREST).tobytes())
for f in range(60):                              # 2 s close-up: talking, then scary heartbeat
    t = 5 + f / 30
    bg = RM.frame(t, 0, 'calm', host_on=0)
    bg = cv2.resize(bg[40:200, 45:135], (180, 320), interpolation=cv2.INTER_NEAREST)
    pi = HO._pal_index(bg); lut = np.array([HO._darker(i) for i in range(18)], np.int16); bg = HO.BGR[lut[lut[pi]]]
    HO.draw(bg, 90, 347 + int(round(math.sin(t * 2 * math.pi / 2.14))), t, amp_at(t) if f < 35 else 0, 1, False, 0, 0, int(t * 15),
            'calm' if f < 35 else 'scary', 'close', aura=True, smoke=30)
    p.stdin.write(cv2.resize(bg, (540, 960), interpolation=cv2.INTER_NEAREST).tobytes())
p.stdin.close(); p.wait(); print('ok')
