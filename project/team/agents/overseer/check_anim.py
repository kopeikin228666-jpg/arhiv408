"""overseer: independent check of the animator's host.py / room.py (round 2). single frames only."""
import sys, math, time, traceback, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/team/agents/overseer')
from pal import off_palette, up, label, red_count, PAL
import host as HO, room as RM
O = '/home/claude/voynich/team/agents/overseer/'
res = []
def chk(name, fn):
    try:
        t0 = time.time(); r = fn(); res.append((name, 'OK', r, time.time() - t0))
    except Exception as e:
        res.append((name, 'FAIL', repr(e) + ' ' + traceback.format_exc().splitlines()[-3], 0))

# --- A. old call signatures (no new kwargs) ---------------------------------------------------------
def old_sprite():
    out = {}
    for pose in ('front', 'point', 'close'):
        for mood in ('calm', 'scary'):
            s = HO.sprite_img(1.3, 0.7, False, 1, 3, mood, pose)
            out[(pose, mood)] = off_palette(s, s[..., 3] > 0)[0]
    s = HO.sprite_img(1.3, 0.7, False, 0, 3, 'calm', 'close', sway=False, anim=False); out['noanim'] = off_palette(s, s[..., 3] > 0)[0]
    s = HO.sprite(1.3, 0.7, False, 0, 3, 'calm', 'stand'); out['legacy sprite()'] = off_palette(s, s[..., 3] > 0)[0]
    return out
chk('A1 sprite_img old args (offpal counts)', old_sprite)
def old_draw():
    g = np.zeros((320, 180, 3), np.uint8); g[:] = PAL[3]
    HO.draw(g, 90, 300, 2.0, 0.5, 1, False, 0, 0, 3, 'calm', 'front')
    HO.draw(g, 90, 300, 2.0, 0.5, 1, False, 0, 0, 3, 'scary', 'point', aura=True, smoke=46, sx=0.9, scale_f=1.05, tilt=2, vx=40)
    HO.draw(g, 90, 345, 2.0, 0, 1, False, 0, 0, 3, 'scary', 'close', aura=False, smoke=20)
    return off_palette(g)
chk('A2 draw old args', old_draw)
def old_room():
    r = {}
    g = RM.frame(3.0, 0.4); r['frame(t,amp)'] = off_palette(g)[0]
    g = RM.frame(3.0, 0.9, 'scary', False, 0, host_pose='point', host_kw=dict(scale_f=1.04)); r['scary point'] = off_palette(g)[0]
    g = RM.frame(3.0, 0.0, anim=False); r['anim=False'] = off_palette(g)[0]
    cv2.imwrite(O + 'room_old.png', up(g, 3)); return r
chk('A3 room.frame old args', old_room)

# --- B. stress reaction, close pose -----------------------------------------------------------------
def stress_close():
    tiles = []; diffs = {}
    for mood in ('calm', 'scary'):
        for st in (0.0, 0.5, 1.0):
            t = 2.0
            s = HO.sprite_img(t, 0.5, False, 0, 3, mood, 'close', stress=st)
            g = np.zeros_like(s[..., :3]); g[:] = PAL[4]; m = s[..., 3] > 0; g[m] = s[..., :3][m]
            if st == 0.0: base = g.copy()
            diffs[(mood, st)] = (int((g != base).any(2).sum()), red_count(g), off_palette(s, m)[0])
            tiles.append(label(up(g[40:160], 3), f'{mood} st={st}'))
    cv2.imwrite(O + 'stress_close.png', np.hstack(tiles)); return diffs
chk('B1 stress close: (changed px vs st=0, red px, offpal)', stress_close)
def stress_kinds():
    # a realistic word list, sample one word every frame: kind must be constant within one word
    words = [dict(key=True, t0=1.237 + 0.73 * i) for i in range(12)]
    out = []
    for w in words:
        ks = []
        for f in range(int((w['t0'] - 0.1) * 30), int((w['t0'] + 0.55) * 30)):
            t = f / 30; st = HO.stress_at(words, t)
            if st > 0.12: ks.append(HO._stress_kind(t, st, True))
        out.append(''.join(map(str, ks)))
    return out
chk('B2 _stress_kind per frame within each stressed word (should be one digit repeated)', stress_kinds)
def stress_env():
    words = [dict(key=True, t0=1.0)]
    return [round(HO.stress_at(words, 1.0 + f / 30), 2) for f in range(-3, 12)]
chk('B3 stress_at envelope frames -3..11 around attack', stress_env)

# --- C. breathing: hood-top row over time ---------------------------------------------------------
def breath_amp():
    r = {}
    for pose in ('close', 'front'):
        for mood in ('calm', 'scary'):
            tops = []; snk = []
            for f in range(0, 30 * 6, 3):
                t = f / 30
                s = HO.sprite_img(t, 0.0, False, 0, 3, mood, pose, sway=False, stress=0.0)
                ys = np.nonzero((s[..., 3] > 0).any(1))[0]; tops.append(int(ys[0]))
                d, _ = HO.breath(t, mood)
            r[(pose, mood)] = (min(tops), max(tops), max(tops) - min(tops))
    return r
chk('C1 breath: hood top row (min,max,range) sprite px; close/front sprite px = grid px', breath_amp)

# --- D. final card ----------------------------------------------------------------------------------
def final():
    rows = []; tiles = []
    for dur in (1.3, 1.25):
        n = int(round(dur * 30)); seq = []
        for fi in range(n):
            tl = fi / 30; t = 51.7 + tl
            g = HO.final_card(t, tl, dur, int(t * 30), eye_y=162)
            op = off_palette(g)[0]
            seq.append((fi, round(dur - tl, 3), red_count(g), op))
            if dur == 1.3 and fi >= n - 12:
                tiles.append(label(up(g[110:230, 30:150], 2), f'{fi} L={dur-tl:.2f}'))
        rows.append(seq[-14:])
    cv2.imwrite(O + 'final_card.png', np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:12])]))
    return rows
chk('D1 final_card last frames (fi, left, redpx, offpal)', final)

# --- E. hook frame 0 stare + dark_portrait ------------------------------------------------------------
def portrait():
    t = []
    for st in (1.0, 0.66, 0.33, 0.0):
        g = HO.dark_portrait(0.0 + (1 - st) * 0.3, 0, eye_y=207, stare=st)
        t.append(label(up(g[150:260, 30:150], 2), f'stare={st} offpal={off_palette(g)[0]}'))
    cv2.imwrite(O + 'stare_portrait.png', np.hstack(t)); return 'saved'
chk('E1 dark_portrait stare ramp', portrait)

# --- F. pip_box -------------------------------------------------------------------------------------
def pipb():
    tiles = []; ops = []
    for k in range(8):
        t = 40.0 + k * 0.29
        b = HO.pip_box(t, 0.8 if k % 2 else 0.0)
        ops.append(off_palette(b)[0]); tiles.append(up(b, 5))
    cv2.imwrite(O + 'pip_box.png', np.hstack(tiles)); return ops
chk('F1 pip_box offpal per tile (54x46 = 2484 px)', pipb)

# --- G. room monitors on stress ----------------------------------------------------------------------
def room_stress():
    a = RM.frame(3.0, 0.4, host_kw=dict(stress=0.0)); b = RM.frame(3.0, 0.4, host_kw=dict(stress=1.0))
    c = RM.frame(3.0, 0.4, host_kw=dict(stress=0.6))
    d = (a != b).any(2)
    ys, xs = np.nonzero(d)
    cv2.imwrite(O + 'room_stress.png', np.hstack([label(up(a, 3), 'st=0'), label(up(c, 3), 'st=.6'), label(up(b, 3), 'st=1')]))
    return dict(changed=int(d.sum()), bbox=(int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if len(ys) else None,
                red_a=red_count(a), red_b=red_count(b), red_c=red_count(c), offpal=off_palette(b)[0])
chk('G1 room stress 0 vs 1', room_stress)

# --- H. lean glow ------------------------------------------------------------------------------------
def lean():
    tiles = []; r = {}
    for sf in (1.0, 1.04, 1.07):
        g = np.zeros((320, 180, 3), np.uint8); g[:] = PAL[2]
        HO.draw(g, 90, 300, 31.6, 0.5, 1, False, 0, 0, 3, 'calm', 'point', scale_f=sf)
        r[sf] = (red_count(g), off_palette(g)[0]); tiles.append(label(up(g[120:310, 20:160], 2), f'sf={sf}'))
    cv2.imwrite(O + 'lean_glow.png', np.hstack(tiles)); return r
chk('H1 lean glow (red px, offpal)', lean)

for n, s, r, dt in res:
    print(f'[{s}] {n} ({dt:.1f}s)\n     {r}')
