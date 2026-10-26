"""apply the animator's call-site patches to a copy of make_ep1e.py:  python3 apply_patch.py SRC DST"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
P = [
 # 1. close-ups: his eyes / hood react exactly on the stressed (*) words
 ("""    kw = kw or {}
    by = 160 - 88 + 273 + 2""",
  """    kw = dict(kw or {})
    kw.setdefault('stress', HO.stress_at(sg.get('words'), t))
    by = 160 - 88 + 273 + 2"""),
 # 2. wide lab shots: the same
 ("""    flick = -1.0 if (int(t * 30) % 47 == 0) else 0.0
    g = RM.frame(""",
  """    flick = -1.0 if (int(t * 30) % 47 == 0) else 0.0
    kw['stress'] = HO.stress_at(sg.get('words'), t)
    g = RM.frame("""),
 # 3. hook frame 0: opens on the wide stare the card ends on, which narrows into the slits in 0.3 s
 ("""        g = close_grid(sg, t, f, mood, None if eyes_on else 0.0, eyes_on=eyes_on)""",
  """        g = close_grid(sg, t, f, mood, None if eyes_on else 0.0, eyes_on=eyes_on, kw=dict(stare=max(0.0, 1 - tl / 0.3)))"""),
 # 4. hook title: his eyes below the title instead of hidden behind «РУКОПИСЬ ВОЙНИЧА»
 ("""            g = np.zeros((GH, GW, 3), np.uint8); g[:] = HO.BGR[HO.VOID]
            HO.draw(g, 90, 345, t, 0, 1, False, 0, 0, f, 'scary', 'close', aura=False, smoke=20)
            pi = HO._pal_index(g); lut = np.array([HO._darker(i) for i in range(18)], np.int16)
            red = (pi == HO.RED) | (pi == HO.RED2)
            pi2 = lut[lut[pi]]; pi2[red] = pi[red]
            g = HO.BGR[pi2]
            rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
            k2 = tl - sg['title']""",
  """            g = HO.dark_portrait(t, f, eye_y=207)
            rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
            k2 = tl - sg['title']"""),
 # 5. the card: eyes below the words (they were hidden behind «ОНА — НЕТ.»), creep in, go out, SNAP wide -> cut
 ("""        g = np.zeros((GH, GW, 3), np.uint8); g[:] = HO.BGR[HO.VOID]
        HO.draw(g, 90, 345, t, 0, 1, False, 0, 0, f, 'scary', 'close', aura=False, smoke=20)
        pi = HO._pal_index(g); lut = np.array([HO._darker(i) for i in range(18)], np.int16)
        red = (pi == HO.RED) | (pi == HO.RED2)
        pi2 = lut[lut[pi]]; pi2[red] = pi[red]
        g = HO.BGR[pi2]
        if tl < 2 / FPS:""",
  """        g = HO.final_card(t, tl, sg['dur'], f)
        if tl < 2 / FPS:"""),
 # 6. the live window: breathing, glances, blinks, hot eyes on loud syllables
 ("""    s_ = HO.sprite_img(t, 0, False, 0, 0, 'calm', 'close', sway=False)
    sp = s_[30:30 + 150]
    small = cv2.resize(sp[..., :3], (w, int(150 * w / 149)), interpolation=cv2.INTER_AREA)
    small = P.quantize(small, dither=0.0, contrast=1.1)[:h]
    box = np.zeros((h, w, 3), np.uint8); box[:] = HO.BGR[HO.VOID]
    box[:small.shape[0]] = small
    ey = int((88 - 30) * w / 149)
    for ex in (int(61 * w / 149), int(88 * w / 149)):
        box[ey, ex - 1:ex + 2] = HO.BGR[HO.RED2]; box[ey, ex] = HO.BGR[HO.RED]
    if amp > 0.05:
        my = int((104 - 30) * w / 149); r_ = np.random.default_rng(int(t * 15))
        for x in range(int(64 * w / 149), int(86 * w / 149), 2):
            hh = int(round(amp * 2 * r_.random()))
            box[my - hh:my + hh + 1, x] = HO.BGR[HO.RED2]
    box[::2] = (box[::2].astype(np.int16) * 3 // 4).astype(np.uint8)""",
  """    box = HO.pip_box(t, amp, w, h)"""),
]
for i, (a, b) in enumerate(P, 1):
    n = s.count(a)
    if n != 1:
        print(f'patch {i}: anchor found {n} times - SKIPPED'); continue
    s = s.replace(a, b); print(f'patch {i}: ok')
open(dst, 'w').write(s)
