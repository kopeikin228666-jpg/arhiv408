def _face_smoke(q, P_, t, k, mood):
    """thin curls of smoke drifting across the void of the face (painted only on the void, under the eyes)"""
    cy, cx, ry, rx = P_['face']
    ry, rx = ry * 1.5, rx * 1.15
    void = BGR[VOID]
    n = 5 if mood == 'scary' else 3
    h, w = q.shape[:2]
    for i in range(n):
        per = 3.1 + i * 0.83
        u = (t / per + i * 0.37) % 1.0
        dirn = 1 if i % 2 == 0 else -1
        xc = cx + dirn * (u * 2 - 1) * rx * 1.1
        yc = cy + (i - (n - 1) / 2) * ry * 0.38 + math.sin(t * 0.9 + i) * ry * 0.12
        L = int(rx * 0.75)
        fade_ = math.sin(math.pi * u)                     # a curl is born, crosses and thins out
        for j in range(0, L, k):
            e = math.sin(math.pi * j / L) * fade_
            if e < 0.25 or ((j // k + i + int(t * 10)) % 3 == 0 and e < 0.6):
                continue
            x = int(xc - L / 2 + j)
            y = int(yc + ry * 0.16 * math.sin(j * 0.22 / k + t * 2.1 + i * 1.7))
            y -= y % k; x -= x % k
            if 0 <= y < h - k and 0 <= x < w - k and (q[y, x] == void).all():
                c = DARK if (e > 0.7 and mood == 'scary') or (e > 0.85) else VOID2
                q[y:y + k, x:x + k] = BGR[c]


def _shift_box(q, m, y0, y1, x0, x1, dy, dx):
    """move a box of the sprite by (dy, dx) px (the hood nodding, an arm beating); the box's trailing edge keeps its pixels"""
    h, w = m.shape
    y0, y1, x0, x1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    sq, sm = q[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].copy()
    ty0, tx0 = y0 + dy, x0 + dx
    a0, b0 = max(0, ty0), max(0, tx0)
    a1, b1 = min(h, ty0 + sq.shape[0]), min(w, tx0 + sq.shape[1])
    if a1 <= a0 or b1 <= b0:
        return
    s_ = (slice(a0 - ty0, a1 - ty0), slice(b0 - tx0, b1 - tx0))
    if dy < 0 or dx != 0:                                  # what moved away leaves the cloak behind it unchanged; the
        pass                                               # moved pixels simply land on top (all near-black, no seam)
    tgt_m = m[a0:a1, b0:b1]
    tgt_q = q[a0:a1, b0:b1]
    keep = sm[s_]
    tgt_q[keep] = sq[s_][keep]
    tgt_m |= keep
    if dy > 0:                                             # nodding down: clear the rows the box vacated at its top
        clr = sm[:dy] & ~np.zeros_like(sm[:dy])
        m[y0:y0 + dy, x0:x1][clr] = False
        m[y0 + dy:y0 + 2 * dy, x0:x1] |= False


def _anim_motion(q, m, P_, t, amp, mood, pose, k, sway):
    """the living layer: hood nods / tilts with the voice, the hood point and rags move in the lamp's draught,
    the pointing arm beats on stressed words and a red signal runs to the finger"""
    h, w = m.shape
    wind = lamp_wind(t)
    beat = speech_beat(t, amp)
    # --- head: a small nod down on stressed syllables, a slow side tilt while he talks
    hy1, hx0, hx1 = P_['head']
    if beat > 0.4:
        _shift_box(q, m, 0, hy1, hx0, hx1, k, 0)
    tilt = (1 if math.sin(t * 0.83 + 0.4) > 0 else -1) * k if amp > 0.3 else 0
    tip = P_['tip']
    for y in range(0, hy1):
        d = 0.0
        if tilt:
            d += tilt * (1 - y / hy1)
        if y < tip:                                        # the hood point bends in the draught, the very tip flutters
            f = (1 - y / tip) ** 1.6
            d += wind * 1.8 * k * f + (0.8 * k * f if _hash(int(t * 12), y // 3) > 0.7 else 0)
        d = int(round(d))
        if d:
            q[y, hx0:hx1] = np.roll(q[y, hx0:hx1], d, 0); m[y, hx0:hx1] = np.roll(m[y, hx0:hx1], d, 0)
    # --- ragged edges flutter downwind (whole little strips of cloth, 12 fps)
    if sway:
        sgn = 1 if wind >= 0 else -1
        f = int(t * 12)
        rows = np.array([_hash(f, (yy // (2 * k)) * 7 + 3) for yy in range(h)])
        zone = np.zeros(h, bool)
        zone[int(tip * 0.4):h - 22 * k] = True
        p = 0.16 + 0.22 * abs(wind) + (0.12 if mood == 'scary' else 0)
        on = (rows < p) & zone
        dw = np.roll(m, sgn * k, 1) & ~m & on[:, None]
        q[dw] = BGR[VOID2]; m |= dw
        nib = m & ~np.roll(m, -sgn, 1) & (rows > 1 - p * 0.6)[:, None] & zone[:, None]
        m &= ~nib                                          # and the upwind edge frays a little
    # --- the pointing arm
    if 'arm' in P_:
        ay0, ay1, ax0 = P_['arm']
        if beat > 0.25:
            _shift_box(q, m, ay0, ay1, ax0, w, -1, 1)
        hand = m.copy(); hand[:ay0 - 2] = False; hand[ay1:] = False; hand[:, :ax0 + 22] = False
        ys, xs = np.nonzero(hand)
        if len(xs):
            i = int(np.argmax(xs)); fy, fx = int(ys[i]), int(xs[i])
            fi = int(t * 15)
            # a faint red signal leaves the fingertip: two dashes travelling outward, flickering
            for j in range(2):
                dx_ = 2 + ((fi + j * 3) % 6) * 1
                if _hash(fi, j + 11) < 0.75:
                    x_ = fx + dx_
                    if x_ + 1 < w:
                        q[fy, x_:x_ + 2 - j] = BGR[RED2 if j else RED2]; m[fy, x_:x_ + 2 - j] = True
            # embers run along the top of the sleeve toward the hand while he speaks
            if amp > 0.05:
                top = m & ~np.roll(m, 1, 0)
                top[:ay0] = False; top[ay1:] = False; top[:, :ax0 + 4] = False; top[:, fx - 1:] = False
                ty, tx = np.nonzero(top)
                sel = ((tx - fi * 2) % 9) == 0
                q[ty[sel], tx[sel]] = BGR[RED2]
            # glitch: on a stressed word a slice of the glove jumps forward with a red fringe
            if beat > 0.25 or _hash(fi, 5) < 0.12:
                gy = fy - 2 + int(_hash(fi, 6) * 5)
                gy0, gy1 = max(0, gy), min(h, gy + 2)
                xa = max(0, fx - 12)
                for yy in range(gy0, gy1):
                    row_q = q[yy, xa:fx + 1].copy(); row_m = m[yy, xa:fx + 1].copy()
                    q[yy, xa + 2:fx + 3][row_m] = row_q[row_m]; m[yy, xa + 2:fx + 3] |= row_m
                    lead = np.nonzero(row_m)[0]
                    if len(lead):
                        q[yy, xa + 2 + lead[-1]] = BGR[RED]
                        if xa + lead[0] < w:
                            q[yy, xa + lead[0]] = BGR[RED2]


def sprite_img(t=0.0, amp=0.0, blink=False, look=0, wave_seed=0, mood='calm', pose='front', sway=True, anim=True):
    """anim=False gives the old, quieter figure (no face smoke, nods, flares, draught, arm beat)"""
    P_ = POSES[pose]
    q, m = _load_img(pose)
    q = q.copy(); m = m.copy()
    k = P_['k']
    pad = P_.get('pad', 0) if anim else 0
    if pad:                                                # room around the pointing hand for its red signal
        q = np.pad(q, ((0, 0), (pad, pad), (0, 0))); m = np.pad(m, ((0, 0), (pad, pad)))
        P_ = dict(P_, eyes=tuple((y, x + pad) for y, x in P_['eyes']), mouth=(P_['mouth'][0], P_['mouth'][1] + pad, P_['mouth'][2] + pad),
                  head=(P_['head'][0], P_['head'][1] + pad, P_['head'][2] + pad), arm=(P_['arm'][0], P_['arm'][1], P_['arm'][2] + pad),
                  face=(P_['face'][0], P_['face'][1] + pad, P_['face'][2], P_['face'][3]))
    h, w = m.shape
    flare = (t % 4.3) < 0.15
    lit = amp > 0.05 or flare
    open_ = fl = 0
    if anim:
        _face_smoke(q, P_, t, k, mood)
        open_ = 1 if amp > 0.8 else 0
        fl = 1 if (mood == 'scary' and heartbeat(t) >= 1.0) or (mood != 'scary' and speech_beat(t, amp) > 0.5) else 0
    if not blink:
        for j, (ey, ex) in enumerate(P_['eyes']):
            _slit(q, int(ey), int(round(ex + look * k)), 1 if j == 0 else -1, k, mood, lit, open_, fl)
    if amp >= 0.04:
        # mouth: a broken signal - dim, low bars, some columns drop out
        my, mx0, mx1 = P_['mouth']
        r = np.random.default_rng(wave_seed)
        ph = r.random(64) * 6.28
        xs = list(range(mx0, mx1 + 1, k))
        n = len(xs)
        for i, x in enumerate(xs):
            if i % 2 or r.random() < 0.25:
                continue
            env = 0.3 + 0.7 * math.sin(math.pi * (i + 0.5) / n)
            v = env * amp * (0.2 + 0.8 * abs(math.sin(t * 19 + ph[(i // 2) % 64] + i * 0.45)))
            hgt = int(round(v * 3.0)) * k
            q[my + 1 - hgt:my + 1 + k + hgt, x:x + k] = BGR[RED2]
            if hgt >= 2 * k:
                q[my + 1:my + 1 + k, x:x + k] = BGR[RED]
    if sway:
        # the cloak moves: rows below the waist swing sideways, more toward the hem, rags flutter; with anim the
        # whole hem also leans into the lamp's draught
        wy = P_['waist']
        wind = lamp_wind(t) if anim else 0.0
        for y in range(wy, h):
            d = (y - wy) / max(1, h - wy)
            sh = int(round(math.sin(t * 2 * math.pi / 2.14 + y * 0.09) * 2.2 * d + math.sin(t * 9 + y * 0.7) * 0.6 * d * d
                           + wind * 1.6 * d * d))
            if sh:
                q[y] = np.roll(q[y], sh, 0); m[y] = np.roll(m[y], sh, 0)
    if anim:
        _anim_motion(q, m, P_, t, amp, mood, pose, k, sway)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = q
    out[..., 3] = m * 255
    return out
