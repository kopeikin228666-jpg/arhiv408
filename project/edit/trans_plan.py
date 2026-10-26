"""Transitions director's plan for version B (plug-in read by render.build()).

plan(ver, shots) -> {shot_name: (kind, frames, frames_before_cut) or None}
    shots = [(name, t0, t1), ...] in film order (cuts already snapped to the music).
    The transition belongs to the INCOMING shot.  Its frames run from fc-pre to fc-pre+n-1
    (fc = cut frame), prog = (k+1)/(n+1); with pre = n // 2 the first frame past the switch
    (prog >= 0.5) is exactly the cut frame, i.e. the beat.

Rules followed: every cut is assigned on purpose (None = a deliberate hard cut), never the same kind on two
consecutive cuts, long transitions (14-18 frames) on Mara's booms / big hits in act II-III, short and punchy
(2-6 frames) in the fast act I, medium (8-14) on weak beats where the meaning wants time.
Kinds: render built-ins tear blocks melt grow swirl scatter; transitions.ALL slice zoom glyph iris sortwipe
flash shutter + new inkbleed pageturn ripple pixelrain blink textdissolve mirror.
"""

FPS = 30


def _c(kind, n):
    """centred on the cut: the switch lands on the cut frame (the beat)"""
    return (kind, n, n // 2 if kind != 'tear' else 0)


# name: (entry, reason)
B_PLAN = {
    # ---------------------------------------------------------------- the drop / act I (fast, 2-6 frames)
    's1':   (None, 'the book intro hands over to the glyph by itself on the drop (book.py); a transition would blur the hit'),
    's2':   (_c('inkbleed', 6), 'the bursting glyph is wet ink: it bleeds into the parchment macro - the script becomes the page'),
    's3':   (_c('zoom', 4), 'big Mara hit: punch through the page into the eye that reads it'),
    's4':   (_c('blink', 6), 'the eye we just saw blinks shut and opens on the museum case: MS 408 is what it was looking at'),
    's5':   (_c('pageturn', 4), '"не хочет, чтобы её читали": the museum book flips a page - and the pages start to strobe'),
    's6':   (_c('slice', 4), 'fast VHS shutter: strobing pages tracked out, the reaching hand pushed in'),
    's7':   (_c('flash', 4), 'the hand burns out to bone, one bone frame, НЕ ЧИТАЙ lands on the cut frame'),
    's8':   (_c('scatter', 6), 'Mara hit: the warning shatters into flying pixels revealing 1912 archive photo'),
    's9':   (_c('tear', 3), '"купили": hard glitch cut onto the auction tag - a price tag is a bureaucratic snap'),
    's10':  (_c('shutter', 6), 'blinds flip from the tag to the gallery of readers - pages of a catalogue of victims'),
    's11':  (_c('textdissolve', 8), '"пытались читать": the readers are transcribed into Voynich script that resolves into the decipherer\'s wall'),
    's12':  (('flash', 2, 0), 'creak window, pinned cut on knock 1: one bone frame exactly on the knock, then the dive'),
    's13':  (_c('glyph', 8), 'knock 2: a giant gallows glyph opens out of the dived-into page onto the 3x3 heads'),
    # ---------------------------------------------------------------- act II (Mara's boom every 8.57 s)
    's15':  (None, 'the 30.0 boom + knock 3 out of total black; s15 builds itself out of blocks - silence-to-boom must be a hard cut'),
    's16':  (_c('pixelrain', 14), '"шестьсот лет назад" coming: the reading room crumbles into falling sand that heaps up - and the hourglass stands behind it'),
    's17':  (_c('swirl', 16), 'BOOM: the hourglass is wound into a time vortex, the profile unwinds out of it (only swirl in the film)'),
    's18':  (_c('glyph', 12), '"буквы двигаются": the letters leaving his mouth become one giant glyph that swallows the screen into the page'),
    's19':  (_c('inkbleed', 18), 'BOOM: the page\'s ink bleeds into blots and the impossible plant soaks through ("растения… неправильные")'),
    's20':  (_c('iris', 14), 'the round pools from above: a circular aperture opens the plant onto the ring-shaped pools'),
    'v21':  (_c('ripple', 16), 'BOOM: a drop falls into the green water, rings run out and the women rise behind them ("в зелёной воде")'),
    's23':  (_c('blink', 10), '"видел то, чего нельзя видеть": a slow blink - the eyelids glow red while shut, open on the scribe'),
    's22':  (_c('melt', 18), 'BOOM: the scribe\'s page slides down in datamosh columns - we sink with it under water'),
    's24':  (_c('sortwipe', 6), 'weak beat, short bridge: a dripping wipe line brings water down onto the glass with the handprint'),
    's26':  (_c('scatter', 8), 'the water surface flies apart in pixels to the face behind the glass ("некоторые страницы")'),
    's25':  (_c('blocks', 6), 'off-grid cut (+0.13 s, word-locked): kept short and quiet - a dithered block wipe to the print, "даже через стекло"'),
    's27':  (_c('ripple', 14), '"после них снятся…": the classic dream wobble, in water rings, into the sleeper'),
    's28':  (_c('grow', 14), 'BOOM on the same sleeper: pixels swell into huge blocks and shrink back - the dream loops ("одни и те же сны")'),
    # ---------------------------------------------------------------- act III / the wall
    's29':  (_c('pageturn', 18), '"книга открывается сама": a slow full page turn - the leaf (Voynich script on its back) turns itself onto the open book'),
    's31':  (_c('slice', 4), 'fast VHS cut onto the whispering lips ("и кто-то шепчет")'),
    's30':  (_c('mirror', 8), '"с другой стороны": the lips fold into their own mirror image, a Rorschach blot, and the man behind the book unfolds'),
    's31b': (_c('textdissolve', 16), 'BOOM on the wall: the man is transcribed into the manuscript\'s script, which reads back out as the whispering lips'),
    's32':  (_c('tear', 3), 'signal lost: glitch-tear into static with the subliminal'),
    's33':  (_c('inkbleed', 12), 'out of the static, ink blots open onto the reader at the bench (MS 408 in his hands)'),
    's34':  (_c('shutter', 4), 'right after "ты слишком долго смотришь…": blinds flip onto the 3x3 wall of watching faces'),
    's35':  (_c('mirror', 16), 'the faces fold into a symmetric red Rorschach blot that becomes her ("она уже… внутри…") - the screenshot frame'),
    's36':  (None, 'last boom: s36 continues s35 (same picture zooming/spinning into her) - a transition would break the continuous motion'),
    's37':  (None, 's36 ends in black and s37 assembles itself from blocks out of the black (its own transition)'),
    's38':  (_c('pageturn', 12), '"ты дочитал.": the last leaf turns over the final image onto the black end page - the book closes the film'),
}


def _clamp(shots, out):
    """never let a transition run past the middle of a short shot or into the neighbouring transition"""
    names = [s[0] for s in shots]
    lens = {s[0]: int(round((s[2] - s[1]) * FPS)) for s in shots}
    for i, nm in enumerate(names):
        tr = out.get(nm)
        if not tr or i == 0:
            continue
        kind, n, pre = tr
        prev = names[i - 1]
        prev_after = 0
        ptr = out.get(prev)
        if ptr:
            prev_after = ptr[1] - ptr[2]
        room_before = max(0, lens[prev] - prev_after - 2)
        nxt = out.get(names[i + 1]) if i + 1 < len(names) else None
        room_after = lens[nm] - (nxt[2] if nxt else 0) - 2
        pre2 = min(pre, room_before)
        post2 = min(n - pre, max(1, room_after))
        if (pre2, post2) != (pre, n - pre):
            out[nm] = (kind, pre2 + post2, pre2)
    return out


def plan(ver, shots):
    if ver != 'B':
        return {}
    out = {}
    for name, _t0, _t1 in shots:
        if name in B_PLAN:
            out[name] = B_PLAN[name][0]
    return _clamp(shots, out)


def check(shots):
    """consecutive-kind check + table; returns rows (name, t0, kind, n, pre)"""
    p = plan('B', shots)
    rows, last = [], None
    for name, t0, t1 in shots:
        tr = p.get(name)
        k = tr[0] if tr else None
        if k is not None:
            assert k != last, f'{name}: {k} twice in a row'
        last = k if k is not None else last
        rows.append((name, t0, t1, tr))
    return rows


if __name__ == '__main__':
    import os, sys
    os.environ.setdefault('VOY_VER', 'B')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render
    sh = render.build()
    for name, t0, t1, tr in check([(s.name, s.t0, s.t1) for s in sh]):
        print(f'{name:5s} {t0:7.3f} {t1 - t0:5.2f}  {tr}   {B_PLAN.get(name, (None, ""))[1][:90]}')
