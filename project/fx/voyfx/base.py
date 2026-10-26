"""Effect framework: a frame-stream effect is a class with `setup(info)` and `process(frame, i, t)`.

Effects that need ffmpeg-level access (real datamosh, genloss, blockcrush ffmpeg mode) implement
`run_file(inp, out, ...)` instead and are not chainable in the single-decode path (chain handles
them by materializing an intermediate).
"""
import time

import numpy as np

from .io import FrameReader, FrameWriter
from .masks import MaskSource, apply_mask, mix
from .params import P, Rng

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


class Effect:
    name = "base"
    help = ""
    # list of (argname, default, help); type inferred from default. Strings with animated
    # specs are converted to Param objects when listed in `animated`.
    args = []
    animated = ("intensity",)
    frame_based = True   # False -> uses run_file
    common = True        # gets --intensity/--mask/--mix/--seed

    def __init__(self, **kw):
        self.kw = {}
        for a in self.all_args():
            self.kw[a[0]] = kw.get(a[0], a[1])
        for k, v in kw.items():
            if k not in self.kw:
                self.kw[k] = v
        self.info = None

    @classmethod
    def all_args(cls):
        base = []
        if cls.common:
            base = [("intensity", "1.0", "effect strength 0..1+ (animatable: '0..1' or '0:0,2:1')"),
                    ("mask", "none", "where the effect applies: none | luma:lo:hi | rect:x,y,w,h | "
                                      "ellipse:cx,cy,rx,ry | grad:bottom | auto | file.png/.mp4 | invert:<spec>"),
                    ("mix", "1.0", "global wet/dry 0..1 (animatable)"),
                    ("seed", 7, "random seed (deterministic re-renders)")]
        return base + list(cls.args)

    # ---- lifecycle
    def bind(self, info):
        self.info = info
        self.p = {}
        for k, v in self.kw.items():
            if k in self.animated or k in ("intensity", "mix"):
                self.p[k] = P(v, info)
        self.rng = Rng(int(self.kw.get("seed", 7)))
        self.mask = MaskSource(self.kw.get("mask", "none"), info.w, info.h, info.fps) if self.common else None
        self.setup(info)
        return self

    def setup(self, info):
        pass

    def val(self, k, t):
        return self.p[k](t)

    def process(self, frame, i, t):
        raise NotImplementedError

    # ---- wrapped processing (mask + mix)
    def __call__(self, frame, i):
        t = i / self.info.fps
        out = self.process(frame, i, t)
        if self.common:
            m = self.mask(frame, i) if self.mask is not None else None
            if m is not None:
                out = apply_mask(frame, out, m)
            mx = self.p["mix"](t)
            if mx < 0.999:
                out = mix(frame, out, mx)
        return out

    def flush(self):
        """Effects with look-ahead may hold frames; return the rest at the end."""
        return []


def render(effects, inp, out, crf=14, preset="medium", size=None, fps=None, max_frames=None,
           verbose=True, lossless=False, audio=True):
    """Single decode -> effect chain -> single encode."""
    rd = FrameReader(inp, size=size, fps=fps)
    info = rd.info
    w, h = rd.size
    info.w, info.h = w, h
    info.fps = rd.fps
    for e in effects:
        e.bind(info)
    wr = FrameWriter(out, w, h, info.fps, crf=crf, preset=preset, audio_from=inp if audio else None,
                     lossless=lossless)
    t0 = time.time()
    n = 0
    for i, f in enumerate(rd):
        if max_frames and i >= max_frames:
            break
        for e in effects:
            f = e(f, i)
        wr.write(f)
        n += 1
        if verbose and n % 30 == 0:
            el = time.time() - t0
            print(f"  [{'+'.join(e.name for e in effects)}] {n}/{info.nframes} frames  {n / el:.2f} fps", flush=True)
    wr.close()
    dt = time.time() - t0
    if verbose:
        print(f"done: {out}  {n} frames in {dt:.1f}s ({n / max(dt, 1e-6):.2f} fps)")
    return dt, n
