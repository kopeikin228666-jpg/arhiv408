"""Animated parameters.

Accepted syntaxes (time in seconds unless noted):
  "0.7"                      constant
  "0.2..1.0" / "0.2->1.0"    linear ramp over the whole clip (or over --start/--end range)
  "0:0.2,1.5:1.0,4:0"        keyframes  time:value
  "0:0, 50%:1, 100%:0"       times can be % of duration
  "0f:0, 45f:1"              times can be in frames
  suffix "@smooth" "@ease" "@step" "@linear" "@expo" picks interpolation (default linear;
  ramps "a..b" default to linear too). Example: "0:0,2:1@smooth"
Values outside the keyframe range hold the first/last value.
"""
import math
import re

import numpy as np


def _ease(kind, u):
    if kind == "linear":
        return u
    if kind in ("smooth", "smoothstep"):
        return u * u * (3 - 2 * u)
    if kind in ("ease", "cos", "sine"):
        return 0.5 - 0.5 * math.cos(math.pi * u)
    if kind == "step":
        return 0.0 if u < 1 else 1.0
    if kind in ("expo", "in"):
        return u * u * u
    if kind == "out":
        return 1 - (1 - u) ** 3
    return u


class Param:
    def __init__(self, spec, duration=5.0, fps=30.0):
        self.spec = str(spec).strip()
        self.duration = float(duration)
        self.fps = float(fps)
        s = self.spec
        self.interp = "linear"
        if "@" in s:
            s, self.interp = s.rsplit("@", 1)
            self.interp = self.interp.strip().lower()
        s = s.strip()
        self.keys = []
        m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*(?:\.\.|->)\s*([-+0-9.eE]+)\s*", s)
        if m:
            self.keys = [(0.0, float(m.group(1))), (self.duration, float(m.group(2)))]
        elif ":" in s:
            for part in s.split(","):
                part = part.strip()
                if not part:
                    continue
                t, v = part.split(":", 1)
                self.keys.append((self._time(t.strip()), float(v)))
            self.keys.sort(key=lambda kv: kv[0])
        else:
            self.keys = [(0.0, float(s))]
        self.const = len(self.keys) == 1

    def _time(self, t):
        if t.endswith("%"):
            return float(t[:-1]) / 100.0 * self.duration
        if t.endswith("f"):
            return float(t[:-1]) / self.fps
        if t.endswith("s"):
            t = t[:-1]
        return float(t)

    def __call__(self, t):
        k = self.keys
        if self.const or t <= k[0][0]:
            return k[0][1]
        if t >= k[-1][0]:
            return k[-1][1]
        for (t0, v0), (t1, v1) in zip(k[:-1], k[1:]):
            if t0 <= t <= t1:
                u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return v0 + (v1 - v0) * _ease(self.interp, u)
        return k[-1][1]

    def __repr__(self):
        return f"Param({self.spec!r})"


def P(spec, info):
    """Shortcut: build a Param bound to a VideoInfo."""
    if isinstance(spec, Param):
        return spec
    return Param(spec, info.duration, info.fps)


def parse_times(spec):
    """'1.2,2.5,3' -> [1.2, 2.5, 3.0]; supports frames '30f'."""
    if spec is None or str(spec).strip() == "":
        return []
    out = []
    for p in str(spec).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out


def time_to_frame(tok, fps, duration=None):
    tok = str(tok).strip()
    if tok.endswith("f"):
        return int(round(float(tok[:-1])))
    if tok.endswith("%") and duration:
        return int(round(float(tok[:-1]) / 100 * duration * fps))
    if tok.endswith("s"):
        tok = tok[:-1]
    return int(round(float(tok) * fps))


class Rng:
    """Deterministic per-frame RNG: same seed + frame -> same randomness (re-renders match)."""

    def __init__(self, seed):
        self.seed = int(seed)

    def at(self, i, salt=0):
        return np.random.default_rng((self.seed * 1000003 + i * 9176 + salt * 7919) & 0xFFFFFFFF)
