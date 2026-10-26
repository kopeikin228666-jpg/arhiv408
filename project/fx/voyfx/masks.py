"""Mask sources. Every mask is float32 HxW in [0,1].

Mask spec strings (used by --mask on many effects):
  none                      full frame (1.0)
  luma:LO:HI[:soft]         pixels whose luma (0..1) is in [LO,HI], soft edge width
  rect:x,y,w,h[:feather]    rectangle in normalized (0..1) coords or pixels (>1)
  ellipse:cx,cy,rx,ry[:f]   ellipse, normalized coords
  grad:top|bottom|left|right[:start:end]   linear ramp
  auto[:face|person|saliency]   automatic subject region (Haar if cascade exists -> HOG people -> saliency)
  path/to/mask.png|.mp4     external mask image or video (white = effect)
  invert:<spec>             inverted mask
"""
import os

import cv2
import numpy as np

from .io import FrameReader, is_image


def luma(img):
    """BGR uint8/float -> luma float32 0..1 (Rec.709)."""
    f = img.astype(np.float32)
    if img.dtype == np.uint8:
        f *= 1.0 / 255.0
    return f[..., 0] * 0.0722 + f[..., 1] * 0.7152 + f[..., 2] * 0.2126


def feather(m, px):
    if px and px > 0:
        k = int(px) | 1
        m = cv2.GaussianBlur(m, (0, 0), px / 2.0)
    return m


def smoothband(x, lo, hi, soft):
    soft = max(soft, 1e-4)
    a = np.clip((x - (lo - soft)) / soft, 0, 1)
    b = np.clip(((hi + soft) - x) / soft, 0, 1)
    return (a * b).astype(np.float32)


_CASCADE_CANDIDATES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
]


def _find_cascade():
    dirs = [getattr(getattr(cv2, "data", None), "haarcascades", ""),
            "/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades",
            os.path.join(os.path.dirname(__file__), "data")]
    for d in dirs:
        for c in _CASCADE_CANDIDATES:
            p = os.path.join(d or "", c)
            if os.path.isfile(p):
                return p
    return None


class AutoSubject:
    """Automatic subject mask: face (Haar) -> person (HOG) -> spectral-residual saliency.

    Detection runs on a downscaled frame every `every` frames; boxes are smoothed over time
    so the mask does not jitter.
    """

    def __init__(self, mode="auto", every=6, smooth=0.75):
        self.mode = mode
        self.every = every
        self.smooth = smooth
        self.i = 0
        self.box = None  # normalized cx, cy, rx, ry
        self.sal = None
        cp = _find_cascade()
        self.face = cv2.CascadeClassifier(cp) if cp and mode in ("auto", "face") else None
        self.hog = None
        if mode in ("auto", "person"):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.salobj = cv2.saliency.StaticSaliencySpectralResidual_create()

    def _detect(self, img):
        h, w = img.shape[:2]
        s = 360.0 / max(h, w)
        sm = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        if self.face is not None:
            f = self.face.detectMultiScale(g, 1.1, 4, minSize=(20, 20))
            if len(f):
                x, y, fw, fh = max(f, key=lambda r: r[2] * r[3])
                # expand face box to head+shoulders
                cx, cy = (x + fw / 2) / sm.shape[1], (y + fh * 0.7) / sm.shape[0]
                return ("box", (cx, cy, fw * 1.1 / sm.shape[1], fh * 1.5 / sm.shape[0]))
        if self.hog is not None:
            rects, weights = self.hog.detectMultiScale(sm, winStride=(8, 8), padding=(8, 8), scale=1.06)
            if len(rects) and max(weights) > 0.5:
                x, y, rw, rh = rects[int(np.argmax(weights))]
                return ("box", ((x + rw / 2) / sm.shape[1], (y + rh / 2) / sm.shape[0],
                                rw * 0.45 / sm.shape[1], rh * 0.5 / sm.shape[0]))
        ok, sal = self.salobj.computeSaliency(sm)
        sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 6)
        sal = (sal - sal.min()) / (np.ptp(sal) + 1e-6)
        return ("sal", sal)

    def __call__(self, img):
        h, w = img.shape[:2]
        if self.i % self.every == 0:
            kind, val = self._detect(img)
            if kind == "box":
                b = np.array(val, np.float32)
                self.box = b if self.box is None else self.box * self.smooth + b * (1 - self.smooth)
                self.sal = None
            else:
                self.sal = val if self.sal is None or self.sal.shape != val.shape else \
                    self.sal * self.smooth + val * (1 - self.smooth)
        self.i += 1
        if self.sal is not None and self.box is None:
            m = cv2.resize(self.sal, (w, h), interpolation=cv2.INTER_LINEAR)
            return np.clip((m - 0.25) / 0.4, 0, 1).astype(np.float32)
        cx, cy, rx, ry = self.box
        return ellipse_mask(w, h, cx, cy, rx, ry, feather_px=0.04 * w)


def ellipse_mask(w, h, cx, cy, rx, ry, feather_px=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / w - cx) / max(rx, 1e-4)) ** 2 + ((yy / h - cy) / max(ry, 1e-4)) ** 2
    d = np.sqrt(d)
    f = max(feather_px / (max(rx, 1e-4) * w), 1e-3)
    return np.clip((1 - d) / f + 0.5, 0, 1).astype(np.float32)


class MaskSource:
    """Callable(frame_bgr, index) -> float32 mask in [0,1] or None (= full frame)."""

    def __init__(self, spec, w, h, fps=30.0):
        self.spec = (spec or "none").strip()
        self.w, self.h = w, h
        self.invert = False
        s = self.spec
        if s.startswith("invert:"):
            self.invert, s = True, s[7:]
        self.kind = "none"
        self.static = None
        self.reader = None
        self.auto = None
        if s in ("", "none", "full"):
            self.kind = "none"
        elif s.startswith("luma:"):
            parts = [float(x) for x in s[5:].split(":")]
            self.lo, self.hi = parts[0], parts[1]
            self.soft = parts[2] if len(parts) > 2 else 0.05
            self.kind = "luma"
        elif s.startswith("rect:"):
            body, *f = s[5:].split(":")
            x, y, rw, rh = [float(v) for v in body.split(",")]
            if max(x, y, rw, rh) <= 1.0:
                x, y, rw, rh = x * w, y * h, rw * w, rh * h
            m = np.zeros((h, w), np.float32)
            m[int(y):int(y + rh), int(x):int(x + rw)] = 1
            self.static = feather(m, float(f[0]) if f else 0)
            self.kind = "static"
        elif s.startswith("ellipse:"):
            body, *f = s[8:].split(":")
            cx, cy, rx, ry = [float(v) for v in body.split(",")]
            self.static = ellipse_mask(w, h, cx, cy, rx, ry, float(f[0]) if f else 0.03 * w)
            self.kind = "static"
        elif s.startswith("grad:"):
            parts = s[5:].split(":")
            side = parts[0]
            a = float(parts[1]) if len(parts) > 1 else 0.0
            b = float(parts[2]) if len(parts) > 2 else 1.0
            if side in ("top", "bottom"):
                ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
            else:
                ramp = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, 0)
            if side in ("top", "left"):
                ramp = 1 - ramp
            self.static = np.clip((ramp - a) / max(b - a, 1e-4), 0, 1)
            self.kind = "static"
        elif s.startswith("auto"):
            mode = s.split(":", 1)[1] if ":" in s else "auto"
            self.auto = AutoSubject(mode)
            self.kind = "auto"
        elif os.path.exists(s):
            if is_image(s):
                m = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
                self.static = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                self.kind = "static"
            else:
                self.reader = iter(FrameReader(s, size=(w, h), fps=fps, loop=True))
                self.kind = "video"
        else:
            raise ValueError("bad mask spec: " + spec)

    @property
    def is_full(self):
        return self.kind == "none" and not self.invert

    def __call__(self, frame, i=0):
        if self.kind == "none":
            m = None if not self.invert else np.zeros((self.h, self.w), np.float32)
            return m
        if self.kind == "luma":
            m = smoothband(luma(frame), self.lo, self.hi, self.soft)
        elif self.kind == "static":
            m = self.static
        elif self.kind == "video":
            fr = next(self.reader)
            m = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            m = self.auto(frame)
        return 1.0 - m if self.invert else m


def apply_mask(orig, fx, m):
    """Blend fx over orig by mask m (float HxW or None)."""
    if m is None:
        return fx
    if m.ndim == 2:
        m = m[..., None]
    o = orig.astype(np.float32)
    return np.clip(o + (fx.astype(np.float32) - o) * m, 0, 255).astype(np.uint8)


def mix(orig, fx, amount):
    """Global wet/dry."""
    if amount >= 0.999:
        return fx
    if amount <= 0.001:
        return orig
    return cv2.addWeighted(fx, float(amount), orig, float(1 - amount), 0)
