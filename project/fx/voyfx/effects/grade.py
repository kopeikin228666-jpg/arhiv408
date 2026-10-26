"""Master grade for the film + .cube export.

Look ("voynich" preset): ~60% desaturation, teal/green-blue shadows (hue ~175), lifted dirty
green-black blacks, low-contrast soft S-curve, highlights rolling into faded grey-yellow parchment,
-25% vignette, halation + bloom on highlights, organic multi-octave luma-weighted film grain,
digital chroma noise in the shadows.

The per-pixel colour part is a pure function of RGB, so it is exported 1:1 as a 33^3 .cube LUT.
Spatial parts (vignette, halation, grain, noise) are not representable in a LUT and are listed
in the README as "add in the NLE".
"""
import math

import cv2
import numpy as np

from ..base import Effect, register
from ..util import big_blur, luma_f, smoothstep

PRESETS = {
    "voynich": dict(desat=0.60, shadow_hue=175.0, shadow_tint=0.55, black_lift=0.055,
                    black_color="0.85,1.25,1.0", contrast=0.80, scurve=0.55,
                    highlight_color="1.0,0.94,0.76", white_point=0.90, highlight_tint=0.75,
                    vignette=0.25, halation=0.35, bloom=0.22, grain=0.75, grain_size=1.0,
                    shadow_noise=0.5, exposure=0.0),
    "clean": dict(desat=0.15, shadow_hue=175.0, shadow_tint=0.0, black_lift=0.015,
                  black_color="1,1,1", contrast=0.95, scurve=0.30,
                  highlight_color="1,1,1", white_point=0.97, highlight_tint=0.0,
                  vignette=0.10, halation=0.0, bloom=0.0, grain=0.0, grain_size=1.0,
                  shadow_noise=0.0, exposure=0.0),
}


def _vec(s):
    return np.array([float(v) for v in str(s).split(",")], np.float32)


def hue_to_bgr(h_deg):
    """Unit-luma colour direction for a hue (HSV, S=1, V=1) returned as BGR float."""
    h = (h_deg % 360) / 60.0
    c = 1.0
    x = c * (1 - abs(h % 2 - 1))
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h) % 6]
    return np.array([b, g, r], np.float32)


class ColorGrade:
    """Pure per-pixel transform on float32 BGR in [0,1]. Shared by the effect and the LUT export."""

    def __init__(self, desat, shadow_hue, shadow_tint, black_lift, black_color, contrast, scurve,
                 highlight_color, white_point, highlight_tint, exposure=0.0, **_):
        self.desat = float(desat)
        self.contrast = float(contrast)
        self.scurve = float(scurve)
        self.exposure = float(exposure)
        bc = _vec(black_color)[::-1]  # rgb -> bgr
        self.black = float(black_lift) * bc / max(bc.mean(), 1e-6)
        hc = _vec(highlight_color)[::-1]
        hc = hc / (0.0722 * hc[0] + 0.7152 * hc[1] + 0.2126 * hc[2])
        self.white = float(white_point) * (1 + (hc - 1) * float(highlight_tint))
        t = hue_to_bgr(shadow_hue)
        t = t / (0.0722 * t[0] + 0.7152 * t[1] + 0.2126 * t[2])   # luma-normalised tint
        self.tint = t
        self.shadow_tint = float(shadow_tint)

    def __call__(self, x, amount=1.0):
        """x: float32 HxWx3 (or Nx3) BGR 0..1 -> graded, blended with original by `amount`."""
        src = x
        if self.exposure:
            x = x * (2.0 ** self.exposure)
        L = luma_f(x)[..., None]
        # 1. desaturate
        x = L + (x - L) * (1.0 - self.desat)
        # 2. soft S-curve on each channel around mid grey, then reduce contrast
        xs = np.clip(x, 0, 1)
        s = xs * xs * (3 - 2 * xs)
        x = xs + (s - xs) * self.scurve
        x = 0.5 + (x - 0.5) * self.contrast
        # 3. shadow split-tone toward teal (luma-preserving)
        L = luma_f(x)[..., None]
        w = (1.0 - smoothstep(0.0, 0.55, L)) * self.shadow_tint
        x = x + (L * self.tint - L) * w * 0.55
        # 4. lift to dirty green-black, roll whites into parchment
        x = np.clip(x, 0, 1)
        x = self.black + (self.white - self.black) * x
        x = np.clip(x, 0, 1)
        if amount < 1.0:
            x = src + (x - src) * amount
        return x.astype(np.float32)


class Grain:
    """Organic film grain: 3 octaves of gaussian noise (fine/medium/clumps) with per-octave blur,
    mostly luma-correlated with slight chroma, amplitude shaped by luma (mid/shadow heavy)."""

    def __init__(self, w, h, size=1.0):
        self.w, self.h = w, h
        self.size = max(0.5, float(size))

    def field(self, rng):
        w, h, s = self.w, self.h, self.size
        acc = None
        # (downscale, blur sigma, weight)
        for d, sig, wt in ((1.0, 0.55, 0.55), (2.0, 0.7, 0.35), (4.0, 0.9, 0.22)):
            d = d * s
            ww, hh = max(8, int(w / d)), max(8, int(h / d))
            n = rng.standard_normal((hh, ww, 3), dtype=np.float32)
            # mostly monochrome grain: mix channels toward their mean
            m = n.mean(axis=2, keepdims=True)
            n = m * 0.8 + n * 0.35
            if sig > 0:
                n = cv2.GaussianBlur(n, (0, 0), sig)
            if (ww, hh) != (w, h):
                n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
            acc = n * wt if acc is None else acc + n * wt
        return acc

    @staticmethod
    def response(L):
        # film grain is most visible in mids/low-mids, less in deep black and in highlights
        return (0.35 + 1.6 * L * (1 - L) ** 1.3 + 0.15 * (1 - L))


@register
class Grade(Effect):
    name = "grade"
    help = "master grade (voynich / clean presets): desat, teal shadows, parchment highlights, halation, grain"
    args = [("preset", "voynich", "voynich | clean"),
            ("desat", "", "desaturation 0..1 (preset voynich=0.6)"),
            ("shadow_hue", "", "hue of shadow tint in degrees (175 = teal/green-blue)"),
            ("shadow_tint", "", "shadow tint strength 0..1"),
            ("black_lift", "", "black level lift 0..0.15"),
            ("contrast", "", "contrast multiplier (<1 = flatter)"),
            ("white_point", "", "highlight roll-off ceiling 0..1"),
            ("highlight_tint", "", "how much highlights go parchment-yellow 0..1"),
            ("exposure", "", "exposure in stops"),
            ("vignette", "", "vignette darkening 0..1 (0.25 = -25%)"),
            ("halation", "", "red-warm halation around highlights 0..1"),
            ("bloom", "", "soft wide bloom on highlights 0..1"),
            ("grain", "", "film grain strength 0..2 (animatable)"),
            ("grain_size", "", "grain scale (1 = 35mm-ish at 1080p, 1.5 = 16mm)"),
            ("shadow_noise", "", "digital chroma noise in the shadows 0..1 (animatable)")]
    animated = ("intensity", "grain", "shadow_noise", "vignette", "halation", "bloom")

    def setup(self, info):
        cfg = dict(PRESETS.get(self.kw["preset"], PRESETS["voynich"]))
        for k in cfg:
            v = self.kw.get(k, "")
            if v not in ("", None):
                cfg[k] = v
        self.cfg = cfg
        for k in self.animated:
            if k in cfg:
                from ..params import P
                self.p[k] = P(cfg[k], info)
        self.cg = ColorGrade(**cfg)
        w, h = info.w, info.h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # elliptical vignette following the 9:16 frame, soft (cos^4-like) fall-off
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.85 + ((yy - h / 2) / (h / 2)) ** 2 * 0.75)
        self.vig_shape = smoothstep(0.35, 1.25, r) ** 1.4
        self.grain = Grain(w, h, float(cfg["grain_size"]))

    def process(self, frame, i, t):
        amt = self.val("intensity", t)
        x = frame.astype(np.float32) * (1.0 / 255.0)
        hal = self.val("halation", t) * amt
        blm = self.val("bloom", t) * amt
        if hal > 0 or blm > 0:
            L = luma_f(x)
            hi = smoothstep(0.62, 1.0, L)[..., None] * x
            if hal > 0:
                h1 = big_blur(hi, 18, 4)
                # halation: red/orange scatter from film base
                x = x + h1 * np.array([0.10, 0.35, 1.0], np.float32) * hal
            if blm > 0:
                h2 = big_blur(hi, 70, 8)
                x = x + h2 * np.array([0.75, 0.92, 1.0], np.float32) * blm
            x = 1 - (1 - np.clip(x, 0, 2.0)) * (x <= 1) - (x > 1) * 0  # keep >1 for soft clip below
            x = np.where(x > 0.85, 0.85 + 0.15 * np.tanh((x - 0.85) / 0.15), x)
        x = self.cg(x, amt)
        v = self.val("vignette", t) * amt
        if v > 0:
            x = x * (1.0 - v * self.vig_shape[..., None])
        L = luma_f(x)[..., None]
        g = self.val("grain", t) * amt
        if g > 0:
            rng = self.rng.at(i)
            field = self.grain.field(rng)
            x = x + field * (0.045 * g) * Grain.response(L)
        sn = self.val("shadow_noise", t) * amt
        if sn > 0:
            rng = self.rng.at(i, 1)
            h, w = x.shape[:2]
            # blocky 2x2 chroma noise like a sensor pushed in the dark, only in shadows
            n = rng.standard_normal((h // 2, w // 2, 3), dtype=np.float32)
            n = n - n.mean(axis=2, keepdims=True) * 0.5
            n = cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
            sh = (1.0 - smoothstep(0.02, 0.28, L))
            x = x + n * 0.022 * sn * sh
        return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def export_cube(path, preset="voynich", size=33, overrides=None, title=None):
    cfg = dict(PRESETS[preset])
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v not in ("", None)})
    cg = ColorGrade(**cfg)
    r = np.linspace(0, 1, size, dtype=np.float32)
    # .cube order: R fastest, then G, then B
    B, G, R = np.meshgrid(r, r, r, indexing="ij")
    rgb = np.stack([R.ravel(), G.ravel(), B.ravel()], 1)
    bgr = rgb[:, ::-1].copy()
    out = cg(bgr[None, :, :])[0][:, ::-1]
    with open(path, "w") as f:
        f.write(f'TITLE "{title or "voynich_" + preset}"\n')
        f.write("# voyfx grade (colour part only: no vignette/grain/halation)\n")
        f.write("# input: Rec.709 display-referred video levels 0..1\n")
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    return path
