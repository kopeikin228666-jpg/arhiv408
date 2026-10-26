"""Small fast image helpers (float32 / cv2 based)."""
import cv2
import numpy as np


def f32(img):
    return img.astype(np.float32) * (1.0 / 255.0)


def u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def luma_f(x):
    """x float BGR -> luma (Rec.709)"""
    return x[..., 0] * 0.0722 + x[..., 1] * 0.7152 + x[..., 2] * 0.2126


def big_blur(x, sigma, down=4):
    """Large gaussian blur computed at reduced resolution (fast halation/bloom)."""
    h, w = x.shape[:2]
    if down > 1:
        sm = cv2.resize(x, (max(1, w // down), max(1, h // down)), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), max(sigma / down, 0.5))
        return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(x, (0, 0), sigma)


def shift_img(img, dx, dy, border=cv2.BORDER_REFLECT):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


def scale_about(img, s, cx, cy, border=cv2.BORDER_REFLECT, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=border)


def hex_color(s, default=(255, 255, 255)):
    """'#aabbcc' / 'white' / 'green' / 'r,g,b' -> BGR tuple of ints."""
    named = {"white": (255, 255, 255), "green": (60, 255, 120), "black": (0, 0, 0),
             "red": (255, 40, 40), "teal": (40, 200, 190), "parchment": (235, 222, 190),
             "cyan": (60, 230, 255), "magenta": (255, 40, 220)}
    if s is None:
        r, g, b = default
    elif str(s).lower() in named:
        r, g, b = named[str(s).lower()]
    elif str(s).startswith("#") or len(str(s)) == 6:
        h = str(s).lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    else:
        r, g, b = [int(float(v)) for v in str(s).split(",")]
    return (b, g, r)


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, q)))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
