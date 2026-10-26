"""Pixel-art converter for the Voynich reel.
Photo/frame -> low-res grid -> curated dark palette with ordered dithering -> nearest upscale.
"""
import numpy as np, cv2

PALETTE_HEX = [
    '0a0c0b', '121715', '1b2220', '26302d', '35413d', '4a5752', '657269', '87938a',
    'aeb7ac', 'd9dccf',              # cold greys / bone
    '1d3a2c', '2e5641', '4f7a5b',    # swamp greens (water)
    '6e6046', 'a08b5f', 'cdb98a',    # parchment / ink
    '7d1b17', 'b8302a',              # blood red accent
]
PALETTE = np.array([[int(h[i:i + 2], 16) for i in (4, 2, 0)] for h in PALETTE_HEX], np.float32)  # BGR

BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16 - 0.5
BAYER8 = None

def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m.astype(np.float32) + 0.5) / m.size - 0.5


def quantize(small, palette=PALETTE, dither=0.10, matrix=None, contrast=1.15, lift=0.0):
    img = small.astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * contrast + 0.5 + lift, 0, 1)
    if dither > 0:
        m = bayer(8) if matrix is None else matrix
        h, w = img.shape[:2]
        th = np.tile(m, (h // m.shape[0] + 1, w // m.shape[1] + 1))[:h, :w]
        img = img + th[..., None] * dither
    flat = img.reshape(-1, 3) * 255.0
    # perceptual-ish distance (weight green more)
    wts = np.array([0.8, 1.2, 1.0], np.float32)
    d = (((flat[:, None, :] - palette[None, :, :]) * wts) ** 2).sum(-1)
    idx = d.argmin(1)
    return palette[idx].reshape(small.shape).astype(np.uint8)


def pixelize(img, grid_w, out_w=1080, out_h=1920, **kw):
    h, w = img.shape[:2]
    gh = int(round(grid_w * out_h / out_w))
    # cover-crop to output aspect
    tgt = out_w / out_h
    if w / h > tgt:
        nw = int(h * tgt); x0 = (w - nw) // 2; img = img[:, x0:x0 + nw]
    else:
        nh = int(w / tgt); y0 = (h - nh) // 2; img = img[y0:y0 + nh]
    small = cv2.resize(img, (grid_w, gh), interpolation=cv2.INTER_AREA)
    q = quantize(small, **kw)
    return cv2.resize(q, (out_w, out_h), interpolation=cv2.INTER_NEAREST), q


def one_bit(img, grid_w, out_w=1080, out_h=1920, ink=(14, 16, 15), paper=(214, 220, 207), dither=0.9):
    h, w = img.shape[:2]
    gh = int(round(grid_w * out_h / out_w))
    small = cv2.resize(img, (grid_w, gh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g = np.clip((g - 0.5) * 1.4 + 0.5, 0, 1)
    m = bayer(8)
    th = np.tile(m, (gh // 8 + 1, grid_w // 8 + 1))[:gh, :grid_w]
    b = (g + th * dither) > 0.5
    out = np.where(b[..., None], np.array(paper, np.uint8), np.array(ink, np.uint8))
    return cv2.resize(out.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
