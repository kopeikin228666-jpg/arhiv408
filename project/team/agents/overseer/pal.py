"""overseer helpers: palette check + nearest-upscale + labels.  import from any check script."""
import sys, numpy as np, cv2
sys.path.insert(0, '/home/claude/voynich/look'); sys.path.insert(0, '/home/claude/voynich/edit')
import pixelize as P
PAL = np.array(P.PALETTE, np.uint8).reshape(-1, 3)
_KEYS = set((PAL[:, 0].astype(np.int64) << 16 | PAL[:, 1].astype(np.int64) << 8 | PAL[:, 2]).tolist())

def off_palette(img, mask=None):
    """-> (count of pixels whose BGR is not one of the 18 palette colours, up to 8 example colours)"""
    a = img[..., :3].reshape(-1, 3).astype(np.int64)
    if mask is not None:
        a = a[mask.reshape(-1)]
    k = a[:, 0] << 16 | a[:, 1] << 8 | a[:, 2]
    bad = ~np.isin(k, list(_KEYS))
    ex = np.unique(a[bad], axis=0)[:8].tolist()
    return int(bad.sum()), ex

def up(img, s=4):
    return cv2.resize(img[..., :3], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

def label(img, txt, col=(0, 255, 0)):
    img = img.copy(); cv2.putText(img, txt, (3, 12), 0, 0.4, col, 1); return img

def red_count(img):
    b, g, r = [img[..., i].astype(int) for i in range(3)]
    return int(((r - g) > 60).sum())
